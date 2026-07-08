

import torch
import torch.nn as nn
import timm


class EfficientNetB4Deepfake(nn.Module):
    
    
    def __init__(self, pretrained=True, freeze_backbone=True):
        super(EfficientNetB4Deepfake, self).__init__()
        
        # Load pretrained EfficientNet-B4 from timm
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0  # Remove classifier, get features only
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        # EfficientNet-B4 outputs 1792 features
        feature_dim = 1792
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, x):
      
        # Extract features from backbone
        features = self.backbone(x)  # (B, 1792)
        
        # Classify
        output = self.classifier(features)  # (B, 1)
        
        return output
    
    def unfreeze_backbone(self, num_layers=-1):
       
        if num_layers == -1:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Unfroze all backbone layers")
        else:
            # Unfreeze last N layers
            layers = list(self.backbone.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f" Unfroze last {num_layers} backbone layers")


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    print("="*60)
    print("EfficientNet-B4 Deepfake Detector")
    print("="*60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Create model
    model = EfficientNetB4Deepfake(pretrained=True, freeze_backbone=True)
    model = model.to(device)
    
    print("\nModel created!")
    print(f"Backbone frozen: {not next(model.backbone.parameters()).requires_grad}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
    
    # Test unfreezing
    print("\nTesting unfreeze...")
    model.unfreeze_backbone(num_layers=3)
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable after unfreeze: {trainable_after:,}")
    
    print("\n Model ready for training!")