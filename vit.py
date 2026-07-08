
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTDeepfake(nn.Module):
   
    def __init__(self, pretrained=True, freeze_backbone=True, dropout=0.3):
        super(ViTDeepfake, self).__init__()

        # Load pretrained ViT-B/16
        if pretrained:
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            print(" Loaded pretrained ViT-B/16 (ImageNet)")
        else:
            self.backbone = vit_b_16(weights=None)
            print(" ViT-B/16 initialized with random weights")

        # ViT-B/16 hidden size is 768
        hidden_size = self.backbone.heads.head.in_features

        # Replace classification head with binary classifier
        self.backbone.heads = nn.Identity()

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
            print(" ViT backbone frozen (only classifier will train)")
        else:
            print(" Full ViT model will train")

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_last_n_blocks=4):
        
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last N encoder blocks
        encoder_blocks = list(self.backbone.encoder.layers.children())
        for block in encoder_blocks[-unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # Always unfreeze the encoder norm
        for param in self.backbone.encoder.ln.parameters():
            param.requires_grad = True

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f" Unfroze last {unfreeze_last_n_blocks} ViT blocks")
        print(f"   Trainable: {trainable:,} / {total:,} params")

    def unfreeze_all(self):
        """Unfreeze entire model for full fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in self.parameters())
        print(f" All ViT params unfrozen: {total:,} params")

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    print("Testing ViTDeepfake model...")

    # Create model
    model = ViTDeepfake(pretrained=True, freeze_backbone=True)

    total, trainable = model.get_param_count()
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    print(" Forward pass OK")