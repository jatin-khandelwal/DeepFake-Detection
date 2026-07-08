
import torch
import torch.nn as nn
import timm



class XceptionDeepfake(nn.Module):
    

    def __init__(self, pretrained=True, freeze_backbone=True, dropout=0.5):
        super(XceptionDeepfake, self).__init__()

        # Load pretrained Xception via timm
        self.backbone = timm.create_model(
            'xception',
            pretrained=pretrained,
            num_classes=0,        # Remove classifier head
            global_pool='avg'     # Global average pooling
        )

        if pretrained:
            print(" Loaded pretrained Xception (ImageNet)")
        else:
            print(" Xception initialized with random weights")

        # Xception feature size is 2048
        in_features = self.backbone.num_features  # 2048

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
            print(" Xception backbone frozen (only classifier will train)")
        else:
            print(" Full Xception model will train")

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_last_n_blocks=2):
       
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze exit_flow always (final block)
        for param in self.backbone.exit_flow.parameters():
            param.requires_grad = True

        # Optionally unfreeze last N middle_flow blocks
        if unfreeze_last_n_blocks > 1:
            middle_blocks = list(self.backbone.middle_flow.children())
            for block in middle_blocks[-(unfreeze_last_n_blocks - 1):]:
                for param in block.parameters():
                    param.requires_grad = True

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f" Unfroze last {unfreeze_last_n_blocks} Xception blocks")
        print(f"   Trainable: {trainable:,} / {total:,} params")

    def unfreeze_all(self):
        """Unfreeze entire model for full fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in self.parameters())
        print(f" All Xception params unfrozen: {total:,} params")

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    print("Testing XceptionDeepfake model...")

    model = XceptionDeepfake(pretrained=True, freeze_backbone=True)

    total, trainable = model.get_param_count()
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    # Test forward pass with 299x299 (recommended for Xception)
    x = torch.randn(4, 3, 299, 299)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    print(" Forward pass OK")

    # Also works with 224x224
    x2 = torch.randn(4, 3, 224, 224)
    out2 = model(x2)
    print(f"\n224x224 also works: {out2.shape} ")