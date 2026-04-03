import torch
import torch.nn as nn


class AutoDriveHead(nn.Module):
    """
    Regression + classification head for AutoDrive.

    Inputs
    ------
    feature_prev : (B, C, H, W)  — P5 from backbone(image_prev)
    feature_curr : (B, C, H, W)  — P5 from backbone(image_curr)

    Compression path
    ----------------
    cat([feature_prev, feature_curr], dim=1)  → (B, 2C, H, W)
    AdaptiveAvgPool2d((1, 2))                   → (B, 2C, 1, 2)
    flatten                                   → (B, 2C*2) = (B, 1024) for C=256

    Shared trunk
    ------------
    FC1  : Linear(1024, 768) + ReLU
    FC2  : Linear(768,  512) + ReLU

    Task branches
    -------------
    distance_head  : Linear(512, 1) + Sigmoid
                     → d ∈ (0, 1);  distance_m = 200 * (1 - d)

    curvature_head : Linear(512, 1)  — raw regression (1/m)

    flag_head      : Linear(512, 2)  — raw logits for CIPO presence
                     (CrossEntropyLoss in training; argmax or softmax at inference)
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()

        # Two P5 maps concatenated on channel → pool to fixed 1024-d vector
        self.pool = nn.AdaptiveAvgPool2d((1, 2))
        flat_dim = in_channels * 2 * 2  # 256 * 4 = 1024

        self.fc1 = nn.Sequential(
            nn.Linear(flat_dim, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

        self.distance_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        self.curvature_head = nn.Linear(512, 1)

        self.flag_head = nn.Linear(512, 2)

    def forward(self, feature_prev: torch.Tensor, feature_curr: torch.Tensor):
        x = torch.cat([feature_prev, feature_curr], dim=1)
        x = self.pool(x)
        x = x.flatten(1)

        x = self.fc1(x)
        x = self.fc2(x)

        d_norm = self.distance_head(x)
        curvature = self.curvature_head(x)
        cipo_presence = self.flag_head(x)

        return d_norm, curvature, cipo_presence

    @staticmethod
    def to_distance_meters(d_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalised sigmoid output → metres.  distance_m = 200 * (1 - d)."""
        return 200.0 * (1.0 - d_norm)
