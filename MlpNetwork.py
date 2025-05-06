import torch
import torch.nn as nn
import torch.nn.functional as F

C = 32
H = W = 32
roi_feat = torch.randn(1, C, H, W)  # (B, C, H, W)

class KernelPredictor(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # (B, C, 1, 1)
            nn.Flatten(),                # (B, C)
            nn.Linear(C, 1024),
            nn.ReLU(),
            nn.Linear(1024, C * C * 3 * 3)  # Output: dynamic kernel
        )
        self.C = C

    def forward(self, x):
        B = x.shape[0]
        weights = self.mlp(x)
        weights = weights.view(B, self.C, self.C, 3, 3)
        return weights

# 커널 추론 및 적용
predictor = KernelPredictor(C)
kernel = predictor(roi_feat)[0]  # shape: [C_out, C_in, 3, 3]
enhanced = F.conv2d(roi_feat, kernel, padding=1)