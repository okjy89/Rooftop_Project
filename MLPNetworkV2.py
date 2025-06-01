import torch
import torch.nn as nn

class KernelPredictor(nn.Module):
    """
    Feature map으로부터 배치별 3×3 blur kernel을 동적으로 생성합니다.
    입력: feat   (Tensor[B, C_feat, H, W])
    출력: kernels (Tensor[B, 1, k, k])
    """
    def __init__(self, in_channels, kernel_size=3, hidden_dim=128):
        super().__init__()
        self.k = kernel_size
        self.pool = nn.AdaptiveAvgPool2d(1)   # → [B, C_feat, 1, 1]
        self.fc = nn.Sequential(
            nn.Flatten(),                     # → [B, C_feat]
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, kernel_size * kernel_size)  # → [B, k*k]
        )

    def forward(self, feat):
        # feat: [B, C_feat, H, W]
        x = self.pool(feat)                  # [B, C_feat, 1, 1]
        k_flat = self.fc(x)                  # [B, k*k]
        kernels = k_flat.view(-1, 1, self.k, self.k)
        return kernels                       # [B, 1, k, k]
