import torch
import torch.nn as nn
import torch.nn.functional as F

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
    



class KernelPredictor2(nn.Module):
    def __init__(self, Cin):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # [B, C, 1, 1]
            nn.Flatten(),              # [B, C]
            nn.Linear(Cin, 128),
            nn.ReLU(),
            nn.Linear(128, 9),         # → 3x3 = 9개 weight
        )

    def forward(self, x):
        kernel = self.mlp(x)           # [B, 9]
        kernel = kernel.view(-1, 1, 3, 3)  # → [B, 1, 3, 3]
        return kernel