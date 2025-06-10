import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init


def get_gaussian_kernel(k=3, sigma=1.0):
    """
    2D Gaussian 커널 생성 함수. 반환 형태: [1, 1, k, k]
    """
    ax = torch.arange(k).float() - (k - 1) / 2.
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, k, k)  # shape: [1, 1, k, k]


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class DeepKernelPredictor(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=3,              # RGB용으로 3개 커널
                 kernel_size=3,
                 hidden_dims=[256, 128, 64, 32],
                 dropout_p=0.2,
                 se_reduction=16):
        super().__init__()
        self.k = kernel_size
        self.out_channels = out_channels

        self.se = SELayer(in_channels, reduction=se_reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)

        layers = []
        prev_dim = in_channels
        for i, h_dim in enumerate(hidden_dims):
            layers += [
                nn.Flatten() if prev_dim == in_channels else nn.Identity(),
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p)
            ]
            prev_dim = h_dim

        # 마지막 fc: RGB 각 채널 × (k×k) 예측
        layers += [
            nn.Linear(prev_dim, out_channels * kernel_size * kernel_size)
        ]
        self.fc = nn.Sequential(*layers)

        #self._init_with_gaussian()

    def _init_with_gaussian(self, sigma=1.0):
        """
        Gaussian 초기값으로 bias 초기화
        """
        gaussian_kernel = get_gaussian_kernel(self.k, sigma).view(-1)  # [k*k]
        full_bias = gaussian_kernel.repeat(self.out_channels)          # [C×k*k]
        last_linear = self.fc[-1]
        assert isinstance(last_linear, nn.Linear)
        with torch.no_grad():
            last_linear.bias.copy_(full_bias)
            init.kaiming_normal_(last_linear.weight)

    def forward(self, feat):
        """
        feat: [B, C_feat, H, W]
        output: [B, 3, 3, 3] → RGB 각 채널별 커널
        """
        feat = self.se(feat)
        x = self.pool(feat).view(feat.size(0), -1)  # [B, C_feat]

        k_flat = self.fc(x)                         # [B, 3×9]
        k_flat = k_flat.view(-1, self.out_channels, self.k * self.k)  # [B, 3, 9]

        # 정규화: 커널 합이 1 되도록
        k_norm = k_flat / (k_flat.sum(dim=-1, keepdim=True) + 1e-8)
        kernels = k_norm.view(-1, self.out_channels, self.k, self.k)  # [B, 3, 3, 3]

        return kernels


import torch
import torch.nn as nn

class FixedIdentityKernel(nn.Module):
    def __init__(self, out_channels=3, kernel_size=3):
        super().__init__()
        self.k = kernel_size
        self.out_channels = out_channels

        # [out_channels, k, k] → R, G, B 각 채널용
        kernel = torch.zeros(out_channels, kernel_size, kernel_size)

        center = kernel_size // 2
        for i in range(out_channels):
            kernel[i, center, center] = 1.0  # 중심만 1

        # register buffer so it's part of the module but not trainable
        self.register_buffer("fixed_kernel", kernel.unsqueeze(0))  # [1, 3, k, k]

    def forward(self, x):
        B = x.size(0)
        # [1, 3, k, k] → [B, 3, k, k] by repeat
        return self.fixed_kernel.repeat(B, 1, 1, 1)
