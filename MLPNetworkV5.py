import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init

# Gaussian으로 초기화
def get_gaussian_kernel(k=3, sigma=1.0):
    """
    2D Gaussian 커널 생성 함수. 반환 형태: [1, 1, k, k]
    """
    ax = torch.arange(k).float() - (k - 1) / 2.
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, k, k)

#CNN channel의 중요도를 판단하기 위해 S-E 추가가

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation 블록:
    입력 피처맵을 채널별로 가중치 조정한 뒤 반환합니다.
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)     # [B, C, 1, 1]
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        # 1) Squeeze: 채널별 평균값 계산 → [B, C]
        y = self.avg_pool(x).view(b, c)
        # 2) Excitation: 채널 축소 → 활성화 → 채널 복원 → 시그모이드
        y = self.fc1(y)       # [B, C//r]
        y = self.relu(y)
        y = self.fc2(y)       # [B, C]
        y = self.sigmoid(y).view(b, c, 1, 1)  # [B, C, 1, 1]
        # 3) Scale: 원본 피처맵에 채널별 가중치 곱하기
        return x * y         # [B, C, H, W]


class DeepKernelPredictor(nn.Module):
    """
    4개의 히든 레이어로 구성된 깊은 MLP를 사용해서
    배치별 3×3 Kernel을 예측합니다.
    Dropout을 각 히든 레이어 뒤에 추가하여 과적합을 방지합니다.
    Squeeze-and-Excitation(SE) 블록을 CNN 피처맵 단계에 추가합니다.
    """
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 hidden_dims=[256, 128, 64, 32],
                 dropout_p=0.2,
                 se_reduction=16):
        super().__init__()
        self.k = kernel_size

        # 1) SE 블록: 채널별 중요도 학습
        self.se = SELayer(in_channels, reduction=se_reduction)

        # 2) Adaptive Average Pooling: [B, C_feat, H, W] → [B, C_feat, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 3) MLP: [B, C_feat] → 히든 레이어 여러 개 → [B, k*k]
        layers = []
        prev_dim = in_channels
        for i, h_dim in enumerate(hidden_dims):
            layers += [
                # 첫 번째 레이어에서는 Flatten을 적용하여 [B, C_feat] 형태로 변환
                nn.Flatten() if prev_dim == in_channels else nn.Identity(),
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),       # 정규화: 학습 안정화
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p)      # Dropout 추가
            ]
            prev_dim = h_dim

        # 마지막으로 k*k 개수만큼 매핑
        layers += [
            nn.Linear(prev_dim, kernel_size * kernel_size)
        ]
        self.fc = nn.Sequential(*layers)

        # Gaussian 초기화
        self._init_with_gaussian()

    def _init_with_gaussian(self, sigma=1.0):
        """
        마지막 Linear layer의 bias를 Gaussian 커널로 초기화
        """
        gaussian_kernel = get_gaussian_kernel(self.k, sigma).view(-1)  # [k*k]
        # 마지막 Linear layer
        last_linear = self.fc[-1]
        assert isinstance(last_linear, nn.Linear), "마지막 계층은 Linear여야 합니다."
        with torch.no_grad():
            last_linear.bias.copy_(gaussian_kernel)
            init.kaiming_normal_(last_linear.weight)  # weight는 일반적인 초기화


    def forward(self, feat):
        """
        feat: [B, C_feat, H, W]
        """
        # 1) SE 블록: 채널별 가중치로 조정
        feat = self.se(feat)               # [B, C_feat, H, W]

        # 2) AdaptiveAvgPool → [B, C_feat, 1, 1] → view → [B, C_feat]
        x = self.pool(feat)                # [B, C_feat, 1, 1]
        x = x.view(x.size(0), -1)          # [B, C_feat]

        # 3) MLP를 통해 3×3 커널 예측 → [B, k*k] → reshape → [B, 1, k, k]
        k_flat = self.fc(x)                # [B, k*k]
        kernels = k_flat.view(-1, 1, self.k, self.k)  # [B, 1, k, k]
        return kernels
