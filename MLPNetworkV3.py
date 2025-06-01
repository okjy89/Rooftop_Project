import torch
import torch.nn as nn

# Hidden Layer를 4개로 늘림
# 기존 MLP는 Network가 얕아 Gradient 학습 낫 굳

class DeepKernelPredictor(nn.Module):
    """
    4개의 히든 레이어로 구성된 깊은 MLP를 사용해서
    배치별 3×3 Kernel을 예측합니다.
    Dropout을 각 히든 레이어 뒤에 추가하여 과적합을 방지합니다.
    Residual이랑 Squeeze
    """
    def __init__(self, in_channels, kernel_size=3, hidden_dims=[256, 128, 64, 32], dropout_p=0.2):
        super().__init__()
        self.k = kernel_size
        self.pool = nn.AdaptiveAvgPool2d(1)  # [B, C_feat, 1, 1]

        layers = []
        prev_dim = in_channels
        for i, h_dim in enumerate(hidden_dims):
            layers += [
                # 첫 번째 레이어에서는 Flatten을 적용하여 [B, C_feat] 형태로 변환
                nn.Flatten() if prev_dim == in_channels else nn.Identity(),
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),          # 정규화: 학습 안정화
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p)         # Dropout 추가
            ]
            prev_dim = h_dim

        # 마지막으로 k*k 개수만큼 매핑
        layers += [
            nn.Linear(prev_dim, kernel_size * kernel_size)
        ]
        self.fc = nn.Sequential(*layers)

    def forward(self, feat):
        """
        feat: [B, C_feat, H, W]
        """
        x = self.pool(feat)                  # [B, C_feat, 1, 1]
        x = x.view(x.size(0), -1)            # [B, C_feat]
        k_flat = self.fc(x)                  # [B, k*k]
        kernels = k_flat.view(-1, 1, self.k, self.k)  # [B, 1, k, k]
        return kernels
