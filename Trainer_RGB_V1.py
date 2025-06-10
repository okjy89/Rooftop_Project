import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ultralytics import YOLO
import os
from torchvision.utils import save_image

from MLPNetwork_RGB_V1 import DeepKernelPredictor

import matplotlib.pyplot as plt
import numpy as np

def visualize_kernels(kernels, epoch=0, batch_idx=0, save_dir="./debug_images/kernels"):
    """
    kernels: torch.Tensor [B, 3, 3, 3]  → RGB별 커널
    """
    os.makedirs(save_dir, exist_ok=True)

    k = kernels[0].detach().cpu().numpy()  # 첫 번째 이미지 기준: [3, 3, 3]

    titles = ['R Kernel', 'G Kernel', 'B Kernel']
    plt.figure(figsize=(9, 3))

    for i in range(3):  # R, G, B 각각 시각화
        plt.subplot(1, 3, i+1)
        plt.imshow(k[i], cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(titles[i])
        plt.axis('off')

    fname = f"kernel_epoch{epoch}_batch{batch_idx}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()


class MLPTrainer:
    def __init__(self, backbone, feature_channels,
                 yolo_weights='master_yolov8.pt',
                 lr=1e-3, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = backbone.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.kernel_pred = DeepKernelPredictor(feature_channels, kernel_size=3).to(self.device)

        self.yolo = YOLO(yolo_weights)
        self.yolo.model.to(self.device)
        for p in self.yolo.model.parameters():
            p.requires_grad = False
        self.yolo.model.eval()

        self.optimizer = optim.Adam(self.kernel_pred.parameters(), lr=lr)

    def train_step(self, images, _targets=None, epoch=0, batch_idx=0):
        self.model.train()
        self.kernel_pred.train()
        self.yolo.model.eval()

        imgs = images.to(self.device)  # [B, 3, H, W]
        feats = self.model(imgs)       # [B, C_feat, Hf, Wf]
        kernels = self.kernel_pred(feats)  # [B, 3, 3, 3]
        visualize_kernels(kernels, epoch=epoch, batch_idx=batch_idx)
        pad = kernels.size(-1) // 2

        B, C, H, W = imgs.shape
        k = kernels.size(-1)

        imgs_reshaped = imgs.view(1, B * C, H, W)  # [1, 48, H, W]
        kernels_reshaped = kernels.view(B * C, 1, k, k)         # [B*3, 1, 3, 3]

        # ✅ 각 채널에 맞는 커널을 적용하기 위해 groups=B*C 추가
        blurred = F.conv2d(imgs_reshaped, kernels_reshaped, padding=pad, groups=B * C)  # ✅

        enhanced = blurred.view(B, C, H, W) 

        raw_preds = self.yolo.model(enhanced)
        preds_list = raw_preds if isinstance(raw_preds, (list, tuple)) else [raw_preds]

        conf_logits_list = []
        for p in preds_list:
            if isinstance(p, (list, tuple)):
                for scale_tensor in p:
                    conf = scale_tensor[..., 4]
                    conf_flat = conf.flatten(1)
                    conf_logits_list.append(conf_flat)
            else:
                conf = p[..., 4]
                conf_flat = conf.flatten(1)
                conf_logits_list.append(conf_flat)

        all_conf_logits = torch.cat(conf_logits_list, dim=1)
        raw_conf = torch.sigmoid(all_conf_logits)

        conf_per_image = raw_conf.mean(dim=1)
        batch_mean_conf = conf_per_image.mean()
        loss = -10 * batch_mean_conf

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        save_root = "./debug_images"
        raw_dir = os.path.join(save_root, "raw")
        filtered_dir = os.path.join(save_root, "filtered")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(filtered_dir, exist_ok=True)

        raw_img = imgs[0].detach().cpu()
        if((epoch == 30) or (epoch == 49)):
            save_image(raw_img, f"{raw_dir}/raw_epoch{epoch}_batch{batch_idx}.png")

        if((epoch == 30) or (epoch == 49)):
            filtered_img = enhanced[0].detach().cpu()
            save_image(filtered_img, f"{filtered_dir}/filtered_epoch{epoch}_batch{batch_idx}.png")

        return loss.item(), batch_mean_conf.item()

    def fit(self, dataloader, epochs=10):
        for epoch in range(1, epochs + 1):
            total_loss, total_conf = 0.0, 0.0
            num_batches = len(dataloader)

            for batch_idx, (imgs, _) in enumerate(dataloader, start=1):
                loss, conf = self.train_step(imgs, epoch=epoch, batch_idx=batch_idx)
                total_loss += loss
                total_conf += conf

                pct = 100.0 * batch_idx / num_batches
                print(
                    f"\rEpoch [{epoch}/{epochs}] "
                    f"Batch [{batch_idx}/{num_batches}] "
                    f"{pct:5.1f}%  "
                    f"Loss: {loss:.4f}  "
                    f"MeanRawConf: {conf:.4f}",
                    end=""
                )

            avg_loss = total_loss / num_batches
            avg_conf = total_conf / num_batches
            print(
                f"\n>> Ep {epoch}/{epochs} 완료 — "
                f"AvgLoss: {avg_loss:.4f}, "
                f"AvgMeanRawConf: {avg_conf:.4f}"
            )

        print("Training 완료.")
