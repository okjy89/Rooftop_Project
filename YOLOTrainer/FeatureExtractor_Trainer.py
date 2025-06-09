import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn import parse_model  # 너가 만든 cnn.py에서 import

# ───────────────────────
# 1. 설정값
# ───────────────────────
VOC_ROOT = r"C:\Users\DELL\Desktop\VOCdevkit"  # VOC2007 폴더
MODEL_YAML = r"C:\Users\DELL\Desktop\Project_Ver1\cnn.yaml"
NUM_CLASSES = 20
INPUT_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
SAVE_PATH = r"C:\Users\DELL\Desktop\Project_Ver1\CNN_weights"

# ───────────────────────
# 2. CNN Backbone + Classifier
# ───────────────────────
class CNNClassifier(nn.Module):
    def __init__(self, backbone, feature_dim=1024, num_classes=20):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)         # [B, C, H, W]
        x = self.pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        x = self.fc(x)               # [B, num_classes]
        return x

# ───────────────────────
# 3. 학습 함수
# ───────────────────────
def train():
    # 모델 로드
    with open(MODEL_YAML, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    backbone = parse_model(model_cfg, ch=[3])
    model = CNNClassifier(backbone, feature_dim=1024, num_classes=NUM_CLASSES).cuda()

    # 데이터
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
    ])
    train_data = datasets.VOCDetection(
        root=VOC_ROOT, image_set='train', year='2007', download=False, transform=transform,
        target_transform=lambda t: int(t['annotation']['object'][0]['name'] in VOC_CLASSES)
    )
    # 위 코드에서 VOC_CLASSES를 쓰고 싶으면 미리 리스트 선언해줘야 함

    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 손실함수 & 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 학습 루프
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.cuda(), labels.cuda()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = outputs.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}, Acc: {acc:.2f}%")

    # 저장
    torch.save(model.backbone.state_dict(), SAVE_PATH)
    print(f"[✓] 모델 저장 완료: {SAVE_PATH}")

if __name__ == "__main__":
    VOC_CLASSES = [  # VOC 20 클래스
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    train()
