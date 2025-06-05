# main.py

import os
import yaml

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image

# ────────────────────────────────────────────────────────────────────────
# 1. 사용자 정의 모듈 import
# ────────────────────────────────────────────────────────────────────────

# VOCDataset: dataloader.py 안에 정의됨
from dataloader import VOCDataset

# parse_model: cnn.py 안에 정의된 함수
from cnn import parse_model

# MLPTrainer: Trainer_V2.py 안에 정의된 클래스 (내부에서 KernelPredictor를 MLPNetworkV2에서 import)
from Trainer_V4 import MLPTrainer


# ────────────────────────────────────────────────────────────────────────
# 2. 하이퍼파라미터 및 경로 설정
# ────────────────────────────────────────────────────────────────────────

# VOC2007 데이터셋 경로 (실제 경로로 바꿔주세요)
VOC_ROOT = r"/home/okjy89/dataset/pascal/train/VOCdevkit/VOC2007"

# # cnn.yaml 파일 경로 (feature extractor 구조 정의)
# MODEL_YAML = r"~/cnn.yaml"
MODEL_YAML = os.path.join(os.getcwd(), "cnn.yaml")
YOLO_WEIGHTS = os.path.join(os.getcwd(), "best.pt")

# Ultralytics YOLOv8 pretrained weight 파일 경로(.pt)
# YOLO_WEIGHTS = r"~/best.pt"

# 이미지 리사이즈 크기 (Height, Width)
INPUT_SIZE = (640, 640)

# DataLoader 관련 설정
BATCH_SIZE = 16
NUM_WORKERS = 4

# 학습 관련 설정
LR = 1e-3
EPOCHS = 20


# ────────────────────────────────────────────────────────────────────────
# 3. 메인 함수 정의
# ────────────────────────────────────────────────────────────────────────

def main():
    # ------------------------------------------------------------------------
    # 3.1. Feature Extractor(backbone) 생성
    # ------------------------------------------------------------------------
    # (a) cnn.yaml을 열어서 dict로 로드
    with open(MODEL_YAML, "r") as f:
        yaml_dict = yaml.safe_load(f)

    # (b) parse_model 호출: ch=[3] → RGB 입력
    backbone = parse_model(yaml_dict, ch=[3])
    backbone.eval()

    # (c) 더미 텐서를 흘려서 feature 채널 수 추론
    with torch.no_grad():
        dummy = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1])
        feat_out = backbone(dummy)  # [1, C_feat, Hf, Wf]
    feature_channels = feat_out.shape[1]
    print(f"[Info] Backbone 생성 완료. Feature 채널 수: {feature_channels}")

    # ------------------------------------------------------------------------
    # 3.2. 이미지 전처리(transform) 정의
    # ------------------------------------------------------------------------
    transform = T.Compose([
        T.Resize(INPUT_SIZE, interpolation=Image.BILINEAR),
        T.ToTensor(),
        # 필요 시 Normalize (ImageNet 기준 등)
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #             std =[0.229, 0.224, 0.225]),
    ])

    # ------------------------------------------------------------------------
    # 3.3. VOCDataset 및 DataLoader 생성
    # ------------------------------------------------------------------------
    dataset = VOCDataset(
        root_dir=VOC_ROOT,
        img_folder="images",
        label_folder="labels",
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"[Info] DataLoader 생성 완료. 전체 이미지 수: {len(dataset)}, 배치 크기: {BATCH_SIZE}")

    # ------------------------------------------------------------------------
    # 3.4. MLPTrainer 생성
    # ------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = MLPTrainer(
        backbone=backbone,
        feature_channels=feature_channels,
        yolo_weights=YOLO_WEIGHTS,
        lr=LR,
        device=device
    )
    print(f"[Info] MLPTrainer 생성 완료. Device: {device}")

    # ------------------------------------------------------------------------
    # 3.5. 학습(Fit) 실행
    # ------------------------------------------------------------------------
    print("[Info] 학습 시작...")
    trainer.fit(dataloader, epochs=EPOCHS)
    print("[Info] 학습 완료.")


# -----------------------------------------------------------------------------
# 4. 스크립트 실행 지점
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
