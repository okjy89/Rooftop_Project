# main.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated.*")

import os
import yaml

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "yolov5"))
from pathlib import Path
from utils.yolo_backbone import load_yolo_backbone  # 새로 만들었던 외부 함수


# ────────────────────────────────────────────────────────────────────────
# 1. 사용자 정의 모듈 import
# ────────────────────────────────────────────────────────────────────────

# VOCDataset: dataloader.py 안에 정의됨
from dataloader import VOCDataset

# MLPTrainer: Trainer_V2.py 안에 정의된 클래스 (내부에서 KernelPredictor를 MLPNetworkV4에서 import)
from Trainer_RGB_V1 import MLPTrainer


# ────────────────────────────────────────────────────────────────────────
# 2. 하이퍼파라미터 및 경로 설정
# ────────────────────────────────────────────────────────────────────────

# VOC2007 데이터셋 경로 (실제 경로로 바꿔주세요)
VOC_ROOT = r"C:\Users\DELL\Desktop\VOCdevkit\VOC2007"

# cnn.yaml 파일 경로 (feature extractor 구조 정의)
MODEL_YAML = r"C:\Users\DELL\Desktop\Project_Ver1\cnn.yaml"

# Ultralytics YOLOv8 pretrained weight 파일 경로(.pt)
YOLO_WEIGHTS = r"C:\Users\DELL\Desktop\Yolov8\yolov8-pytorch\runs\train\exp11\weights\best.pt"

# Backbone Yolo 가중치 경로; Feature Extractor
YOLOV5_BACKBONE_WEIGHTS = r"C:\Users\DELL\Desktop\Project_Ver1\yolov5\runs\train\yolov5n_voc2\weights\best.pt"
YOLOV5_CFG = r"C:\Users\DELL\Desktop\Project_Ver1\yolov5\models\yolov5n.yaml"

# 이미지 리사이즈 크기 (Height, Width)
INPUT_SIZE = (640, 640)

# DataLoader 관련 설정
BATCH_SIZE = 16
NUM_WORKERS = 4

# 학습 관련 설정
LR = 1e-2
EPOCHS = 50


# ────────────────────────────────────────────────────────────────────────
# 3. 메인 함수 정의
# ────────────────────────────────────────────────────────────────────────

def main():
    # ------------------------------------------------------------------------
    # 3.1. Feature Extractor(backbone) 생성 (YOLOv5n 백본 사용)
    # ------------------------------------------------------------------------
    # (a) YOLOv5n 백본 로드
    backbone = load_yolo_backbone(
        model_cfg=YOLOV5_CFG,
        model_weights=YOLOV5_BACKBONE_WEIGHTS,
        ch=3  # RGB 입력 채널
    )
    backbone.eval()
    #Backbone Parameter 추적 끄기(V2에서 Updated)
    for p in backbone.parameters():
        p.requires_grad = False

    # (b) 더미 텐서로 feature 채널 수 확인
    with torch.no_grad():
        dummy = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1])
        outputs  = backbone(dummy)
        feat_out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    feature_channels = feat_out.shape[1]
    print(f"[Info] YOLOv5 백본 로드 완료. Feature 채널 수: {feature_channels}")

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
