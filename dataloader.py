# dataloader.py

import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class VOCDataset(Dataset):
    """
    VOC2007/images 폴더에서 이미지만 읽어오는 Dataset.
    라벨(.txt)은 사용하지 않으므로 더미(target)을 반환합니다.
    """
    def __init__(self, root_dir, img_folder="images", label_folder="labels", transform=None):
        super().__init__()
        self.img_dir = os.path.join(root_dir, img_folder)
        self.lbl_dir = os.path.join(root_dir, label_folder)
        self.transform = transform

        # 이미지 파일(.jpg, .png 등) 목록만 따로 모아서 정렬
        self.image_names = [
            fname for fname in os.listdir(self.img_dir)
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        self.image_names.sort()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # PIL로 읽고 RGB로 변환
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 라벨은 실제로 사용하지 않으므로 더미 값 반환
        dummy_target = torch.tensor(0)
        return image, dummy_target
