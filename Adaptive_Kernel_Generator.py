import cv2
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image

from get_roi_featuremap import roi_feature_map
from MlpNetwork import KernelPredictor

def main():
    # 이미지 읽기
    img_path = 'download.jpeg' 
    path = os.path.join(os.path.dirname(__file__), "download.jpeg")
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        print(f"이미지 파일을 찾을 수 없습니다: {img_path}")
        return

    # get roi_feature_map
    feature_map, roi_img = roi_feature_map(img_rgb)

    kernel_predictor = KernelPredictor(32).cuda()
    kernel = kernel_predictor(feature_map)[0]  # shape: [C_out, C_in, 3, 3]

    if feature_map is not None:
        print("ROI feature map size:", feature_map.shape)
    else:
        print("ROI feature map이 없습니다.")

if __name__ == "__main__":
    main()