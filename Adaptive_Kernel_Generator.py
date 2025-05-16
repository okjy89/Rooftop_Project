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
    feature_map = roi_feature_map(img_rgb)

    kernel_predictor = KernelPredictor(32).cuda()
    kernel = kernel_predictor(feature_map)[0]  # shape: [C_out, C_in, 3, 3]
    enhanced = F.conv2d(feature_map.clone(), kernel, padding=1)

    # feature map을 이미지로 변환
    feature_map_img = feature_map_to_image(enhanced)
    feature_map_img.save("feature_map.jpg")
    feature_map_img.show()

    if feature_map is not None:
        print("ROI feature map size:", feature_map.shape)
    else:
        print("ROI feature map이 없습니다.")

def feature_map_to_image(feature_map):
    # feature map을 이미지로 변환하는 함수
    feature_map = feature_map[0][0]  # (B, C, H, W) -> (H, W)
    feature_map = feature_map - feature_map.min()  # 최소값 0으로 맞춤
    feature_map = feature_map / feature_map.max() + 1e-5  # 최대값 1로 맞춤
    feature_map = feature_map.detach().cpu().numpy()
    feature_map_uint8 = (feature_map * 255).astype(np.uint8)
    img = Image.fromarray(feature_map_uint8)
    return img



if __name__ == "__main__":
    main()