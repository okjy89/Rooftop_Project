import torch
import cv2
import time
import numpy as np

def roi_feature_map(img):
    # 모델 로드
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model.eval()  # 평가 모드로 설정

    # 이미지 크기 변경 (320x320)
    img_resized = cv2.resize(img, (320, 320))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)


    # feature map 추출 ready
    feature_maps = []

    def hook_fn(module, input, output):
        feature_maps.append(output)
    # 모델의 feature map을 추출하기 위한 hook 설정
        # model.model.model.model[0] ~ [9] : cnn
        # model.model.model.model[10] ~ : fcn
    model.model.model.model[1].register_forward_hook(hook_fn)

    # 추론
    with torch.no_grad():
        results = model(img_resized)

    # RoI 추출
    pred = results.pred[0]  # (x1, y1, x2, y2, objectness, class_score)
    rois = []
    conf_threshold = 0.3 

    for det in pred:
        x1, y1, x2, y2, obj, cls = det[:6]
        conf = obj  

        if conf > conf_threshold:
            roi = (int(x1), int(y1), int(x2), int(y2))  # (x1, y1, x2, y2) 좌표
            rois.append(roi)
            print(f"RoI 좌표: ({roi[0]}, {roi[1]}) ~ ({roi[2]}, {roi[3]}), 신뢰도: {conf:.2f}, object: {det[5]}")

    # 모든 RoI를 포함하는 큰 바운딩 박스 계산
    if rois:
        min_x1 = min([roi[0] for roi in rois])
        min_y1 = min([roi[1] for roi in rois])
        max_x2 = max([roi[2] for roi in rois])
        max_y2 = max([roi[3] for roi in rois])
        large_roi = (min_x1, min_y1, max_x2, max_y2)
        print(f"Large bounding box: {large_roi}")

        # Feature map 가져오기
        feat_map = feature_maps[0]  # (B, C, H, W) 

        # 이미지와 feature map 사이의 축소 비율
        B, C, H, W = feat_map.shape
        img_h, img_w = img_resized.shape[:2]
        stride_x = img_w / W
        stride_y = img_h / H

        # 원본 좌표를 feature map 좌표로
        fx1 = int(large_roi[0] / stride_x)
        fy1 = int(large_roi[1] / stride_y)
        fx2 = int(large_roi[2] / stride_x)
        fy2 = int(large_roi[3] / stride_y)

        # RoI에 해당하는 feature map만 crop
        roi_feat_map = feat_map[0, :, fy1:fy2, fx1:fx2]  # (C, h, w)
        roi_feat_map = roi_feat_map.unsqueeze(0) # (B, C, h, w)
        print("RoI feature map shape:", roi_feat_map.shape)

         # ROI 내부 영역 crop 후 저장
        roi_img = img[large_roi[1]:large_roi[3], large_roi[0]:large_roi[2]]

        return roi_feat_map, roi_img
    else:
        print("No RoI detected.")
        return None