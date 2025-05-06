import torch
import cv2
import time

# 0. 시간 측정 시작
start_time = time.time()

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# 이미지 읽기
img = cv2.imread('/home/victus04/intern_ws/yolo/download.jpeg')  # 너가 입력하고 싶은 이미지 경로
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 추론
results = model(img_rgb)

# 6. RoI 추출
pred = results.pred[0]  # (x1, y1, x2, y2, objectness, class_score)
rois = []

conf_threshold = 0.3  # 신뢰도 임계값 onnx랑 맞춤

for det in pred:
    x1, y1, x2, y2, obj, cls = det[:6]
    conf = obj  # onnx처럼 object_conf * class_conf

    if conf > conf_threshold:
        roi = (int(x1), int(y1), int(x2), int(y2))  # 그냥 그대로 x1~x2 좌표
        rois.append(roi)
        print(f"RoI 좌표: ({roi[0]}, {roi[1]}) ~ ({roi[2]}, {roi[3]}), 신뢰도: {conf:.2f}, object: {det[5]}")

# 7. 모든 RoI를 포함하는 큰 바운딩 박스 계산
if rois:
    min_x1 = min([roi[0] for roi in rois])
    min_y1 = min([roi[1] for roi in rois])
    max_x2 = max([roi[2] for roi in rois])
    max_y2 = max([roi[3] for roi in rois])
    large_roi = (min_x1, min_y1, max_x2, max_y2)
    print(f"Large bounding box: {large_roi}")

    # 8. 큰 바운딩 박스를 이미지에 그리기
    cv2.rectangle(img, (large_roi[0], large_roi[1]), (large_roi[2], large_roi[3]), (0, 255, 0), 2)

# 9. 결과 이미지 저장
cv2.imwrite('result_with_large_roi_yolo_fixed.jpg', img)

# 10. 시간 측정 완료
end_time = time.time()
print("걸리는 시간 : ", end_time-start_time)