import onnxruntime as ort
import cv2
import numpy as np
import time 

# 시간 측정 시작
start_time = time.time()

# 모델 불러오기
session = ort.InferenceSession('yolov5n.onnx', providers=['CPUExecutionProvider'])

# 이미지 불러오기
img = cv2.imread('/home/victus04/intern_ws/yolo/download.jpeg')
if img is None:
    raise ValueError("Image not found or unable to read the image file.")

# 이미지 전처리
img_resized = cv2.resize(img, (640, 640))  # 640x640 크기로 리사이즈
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # RGB로 변환
img_rgb = img_rgb.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
img_rgb = img_rgb / 255.0  # 0-1로 정규화
img_rgb = np.expand_dims(img_rgb, axis=0).astype(np.float32)  # 배치 차원 추가

# 모델 입력 확인 (ONNX 모델의 입력 이름을 확인)
input_name = session.get_inputs()[0].name  # 첫 번째 입력 이름
output_name = session.get_outputs()[0].name  # 첫 번째 출력 이름

# 추론 실행
outputs = session.run([output_name], {input_name: img_rgb})

# RoI 추출
pred = outputs[0]
conf_threshold = 0.3  # 신뢰도 기준 설정
rois = []

for det in pred[0]:  # 첫 번째 이미지에 대한 예측
    x1, y1, x2, y2, obj, cls = det[:6]
    class_conf = det[5:].max()  # 클래스 신뢰도 (최대값)
    class_idx = np.argmax(det[5:])
    conf = obj

    if conf > conf_threshold:  # 신뢰도가 임계값을 넘으면
        x_center, y_center, w, h = det[0], det[1], det[2], det[3]
        x1 = (x_center - w / 2) * img.shape[1] / 640  # 원본 이미지에 맞게 좌표 변환
        y1 = (y_center - h / 2) * img.shape[0] / 640
        x2 = (x_center + w / 2) * img.shape[1] / 640
        y2 = (y_center + h / 2) * img.shape[0] / 640
        roi = (int(x1), int(y1), int(x2), int(y2))
        rois.append(roi)
        print(f"RoI 좌표: ({int(x1)}, {int(y1)}) ~ ({int(x2)}, {int(y2)}), 신뢰도: {conf:.2f}, object: {class_idx}")


# 모든 RoI를 포함하는 큰 바운딩 박스 계산
if rois:
    min_x1 = min([roi[0] for roi in rois])
    min_y1 = min([roi[1] for roi in rois])
    max_x2 = max([roi[2] for roi in rois])
    max_y2 = max([roi[3] for roi in rois])
    large_roi = (min_x1, min_y1, max_x2, max_y2)
    print(f"Large bounding box: {large_roi}")

    # 큰 바운딩 박스를 이미지에 그리기
    cv2.rectangle(img, (large_roi[0], large_roi[1]), (large_roi[2], large_roi[3]), (0, 255, 0), 2)

# 결과 이미지 저장
cv2.imwrite('result_with_large_roi.jpg', img)  # 결과 이미지 저장

# 시간 측정 완료
end_time = time.time()
print("걸리는 시간 : ", end_time-start_time)