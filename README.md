# Adaptive Kernel Generator 

## Main Ideas
### Edge Device
- Edge Device의 연산 부담 최소화
- Edge Device에서의 서버 전송 속도 최적화 (10MB/s 기준, 최대 3배 향상)
### Cloud Device
- pre-trained 된 yolov5의 CNN으로 input image의 Feature map 생성 
- MLP로 해당 Feature map에 고유한 3x3 kernel 생성  
- Input image와 kernel을 conv

## Main code
- main.py