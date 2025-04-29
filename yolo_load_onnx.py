import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)  # YOLO 기본 입력 크기
torch.onnx.export(model, dummy_input, "yolov5n.onnx", opset_version=12)
