import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, draw_bounding_boxes
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

from ultralytics import YOLO
from dataloader import VOCDataset  # 기존 dataloader 재활용

class RawTester:
    def __init__(self, yolo, device):
        self.device = device
        self.yolo = yolo
        self.yolo.model.to(device).eval()

    @torch.no_grad()
    def evaluate(self, dataloader, conf_threshold=0.5, iou_threshold=0.5, save_img=True):
        all_conf = []

        for idx, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(self.device)

            # YOLOv8 추론
            preds = self.yolo(imgs, conf=conf_threshold, iou=iou_threshold, verbose=False)

            for i, pred in enumerate(preds):
                boxes = pred.boxes.xyxy.cpu()
                confs = pred.boxes.conf.cpu()
                mean_conf = confs.mean().item() if len(confs) > 0 else 0.0
                all_conf.append(mean_conf)

                if save_img:
                    img_cpu = (imgs[i].cpu() * 255).byte()
                    drawn_img = draw_bounding_boxes(img_cpu, boxes, width=2, colors="blue")
                    save_dir = Path("./raw_test_results")
                    save_dir.mkdir(exist_ok=True)
                    save_image(drawn_img.float()/255, save_dir/f"raw_result_{idx}_{i}.png")

            print(f"[Raw] Processed batch {idx+1}/{len(dataloader)}")

        avg_confidence = sum(all_conf) / len(all_conf)

        # mAP 측정
        metrics = self.yolo.val(data=r"C:\Users\DELL\Desktop\Project_Ver1\yolov5\data\voc.yaml", imgsz=640, batch=16)
        mAP50 = metrics.box.map50
        mAP = metrics.box.map

        print(f"\n[Raw Image Results] mAP@0.5: {mAP50:.4f}, mAP@0.5:0.95: {mAP:.4f}, Avg Confidence: {avg_confidence:.4f}")


def main():
    VOC_ROOT = r"C:\Users\DELL\Desktop\VOCdevkit\VOC2007"
    YOLO_WEIGHTS = r"C:\Users\DELL\Desktop\Yolov8\yolov8-pytorch\runs\train\exp11\weights\best.pt"
    INPUT_SIZE = (640, 640)
    BATCH_SIZE = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # YOLOv8 로드
    yolo = YOLO(YOLO_WEIGHTS)

    # JPEG 데이터셋 로드
    transform = T.Compose([
        T.Resize(INPUT_SIZE, interpolation=Image.BILINEAR),
        T.ToTensor(),
    ])
    dataset = VOCDataset(root_dir=VOC_ROOT, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 평가
    tester = RawTester(yolo, device)
    tester.evaluate(dataloader)

if __name__ == "__main__":
    main()
