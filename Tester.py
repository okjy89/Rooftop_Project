import os
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

def evaluate_images_in_folder(
    model_weights_path: str,
    image_folder: str,
    imgsz: int = 640
) -> tuple[float, float]:
    """
    이미지 폴더 내 모든 .jpg 파일에 대해 YOLO 추론 수행 후
    평균 confidence와 평균 mAP50 반환

    - model_weights_path: 학습된 YOLOv8 모델 가중치 경로
    - image_folder: .jpg 이미지들이 있는 디렉터리
    - imgsz: YOLO 입력 사이즈
    """
    print("[INFO] YOLO 모델 로딩 중...")
    model = YOLO(model_weights_path)
    print("[INFO] 모델 로딩 완료")

    # 이미지 리스트 구성
    valid_exts = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(image_folder)
                if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(image_folder, f))]


    if not image_files:
        print("[경고] 이미지가 없습니다:", image_folder)
        return 0.0, 0.0

    print(f"[INFO] 총 {len(image_files)}장 이미지 평가 시작")

    all_confidences = []
    all_maps = []

    for fname in tqdm(image_files):
        img_path = os.path.join(image_folder, fname)

        try:
            results = model.predict(source=img_path, imgsz=imgsz, device='cuda', verbose=False)
        except Exception as e:
            print(f"[에러] 추론 중 문제 발생 ({fname}): {e}")
            continue

        # confidence 수집
        confs = results[0].boxes.conf.cpu().numpy().tolist() if results[0].boxes else []
        all_confidences.extend(confs)

        # mAP50 수집 (Ultralytics에서 지원될 경우)
        metrics = getattr(results[0], "metrics", {})
        if isinstance(metrics, dict) and "map50" in metrics:
            all_maps.append(metrics["map50"])

    # 평균 계산
    avg_conf = float(np.mean(all_confidences)) if all_confidences else 0.0
    avg_map50 = float(np.mean(all_maps)) if all_maps else 0.0

    print(f"[결과] 평균 confidence: {avg_conf:.4f}")
    print(f"[결과] 평균 mAP50:     {avg_map50:.4f}")
    return avg_conf, avg_map50


if __name__ == "__main__":
    # 사용자 수정 영역
    model_weights = r"C:\Users\DELL\Desktop\Yolov8\yolov8-pytorch\runs\train\exp11\weights\best.pt"
    image_folder = r"C:\Users\DELL\Desktop\Project_Ver1\debug_images\filtered"  # 여기에 .jpg들만 있으면 됨

    # 평가 실행
    avg_conf, avg_map = evaluate_images_in_folder(
        model_weights_path=model_weights,
        image_folder=image_folder,
        imgsz=640
    )

    print(f"\n📊 최종 결과:\n- 평균 confidence: {avg_conf:.4f}\n- 평균 mAP50: {avg_map:.4f}")
