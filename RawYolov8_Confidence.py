import os
from ultralytics import YOLO
import numpy as np

def compute_average_confidence(
    model_weights_path: str,
    voc_root: str,
    val_list_file: str,
    image_folder: str,
    imgsz: int = 640
) -> float:
    """
    - model_weights_path: YOLOv8n으로 학습된 weights(.pt) 경로
    - voc_root:           VOC2007 디렉터리 최상위 경로
    - val_list_file:      val 이미지 목록(txt) 경로
    - image_folder:       실제 이미지(.jpg)들이 들어있는 폴더 경로
    - imgsz:              모델 입력 사이즈

    반환값: val_list_file에 정의된 이미지들에서 검출된 
           모든 바운딩박스들의 confidence 평균 (float)
    """

    # 1) 모델 로드
    print("[디버그] 모델 로드 시작 →", model_weights_path)
    model = YOLO(model_weights_path)
    print("[디버그] 모델 로드 완료")

    # 2) val.txt 읽어서 이미지 ID 리스트 가져오기
    print("[디버그] val_list_file 경로:", val_list_file)
    with open(val_list_file, 'r') as f:
        lines = f.read().splitlines()
    # 공백이나 빈 줄 제거
    image_ids = [line.strip() for line in lines if line.strip()]
    print(f"[디버그] val.txt에서 읽어온 ID 개수: {len(image_ids)}")
    if len(image_ids) > 0:
        print("[디버그] 첫 5개 ID 예시:", image_ids[:5])

    all_confidences = []

    # 3) 한 장씩 순회하며 추론 → confidence 값 수집
    for idx, img_id in enumerate(image_ids):
        img_path = os.path.join(image_folder, f"{img_id}.jpg")
        # 이미지 파일 존재 여부 확인
        if not os.path.isfile(img_path):
            print(f"[경고] ({idx+1}/{len(image_ids)}) 이미지 파일이 없음: {img_path}")
            continue
        else:
            print(f"[디버그] ({idx+1}/{len(image_ids)}) 이미지 파일 발견: {img_path}")

        # inference 시작
        try:
            results = model.predict(source=img_path, imgsz=imgsz, device='cuda', verbose=False)
        except Exception as e:
            print(f"[에러] 모델 추론 중 예외 발생 (ID={img_id}): {e}")
            continue

        # 검출된 confidence 확인
        confidences = results[0].boxes.conf.cpu().numpy().tolist()
        print(f"[디버그] ({idx+1}/{len(image_ids)}) 검출된 confidence 개수: {len(confidences)}")

        if confidences:
            all_confidences.extend(confidences)

    # 4) 평균 계산
    print(f"[디버그] 최종 누적된 confidence 개수: {len(all_confidences)}")
    if not all_confidences:
        print("[경고] 검출된 바운딩박스가 하나도 없습니다. 모델/데이터셋을 확인하세요.")
        return 0.0

    all_confidences = np.array(all_confidences, dtype=np.float32)
    avg_conf = float(np.mean(all_confidences))
    print(f"[디버그] 평균 confidence 계산 완료: {avg_conf:.4f}")
    return avg_conf


if __name__ == "__main__":
    # ────────────────────────────────────────────────────────────────────
    # 아래 4개 변수만 실제 경로에 맞게 수정하세요.
    # ────────────────────────────────────────────────────────────────────

    # 1) YOLOv8n 학습 결과 가중치(.pt) 경로
    model_weights = r"C:\Users\DELL\Desktop\Yolov8\yolov8-pytorch\runs\train\exp11\weights\best.pt"

    # 2) VOC2007 최상위 경로 (ImageSets/Main/val.txt와 images/가 여기 아래 있어야 함)
    voc_root_dir = r"home/okjy89/dataset/pascal/train/VOCdevkit/VOC2007"

    # 3) val.txt 파일 경로
    val_txt_path = os.path.join(voc_root_dir, "ImageSets", "Main", "val.txt")

    # 4) 이미지(.jpg) 폴더 경로
    jpeg_images = os.path.join(voc_root_dir, "images")
    # ────────────────────────────────────────────────────────────────────

    # (디버그) 경로 존재 여부 출력
    print(">>> 사용할 val.txt 경로:", val_txt_path)
    print(">>> val.txt 파일 존재 여부:", os.path.isfile(val_txt_path))
    print(">>> 사용할 이미지 폴더 경로:", jpeg_images)
    print(">>> 이미지 폴더 존재 여부:", os.path.isdir(jpeg_images))

    # 평균 confidence 계산
    avg_confidence = compute_average_confidence(
        model_weights_path=model_weights,
        voc_root=voc_root_dir,
        val_list_file=val_txt_path,
        image_folder=jpeg_images,
        imgsz=640
    )

    print(f"\nVOC2007 Val Set 전체 바운딩박스의 평균 confidence: {avg_confidence:.4f}")
