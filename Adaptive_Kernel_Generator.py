import cv2
from get_roi_featuremap import roi_feature_map

def main():
    # 이미지 읽기
    img_path = 'download.jpeg' 
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        print(f"이미지 파일을 찾을 수 없습니다: {img_path}")
        return

    # roi_feature_map
    feature_map = roi_feature_map(img_rgb)


    if feature_map is not None:
        print("ROI feature map size:", feature_map.shape)
    else:
        print("ROI feature map이 없습니다.")

if __name__ == "__main__":
    main()