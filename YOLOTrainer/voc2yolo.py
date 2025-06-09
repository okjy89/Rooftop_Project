import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil

# VOC 20 클래스
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def convert_annotation(xml_path, class_list, image_width, image_height):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in class_list:
            continue
        cls_id = class_list.index(cls)
        xmlbox = obj.find('bndbox')
        x_min = float(xmlbox.find('xmin').text)
        y_min = float(xmlbox.find('ymin').text)
        x_max = float(xmlbox.find('xmax').text)
        y_max = float(xmlbox.find('ymax').text)

        x_center = (x_min + x_max) / 2.0 / image_width
        y_center = (y_min + y_max) / 2.0 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return labels


def voc2yolo(voc_root, output_root, image_set='train'):
    anno_dir = os.path.join(voc_root, 'VOC2007', 'Annotations')
    image_dir = os.path.join(voc_root, 'VOC2007', 'JPEGImages')
    set_path = os.path.join(voc_root, 'VOC2007', 'ImageSets', 'Main', f"{image_set}.txt")

    save_img_dir = os.path.join(output_root, 'images', image_set)
    save_lbl_dir = os.path.join(output_root, 'labels', image_set)
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_lbl_dir, exist_ok=True)

    with open(set_path, 'r') as f:
        image_ids = [x.strip() for x in f.readlines()]

    for image_id in tqdm(image_ids, desc=f"Converting {image_set}"):
        xml_file = os.path.join(anno_dir, f"{image_id}.xml")
        jpg_file = os.path.join(image_dir, f"{image_id}.jpg")
        label_file = os.path.join(save_lbl_dir, f"{image_id}.txt")

        if not os.path.exists(xml_file) or not os.path.exists(jpg_file):
            continue

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
        except Exception as e:
            print(f"[ERROR] {image_id}: {e}")
            continue

        labels = convert_annotation(xml_file, VOC_CLASSES, w, h)
        if labels:
            with open(label_file, 'w') as f:
                f.write('\n'.join(labels))
            shutil.copy(jpg_file, save_img_dir)  # 복사

if __name__ == "__main__":
    voc_root = r"C:\Users\DELL\Desktop\VOCdevkit"
    yolo_output_root = r"C:\Users\DELL\Desktop\VOC_YOLOv5_Format"  # 새 폴더

    #voc2yolo(voc_root, yolo_output_root, image_set='train')
    voc2yolo(voc_root, yolo_output_root, image_set='val')

