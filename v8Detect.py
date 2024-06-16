from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch
import queue
import time

# 输入和输出文件夹路径
input_folder = r"D:\Desktop\data(add camera)\00_light_change\00_light_change\sawblade_data"
output_folder = r"D:\Desktop\data(add camera)\00_light_change\00_light_change\masked_data03"


def seg(raw_image, conf_threshold=0.75):
    """ mask image """
    print("=======================================================================================================")
    print("segment start...")
    box_selection = YOLO("pt/seg/best.pt")
    box_results = box_selection(raw_image)
    gray_img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)  # 转灰度图
    mask_box = np.zeros_like(gray_img, dtype=np.uint8)
    _h, _w = gray_img.shape[0], gray_img.shape[1]

    for _result in box_results:
        _boxes = _result.boxes
        for _box in _boxes:
            if _box.conf.item() < conf_threshold:  # 置信度
                continue
            x1, y1, x2, y2 = map(int, _box.xyxy[0])
            x1, y1 = max(x1 - 10, 0), max(y1 - 10, 0)
            x2, y2 = min(x2 + 10, _w - 1), min(y2 + 10, _h - 1)
            # box_width, box_height = abs(x2 - x1), abs(y2 - y1)
            # if box_width / box_height < 1.5:
            #     continue
            mask_box[y1:y2 + 1, x1:x2 + 1] = 255

    gray_img = np.bitwise_and(mask_box, gray_img)
    _res = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    return _res


def segmentDetect(raw_img, seg_model, det_model, conf_threshold=0.75):
    """ mask image and return the detect results """
    print("=======================================================================================================")
    print("segment image...")
    box_results = seg_model(raw_img)
    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)  # RGB 2 gray
    box_mask = np.zeros_like(gray_img, dtype=np.uint8)
    _h, _w = gray_img.shape[0], gray_img.shape[1]

    for _result in box_results:
        _boxes = _result.boxes
        for _box in _boxes:
            if _box.conf.item() < conf_threshold:  # Confidence
                continue
            x1, y1, x2, y2 = map(int, _box.xyxy[0])
            x1, y1 = max(x1 - 10, 0), max(y1 - 10, 0)
            x2, y2 = min(x2 + 10, _w - 1), min(y2 + 10, _h - 1)
            # box_width, box_height = abs(x2 - x1), abs(y2 - y1)
            # if box_width / box_height < 1.5:
            #     continue
            box_mask[y1:y2 + 1, x1:x2 + 1] = 255

    masked_img = cv2.cvtColor(np.bitwise_and(box_mask, gray_img), cv2.COLOR_GRAY2BGR)  # gray 2 RGB
    print("segment finished.\n")

    print("detect start...")

    # Saves cropped images of detections. Useful for dataset augmentation, analysis,
    # or creating focused datasets for specific objects.
    #
    # detect_results = det_model(masked_img, save_crop=True)

    detect_results = det_model(masked_img)
    print("detect finished.\n")
    return detect_results


if __name__ == "__main__":
    print("system_02 start...\n")
    box_selection_model = YOLO("pt/segmentation/best.pt")  # segment
    detect_model = YOLO("pt/detection/best.pt")  # detect

    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 构建输入文件和输出文件路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取输入图像
            image = cv2.imread(input_path)

            # 实例分割及缺陷检测
            results = segmentDetect(image, box_selection_model, detect_model)
            print(results)

    print("\n mask finish!")
