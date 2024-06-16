import os
import cv2
from ultralytics import YOLO
from v8Detect import segmentDetect  # Assuming segmentDetect is a function from your v8Detect module


print("Try_system start...\n")
box_selection_model = YOLO("pt/segmentation/best.pt")  # segment
detect_model = YOLO("pt/detection/best.pt")  # detect

# 文件夹路径
folder_to_watch = r"D:\Desktop\data(add camera)\data2\data2\(测试用)大8左_报废_杂"  # 替换为实际的文件夹路径
# folder_to_watch = r"D:\Desktop\Code\01_sawBlade_Detection\images1"  # 替换为实际的文件夹路径


# 获取文件夹中的所有文件
image_files = [f for f in os.listdir(folder_to_watch) if f.endswith('.bmp')]

# 遍历所有图像文件
for image_file in image_files:
    image_path = os.path.join(folder_to_watch, image_file)
    img = cv2.imread(image_path)

    # 如果图像读取失败，则跳过
    if img is None:
        print(f"Failed to read image: {image_file}")
        continue

    # 传入模型进行检测
    detections = segmentDetect(img, box_selection_model, detect_model)
    # print(detections)

    if isinstance(detections, list):
        results = detections
    else:
        results = [detections]

    # 初始化检测结果
    is_bad = False

    # 遍历每个检测结果
    for result in results:
        for box in result.boxes:
            # 如果置信度低于0.5，则跳过
            if box.conf < 0.5:
                continue
            # 判断类别并打印信息
            elif box.cls == 0:
                print("0: tail_unqualified")
                is_bad = True
            elif box.cls == 2:
                print("2: head_unqualified")
                is_bad = True
            elif box.cls == 5:
                print("5: blade_unqualified")
                is_bad = True
            elif box.cls == 6:
                print("6: crack")
                is_bad = True

    # 根据检测结果判断好坏
    if is_bad:
        print(f"{image_file} IS BAD!\n")
    else:
        print(f"{image_file} IS GOOD!\n")
