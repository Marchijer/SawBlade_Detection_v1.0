from ultralytics import YOLO
from utils.torch_utils import select_device
import os
import cv2

if __name__ == "__main__":
    device = "0"
    device = select_device(device)

    model = YOLO(r'.\models\sawBlade-seg\best.pt')
    model2 = YOLO(r'.\models\sawBladev4\best.pt')

    path = r"E:\sawBlade\all"
    paths = os.listdir(path)
    for file in paths:
        filepath = os.path.join(path, file)
        img = cv2.imread(filepath)
        results = model(img)

        # print(result)
        img = results[0].plot()
        cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("pic", 1000, 800)
        cv2.imshow("pic", img)
        cv2.waitKey()
