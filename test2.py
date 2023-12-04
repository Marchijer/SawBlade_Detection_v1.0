from ultralytics import YOLO
import threading
import os
import cv2
import time
import numpy as np


def segImage(image):
    model = YOLO(r'.\models\sawBlade-seg\best.pt')
    results = model(img)
    # res = results[0].plot()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = np.zeros_like(gray)
    for result in results:
        idx = 0
        masks = result.masks
        for mask in masks:
            print(result.boxes.conf[idx].item())
            if result.boxes.conf[idx].item() < 0.6:
                idx += 1
                continue
            idx += 1
            filled_img = np.zeros_like(gray)
            cv2.fillPoly(filled_img, [np.array(mask.xy, dtype=np.int32)], color=255)
            res = cv2.bitwise_or(filled_img, res)
    res = cv2.bitwise_and(res, gray)

if __name__ == "__main__":

    model2 = YOLO(r'.\models\sawBladev4\best.pt')

    path = r"E:\sawBlade\all"
    paths = os.listdir(path)
    for file in paths:
        filepath = os.path.join(path, file)
        img = cv2.imread(filepath)
        res = model2(img)
        print(res)
        time.sleep(5)

        # # t1 = threading.Thread(target=detect1, args=(img, model1))
        # t2 = threading.Thread(target=detect2, args=(img, model2))
        #
        # # t1.start()
        # t2.start()
        #
        # # t1.join()
        # t2.join()




