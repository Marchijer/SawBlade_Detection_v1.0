from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch
import queue
import time


def segmentDetect(image):
    global imgSave, event
    print("segment started...")
    _model = YOLO("models/seg/best.pt")
    _model_D = YOLO("models/segAndDetect/best.pt")
    _results = _model(image)
    _gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _res = np.zeros_like(_gray)
    for _result in _results:
        idx = 0
        masks = _result.masks
        for mask in masks:
            print(_result.boxes.conf[idx].item())
            if _result.boxes.conf[idx].item() < 0.6:
                idx += 1
                continue
            idx += 1
            filled_img = np.zeros_like(_gray)
            cv2.fillPoly(filled_img, [np.array(mask.xy, dtype=np.int32)], color=255)
            _res = cv2.bitwise_or(filled_img, _res)
    _res = cv2.bitwise_and(_res, _gray)
    _res = cv2.cvtColor(_res, cv2.COLOR_GRAY2BGR)
    _result = _model_D(_res)
    return _result


if __name__ == "__main__":
    dirName = r"D:\sawBlade\image"
    cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
    fileNames = os.listdir(dirName)
    for fileName in fileNames:
        path = os.path.join(dirName, fileName)
        img = cv2.imread(path)
        cv2.resizeWindow("pic", img.shape[1] // 2, img.shape[0] // 2)
        results = segmentDetect(img)
        print(results[0].boxes.cls)
        res = results[0].plot()
        cv2.imshow("pic", res)
        cv2.waitKey()
