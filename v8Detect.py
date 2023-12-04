from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch


def segment(image):
    _model = YOLO("models/seg/best.pt")
    _results = _model(image)
    _gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    return _res


def detect(image):
    _model = YOLO("models/segAndDetect/best.pt")
    _result = _model(image)
    return _result


# 使用新模型抠图-分类-训练
if __name__ == "__main__":
    dirName = r"E:\sawBlade\all"
    # saveDir = r"D:\sawBlade\sawBlade\images\val-seg"
    cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
    fileNames = os.listdir(dirName)
    for fileName in fileNames:
        path = os.path.join(dirName, fileName)
        img = cv2.imread(path)
        cv2.resizeWindow("pic", img.shape[1] // 2, img.shape[0] // 2)
        img = segment(img)
        results = detect(img)
        print(results[0].boxes.cls)
        res = results[0].plot()
        cv2.imshow("pic", res)
        cv2.waitKey()
