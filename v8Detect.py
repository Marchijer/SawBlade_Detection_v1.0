from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch
import queue
import time


def seg(image):
    print("segment started...")
    _model = YOLO("models/seg/best.pt")
    _results = _model(image)
    _gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _mask = np.zeros_like(_gray, dtype=np.uint8)
    _h, _w = _gray.shape[0], _gray.shape[1]
    for _result in _results:
        _boxes = _result.boxes
        for _box in _boxes:
            if _box.conf.item() < 0.2:
                continue
            _xyxy = _box.xyxy
            x1, y1 = max(int(_xyxy[0, 0] - 10), 0), max(int(_xyxy[0, 1] - 10), 0)
            x2, y2 = min(int(_xyxy[0, 2] + 10), _w - 1), min(int(_xyxy[0, 3] + 10), _h - 1)
            box_width, box_height = abs(x2 - x1), abs(y2 - y1)
            if box_width / box_height < 1.5:
                continue
            _mask[y1:y2 + 1, x1:x2 + 1] = 255
    _gray = np.bitwise_and(_mask, _gray)
    _res = cv2.cvtColor(_gray, cv2.COLOR_GRAY2BGR)
    return _res


def segmentDetect(image):
    print("segment started...")
    _model = YOLO("models/seg/best.pt")
    _model_D = YOLO("pt/train3/weights/best.pt")
    _results = _model(image)
    _gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _mask = np.zeros_like(_gray, dtype=np.uint8)
    _h, _w = _gray.shape[0], _gray.shape[1]
    for _result in _results:
        _boxes = _result.boxes
        for _box in _boxes:
            if _box.conf.item() < 0.2:
                continue
            _xyxy = _box.xyxy
            x1, y1 = max(int(_xyxy[0, 0] - 10), 0), max(int(_xyxy[0, 1] - 10), 0)
            x2, y2 = min(int(_xyxy[0, 2] + 10), _w - 1), min(int(_xyxy[0, 3] + 10), _h - 1)
            box_width, box_height = abs(x2 - x1), abs(y2 - y1)
            if box_width / box_height < 1.5:
                continue
            _mask[y1:y2 + 1, x1:x2 + 1] = 255
    _gray = np.bitwise_and(_mask, _gray)
    _res = cv2.cvtColor(_gray, cv2.COLOR_GRAY2BGR)
    _result = _model_D(_res)
    return _result


if __name__ == "__main__":
    dirName = r"D:\sawBlade\sawBlade\image\01_good"
    # cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
    fileNames = os.listdir(dirName)
    for fileName in fileNames:
        path = os.path.join(dirName, fileName)
        print(f"path: {path}")
        img = cv2.imread(path)
        cv2.resizeWindow("pic", img.shape[1] // 2, img.shape[0] // 2)
        results = segmentDetect(img)
        print(results[0].boxes.cls)
        res = results[0].plot()
        cv2.imshow("pic", res)
        cv2.waitKey()

        # res = seg(img)
        # saveDir = r"D:\sawBlade\sawBlade\img-seg\01"
        # saveFile = os.path.join(saveDir, fileName)
        # cv2.imwrite(saveFile, res)
        # print("save:", saveFile, end='\n\n=========\n')
