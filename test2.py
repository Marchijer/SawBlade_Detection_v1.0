from ultralytics import YOLO
import threading
import os
import cv2
import time
import numpy as np
from utils.plots import Annotator, colors


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

    model = YOLO("models/seg/best.pt")

    path = r"D:\sawBlade\sawBlade\images\train"
    paths = os.listdir(path)
    cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
    for file in paths:
        filepath = os.path.join(path, file)
        img = cv2.imread(filepath)
        cv2.resizeWindow("pic", img.shape[1] // 2, img.shape[0] // 2)
        results = model(img)
        boxes = results[0].boxes
        print(boxes.conf)
        cv2.imshow("pic", results[0].plot())
        cv2.waitKey()
        # 如何找到所有小于0.8的框，然后将其画出来呢
        annotator = Annotator(im0, line_width=10, example=str(names))
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (model.names[c] if hide_conf else f'{model.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))




