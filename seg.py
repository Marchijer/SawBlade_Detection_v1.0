from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch

# 使用新模型抠图-分类-训练
if __name__ == "__main__":
    model = YOLO('models/seg/best.pt')
    dirName = r"D:\sawBlade\sawBlade\images\val"
    saveDir = r"D:\sawBlade\sawBlade\images\val-seg"
    # cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
    fileNames = os.listdir(dirName)
    for fileName in fileNames:
        path = os.path.join(dirName, fileName)
        img = cv2.imread(path)
        # cv2.resizeWindow("pic", img.shape[1] // 2, img.shape[0] // 2)
        # cv2.imshow("pic", img)
        # cv2.waitKey()
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
        # cv2.imshow("pic", res)
        # cv2.waitKey()
        save = os.path.join(saveDir, fileName)
        cv2.imwrite(save, res)
        print(save)
