# from ultralytics import YOLO
# import threading
# import os
# import cv2
# import time
#
# if __name__ == "__main__":
#
#     # model1 = YOLO(r'.\models\sawBlade-seg\best.pt')
#     model2 = YOLO(r'.\models\sawBladev4\best.pt')
#
#     def detect1(dec_image, dec_model):
#         results = dec_model(dec_image)
#         print(results)
#         # res = results[0].plot()
#         # cv2.namedWindow("pic1", cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow("pic1", 1000, 800)
#         # cv2.imshow("pic1", res)
#         # cv2.waitKey()
#
#     def detect2(dec_image, dec_model):
#         results = dec_model(dec_image)
#         print(results)
#         # res = results[0].plot()
#         # cv2.namedWindow("pic2", cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow("pic2", 1000, 800)
#         # cv2.imshow("pic2", res)
#         # cv2.waitKey()
#
#     path = r"E:\sawBlade\all"
#     paths = os.listdir(path)
#     for file in paths:
#         filepath = os.path.join(path, file)
#         img = cv2.imread(filepath)
#         res = model2(img)
#         print(res)
#         time.sleep(5)
#
#         # # t1 = threading.Thread(target=detect1, args=(img, model1))
#         # t2 = threading.Thread(target=detect2, args=(img, model2))
#         #
#         # # t1.start()
#         # t2.start()
#         #
#         # # t1.join()
#         # t2.join()
#


import cv2
import os
from Detect import detect

if __name__ == "__main__":
    path = r"E:\sawBlade\all"
    paths = os.listdir(path)
    img = cv2.imread(os.path.join(path, paths[0]))
    img = detect(img)
    cv2.imshow("pic", img)
    cv2.waitKey()

