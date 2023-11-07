from Detect import detect
import cv2

img = cv2.imread(r"E:\sawBlade\all\00001.bmp")
img = detect(img)
cv2.namedWindow("pic", cv2.WINDOW_NORMAL)
cv2.resizeWindow("pic", 1000, 800)
cv2.imshow("pic", img)
cv2.waitKey()
