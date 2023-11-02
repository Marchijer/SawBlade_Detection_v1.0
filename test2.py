from Detect import detect
import cv2

if __name__ == "__main__":
    path = r"E:\sawBlade\unqualified\big8\big8_right_scrap\top\039.bmp"
    mat = cv2.imread(path)
    # bond, classes = detect2(mat)
    # print(bond)
    # print(classes)
    img = detect(mat)
    cv2.imshow("pic", img)
    cv2.waitKey(0)
