import os
import time
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from v8Detect import segmentDetect
# import serial

# ser = serial.Serial('COM7',19200,timeout = 1)


F1ImgGet = False
F2ImgGet = False


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("HDU_Detect.ui", self)
        self.image1.setAlignment(Qt.AlignCenter)
        self.image2.setAlignment(Qt.AlignCenter)
        self.image1_detect.setAlignment(Qt.AlignCenter)
        self.image2_detect.setAlignment(Qt.AlignCenter)

        # 创建 QTimer 对象
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(300)

        # 文件夹路径
        self.folder_to_watch = r".\images1"  # 替换为实际的文件夹路径
        self.folder_to_watch2 = r".\images2"  # 替换为实际的文件夹路径
        self.observed_files = set()
        self.observed_files2 = set()
        self.latest_file = None  # 添加一个变量来保存上一次的 latest_file
        self.latest_file2 = None  # 添加一个变量来保存上一次的 latest_file
        self.start_watchdog()  # 启动监视

    def start_watchdog(self):
        # 创建监视器对象
        event_handler = MyHandler(self)
        self.observer = Observer()
        # 设置 recursive 参数为 False，表示不递归监视子文件夹。
        self.observer.schedule(event_handler, path=self.folder_to_watch, recursive=False)
        self.observer.start()
        event_handler2 = MyHandler2(self)
        self.observer2 = Observer()
        # 设置 recursive 参数为 False，表示不递归监视子文件夹。
        self.observer2.schedule(event_handler2, path=self.folder_to_watch2, recursive=False)
        self.observer2.start()

    def update_frame(self):
        # 判断是否需要更新帧
        global F1ImgGet, F2ImgGet
        if not F1ImgGet or not F2ImgGet:
            return

        # 在界面上显示最新的图片
        latest_file = max(self.observed_files, key=os.path.getctime, default=None)
        latest_file2 = max(self.observed_files2, key=os.path.getctime, default=None)
        if latest_file:
            if latest_file == self.latest_file:
                print("Same")
            else:
                # 使用 OpenCV 读取图片
                img1 = cv2.imread(latest_file)
                img2 = cv2.imread(latest_file2)

                height, width, channel = img1.shape
                bytes_per_line = 3 * width

                # 展示图片
                pixmap1 = QPixmap.fromImage(QImage(img1.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped())
                self.image1.setPixmap(pixmap1.scaled(self.image1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

                pixmap2 = QPixmap.fromImage(QImage(img2.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped())
                self.image2.setPixmap(pixmap2.scaled(self.image2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

                res1 = segmentDetect(img1)
                res2 = segmentDetect(img2)

                resImg1 = res1[0].plot()
                resImg2 = res2[0].plot()

                # 展示检测结果
                height, width, channel = resImg1.shape
                bytes_per_line = 3 * width

                pixmap1 = QPixmap.fromImage(QImage(resImg1.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped())
                self.image1_detect.setPixmap(pixmap1.scaled(self.image1_detect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

                pixmap2 = QPixmap.fromImage(QImage(resImg2.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped())
                self.image2_detect.setPixmap(pixmap2.scaled(self.image2_detect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

                # 判断该零件是否有缺陷
                def isBadSaw(results):
                    bad = False
                    for result in results:
                        for box in result.boxes:
                            if box.cls == 0 or box.cls == 2:
                                bad = True
                            elif box.conf < 0.5:
                                bad = True
                    return bad

                bad_saw = isBadSaw(res1) and isBadSaw(res2)

                if bad_saw:
                    self.result.setStyleSheet("font-size: 36px; color: red;background-color: white;")
                    self.result.setText("有缺陷")
                    data = bytes.fromhex('FF00F1')
                    # ser.write(data)
                else:
                    self.result.setStyleSheet("font-size: 36px; color: green;background-color: white;")
                    self.result.setText("无缺陷")

            F1ImgGet, F2ImgGet = False, False


class MyHandler(FileSystemEventHandler):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def on_created(self, event):
        global F1ImgGet  # 声明全局变量
        if not F1ImgGet:
            if not event.is_directory and event.src_path.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                # 添加文件到观察集合，确保文件已完全写入磁盘
                time.sleep(0.5)  # 这里的延迟可以根据需要调整
                print("文件夹1图片路径：", event.src_path)
                self.main_window.observed_files.add(event.src_path)
                F1ImgGet = True # 第一个文件夹已经进行了更新


class MyHandler2(FileSystemEventHandler):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def on_created(self, event):
        global F2ImgGet  # 声明全局变量
        if not F2ImgGet:
            if not event.is_directory and event.src_path.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                # 添加文件到观察集合，确保文件已完全写入磁盘
                time.sleep(0.5)  # 这里的延迟可以根据需要调整
                print("文件夹2图片路径：", event.src_path)
                self.main_window.observed_files2.add(event.src_path)
                F2ImgGet = True # 第一个文件夹已经进行了更新

if __name__ == "__main__":
    try:
        app = QApplication([])
        window = MainWindow()
        window.show()
        app.exec_()
    except Exception as e:
        print("An error occurred:", str(e))