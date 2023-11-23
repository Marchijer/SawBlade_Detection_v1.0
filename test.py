# 这个文件主要负责将图片检测出来
import multiprocessing

import numpy as np
import torch
from scipy.constants import pt
from torch.masked.maskedtensor._ops_refs import stride

from utils.augmentations import letterbox

from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, cv2, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

weights1 = r'./models/top/best.pt'  # 权重文件地址   .pt文件
weights2 = r'./models/left/best.pt'
data1 = r'./models/top/sawBlade.yaml'  # 标签文件地址   .yaml文件
data2 = r'./models/left/sawBlade.yaml'

imgsz = (640, 640)  # 输入图片的大小 默认640(pixels)
conf_thres = 0.4  # object置信度阈值 默认0.25  用在nms中
iou_thres = 0.45  # 做nms的iou阈值 默认0.45   用在nms中
max_det = 1000  # 每张图片最多的目标数量  用在nms中
device = '0'  # 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
classes = None  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留 --class 0, or --class 0 2 3
agnostic_nms = False  # 进行nms是否也除去不同类别之间的框 默认False
augment = False  # 预测是否也要采用数据增强 TTA 默认False
visualize = False  # 特征图可视化 默认FALSE
half = False  # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
dnn = False  # 使用OpenCV DNN进行ONNX推理
line_thickness = 1
hide_conf = False
hide_labels = False

# 获取设备
device = select_device(device)

# 载入模型
modelL = DetectMultiBackend(weights1, device=device, dnn=dnn, data=data1)
modelT = DetectMultiBackend(weights2, device=device, dnn=dnn, data=data2)


# 传入名为img的图片,img原图，im0
def detect(model, img):
    global imgsz, half
    half &= (model.pt or model.jit or model.onnx or model.engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if model.pt or model.jit:
        model.model.half() if half else model.model.float()
    # model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    im0 = img
    imgsz = check_img_size(imgsz, s=model.stride)
    im = letterbox(im0, imgsz, model.stride, auto=model.pt)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1
    # 预测
    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3
    im0 = np.ascontiguousarray(im0)
    annotator = Annotator(im0, line_width=10, example=str(model.names))
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
    im0 = annotator.result()
    LOGGER.info(f'({t3 - t2:.3f}s)')
    return im0


def inference1(model, img):
    img = detect(model, img)
    cv2.imshow("left", img)
    cv2.waitKey()


def inference2(model, img):
    img = detect(model, img)
    cv2.imshow("top", img)
    cv2.waitKey()


if __name__ == "__main__":
    image1 = cv2.imread(r"E:\sawBlade\all\00019.bmp")
    image2 = cv2.imread(r"E:\sawBlade\all\00285.bmp")
    t1 = multiprocessing.Process(target=inference1, args=(modelL, image1))
    t2 = multiprocessing.Process(target=inference2, args=(modelT, image2))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
