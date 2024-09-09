from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon

import sys
sys.path.insert(0, './yolov5')
import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import time
import cv2
import platform
import shutil

import contextlib
import platform
import argparse
import re
import subprocess
from pathlib import Path
import pandas as pd
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.yolo import  Detect, DetectionModel
from utils.general import (LOGGER, Profile, check_dataset, check_img_size, check_requirements, check_version,
                           check_yaml, colorstr, file_size, get_default_args, print_args, url2file, yaml_save)

from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import colors, plot_one_box, plot_one_box_PIL
from utils.plots import Annotator, colors, save_one_box

from utils.torch_utils import select_device, time_sync, smart_inference_mode
from utils.capnums import Camera
from dialog.rtsp_win import Window
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # emit：detecting/pause/stop/finished/error msg
    send_probability = pyqtSignal(list)
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './best(1).pt'
        self.current_weight = './best(1).pt'
        self.source = '0'
        self.imgsz = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False  # jump out of the loop
        self.is_continue = True  # continue/pause
        self.percent_length = 1000  # progress bar
        self.rate_check = True  # Whether to enable delay
        self.rate = 100
        self.save_fold = './result'
        self.device = 'cuda'
        self.save_txt = True
        self.classes = [0]
        self.config_deepsort = "deep_sort_pytorch/configs/deep_sort.yaml"
        self.augment = False
        self.agnostic_nms = False
        self.view_img = False
        self.output = 'inference/output'

    def bbox_rel(self, *xyxy):
            bbox_left = min([xyxy[0].item(), xyxy[2].item()])
            bbox_top = min([xyxy[1].item(), xyxy[3].item()])
            bbox_w = abs(xyxy[0].item() - xyxy[2].item())
            bbox_h = abs(xyxy[1].item() - xyxy[3].item())
            x_c = (bbox_left + bbox_w / 2)
            y_c = (bbox_top + bbox_h / 2)
            w = bbox_w
            h = bbox_h
            return x_c, y_c, w, h

    def compute_color_for_labels(self, label):
            color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
            return tuple(color)

    def draw_boxes(self, img, bbox, identities=None, offset=(0, 0)):
            for i, box in enumerate(bbox):
                x1, y1, x2, y2 = [int(i) for i in box]
                x1, y1, x2, y2 = [int(i) for i in box]
                x1 += offset[0]
                x2 += offset[0]
                y1 += offset[1]
                y2 += offset[1]
                # box text and bar
                id = int(identities[i]) if identities is not None else 0
                color = self.compute_color_for_labels(id)
                label = '{}{:d}'.format("", id)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(
                    img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
                cv2.putText(img, label, (x1, y1 +
                                         t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
            return img

        # xyxy2tlwh函数  这个函数一般都会自带
    def xyxy2tlwh(self, x):
            y = torch.zeros_like(x) if isinstance(x,
                                                  torch.Tensor) else np.zeros_like(x)
            y[:, 0] = x[:, 0]
            y[:, 1] = x[:, 1]
            y[:, 2] = x[:, 2] - x[:, 0]
            y[:, 3] = x[:, 3] - x[:, 1]
            return y

    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            save_img=False
            ):
     try:
        # Initialize
        cfg = get_config()
        cfg.merge_from_file(self.config_deepsort)
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

        device = select_device(device)
        #self.half &= device.type != 'cpu'  # half precision only supported on CUDA
        half &= device.type != 'cpu'  # half precision only supported on CUDA

         # Load model
        self.detector = attempt_load(self.weights, map_location=device)  # load FP32 model

        self.detector.to(self.device).eval()
        num_params = 0
        for param in self.detector.parameters():
                num_params += param.numel()
        stride = int(self.detector.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names  # get class names
        if half:
            self.detector.half()  # to FP16

        if os.path.isdir(self.source):
            # Dataloader
            print(self.source)
            if device.type != 'cpu':
                self.detector(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(self.detector.parameters()))) # run once
            count = 0
            jump_count = 0
            start_time = time.time()
            while True:
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('Stop')
                    if hasattr(self, 'out'):
                        self.out.release()
                    break
                # change model
                if self.current_weight != self.weights:
                    # Load model
                    self.detector = attempt_load(self.weights, map_location=device)  # load FP32 model
                    num_params = 0
                    for param in self.detector.parameters():
                        num_params += param.numel()
                    stride = int(self.detector.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names  # get class names
                    if half:
                        self.detector.half()  # to FP16
                    # Run inference
                    if device.type != 'cpu':
                        self.detector(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(self.detector.parameters())))  # run once
                    self.current_weight = self.weights
                if self.is_continue:
                    for filename in os.listdir(self.source):
                        print(filename)
                        filename = self.source + "/" + filename
                        print(filename)
                        dataset = LoadImages(filename, img_size=imgsz, stride=stride)
                        dataset = iter(dataset)
                        path, img, im0s, self.vid_cap = next(dataset)
                        print(1)
                        count += 1
                        if count % 30 == 0 and count >= 30:
                            fps = int(30 / (time.time() - start_time))
                            self.send_fps.emit('fps：' + str(fps))
                            start_time = time.time()
                        if self.vid_cap:
                            percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)
                            self.send_percent.emit(percent)
                        else:
                            percent = self.percent_length
                        shipnames = ['ore carrier', 'fishing boat', 'bulk cargo carrier', 'general cargo ship',
                                     'container ship', 'passenger ship']
                        statistic_dic = {name: 0 for name in shipnames}
                        probability = []
                        img = torch.from_numpy(img).to(device)
                        img = img.half() if half else img.float()  # uint8 to fp16/32
                        img /= 255.0  # 0 - 255 to 0.0 - 1.0
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)
                        pred = self.detector(img, augment=augment)[0]

                        # Apply NMS
                        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms,
                                                   max_det=max_det)
                        # Process detections
                        for i, det in enumerate(pred):  # detections per image
                            im0 = im0s.copy()
                            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                                # Write results
                                for *xyxy, conf, cls in reversed(det):
                                    c = int(cls)  # integer class
                                    statistic_dic[shipnames[c]] += 1
                                    label = None if hide_labels else (
                                        shipnames[c] if hide_conf else f'{shipnames[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                    probability.append(f'{shipnames[c]}')
                                    probability.append(f'{conf:.2f}')
                                    probability.append(f'{xyxy[0]:.2f}')
                                    probability.append(f'{xyxy[1]:.2f}')
                                    probability.append(f'{xyxy[2]:.2f}')
                                    probability.append(f'{xyxy[3]:.2f}')
                        if self.rate_check:
                            time.sleep(1 / self.rate)
                        im0 = annotator.result()
                        self.send_img.emit(im0)
                        self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                        self.send_statistic.emit(statistic_dic)
                        self.send_probability.emit(probability)
                        if self.save_fold:
                            os.makedirs(self.save_fold, exist_ok=True)
                            if self.vid_cap is None:
                                save_path = os.path.join(self.save_fold,
                                                         str(count) + time.strftime(' %Y_%m_%d_%H_%M_%S',
                                                                       time.localtime()) + '.jpg')
                                cv2.imwrite(save_path, im0)

                    if percent == self.percent_length:
                        print(count)
                        self.send_percent.emit(0)
                        self.send_msg.emit('finished')
                        if hasattr(self, 'out'):
                            self.out.release()
                        break


        else:
            if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                # bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

            # Run inference
            if device.type != 'cpu':
                self.detector(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(self.detector.parameters())))  # run once

            count = 0
            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)
            save_path = str(Path(self.output))
            dict_box = dict()

            frame_idx = 0

            while True:
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('Stop')
                    if hasattr(self, 'out'):
                        self.out.release()
                    break

                if self.current_weight != self.weights:
                    # Load model
                    self.detector = attempt_load(self.weights, map_location=device)  # load FP32 model
                    num_params = 0
                    for param in self.detector.parameters():
                        num_params += param.numel()
                    stride = int(self.detector.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names  # get class names
                    if half:
                        self.detector.half()  # to FP16
                    # Run inference
                    if device.type != 'cpu':
                        self.detector(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(self.detector.parameters())))  # run once
                    self.current_weight = self.weights

                if self.is_continue:
                    path, img, im0s, self.vid_cap = next(dataset)
                    count += 1
                    if count % 30 == 0 and count >= 30:
                        fps = int(30/(time.time()-start_time))
                        self.send_fps.emit('fps：'+str(fps))
                        start_time = time.time()
                        print(1)
                    if self.vid_cap:
                        percent = int(count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length)
                        self.send_percent.emit(percent)
                        print(2)
                    else:
                        percent = self.percent_length
                        print(3)
                    shipnames = ['ore carrier', 'fishing boat', 'bulk cargo carrier', 'general cargo ship',
                                 'container ship',  'passenger ship']
                    statistic_dic = {name: 0 for name in shipnames}
                    probability = []
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    print(9)

                    pred =self.detector(img, augment=augment)[0]

                    # Apply NMS
                    print(10)
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)
                    print('MOON')

                    # Process detections
                    if self.vid_cap:
                        for i, det in enumerate(pred):  # detections per image
                            im1 = im0s.copy()
                            p, s, im0 = path, '', im0s
                            s += '%gx%g ' % img.shape[2:]  # print string
                            annotator = Annotator(im1, line_width=line_thickness, example=str(names))
                            if det is not None and len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                                for *xyxy, conf, cls in reversed(det):
                                    c = int(cls)  # integer class
                                    statistic_dic[shipnames[c]] += 1
                                    label = None if hide_labels else (
                                    shipnames[c] if hide_conf else f'{shipnames[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                    probability.append(f'{shipnames[c]}')
                                    probability.append(f'{conf:.2f}')
                                    probability.append(f'{xyxy[0]:.2f}')
                                    probability.append(f'{xyxy[1]:.2f}')
                                    probability.append(f'{xyxy[2]:.2f}')
                                    probability.append(f'{xyxy[3]:.2f}')

                                # Print results
                                for c in det[:, -1].unique():
                                    n = (det[:, -1] == c).sum()  # detections per class
                                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                                bbox_xywh = []
                                confs = []
                                COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0),
                                             (210, 105, 30),
                                             (220, 20, 60),
                                             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139),
                                             (100, 149, 237),
                                             (138, 43, 226), (238, 130, 238),
                                             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205),
                                             (255, 140, 0),
                                             (255, 239, 213),
                                             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205),
                                             (176, 196, 222),
                                             (65, 105, 225), (173, 255, 47),
                                             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133),
                                             (148, 0, 211),
                                             (255, 99, 71), (144, 238, 144),
                                             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0),
                                             (189, 183, 107),
                                             (255, 255, 224), (128, 128, 128),
                                             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128),
                                             (72, 209, 204),
                                             (139, 69, 19), (255, 245, 238),
                                             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235),
                                             (0, 191, 255),
                                             (176, 224, 230), (0, 250, 154),
                                             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139),
                                             (143, 188, 143),
                                             (255, 0, 0), (240, 128, 128),
                                             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42),
                                             (178, 34, 34),
                                             (175, 238, 238), (255, 248, 220),
                                             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96),
                                             (210, 105, 30)]
                                # Adapt detections to deep sort input format
                                for *xyxy, conf, cls in det:
                                    x_c, y_c, bbox_w, bbox_h = self.bbox_rel(*xyxy)
                                    obj = [x_c, y_c, bbox_w, bbox_h]

                                    bbox_xywh.append(obj)

                                    confs.append([conf.item()])

                                xywhs = torch.Tensor(bbox_xywh)
                                confss = torch.Tensor(confs)

                                # Pass detections to deepsort
                                outputs = deepsort.update(xywhs, confss, im0)

                                # outputs = [x1, y1, x2, y2, track_id]
                                if len(outputs) > 0:
                                    bbox_xyxy = outputs[:, :4]  # 提取前四列  坐标
                                    identities = outputs[:, -1]  # 提取最后一列 ID
                                    self.draw_boxes(im0, bbox_xyxy, identities, offset=(0, 0))
                                    box_xywh = self.xyxy2tlwh(bbox_xyxy)
                                    for j in range(len(box_xywh)):
                                        x_center = box_xywh[j][0] + box_xywh[j][2] / 2  # 求框的中心x坐标
                                        y_center = box_xywh[j][1] + box_xywh[j][3] / 2  # 求框的中心y坐标
                                        id = outputs[j][-1]
                                        center = [x_center, y_center]
                                        dict_box.setdefault(id, []).append(center)  # 这个字典需要提前定义 dict_box = dict()
                                    # 以下为画轨迹，原理就是将前后帧同ID的跟踪框中心坐标连接起来
                                    if frame_idx > 2:  # 这里可以调整判断条件
                                        max_trajectory_length = 50  # 设置轨迹的最大长度
                                        temp_dict_box = dict_box.copy()  # 创建字典的副本
                                        for key, value in temp_dict_box.items():
                                            if key in identities:  # 判断目标是否存在
                                                if len(value) > max_trajectory_length:  # 限制轨迹长度
                                                    value = value[-max_trajectory_length:]  # 截断保留最近的一部分轨迹
                                                for a in range(len(value) - 1):
                                                    color = self.compute_color_for_labels(key)  # 使用固定颜色
                                                    index_start = a
                                                    index_end = index_start + 1
                                                    cv2.line(im0, tuple(map(int, value[index_start])),
                                                             tuple(map(int, value[index_end])),
                                                             color, thickness=5, lineType=8)
                                            else:
                                                continue
                                frame_idx += 1

                            else:
                                deepsort.increment_ages()

                        if self.rate_check:
                            time.sleep(1 / self.rate)
                        self.send_img.emit(im1)  # 显示在右边
                        self.send_raw.emit(im0)#显示在左边
                        self.send_statistic.emit(statistic_dic)
                        self.send_probability.emit(probability)
                        if self.save_fold:
                            os.makedirs(self.save_fold, exist_ok=True)
                            if count == 1:
                                ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                                if ori_fps == 0:
                                    ori_fps = 25
                                # width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                # height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                width, height = im0.shape[1], im0.shape[0]
                                w = im1.shape[1]
                                h = im1.shape[0]
                                save_detect_path = os.path.join(self.save_fold, time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                                              time.localtime()) + '.mp4')
                                save_track_path = os.path.join(self.output, time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                                          time.localtime()) + '.mp4')
                                self.out = cv2.VideoWriter(save_detect_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                           (width, height))
                                vid_writer = cv2.VideoWriter(save_track_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                             (w, h))
                            self.out.write(im0)
                            vid_writer.write(im1)

                        if percent == self.percent_length:
                            print(count)
                            self.send_percent.emit(0)
                            self.send_msg.emit('finished')
                            if hasattr(self, 'out'):
                                self.out.release()
                            break
                    else:
                        for i, det in enumerate(pred):  # detections per image
                            im0 = im0s.copy()
                            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                                # Write results
                                for *xyxy, conf, cls in reversed(det):
                                    c = int(cls)  # integer class
                                    statistic_dic[shipnames[c]] += 1
                                    label = None if hide_labels else (
                                    shipnames[c] if hide_conf else f'{shipnames[c]} {conf:.2f}')
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                    probability.append(f'{shipnames[c]}')
                                    probability.append(f'{conf:.2f}')
                                    probability.append(f'{xyxy[0]:.2f}')
                                    probability.append(f'{xyxy[1]:.2f}')
                                    probability.append(f'{xyxy[2]:.2f}')
                                    probability.append(f'{xyxy[3]:.2f}')

                        if self.rate_check:
                            time.sleep(1 / self.rate)
                        im0 = annotator.result()
                        self.send_img.emit(im0)
                        self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                        self.send_statistic.emit(statistic_dic)
                        self.send_probability.emit(probability)
                        if self.save_fold:
                            os.makedirs(self.save_fold, exist_ok=True)
                            if self.vid_cap is None:
                                save_path = os.path.join(self.save_fold,
                                                         time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                       time.localtime()) + '.jpg')
                                cv2.imwrite(save_path, im0)
                        if percent == self.percent_length:
                            print(count)
                            self.send_percent.emit(0)
                            self.send_msg.emit('finished')
                            if hasattr(self, 'out'):
                                self.out.release()

                            break
     except Exception as e:
            self.send_msg.emit('%s' % e)

class export(QThread):
    send_msg = pyqtSignal(str)
    def __init__(self):
        super(export, self).__init__()
        self.weights = './best(1).pt'
        self.current_weight = './best(1).pt'
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False                   # jump out of the loop
        self.is_continue = True                 # continue/pause
        self.percent_length = 1000              # progress bar
        self.rate_check = True                  # Whether to enable delay
        self.rate = 100
        self.save_fold = 'C:/'
        self.include = ['torchscript']

    def export_formats(self):
        # YOLOv5 export formats
        x = [
            ['PyTorch', '-', '.pt', True, True],
            ['TorchScript', 'torchscript', '.torchscript', True, True],
            ['ONNX', 'onnx', '.onnx', True, True],
            ['OpenVINO', 'openvino', '_openvino_model', True, False],
            ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
            ['TensorFlow GraphDef', 'pb', '.pb', True, True],
            ['TensorFlow Lite', 'tflite', '.tflite', True, False],
            ['TensorRT', 'engine', '.engine', False, True]]
        return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])

    def try_export(inner_func):
        # YOLOv5 export decorator, i..e @try_export
        inner_args = get_default_args(inner_func)

        def outer_func(*args, **kwargs):
            prefix = inner_args['prefix']
            try:
                with Profile() as dt:
                    f, model = inner_func(*args, **kwargs)
                LOGGER.info(f'{prefix} export success ✅ , saved as {f} ({file_size(f):.1f} MB)')
                return f, model
            except Exception as e:
                LOGGER.info(f'{prefix} export failure ❌ : {e}')
                return None, None

        return outer_func

    @try_export
    def export_torchscript(self, model, im, file, optimize, prefix=colorstr('TorchScript:')):
        # YOLOv5 TorchScript model export
        LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = file.with_suffix('.torchscript')
        print(f)
        fm = os.path.basename(f)
        print(fm)
        open_fold = self.save_fold
        open_fold = open_fold + '/' + fm
        print(open_fold)
        ts = torch.jit.trace(model, im, strict=False)
        d = {'shape': im.shape, 'stride': int(max(model.stride)), 'names': model.names}
        extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
        if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            optimize_for_mobile(ts)._save_for_lite_interpreter(open_fold, _extra_files=extra_files)
        else:
            ts.save(open_fold, _extra_files=extra_files)
        self.send_msg.emit(f'export success ✅ , saved as {open_fold} ({file_size(open_fold):.1f} MB)')
        return open_fold, None

    @try_export
    def export_onnx(self, model, im, file, opset, dynamic, simplify, prefix=colorstr('ONNX:')):
        # YOLOv5 ONNX export
        import onnx

        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = str(file.with_suffix('.onnx'))
        print(f)
        fm = os.path.basename(f)
        print(fm)
        open_fold = self.save_fold
        open_fold = open_fold + '/' + fm
        print(open_fold)

        output_names = ['output0']
        if dynamic:
            dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
            if isinstance(model, DetectionModel):
                dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

        torch.onnx.export(
            model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
            im.cpu() if dynamic else im,
            f,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
            input_names=['images'],
            output_names=output_names,
            dynamic_axes=dynamic or None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Metadata
        d = {'stride': int(max(model.stride)), 'names': model.names}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(model_onnx, open_fold)

        # Simplify
        if simplify:
            try:
                cuda = torch.cuda.is_available()
                check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
                import onnxsim

                LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'assert check failed'
                onnx.save(model_onnx, open_fold)
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')
        self.send_msg.emit(f'export success ✅ , saved as {open_fold} ({file_size(open_fold):.1f} MB)')
        return open_fold, model_onnx

    @try_export
    def export_openvino(self, file, metadata, half, int8, data, prefix=colorstr('OpenVINO:')):
        # YOLOv5 OpenVINO export
        import openvino.runtime as ov  # noqa
        from openvino.tools import mo  # noqa

        LOGGER.info(f'\n{prefix} starting export with openvino {ov.__version__}...')
        open_fold = self.save_fold
        f = str(file).replace(file.suffix, f'_openvino_model{os.sep}')
        f_onnx = file.with_suffix('.onnx')
        print(f_onnx)
        f_ov = str(Path(f) / file.with_suffix('.xml').name)
        print(f_ov)
        if int8:
            check_requirements('nncf>=2.4.0')  # requires at least version 2.4.0 to use the post-training quantization
            import nncf
            import numpy as np
            from openvino.runtime import Core

            from utils.dataloaders import create_dataloader
            core = Core()
            onnx_model = core.read_model(f_onnx)  # export

            def prepare_input_tensor(image: np.ndarray):
                input_tensor = image.astype(np.float32)  # uint8 to fp16/32
                input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

                if input_tensor.ndim == 3:
                    input_tensor = np.expand_dims(input_tensor, 0)
                return input_tensor

            def gen_dataloader(yaml_path, task='train', imgsz=640, workers=4):
                data_yaml = check_yaml(yaml_path)
                data = check_dataset(data_yaml)
                dataloader = create_dataloader(data[task],
                                               imgsz=imgsz,
                                               batch_size=1,
                                               stride=32,
                                               pad=0.5,
                                               single_cls=False,
                                               rect=False,
                                               workers=workers)[0]
                return dataloader

            # noqa: F811

            def transform_fn(data_item):
                img = data_item[0].numpy()
                input_tensor = prepare_input_tensor(img)
                return input_tensor

            ds = gen_dataloader(data)
            quantization_dataset = nncf.Dataset(ds, transform_fn)
            ov_model = nncf.quantize(onnx_model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED)
        else:
            ov_model = mo.convert_model(f_onnx, model_name=file.stem, framework='onnx', compress_to_fp16=half)  # export
            # shutil.move(ov_model, open_fold)
        ov.serialize(ov_model, f_ov)  # save
        yaml_save(Path(f) / file.with_suffix('.yaml').name, metadata)  # add metadata.yaml
        shutil.move(f, open_fold)
        self.send_msg.emit(f'export success ✅ , saved as {open_fold} ({file_size(open_fold):.1f} MB)')
        return open_fold, None

    def export_engine(self, model, im, file, half, dynamic, simplify, workspace=4, verbose=False,
                      prefix=colorstr('TensorRT:')):
        # YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt
        assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
        try:
            import tensorrt as trt
        except Exception:
            if platform.system() == 'Linux':
                check_requirements('nvidia-tensorrt', cmds='-U --index-url https://pypi.ngc.nvidia.com')
            import tensorrt as trt
        if trt.__version__[0] == '7':  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
            grid = model.model[-1].anchor_grid
            model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
            export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
            model.model[-1].anchor_grid = grid
        else:  # TensorRT >= 8
            check_version(trt.__version__, '8.0.0', hard=True)  # require tensorrt>=8.0.0
            print(8)
            self.export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
            print(9)
        onnx = file.with_suffix('.onnx')

        LOGGER.info(f'\n{prefix} starting export with TensorRT {trt.__version__}...')
        assert onnx.exists(), f'failed to export ONNX file: {onnx}'
        f = file.with_suffix('.engine')  # TensorRT engine file
        fm = os.path.basename(f)
        open_fold = self.save_fold
        open_fold = open_fold + '/' + fm
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

        if dynamic:
            if im.shape[0] <= 1:
                LOGGER.warning(f'{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument')
            profile = builder.create_optimization_profile()
            for inp in inputs:
                profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
            config.add_optimization_profile(profile)

        LOGGER.info(
            f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {open_fold}')
        if builder.platform_has_fast_fp16 and half:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(open_fold, 'wb') as t:
            t.write(engine.serialize())
        self.send_msg.emit(f'export success ✅ , saved as {open_fold} ({file_size(open_fold):.1f} MB)')
        return open_fold, None

    @try_export
    def export_saved_model(self,
                           model,
                           im,
                           file,
                           dynamic,
                           tf_nms=False,
                           agnostic_nms=False,
                           topk_per_class=100,
                           topk_all=100,
                           iou_thres=0.45,
                           conf_thres=0.25,
                           keras=False,
                           prefix=colorstr('TensorFlow SavedModel:')):
        # YOLOv5 TensorFlow SavedModel export
        try:
            import tensorflow as tf
        except Exception:
            import tensorflow as tf
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        from models.tf import TFModel

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = str(file).replace('.pt', '_saved_model')
        batch_size, ch, *imgsz = list(im.shape)  # BCHW

        tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
        im = tf.zeros((batch_size, *imgsz, ch))  # BHWC order for TensorFlow
        _ = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
        inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if dynamic else batch_size)
        outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        keras_model.trainable = False
        keras_model.summary()
        if keras:
            keras_model.save(f, save_format='tf')
        else:
            spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
            m = tf.function(lambda x: keras_model(x))  # full model
            m = m.get_concrete_function(spec)
            frozen_func = convert_variables_to_constants_v2(m)
            tfm = tf.Module()
            tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if tf_nms else frozen_func(x), [spec])
            tfm.__call__(im)
            open_fold = self.save_fold

            tf.saved_model.save(tfm, open_fold, options=tf.saved_model.SaveOptions(experimental_custom_gradients=False))
        self.send_msg.emit(f'export success ✅ , saved as {open_fold} ({file_size(open_fold):.1f} MB)')
        return open_fold, keras_model

    @try_export
    def export_pb(self, keras_model, file, prefix=colorstr('TensorFlow GraphDef:')):
        # YOLOv5 TensorFlow GraphDef *.pb export https://github.com/leimao/Frozen_Graph_TensorFlow
        import tensorflow as tf
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        f = file.with_suffix('.pb')

        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(m)
        frozen_func.graph.as_graph_def()
        open_fold = self.save_fold
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=open_fold, name=f.name, as_text=False)
        self.send_msg.emit(f'export success ✅ , saved as {open_fold} ({file_size(open_fold):.1f} MB)')
        return open_fold, None

    @try_export
    def export_tflite(self, keras_model, im, file, int8, data, nms, agnostic_nms, prefix=colorstr('TensorFlow Lite:')):
        # YOLOv5 TensorFlow Lite export
        import tensorflow as tf

        LOGGER.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
        batch_size, ch, *imgsz = list(im.shape)  # BCHW
        f = str(file).replace('.pt', '-fp16.tflite')
        print(f)
        fm = os.path.basename(f)
        print(fm)
        open_fold = self.save_fold
        open_fold = open_fold + '/' + fm
        print(open_fold)
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if int8:
            from models.tf import representative_dataset_gen
            dataset = LoadImages(check_dataset(check_yaml(data))['train'], img_size=imgsz, auto=False)
            converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib=100)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = []
            converter.inference_input_type = tf.uint8  # or tf.int8
            converter.inference_output_type = tf.uint8  # or tf.int8
            converter.experimental_new_quantizer = True
            f = str(file).replace('.pt', '-int8.tflite')
        if nms or agnostic_nms:
            converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

        tflite_model = converter.convert()
        open(open_fold, 'wb').write(tflite_model)
        self.send_msg.emit(f'export success ✅ , saved as {open_fold} ({file_size(open_fold):.1f} MB)')
        return open_fold, None

    def add_tflite_metadata(self, file, metadata, num_outputs):
        # Add metadata to *.tflite models per https://www.tensorflow.org/lite/models/convert/metadata
        with contextlib.suppress(ImportError):
            # check_requirements('tflite_support')
            from tflite_support import flatbuffers
            from tflite_support import metadata as _metadata
            from tflite_support import metadata_schema_py_generated as _metadata_fb

            tmp_file = Path('/tmp/meta.txt')
            with open(tmp_file, 'w') as meta_f:
                meta_f.write(str(metadata))

            model_meta = _metadata_fb.ModelMetadataT()
            label_file = _metadata_fb.AssociatedFileT()
            label_file.name = tmp_file.name
            model_meta.associatedFiles = [label_file]

            subgraph = _metadata_fb.SubGraphMetadataT()
            subgraph.inputTensorMetadata = [_metadata_fb.TensorMetadataT()]
            subgraph.outputTensorMetadata = [_metadata_fb.TensorMetadataT()] * num_outputs
            model_meta.subgraphMetadata = [subgraph]

            b = flatbuffers.Builder(0)
            b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
            metadata_buf = b.Output()

            populator = _metadata.MetadataPopulator.with_model_file(file)
            populator.load_metadata_buffer(metadata_buf)
            populator.load_associated_files([str(tmp_file)])
            populator.populate()
            tmp_file.unlink()

    # 'torchscript', 'onnx', 'openvino', 'saved_model', 'pb', 'tflite', 'engine'
    @smart_inference_mode()
    def run(self,
            data='data/ship.yaml',  # 'dataset.yaml path'
            imgsz=(640, 640),  # image (height, width)
            batch_size=1,  # batch size
            device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            half=False,  # FP16 half-precision export
            inplace=False,  # set YOLOv5 Detect() inplace=True
            keras=False,  # use Keras
            optimize=False,  # TorchScript: optimize for mobile
            int8=False,  # CoreML/TF INT8 quantization
            dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
            simplify=False,  # ONNX: simplify model
            opset=13,  # ONNX: opset version
            verbose=False,  # TensorRT: verbose log
            workspace=8,  # TensorRT: workspace size (GB)
            nms=False,  # TF: add NMS to model
            agnostic_nms=False,  # TF: add agnostic NMS to model
            topk_per_class=100,  # TF.js NMS: topk per class to keep
            topk_all=100,  # TF.js NMS: topk for all classes to keep
            iou_thres=0.45,  # TF.js NMS: IoU threshold
            conf_thres=0.25,  # TF.js NMS: confidence threshold
    ):
        t = time.time()
        self.include = [x.lower() for x in self.include]  # to lowercase
        fmts = tuple(self.export_formats()['Argument'][1:])  # --include arguments
        flags = [x in self.include for x in fmts]
        assert sum(flags) == len(self.include), f'ERROR: Invalid --include {self.include}, valid --include arguments are {fmts}'
        jit, onnx, xml, saved_model, pb, tflite, engine = flags  # export booleans
        file = Path(url2file(self.weights) if str(self.weights).startswith(('http:/', 'https:/')) else self.weights)  # PyTorch weights
        # Load PyTorch model
        device = select_device(device)
        print(2)
        if half:
            assert device.type != 'cpu' , '--half only compatible with GPU export, i.e. use --device 0'
            assert not dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both'
        print(3)
        if self.weights:
           model = attempt_load(self.weights, map_location=device, inplace=True, fuse=True)  # load FP32 model
        else:
            print(4)
        # Checks
        print(5)
        imgsz *= 2 if len(imgsz) == 1 else 1  # expand
        if optimize:
            assert device.type == 'cpu', '--optimize not compatible with cuda devices, i.e. use --device cpu'

        # Input
        gs = int(max(model.stride))  # grid size (max stride)
        imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
        im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

        # Update model
        model.eval()
        for k, m in model.named_modules():
            if isinstance(m, Detect):
                m.inplace = inplace
                m.dynamic = dynamic
                m.export = True

        for _ in range(2):
            y = model(im)  # dry runs
        if half:
            im, model = im.half(), model.half()  # to FP16
        shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
        metadata = {'stride': int(max(model.stride)), 'names': model.names}  # model metadata
        LOGGER.info(
            f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")
        while True:
        # Exports
          f = [''] * len(fmts)  # exported filenames
          warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
          if jit:  # TorchScript
            f[0], _ = self.export_torchscript(model, im, file, optimize)
          if onnx or xml:  # OpenVINO requires ONNX
            f[1], _ = self.export_onnx(model, im, file, opset, dynamic, simplify)
          if xml:  # OpenVINO
            f[2], _ = self.export_openvino(file, metadata, half, int8, data)
          if any((saved_model, pb, tflite)):  # TensorFlow formats
            f[3], s_model = self.export_saved_model(model.cpu(),
                                               im,
                                               file,
                                               dynamic,
                                               tf_nms=nms or agnostic_nms,
                                               agnostic_nms=agnostic_nms,
                                               topk_per_class=topk_per_class,
                                               topk_all=topk_all,
                                               iou_thres=iou_thres,
                                               conf_thres=conf_thres,
                                               keras=keras)
            if pb:  # pb prerequisite to tfjs
                f[4], _ = self.export_pb(s_model, file)
            if tflite:
                f[5], _ = self.export_tflite(s_model, im, file, int8, data=data, nms=nms, agnostic_nms=agnostic_nms)
                self.add_tflite_metadata(f[5], metadata, num_outputs=len(s_model.outputs))
          if engine:  # TensorRT required before ONNX
            print('Moon')
            f[6], _ = self.export_engine(model, im, file, half, dynamic, simplify, workspace, verbose)
            print(6)
        # Finish
          f = [str(x) for x in f if x]  # filter out '' and None
          if any(f):
            det = (isinstance(model, DetectionModel))  # type
            h = '--half' if half else ''  # --half FP16 inference arg
          break


class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False

        # style 1: window can be stretched
        # self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)

        # style 2: window can not be stretched
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                            | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        # self.setWindowOpacity(0.85)  # Transparency of window

        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        # show Maximized window
        self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close)

        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # search models automatically
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        self.comboBox_2.clear()
        self.convert_list = ['torchscript', 'onnx', 'openvino', 'saved_model', 'pb', 'tflite', 'engine']
        self.comboBox_2.clear()
        self.comboBox_2.addItems(self.convert_list)

        # yolov5 thread
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.det_thread.source = '0'
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_probability.connect(self.show_probability)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))
        self.det_thread.save_fold = './result'

        self.modelButton.clicked.connect(self.open_model)
        self.savefileButton.clicked.connect(self.save_file)
        self.fileButton.clicked.connect(self.open_file)
        self.imagefileButton.clicked.connect(self.open_imagefile)
        self.cameraButton.clicked.connect(self.chose_cam)
        self.rtspButton.clicked.connect(self.chose_rtsp)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.saveCheckBox.clicked.connect(self.is_save)
        self.load_setting()

        self.export = export()
        self.pushButton.clicked.connect(self.run_export)
        self.export.send_msg.connect(lambda x: self.show_msg(x))
        self.model_convert = self.comboBox_2.currentText()
        self.export.weights = "./pt/%s" % self.model_type
        self.comboBox_2.currentTextChanged.connect(self.convert_model)

    def convert_model(self, x):
        self.model_convert = self.comboBox_2.currentText()
        list=[]
        list.append(self.model_convert)
        print(list)
        self.export.include = list
        self.statistic_msg('Convert model to %s' % x)

    def run_export(self):
        if self.pushButton.isChecked():
            #self.saveCheckBox.setEnabled(False)
            self.export.is_continue = True
            print(1)
            if not self.export.isRunning():
                self.export.start()
        else:
            self.export.is_continue = False
            #self.statistic_msg('Pause')

    def save_file(self):
        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        save_file = config['save_file']
        if not os.path.exists(save_file):
            save_file = os.getcwd()
        savefilename = QFileDialog.getExistingDirectory(self, 'savefile', save_file)
        print(savefilename)
        if savefilename:
            self.det_thread.save_fold = savefilename
            self.export.save_fold = savefilename
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(savefilename)))
            config['save_file'] = os.path.dirname(savefilename)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()



    def open_model(self):

        config_file = 'config/fold.json'
        src_file = "C:/Users/Administrator/Desktop/YOLOV5尝试加入带轨迹捕捉的deepsort算法/pt"
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_model = config['open_model']
        if not os.path.exists(open_model):
            open_model = os.getcwd()
        modelname, _ = QFileDialog.getOpenFileName(self, 'modelname', open_model, "Model File(*.pt))")
        print(modelname)
        if modelname:
            shutil.move(modelname, src_file)
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(modelname)))
            config['open_model'] = os.path.dirname(modelname)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def show_probability(self, probability):
        try:
            list1 = []
            num = int(len(probability) / 6)
            for i in range(num):
                list1.append('')
                list1.append(' target' + str(i + 1) + ': ' + str(probability[6 * i]))
                list1.append(' class: ' + str(probability[6 * i + 1]))
                list1.append(' xmin: ' + str(probability[6 * i + 2]))
                list1.append(' ymin: ' + str(probability[6 * i + 3]))
                list1.append(' xmax: ' + str(probability[6 * i + 4]))
                list1.append(' ymax: ' + str(probability[6 * i + 5]))
            self.resultWidget.addItems(list1)

        except Exception as e:
            print(repr(e))

    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def is_save(self):
        if self.saveCheckBox.isChecked():
            self.det_thread.save_fold = './result'
        else:
            self.det_thread.save_fold = None

    def checkrate(self):
        if self.checkBox.isChecked():
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.67:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='Tips', text='Loading rtsp stream', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    def chose_cam(self):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='Tips', text='Loading camera', time=2000, auto=True).exec_()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('Loading camera：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            savecheck = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "check": check,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                check = 0
                savecheck = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                check = config['check']
                savecheck = config['savecheck']
        self.confSpinBox.setValue(conf)
        self.iouSpinBox.setValue(iou)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check
        self.saveCheckBox.setCheckState(savecheck)
        self.is_save()

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
            self.det_thread.conf_thres = x/100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x*100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x/100)
            self.det_thread.iou_thres = x/100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "finished":
            self.saveCheckBox.setEnabled(True)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.export.weights = "./pt/%s" % self.model_type
        self.statistic_msg('Change model to %s' % x)

    def open_file(self):

        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def open_imagefile(self):

        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_imagefile = config['open_imagefile']
        if not os.path.exists(open_imagefile):
            open_imagefile = os.getcwd()
        filename = QFileDialog.getExistingDirectory(self, 'imagefile', open_imagefile)
        if filename:
            self.det_thread.source = filename
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(filename)))
            config['open_imagefile'] = os.path.dirname(filename)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.statistic_msg('Detecting >> model：{}，file：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('Pause')

    def stop(self):
        self.det_thread.jump_out = True
        self.saveCheckBox.setEnabled(True)

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        self.det_thread.jump_out = True
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config['savecheck'] = self.saveCheckBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='Tips', text='Closing the program', time=2000, auto=True).exec_()
        sys.exit(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    # myWin.showMaximized()
    sys.exit(app.exec_())
