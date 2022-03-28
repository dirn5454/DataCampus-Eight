import sys
import os
import cv2
import random
import argparse
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from PyQt5 import QtCore, QtGui, QtWidgets
from models.experimental import attempt_load
from utils.general import check_img_size,  non_max_suppression, apply_classifier, \
    scale_coords, set_logging

from utils.torch_utils import select_device, load_classifier, time_sync
from utils.datasets import letterbox
from utils.plots import plot_one_box

from segmentation.seg_config import Config
from segmentation import seg_utils
from segmentation.seg_model import MaskRCNN
from pathlib import Path

import playsound
from pygame import mixer
import skimage.draw
from PIL import ImageFont, ImageDraw, Image

import mediapipe as mp

class InferenceConfig(Config):

    NAME = "person"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 2
    USE_MINI_MASK = True

def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = int(np.abs(radians * 180 / np.pi))

        if angle > 180:
            angle = 360 - angle
        return angle

def seg_people(img, value):
        font_count = ImageFont.truetype('Maplestory Light.ttf', 25)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        x, y, (w, h) = 27, 27, font_count.getsize(value)
        draw.rectangle((x, y, x + w + 5, y + h + 5), fill='black')
        draw.text((30, 30), value, font=font_count, fill=(255, 255, 255, 0))
        img = np.array(img_pil)
        return img

def pose_detection(img, showimg):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:

                landmarks = results.pose_landmarks.landmark

                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                # 고개를 숙이는/젖히는 위험 자세
                left_eye_inner = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y]
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                        landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                right_eye_inner = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y]

                # 각도 계산
                angle1 = calculate_angle(left_elbow, left_shoulder, left_hip)
                angle2 = calculate_angle(right_elbow, right_shoulder, right_hip)
                angle3 = calculate_angle(left_eye_inner, nose, right_eye_inner)

                cv2.putText(showimg, str(angle1),
                            tuple(np.multiply(left_elbow, [showimg.shape[1], showimg.shape[0]]).astype(
                                int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(showimg, str(angle2),
                            tuple(np.multiply(right_elbow, [showimg.shape[1], showimg.shape[0]]).astype(
                                int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                angle_list = []
                angle_list.append(angle1)
                angle_list.append(angle2)
                print("angle1 = ", angle1,"angle2 = ", angle2,"angle3 = ", angle3, "\n")
                for item in angle_list:

                    # 위험자세의 각도 조절 가능
                    if item < 15 or item > 80 or angle3 < 40:
                        # 경고 메세지
                        print("Warning! \n")

                        cv2.putText(showimg, "Warning!", tuple(
                            np.multiply([0.2, 0.5], [showimg.shape[1], showimg.shape[0]]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                    else:
                        pass
            except:
                pass

            mp_drawing.draw_landmarks(showimg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1,
                                                             circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                             circle_radius=2))
            return showimg

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None
        self.time_count = 0
        self.flag = False
        self.mixer = mixer
        self.mixer.init()
        self.mixer.music.load("sound/warning_message.mp3")

        parser = argparse.ArgumentParser(description='Train for Person and Helmet data.')

        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--rcnnweights', required=True, metavar="/path/to/weights.h5",
                            help="Path to weights .h5 file or 'coco'")
        parser.add_argument('--yoloweights', required=True, nargs='+', type=str, default='yolov5s.pt',
                            help='model.pt path(s)')
        parser.add_argument('--logs', required=False, default="/log", metavar="/path/to/logs/",
                            help='Logs directory (default=logs/)')
        self.opt = parser.parse_args()

        m_weights, y_weights, imgsz = self.opt.rcnnweights, self.opt.yoloweights, self.opt.img_size
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'

        cudnn.benchmark = True

        # Mask R-CNN

        config = InferenceConfig()

        self.m_model = MaskRCNN(mode="inference", config=config,
                            model_dir=self.opt.logs)

        self.m_model.load_weights(m_weights, by_name=True)

        # YOLO

        set_logging()
        y_weights = self.opt.yoloweights
        self.y_model = attempt_load(y_weights, map_location=self.device)

        stride = int(self.y_model.stride.max())
        self.imgsz = check_img_size(imgsz, s=stride)

        if self.half:
            self.m_model.half()
            self.y_model.half()

        # Get names and colors
        self.names = self.y_model.module.names if hasattr(self.y_model, 'module') else self.y_model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, -1, 0, -1)
        self.verticalLayout.setSpacing(80)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_img = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_img.sizePolicy().hasHeightForWidth())
        self.pushButton_img.setSizePolicy(sizePolicy)
        self.pushButton_img.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_img.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_img.setFont(font)
        self.pushButton_img.setObjectName("pushButton_img")
        self.verticalLayout.addWidget(self.pushButton_img, 0, QtCore.Qt.AlignHCenter)
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_camera.sizePolicy().hasHeightForWidth())
        self.pushButton_camera.setSizePolicy(sizePolicy)
        self.pushButton_camera.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_camera.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_camera.setFont(font)
        self.pushButton_camera.setObjectName("pushButton_camera")
        self.verticalLayout.addWidget(self.pushButton_camera, 0, QtCore.Qt.AlignHCenter)
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_video.sizePolicy().hasHeightForWidth())
        self.pushButton_video.setSizePolicy(sizePolicy)
        self.pushButton_video.setMinimumSize(QtCore.QSize(150, 100))
        self.pushButton_video.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.pushButton_video.setFont(font)
        self.pushButton_video.setObjectName("pushButton_video")
        self.verticalLayout.addWidget(self.pushButton_video, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "킥라니 멈춰!"))
        self.setWindowIcon(QtGui.QIcon('pyqt_ui/titleUi.png'))
        self.pushButton_img.setIcon(QtGui.QIcon('pyqt_ui/imageUi.png'))
        self.pushButton_img.setIconSize(QtCore.QSize(160,110))
        self.pushButton_camera.setIcon(QtGui.QIcon('pyqt_ui/cameraUi.png'))
        self.pushButton_camera.setIconSize(QtCore.QSize(160,110))
        self.pushButton_video.setIcon(QtGui.QIcon('pyqt_ui/videoUi.png'))
        self.pushButton_video.setIconSize(QtCore.QSize(160,110))
        self.label.setText(_translate("MainWindow", "TextLabel"))

    def init_slots(self):
        self.pushButton_img.clicked.connect(self.button_image_open)
        self.pushButton_video.clicked.connect(self.button_video_open)
        self.pushButton_camera.clicked.connect(self.button_camera_open)
        self.timer_video.timeout.connect(self.show_video_frame)

    def init_logo(self):
        pix = QtGui.QPixmap('pyqt_ui/projectUi.png')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    def button_image_open(self):
        print('button_image_open')
        name_list = []

        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "open image", "", "*.JPG;;*.jpg;;*.png;;All Files(*)")
        img = cv2.imread(img_name)

        if img is not None:

            # segmentation 진행
            r = self.m_model.detect([img], verbose=0)[0]
            if r['masks'].shape[-1] == 0:
                value = '얼굴을 비춰주세요'  # 사람 얼굴이 없을 경우 얼굴을 비춰주세요
            elif r['masks'].shape[-1] > 1:
                value = '탑승인원을 준수해주세요'  # 탑승인원이 2인 이상인 경우 탑승인원을 준수해주세요
            else:
                value = "인원:" + str(r['masks'].shape[-1]) + '명'

            showimg = img

            with torch.no_grad():

                # yolov5 detect를 위한 image 변환 작업
                img = cv2.resize(img, (self.opt.img_size, self.opt.img_size))
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                pred = self.y_model(img, augment=self.opt.augment)[0]

                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)

                # detections
                for i, det in enumerate(pred):
                    if det is not None and len(det):

                        # Rescale boxes from img_size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()

                        # Save results
                        for *xyxy, conf, cls in reversed(det):
                                c = int(cls)
                                n = (det[:, -1] == c).sum()

                                label = self.names[int(c)]
                                if label.find('WithHelmet') != -1:
                                    label = '헬멧 착용'

                                elif label.find('WithoutHelmet') != -1:
                                    label = '헬멧 미착용'

                                name_list.append(label)
                                label = f'{label} {conf:.2f}'
                                showimg = plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)],
                                                line_thickness=2)

            # segmentation 적용
            showimg = seg_people(showimg, value)

            cv2.imwrite('prediction.jpg', showimg)
            self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
            self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
            self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                      QtGui.QImage.Format_RGB32)
            self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

        else:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"Can't open image", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            self.init_logo()

    def button_video_open(self):

        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open video", "", "*.mp4;;*.avi;;All Files(*)")
        flag = self.cap.open(video_name)
        if flag == False:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"Can't open video", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30,
                                       (int(self.cap.get(3)), int(self.cap.get(4))))
            self.timer_video.start(30)
            self.pushButton_video.setDisabled(True)
            self.pushButton_img.setDisabled(True)
            self.pushButton_camera.setDisabled(True)

    def button_camera_open(self):

        if not self.timer_video.isActive():
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"Camera not available",
                                              buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30,
                                           (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timer_video.start(30)
                self.pushButton_video.setDisabled(True)
                self.pushButton_img.setDisabled(True)

        else:
            self.mixer.music.stop()
            self.time_count = 0.0
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.init_logo()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)

    def show_video_frame(self):

        name_list = []
        flag, img = self.cap.read()

        if img is not None:

            showimg = img
            org_img = img # for pose

            r = self.m_model.detect([img], verbose=0)[0]
            if r['masks'].shape[-1] == 0:
                value = '얼굴을 비춰주세요'  # 사람 얼굴이 없을 경우 얼굴을 비춰주세요
            elif r['masks'].shape[-1] > 1:
                value = '탑승인원을 준수해주세요'  # 탑승인원이 2인 이상인 경우 탑승인원을 준수해주세요
            else:
                value = "인원:" + str(r['masks'].shape[-1]) + '명'

            with torch.no_grad():

                # Convert
                img = letterbox(img, new_shape=self.opt.img_size)[0]
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()
                img /= 255.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # detections
                t1 = time_sync()
                pred = self.y_model(img, augment=self.opt.augment)[0]
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                t2 = time_sync()
                warning_message = '헬멧을 착용하세요.'

                for i, det in enumerate(pred):
                    if det is not None and len(det):

                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()

                        for *xyxy, conf, cls in reversed(det):

                            c = int(cls)
                            n = (det[:, -1] == c).sum()  # detections per class
                            label = self.names[int(c)]

                            if label.find('WithHelmet') != -1:
                                self.time_count = 0.0
                                label = '헬멧 착용'
                                self.flag = False
                                self.mixer.music.stop()

                            elif label.find('WithoutHelmet') != -1:
                                self.time_count += t2 - t1
                                print('헬멧 미착용 시간', self.time_count)
                                label = '헬멧 미착용'

                            name_list.append(label)
                            label = f'{label} {conf:.2f}'

                            if self.time_count >= 5.0:
                                showimg = plot_one_box(xyxy, showimg, label=warning_message, color=self.colors[int(cls)],
                                                        line_thickness=5)

                            else:
                                showimg = plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)],
                                                        line_thickness=4)
                    # segmentation 적용
                    showimg = seg_people(showimg, value)

                    # pose
                    showimg = pose_detection(org_img, showimg)

                    if self.time_count >= 5.0:
                        if self.flag == False: # 음악이 재생되고 있지 않다면 PLAY
                            self.flag = True
                            self.mixer.music.play(-1)
            
            # video 저장 및 출력
            self.out.write(showimg)
            self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                    QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setDisabled(False)
            self.init_logo()

if __name__ == '__main__':

    # implement
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()

    sys.exit(app.exec_())
    self.mixer.music.stop()