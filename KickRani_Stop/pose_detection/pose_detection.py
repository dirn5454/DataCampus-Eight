import PIL
from PIL import ImageFont, ImageDraw, Image
from pygame import mixer
import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
from gtts import gTTS
import time

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = int(np.abs(radians * 180 / np.pi))

    if angle > 180:
        angle = 360 - angle

    return angle

def pose_detect(im0):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    font = ImageFont.truetype('Maplestory Light.ttf', 15)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(im0)
        pil_img = Image.fromarray(im0)

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
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            right_eye_inner = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y]

            # 각도 계산
            angle1 = calculate_angle(left_elbow, left_shoulder, left_hip)
            angle2 = calculate_angle(right_elbow, right_shoulder, right_hip)
            angle3 = calculate_angle(left_eye_inner, nose, right_eye_inner)

            x = str(angle1) + '˚'
            y = str(angle2) + '˚'

            draw = ImageDraw.Draw(pil_img)
            draw.text(tuple(np.multiply(left_elbow, [640, 480]).astype(int)), x,
                      font=font, fill=(255, 255, 255, 0))
            draw.text(tuple(np.multiply(right_elbow, [640, 480]).astype(int)), y,
                      font=font, fill=(255, 255, 255, 0))
            im0 = np.array(pil_img)

            # 위험자세 경고음
            angle_list = []
            angle_list.append(angle1)
            angle_list.append(angle2)
            for item in angle_list:

                # 웹캠 위치에 따라 위험자세의 각도 조절 가능
                if item < 20 or item > 80 or angle3 < 50:
                    mixer.music.play()
                else:
                    mixer.music.pause()

        except:
            pass

        mp_drawing.draw_landmarks(im0, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                                  )