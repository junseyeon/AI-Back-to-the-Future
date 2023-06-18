import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from pytube import YouTube
import cv2
import numpy as np
from keras.models import model_from_json

from Colorizer import *


form = uic.loadUiType("form.ui")[0]

class WindowView(QWidget, form) :
    def __init__(self) :
        super().__init__()
        self.initUI()
        self.show()

    def initUI(self):
        self.setupUi(self)
        self.setWindowTitle("딥러닝 과제")
        self.btn_convert_color.clicked.connect(lambda: self.get_video_color())
        self.btn_train.clicked.connect(lambda: self.get_video_train())
        self.btn_train_emotion.clicked.connect(lambda: self.get_video_train_emotion())

    def get_video_color(self):
        # video_download_url = self.input_url.toPlainText()

        # download_path = "./video/"
        # file_name = "video.mp4"

        # yt = YouTube(video_download_url)
        # stream = yt.streams.get_highest_resolution()
        # stream.download(download_path, file_name)

        colorizer = Colorizer(width=680, height=780)
        colorizer.processVideo("video/video.mp4")

    
    def get_video_train(self):
        cascade_filename = 'haarcascade_frontalface_alt.xml'

        cascade = cv2.CascadeClassifier(cascade_filename)

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

        age_net = cv2.dnn.readNetFromCaffe(
            'deploy_age.prototxt',
            'age_net.caffemodel')

        gender_net = cv2.dnn.readNetFromCaffe(
            'deploy_gender.prototxt',
            'gender_net.caffemodel')

        age_list = ['(0 ~ 2)', '(4 ~ 6)', '(8 ~ 12)', '(15 ~ 20)', '(25 ~ 32)', '(38 ~ 43)', '(48 ~ 53)', '(60 ~ 100)']
        gender_list = ['Male', 'Female']

        cam = cv2.VideoCapture('video.mp4')
        self.videoDetector(cam, cascade, age_net, gender_net, MODEL_MEAN_VALUES, age_list, gender_list)


    def get_video_train_emotion(self):
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        # load json and create model
        json_file = open('model/emotion_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        emotion_model = model_from_json(loaded_model_json)

        # load weights into new model
        emotion_model.load_weights("model/emotion_model.h5")
        print("Loaded model from disk")

        cap = cv2.VideoCapture("./video.mp4")

        while True:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1280, 720))
            if not ret:
                break
            face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces available on camera
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            # take each face available on the camera and Preprocess it
            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                # predict the emotions
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def videoDetector(self, cam, cascade, age_net, gender_net, MODEL_MEAN_VALUES, age_list, gender_list):
        while True:
            ret, img = cam.read()

            try:
                img = cv2.resize(img, dsize=None, fx=1.0, fy=1.0)
            except:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results = cascade.detectMultiScale(gray,  # 입력 이미지
                                            scaleFactor=1.1,  # 이미지 피라미드 스케일 factor
                                            minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                            minSize=(20, 20)  # 탐지 객체 최소 크기
                                            )

            for box in results:
                x, y, w, h = box
                face = img[int(y):int(y + h), int(x):int(x + h)].copy()
                blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                # 성별
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_preds.argmax()

                # 나이
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_preds.argmax()

                info = gender_list[gender] + ' ' + age_list[age]

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
                cv2.putText(img, info, (x, y - 15), 0, 0.5, (0, 255, 0), 1)


            cv2.imshow('train_video', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    window_view = WindowView()
    window_view.show()
    app.exec_()
