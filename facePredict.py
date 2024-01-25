import cv2 as cv
import numpy as numpy
import argparse

font = cv.FONT_HERSHEY_SIMPLEX


def detectanddisplay(frame):

    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_image = cv.equalizeHist(gray_image)
    faces = face_classifier.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, ((w + x) // 7, (h + y) // 4), 0, 0, 360, (255, 0, 0), 1, 4)
        id_number, conf = rec.predict(gray_image[y:y + h, x:x + w])
        cv.putText(frame, str(id_number), (x, y + h), font, 1, (255, 255, 255), 1, cv.LINE_AA)
        face_roi = gray_image[y:y + h, x:x + w]
        eyes = eyes_classifier.detectMultiScale(face_roi)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            frame = cv.ellipse(frame, eye_center, (w2 // 2, h2 // 3), 0, 0, 360, (255, 0, 0), 1, 4)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    cv.imshow('Capture - Face detection', frame_rgb)


face_classifier = cv.CascadeClassifier('database/haarcascade_frontalface_default.xml')
eyes_classifier = cv.CascadeClassifier('database/haarcascade_eye_tree_eyeglasses.xml')
video_capture = cv.VideoCapture(0)
rec = cv.face.LBPHFaceRecognizer_create()
print(rec.read('trainingData.yml'))
if not face_classifier:
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_classifier:
    print('--(!)Error loading eyes cascade')
    exit(0)
video_capture = cv.VideoCapture(0)
if not video_capture.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = video_capture.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectanddisplay(frame)
    if cv.waitKey(10) == 27:
        break

video_capture.release()
cv.destroyAllWindows()
