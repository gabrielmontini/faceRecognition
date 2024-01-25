import cv2 as cv
import os
import numpy

face_detection = cv.CascadeClassifier('database/haarcascade_frontalface_default.xml')
video_capture = cv.VideoCapture(0)
# CPF will be used as a unique identifier for user frames saved
cpf = input('Enter CPF:')
device = input('Enter Device Number:')
# It will stop to record user face when 50 samples were stored
sampleNum = 0

# API FLOW:
# - check if device number and/or cpf already exists
# -


while True:
    ret, frame = video_capture.read()
    # CONVERT IMAGE
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray_image)
    for (x, y, h, w) in faces:
        sampleNum = sampleNum + 1
        # Data saved
        # newPathCPF = ('dataset/' + str(cpf))
        # newPathDevice = ('dataset/' + str(cpf) + '/' + str(device))
        # isExist = os.path.exists()
        #
        # # verificando se o CPF ja tem registro
        # if isExist(newPathCPF):
        #
        #     # checando se temos registro facial feito pelo próprio
        #     # usuário
        #     if len (os.listdir(newPathCPF)) == 0:
        #
        #     if isExist(newPathDevice):
        #         print('--prosseguir para validar rosto')
        #     else:
        #
        #         cv.imwrite(newPathDevice + "/User." + str(device) + "." + str(cpf) + "." + str(sampleNum) + ".jpg",
        #                    gray_image[y:y + h, x:x + w])

        cv.imwrite("dataset/User." + str(device) + "." + str(cpf) + "." + str(sampleNum) + ".jpg", gray_image[y:y + h, x:x + w])
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.waitKey(100)
        pass
    cv.imshow("Faces", frame)
    cv.waitKey(1)
    if sampleNum > 50:
        break
    pass


