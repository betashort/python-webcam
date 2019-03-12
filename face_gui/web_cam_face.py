import cv2
import numpy as np

#Webカメラから入力
cap = cv2.VideoCapture(0)

# カスケードファイルを指定して、検出器を作成
face_cascade_file = "haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_file)

eye_cascade_file = "haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(eye_cascade_file)

while True:
    #画像を取得
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 3)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), -1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), -1)



    cv2.imshow('img', img)

    #ESCかEnterキーが押されたら終了
    k = cv2.waitKey(1)
    if k == 13:
        break

cap.release()
cv2.destroyAllWindows()
