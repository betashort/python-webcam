import cv2
import numpy as np

#Webカメラから入力
cap = cv2.VideoCapture(0)

img_last = None
green = (0, 255, 0)

while True:
    #画像を取得
    _, frame = cap.read()
    frame = cv2.resize(frame, (500, 300))

    #白黒画像に変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    img_b = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

    #差分を確認する
    if img_last is None:
        img_last = img_b
        continue

    frame_diff = cv2.absdiff(img_last, img_b)
    cnts = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    #差分があった点を画面に描く
    for pt in cnts:
        x, y, w, h = cv2.boundingRect(pt)
        if w < 30:
            continue
        cv2.rectangle(frame, (x, y), (x+y, y+h), green, 2)

    #今回のフレームを保存
    img_last = img_b

    #ウィンドウに画像を出力
    cv2.imshow('OpenCV Web Camera', frame)
    cv2.imshow('diff camera', frame_diff)
    #ESCかEnterキーが押されたら終了
    k = cv2.waitKey(1)
    if k == 13:
        break

cap.release()
cv2.destroyAllWindows()
