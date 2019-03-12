import cv2
import numpy as np

#Webカメラから入力
cap = cv2.VideoCapture(0)

#動画書き出し用のオブジェクト
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

fps = 15.0
size = (640, 360)
writer = cv2.VideoWriter('out1.m4v', fmt, fps, size)

'''
カスケードファイルを指定して、検出器を作成
'''
#face1
face_cascade_file1 = "haarcascade_frontalface_alt.xml"
face_cascade_1 = cv2.CascadeClassifier(face_cascade_file1)

#face2
face_cascade_file2 = "haarcascade_frontalcatface.xml"
face_cascade_2 = cv2.CascadeClassifier(face_cascade_file2)

#face3
face_cascade_file3 = "haarcascade_frontalcatface_extended.xml"
face_cascade_3 = cv2.CascadeClassifier(face_cascade_file3)

#face4
face_cascade_file4 = "haarcascade_frontalface_alt2.xml"
face_cascade_4 = cv2.CascadeClassifier(face_cascade_file4)

#face5
face_cascade_file5 = "haarcascade_frontalface_default.xml"
face_cascade_5 = cv2.CascadeClassifier(face_cascade_file5)

#eye
eye_cascade_file = "haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(eye_cascade_file)

#anime画像
anime_file = "jeanne.jpg"
anime_face = cv2.imread(anime_file)

#画像
iqut_file = "out_face.jpg"
iqut_face = cv2.imread(iqut_file)

anime2_file = "IMG_6138.JPG"
anime2_face = cv2.imread(anime2_file)

'''
モザイク
'''
def mosaic(img, rect, size):
    (x1, y1, x2, y2) = rect
    w = x2 - x1
    h = y2 - y1
    i_rect = img[y1:y2, x1:x2]

    i_small = cv2.resize(i_rect, (size, size))
    i_mos = cv2.resize(i_small, (w, h), interpolation=cv2.INTER_AREA)

    img2 = img.copy()
    img2[y1:y2, x1:x2] = i_mos
    return img2


def anime_face_func(img, rect):
    (x1, y1, x2, y2) = rect
    w = x2 - x1
    h = y2 - y1
    if (w <= 150):
        img_face = cv2.resize(anime_face, (w, h))
    else:
        img_face = cv2.resize(anime2_face, (w, h))

    img2 = img.copy()
    img2[y1:y2, x1:x2] = img_face
    return img2





'''
動画処理
'''
while True:
    #画像を取得
    _, img = cap.read()
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_5.detectMultiScale(gray, 1.1, 2)
    '''
    for (x, y, w, h) in faces:
        img = mosaic(img, (x, y, x+w, y+h), 5)
    '''


    for (x, y, w, h) in faces:
        img = anime_face_func(img, (x, y, x+w, y+h))



    writer.write(img)

    cv2.imshow('img', img)

    #ESCかEnterキーが押されたら終了
    k = cv2.waitKey(1)
    if k == 13:
        break

writer.release()
cap.release()
cv2.destroyAllWindows()


#print("幅：{:.2f}¥n高さ：{:.2f}¥n".format(w, h))