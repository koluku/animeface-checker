import cv2
import os
from tqdm import tqdm
from time import time

def check(file):
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = 'lbpcascade_animeface.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(96, 96))
    return facerect

if __name__ == '__main__':
    start = time()
    image_count = 0
    detected_count = 0
    face_count = 0
    files = os.listdir('./')
    for file in tqdm(files):
        splited_name = os.path.splitext(file)
        file_extension = splited_name[1]
        image_extension = {'.jpg', '.jpeg', '.gif', '.png'}
        for i in image_extension:
            if file_extension == i:
                image_count += 1
                facerect = check(file)
                faces = len(facerect)
                if faces != 0:
                    face_count += faces
                    detected_count += 1
            else:
                continue
    end = time()
    time = round(end - start,1)
    per = round(detected_count / image_count * 100, 1)
    print("画像枚数: {}枚".format(image_count))
    print("検出された画像: {}枚".format(detected_count))
    print("検出された顔: {}個".format(face_count))
    print("認識精度: {}%".format(per))
    print("計算時間: {}秒".format(time))
