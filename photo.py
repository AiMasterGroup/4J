import numpy as np
import cv2
import tensorflow as tf
from PIL import ImageDraw, Image
import os
from keras.models import model_from_json
from keras.optimizers import SGD

test_y = np.load('test_label_array_18*32.npy')
#test_x = np.load('test_pixel1_array.npy')
begincwd = os.getcwd()
os.chdir('model')
model = model_from_json(open('yolo_model_loss1_batch_size_36_epoch_2**9_with_cv_4.json').read())
model.load_weights('yolo_model_loss1_batch_size_36_epoch_2**9_with_cv_4.h5')

os.chdir(begincwd)

def call_img_pixel(kk, tt):
    os.chdir(begincwd + '/egohands_dataset/_LABELLED_SAMPLES/' + kk)
    dir = os.listdir()
    img = cv2.imread(dir[tt])
    return img

def photo(kk,tt,box):
    os.chdir(begincwd + '/egohands_dataset/_LABELLED_SAMPLES/' + kk)
    dir = os.listdir()
    img = cv2.imread(dir[tt])
    for i in range(18):
        for j in range(32):
            p = box[0][i,j]
            if p>0.2:
                bx = box[1][i,j]
                by = box[2][i,j]
                bw = box[3][i,j]
                bh = box[4][i,j]
                cv2.rectangle(img, (int(bx+bw/2), int(by+bh/2)), (int(bx-bw/2), int(by-bh/2)), (0, 255, 0), 2)
                cv2.putText(img, str(np.around(p,2)), (int(bx),int(by)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pred_show(place,number):
    pred = model.predict(np.array([call_img_pixel(place,number)]))
    def sigmoid(x, derivative=False):
        sigm = 1. / (1. + np.exp(-x))
        if derivative:
            return sigm * (1. - sigm)
        return sigm
    pred_ad = np.vstack([sigmoid(pred[0][0:3]),np.exp(pred[0][3:5])])

    IMAGE_H, IMAGE_W = 720, 1280
    grid = 40
    GRID_H,  GRID_W = 18, 32
    NO_OBJECT_SCALE = 1.0
    OBJECT_SCALE = 5.0
    COORD_SCALE = 1.0
    CLASS_SCALE = 1.0
    LOSS_noobj = 0.5
    IOU_THRESHOLD = 0.2
    x_cord = np.tile(np.arange(32),[18,1])
    y_cord = np.tile(np.reshape(np.arange(18),[18,1]),[1,32])
    pred_nad_1 = (pred_ad[1]+x_cord)*grid
    pred_nad_2 = (pred_ad[2]+y_cord)*grid
    pred_nad_3 = (pred_ad[3])*IMAGE_W
    pred_nad_4 = (pred_ad[4])*IMAGE_H
    pred_nad = np.array([pred_ad[0],pred_nad_1, pred_nad_2, pred_nad_3, pred_nad_4])
    return photo(place,number,pred_nad)

def camara():
  cap = cv2.VideoCapture(0)

  while(True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()
    pred = model.predict(np.array([frame]))
    def sigmoid(x, derivative=False):
        sigm = 1. / (1. + np.exp(-x))
        if derivative:
            return sigm * (1. - sigm)
        return sigm
    IMAGE_H, IMAGE_W = 720, 1280
    grid = 40
    GRID_H, GRID_W = 18, 32
    NO_OBJECT_SCALE = 1.0
    OBJECT_SCALE = 5.0
    COORD_SCALE = 1.0
    CLASS_SCALE = 1.0
    LOSS_noobj = 0.5
    IOU_THRESHOLD = 0.2
    x_cord = np.tile(np.arange(32), [18, 1])
    y_cord = np.tile(np.reshape(np.arange(18), [18, 1]), [1, 32])
    pred_ad = np.vstack([sigmoid(pred[0][0:3]),np.exp(pred[0][3:5])])
    pred_nad_1 = (pred_ad[1] + x_cord) * grid
    pred_nad_2 = (pred_ad[2] + y_cord) * grid
    pred_nad_3 = (pred_ad[3]) * IMAGE_W
    pred_nad_4 = (pred_ad[4]) * IMAGE_H
    box = np.array([pred_ad[0], pred_nad_1, pred_nad_2, pred_nad_3, pred_nad_4])
    for i in range(18):
        for j in range(32):
            p = box[0][i,j]
            if p>0.2:
                bx = box[1][i,j]
                by = box[2][i,j]
                bw = box[3][i,j]
                bh = box[4][i,j]
                cv2.rectangle(frame, (int(bx+bw/2), int(by+bh/2)), (int(bx-bw/2), int(by-bh/2)), (0, 255, 0), 2)
                cv2.putText(frame, str(np.around(p,2)), (int(bx),int(by)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)
    print(pred)

    # 顯示圖片
    cv2.imshow('frame', frame)
    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # 釋放攝影機
  cap.release()

  # 關閉所有 OpenCV 視窗
  cv2.destroyAllWindows()

