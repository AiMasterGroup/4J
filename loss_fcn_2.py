import tensorflow as tf
import numpy as np

IMAGE_H, IMAGE_W = 720, 1280
grid = 40
GRID_H,  GRID_W = 18, 32
NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
COORD_SCALE = 1.0
CLASS_SCALE = 1.0
LOSS_noobj = 0.5
IOU_THRESHOLD = 0.2
bbox = 5
anchor_box = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

y_pred = tf.constant(np.random.rand(bbox*5, GRID_H, GRID_W))
train2 = np.load('train_label_array_18*32.npy')
y_true = tf.constant(train2[1])

true = tf.equal(tf.ones([GRID_H,  GRID_W], tf.float32), 1)
false = tf.equal(tf.ones([GRID_H,  GRID_W], tf.float32), 0)
true_mat = tf.tile(tf.concat([tf.tile([true],[3,1,1]),tf.tile([false],[2,1,1])],0),[bbox,1,1])
xy_pred = tf.where(true_mat,tf.sigmoid(y_pred),tf.zeros(bbox*5, GRID_H, GRID_W))
wh_pred = tf.where(true_mat,tf.zeros(bbox*5, GRID_H, GRID_W), tf.exp(y_pred))
adjusted_y_pred = xy_pred+wh_pred

less = tf.equal(y_true[0], 0)
#loss 4th 行
loss_no_object_1 = tf.where(less, LOSS_noobj*(y_true[0]-y_pred[0])**2, tf.zeros([GRID_H,  GRID_W],tf.float32))
loss_no_object_2 = tf.where(less, LOSS_noobj*(y_true[0]-y_pred[5])**2, tf.zeros([GRID_H,  GRID_W],tf.float32))
loss_no_object_3 = tf.where(less, LOSS_noobj*(y_true[0]-y_pred[10])**2, tf.zeros([GRID_H,  GRID_W],tf.float32))
loss_no_object_4 = tf.where(less, LOSS_noobj*(y_true[0]-y_pred[15])**2, tf.zeros([GRID_H,  GRID_W],tf.float32))
loss_no_object_5 = tf.where(less, LOSS_noobj*(y_true[0]-y_pred[20])**2, tf.zeros([GRID_H,  GRID_W],tf.float32))

#loss 3rd 行
loss_object_1 = tf.where(less, tf.zeros([GRID_H,  GRID_W],tf.float32), (y_true[0]-tf.sigmoid(y_pred[0]))**2)
loss_object_2 = tf.where(less, tf.zeros([GRID_H,  GRID_W],tf.float32), (y_true[0]-tf.sigmoid(y_pred[5]))**2)
loss_object_3 = tf.where(less, tf.zeros([GRID_H,  GRID_W],tf.float32), (y_true[0]-tf.sigmoid(y_pred[10]))**2)
loss_object_4 = tf.where(less, tf.zeros([GRID_H,  GRID_W],tf.float32), (y_true[0]-tf.sigmoid(y_pred[15]))**2)
loss_object_5 = tf.where(less, tf.zeros([GRID_H,  GRID_W],tf.float32), (y_true[0]-tf.sigmoid(y_pred[20]))**2)



#nms
def iou(box1,box2):
    box1_xh = box1[1] + box1[3]/2
    box1_yh = box1[2] + box1[4]/2
    box1_xl = box1[1] - box1[3]/2
    box1_yl = box1[2] - box1[4]/2
    box2_xh = box2[1] + box2[3]/2
    box2_yh = box2[2] + box2[4]/2
    box2_xl = box2[1] - box2[3]/2
    box2_yl = box2[2] - box2[4]/2

    # 交集
    intersect_x = tf.math.minimum(box1_xh, box2_xh)-tf.math.maximum(box1_xl, box2_xl)
    intersect_y = tf.math.minimum(box1_yh, box2_yh)-tf.math.maximum(box1_yl, box2_yl)
    less_x = tf.less(intersect_x, 0)
    less_y = tf.less(intersect_y, 0)
    intersect_x_positive = tf.where(less_x, tf.zeros([GRID_H,  GRID_W],tf.float64),intersect_x)
    intersect_y_positive = tf.where(less_y, tf.zeros([GRID_H,  GRID_W], tf.float64), intersect_y)
    intersect = tf.multiply(intersect_x_positive,intersect_y_positive)
    # 聯集
    box1_area = box1[3] * box1[4]
    box2_area = box2[3] * box2[4]
    iou = intersect / (box1_area + box2_area - intersect)
    return iou

great = tf.equal(y_true[0], 1)




#loss 2nd 行







