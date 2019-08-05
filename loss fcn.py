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


# 測試資料
y_pred = tf.constant(np.random.rand(5, GRID_H, GRID_W))
train2 = np.load('train_label_array_18*32.npy')
y_true = tf.constant(train2[1])

## function start

# true data
cen_x = tf.boolean_mask(y_true[1], tf.equal(y_true[0], 1))
cen_y = tf.boolean_mask(y_true[2], tf.equal(y_true[0], 1))
box_w = tf.boolean_mask(y_true[3], tf.equal(y_true[0], 1))
box_h = tf.boolean_mask(y_true[4], tf.equal(y_true[0], 1))
no_box = tf.reduce_sum(y_true[0])

x_cord = tf.reshape(tf.tile(tf.cast(tf.range(GRID_W),tf.float64),[GRID_H]),[GRID_H,GRID_W])
y_cord = tf.tile(tf.reshape(tf.cast(tf.range(GRID_H),tf.float64),[GRID_H,1]),[1,GRID_W])

# iou function
def iou(box1,box2):
    box1_xh = (box1[1]+x_cord)*grid + box1[3]*IMAGE_W/2
    box1_yh = (box1[2]+y_cord)*grid + box1[4]*IMAGE_H/2
    box1_xl = (box1[1]+x_cord)*grid - box1[3]*IMAGE_W/2
    box1_yl = (box1[2]+y_cord)*grid - box1[4]*IMAGE_H/2
    box2_xh = (box2[1]+x_cord)*grid + box2[3]*IMAGE_W/2
    box2_yh = (box2[2]+y_cord)*grid + box2[4]*IMAGE_H/2
    box2_xl = (box2[1]+x_cord)*grid - box2[3]*IMAGE_W/2
    box2_yl = (box2[2]+y_cord)*grid - box2[4]*IMAGE_H/2

    # 交集
    intersect_x = tf.math.minimum(box1_xh, box2_xh)-tf.math.maximum(box1_xl, box2_xl)
    intersect_y = tf.math.minimum(box1_yh, box2_yh)-tf.math.maximum(box1_yl, box2_yl)
    less_x = tf.less(intersect_x, 0)
    less_y = tf.less(intersect_y, 0)
    intersect_x_positive = tf.where(less_x, tf.zeros([GRID_H,  GRID_W],tf.float64),intersect_x)
    intersect_y_positive = tf.where(less_y, tf.zeros([GRID_H,  GRID_W], tf.float64), intersect_y)
    intersect = tf.multiply(intersect_x_positive,intersect_y_positive)
    # 聯集
    box1_area = box1[3] * box1[4] * 720 * 1280
    box2_area = box2[3] * box2[4] * 720 * 1280
    iou = intersect / (box1_area + box2_area - intersect)
    return iou

# tf_while_loop
iou_final = tf.zeros([GRID_H,  GRID_W], tf.float64)
box_final = -1*tf.ones([GRID_H,  GRID_W], tf.int32)
i = tf.constant(0,tf.int32)

def cond(i, iou_final, box_final):
    return tf.less(i, tf.cast(no_box, tf.int32))

def body(i,iou_final,box_final):
    need = tf.tile(tf.concat([[[[tf.cast(i, tf.float64)]]],
                       [[[cen_x[i]]]],
                       [[[cen_y[i]]]],
                       [[[box_w[i]]]],
                       [[[box_h[i]]]]], 0), [1, GRID_H,  GRID_W])
    competitor = iou(need, y_pred)
    greater_iou_logic = tf.math.greater(competitor, iou_final)
    iou_final = tf.where(greater_iou_logic, competitor, iou_final)
    box_final = tf.where(greater_iou_logic, i*tf.ones([GRID_H,  GRID_W], tf.int32), box_final)
    i += 1
    return i, iou_final, box_final

i, iou_final, box_final = tf.while_loop(cond, body, [i, iou_final, box_final])
sess = tf.Session()
combination = tf.concat([y_pred,[tf.cast(box_final,tf.float64)],[iou_final]],0) # 與 y_pred 合併


##second half
less = tf.less(combination[6], IOU_THRESHOLD)
loss_less = tf.where(less, LOSS_noobj*(y_true[0]-combination[0])**2, tf.zeros([GRID_H,  GRID_W],tf.float64))

#while
no_box = tf.reduce_max(combination[5])
i = tf.constant(0,tf.float64)
def cond(i):
    return tf.less(i, tf.cast(no_box, tf.float64))

def body():

    i += 1
    return i,

#box==0
i = tf.constant(0,tf.float64)
greater = tf.greater_equal(combination[6], IOU_THRESHOLD)
selected_box = tf.equal(combination[5], i) & greater
selected_box_4 = tf.tile(tf.reshape(selected_box, [18, 32, 1]), [1, 1, 4])


def diag_box(box1):
    box1_xh = (box1[1] + x_cord) * grid + box1[3] * IMAGE_W / 2
    box1_yh = (box1[2] + y_cord) * grid + box1[4] * IMAGE_H / 2
    box1_xl = (box1[1] + x_cord) * grid - box1[3] * IMAGE_W / 2
    box1_yl = (box1[2] + y_cord) * grid - box1[4] * IMAGE_H / 2
    # 要改stack方式使它變成shape = [no_selected_box,4]
    return tf.stack([box1_yh, box1_xh, box1_yl, box1_xl], axis=2)

# 把對應的box取出，非對應的給0 (改fcn diag_box)
boxes = tf.boolean_mask(diag_box(combination), selected_box)
scores = tf.boolean_mask(combination[6], selected_box)


# nms input shape 有問題 (改fcn diag_box)
nms = tf.image.non_max_suppression(tf.cast(boxes,tf.float32),tf.cast(scores,tf.float32),iou_threshold=0.6,max_output_size=1)
final_box = tf.gather(boxes, nms)
box_h = (final_box[0, 0] - final_box[0, 2])/720
box_w = (final_box[0, 1] - final_box[0, 3])/1280
box_x = (final_box[0, 1] + final_box[0, 3])/2


## 可能不用了
loss_greater_1 = tf.where(final_box,OBJECT_SCALE * ((y_true[1] - combination[1]) ** 2 + (y_true[2] - combination[2]) ** 2),tf.zeros([GRID_H, GRID_W], tf.float64))
loss_greater_2 = tf.where(box, OBJECT_SCALE * ((y_true[3] ** 0.5 - combination[3] ** 0.5) ** 2 + (y_true[4] ** 0.5 - combination[4] ** 0.5) ** 2),tf.zeros([GRID_H, GRID_W], tf.float64))
loss_greater_3 = tf.where(box, (y_true[0] - combination[0]) ** 2, tf.zeros([GRID_H, GRID_W], tf.float64))
loss_greater = loss_greater_1 + loss_greater_2 + loss_greater_3




print(sess.run(tf.shape(boxes)))
