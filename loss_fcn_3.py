import tensorflow as tf
import numpy as np

IMAGE_H, IMAGE_W = 720, 1280
grid = 40
GRID_H,  GRID_W = 18, 32
NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
COORD_SCALE = 1.0
LOSS_noobj = 0.5
no_batch = 2

y_pred = tf.reshape(tf.constant(np.random.rand(10, GRID_H, GRID_W)),[2,5, GRID_H, GRID_W])
train2 = np.load('train_label_array_18*32.npy')
y_true = tf.reshape(tf.constant(train2[1:3]),[2,5, GRID_H, GRID_W])



true = tf.equal(tf.ones([GRID_H,  GRID_W], tf.float32), 1)
false = tf.equal(tf.ones([GRID_H,  GRID_W], tf.float32), 0)
true_mat = tf.tile([tf.concat([tf.tile([true], [3, 1, 1]), tf.tile([false], [2, 1, 1])], 0)], [no_batch,1,1,1])
xy_pred = tf.where(true_mat, tf.sigmoid(y_pred), tf.zeros([no_batch, 5, GRID_H, GRID_W],tf.float64))
wh_pred = tf.where(true_mat, tf.zeros([no_batch, 5, GRID_H, GRID_W],tf.float64), tf.exp(y_pred))
adjusted_y_pred = xy_pred+wh_pred
true_2 = tf.equal(y_true[:,0], tf.ones([no_batch, GRID_H,  GRID_W], tf.float64))
no_object_loss = tf.where(true_2, tf.zeros([no_batch,GRID_H,  GRID_W], tf.float64), LOSS_noobj*(y_true[:,0]-adjusted_y_pred[:,0])**2)
object_loss = tf.where(true_2,(y_true[:,0]-adjusted_y_pred[:,0])**2
                       +OBJECT_SCALE*(y_true[:,1]-adjusted_y_pred[:,1])**2
                       +OBJECT_SCALE*(y_true[:,2]-adjusted_y_pred[:,2])**2
                       +OBJECT_SCALE*(y_true[:,3]**0.5-adjusted_y_pred[:,3]**0.5)**2
                       +OBJECT_SCALE*(y_true[:,4]**0.5-adjusted_y_pred[:,4]**0.5)**2
                       , tf.zeros([no_batch,GRID_H,  GRID_W], tf.float64))
total_loss = tf.reduce_sum(no_object_loss + object_loss)

sess = tf.Session()
print(sess.run(total_loss))


