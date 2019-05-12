
import numpy as np


weight_table= 5*np.ones((5, 18, 32))
weight_table[0]=np.ones((18, 32))




def yololoss(y_train, y_pred,y_weights):
    k = y_weights[2]*y_train[1]*(y_train[2]-y_pred[2])**2+y_weights[3]*y_train[1]*(y_train[3]-y_pred[3])**2
    p = y_weights[4]*y_train[1]*(y_train[4]**0.5-y_pred[4]**0.5)**2+y_weights[5]*y_train[1]*(y_train[5]**0.5-y_pred[5]**0.5)**2
    c = y_train[1]*(y_train[1]-y_pred[1])**2
    g = 0.5*(-1)*(y_train[1]-np.ones(y_train[1].shape))*(y_train[2]-y_pred[2])**2
    return k+p+c+g


