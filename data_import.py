import os
import scipy.io
import cv2
import numpy as np

begincwd = os.getcwd()
b = begincwd + '/egohands_with_import/_LABELLED_SAMPLES'
os.chdir(b)
a = os.listdir()
a.remove('.DS_Store')

all_dic = {}
for i in range(len(a)):
    os.chdir(a[i])
    mat = scipy.io.loadmat('polygons.mat')
    img_name = os.listdir()
    ls = list(mat['polygons'][0])
    dic = {}
    for j in range(len(ls)):
        need = {'1': ls[j][0], '2': ls[j][1], '3': ls[j][2], '4': ls[j][3]}
        name = img_name[j]
        dic[name] = need
    all_name = a[i]
    all_dic[all_name] = dic
    os.chdir(b)
os.chdir(begincwd)


def call_data(k, t):
    return all_dic[k][t]


def call_img(k, t):
    b = begincwd + '/egohands_with_import/_LABELLED_SAMPLES/' + k
    os.chdir(b)
    img = cv2.imread(t)
    cv2.imshow(t, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


all_place = list(all_dic.keys())
all_center = dict()
all_box = dict()
for i in range(len(all_place)):
    all_frame = list(all_dic[all_place[i]].keys())
    dic_box = {}
    dic_center = {}
    for j in range(len(all_frame)):
        sublist = ['1', '2', '3', '4']
        for k in sublist:
            length = len(call_data(all_place[i], all_frame[j])[k])
            if length > 1:
                x_array = call_data(all_place[i], all_frame[j])[k][range(length), 0]
                y_array = call_data(all_place[i], all_frame[j])[k][range(length), 1]
                max_x = max(np.array(x_array))
                max_y = max(np.array(y_array))
                min_x = min(np.array(x_array))
                min_y = min(np.array(y_array))
                nam = 'array' + k
                nam2 = 'arrayb' + k
                globals()[nam] = np.array([(max_x+min_x)/2, (max_y+min_y)/2])
                globals()[nam2] = np.array([(max_x - min_x), (max_y - min_y)])
            if length == 1:
                nam = 'array' + k
                nam2 = 'arrayb' + k
                globals()[nam] = np.array([])
                globals()[nam2] = np.array([])
        dic_center[all_frame[j]] = np.array([array1, array2, array3, array4])
        dic_box[all_frame[j]] = np.array([arrayb1, arrayb2, arrayb3, arrayb4])
    all_center[all_place[i]] = dic_center
    all_box[all_place[i]] = dic_box

all_ans = dict()
for i in range(len(all_place)):
    all_frame = list(all_dic[all_place[i]].keys())
    dic_ans = {}
    for j in range(len(all_frame)):
        real_center = all_center[all_place[i]][all_frame[j]]
        real_box = all_box[all_place[i]][all_frame[j]]
        real_ans = np.zeros([5, 18, 32])
        for k in range(4):
            if len(real_center[k]) > 1:
                chan_x = int(real_center[k][0] / 40)
                chan_y = int(real_center[k][1] / 40)
                real_ans[0, chan_y, chan_x] = 1
                real_ans[1, chan_y, chan_x] = real_center[k][0]/40-chan_x
                real_ans[2, chan_y, chan_x] = real_center[k][1]/40-chan_y
                real_ans[3, chan_y, chan_x] = real_box[k][0]
                real_ans[4, chan_y, chan_x] = real_box[k][1]
        dic_ans[all_frame[j]] = real_ans
    all_ans[all_place[i]] = dic_ans


def call_data_ans(k, t):
    return all_ans[k][t]
