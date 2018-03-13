import numpy as np
import tensorflow as tf
import random as rn
import os

seed = 5

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(seed)
rn.seed(seed)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(seed)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import models64 as m
import preprocess2 as pr2
from others import shuffle_arrays, test_model, normalize_set

width = 64
height = 64
epochs = 1500
verbose = 1
numvars = 10
model = m.model2(width, height)

startW = model.get_weights()

[o_x1, o_y1, o_x2, o_y2] = pr2.getData2(width, height, numvars)

shuffle_arrays(o_x1, o_y1, seed)
shuffle_arrays(o_x2, o_y2, seed)

o_x1 = np.reshape(o_x1, (o_x1.shape[0], height, width, 1))
o_x2 = np.reshape(o_x2, (o_x2.shape[0], height, width, 1))

o_y1[:,0] = o_y1[:,0] / 90
o_y1[:,1] = o_y1[:,1] / 90

o_y2[:,0] = o_y2[:,0] / 90
o_y2[:,1] = o_y2[:,1] / 90

x1 = o_x1
x2 = o_x2

y1 = o_y1
y2 = o_y2

[avg, std] = normalize_set(x1)
print("Avg: %f\nStd: %f" % (avg, std))
normalize_set(x2, avg, std)

nombre = input("Introduzca el nombre del modelo a guardar (sin extension): ")

for i in range(5, 30, 5):
    print("%d variaciones por imagen..." % i)
    [err1, err2, score] = test_model(model, x1, y1, x2, y2, epochs, nombre + "_" + str(i), verbose=verbose)
    print("MSE: %f" % score)
    print("Error medio:\nTilt: %f Pan: %f\n" % (err1, err2))
    model.set_weights(startW)
