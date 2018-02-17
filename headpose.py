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

import models_correct as m
import preprocess as pr
from others import shuffle_arrays, rerange, test_model, normalize_set

width = 46
height = 46
epochs = 500
verbose = 1
model = m.model4(width, height)

[o_x1, o_y1, o_x2, o_y2] = pr.getData(width, height)

shuffle_arrays(o_x1, o_y1, seed)
shuffle_arrays(o_x2, o_y2, seed)

o_x1 = np.reshape(o_x1, (o_x1.shape[0], height, width, 1))
o_x2 = np.reshape(o_x2, (o_x2.shape[0], height, width, 1))

o_y1[:,0] = o_y1[:,0]/90
o_y1[:,1] = o_y1[:,1]/90

o_y2[:,0] = o_y2[:,0]/90
o_y2[:,1] = o_y2[:,1]/90

x1 = o_x2
x2 = o_x1

y1 = o_y2
y2 = o_y1

[avg, std] = normalize_set(x1)
normalize_set(x2, avg, std)

[err1_1, err2_1, score] = test_model(model, x1, y1, x2, y2, epochs, verbose=verbose)
print("MSE: %f" % score)
print("Error medio:\nTilt: %f Pan: %f\n" % (err1_1, err2_1))
