#coding=utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels

# def get_image(img):
#     for i in 1:img.shape[0]
# 		im = []
# 		im.append(np.reshape(img[i, :], (28, 28)))
# 	return im


# def get_label(buf2): #	
#     label_index = 0
#     label_index += struct.calcsize('>II')
#     return struct.unpack_from('>9B', buf2, label_index)

for i in range(9):
    plt.subplot(3, 3, i + 1)
    curr_img   = np.reshape(trainimg[i, :], (28, 28)) # 28 by 28 matrix 
    curr_label = np.argmax(trainlabel[i, :] ) # Label
    title = "" + str(i) + "Label is " + str(curr_label)
    plt.title(title, fontproperties='SimHei')
    plt.imshow(curr_img, cmap='gray')
plt.show()