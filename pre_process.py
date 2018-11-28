import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import tensorflow as tf

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def reshape(image):
    out = np.zeros(shape=[32,32,3])
    for i in range(3):
        out[:,:,i] = image[1024*i:1024*(i+1)].reshape([32,32])
    return out


def load_data(dir):
    train_x = np.array(unpickle(dir+'data_batch_1')[b'data'])
    train_y = np.array(unpickle(dir+'data_batch_1')[b'labels'])
    for i in range(2,6):
        train_x = np.concatenate((train_x, np.array(unpickle(dir+'data_batch_'+str(i))[b'data'])), axis = 0)
        train_y = np.concatenate((train_y, np.array(unpickle(dir+'data_batch_'+str(i))[b'labels'])),axis = 0)


    test_x = np.array(unpickle(dir+'test_batch')[b'data'])
    test_y = np.array(unpickle(dir+'test_batch')[b'labels'])   
    labels = unpickle(dir+'batches.meta')[b'label_names']
    return train_x, train_y, test_x, test_y, labels

def pre_process_x(x):
    new_x = np.zeros(shape=[len(x),32,32,3])
    for i in range(len(x)):
        new_x[i] = reshape(x[i])
    new_x = new_x/255
    return new_x

def pre_process_y(y):
    new_y = np.zeros([len(y),10])
    for i in range(len(y)):
        new_y[i][y[i]-1] = 1
    return new_y

ori_train_x, ori_train_y, ori_test_x, ori_test_y, labels = load_data('data/')

train_x = pre_process_x(ori_train_x)
test_x = pre_process_x(ori_test_x)
train_y = pre_process_y(ori_train_y)
test_y = pre_process_y(ori_test_y)

print("===>training data shape is "+ str(train_x.shape))
print("===>test data shape is "+ str(test_x.shape))
print("===>training label shape is "+ str(train_y.shape))
print("===>test label shape is "+ str(test_y.shape))

fig, axes = plt.subplots(2, 5)
fig.subplots_adjust(hspace=0.3, wspace=0.3)
for i, ax in enumerate(axes.flat):
    # Plot image and smooth it
    ax.imshow(train_x[i])
    label = labels[ori_train_y[i]]
    ax.set_xlabel(label)

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.2, random_state = 0)