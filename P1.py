import numpy as np 
import math
from keras.datasets import mnist
from utils import *

'''Hyper parameters
'''
EPOCH = 1
BATCH_SIZE = 100
LEARNINT_RATE = 0.01
NUM_OF_CLASSIFIERS = 10

'''Data preparation
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()  #x_train(60000, 28, 28), y_train(60000), x_test(10000, 28, 28), y_test(10000,)  datatype: ndarrays
# print(x_train.shape)
# print(y_train)
# print(x_test.shape)
# print(y_test.shape)

#reshaping training ans test examples
x_train_flatten = x_train.reshape(x_train.shape[0], -1).T   #(784, 60000)
y_train_flatten = y_train.reshape(-1, y_train.shape[0])  #(1, 60000)
x_test_flatten = x_test.reshape(x_test.shape[0], -1).T      #(784, 10000)
y_test_flatten = y_test.reshape(-1, y_test.shape[0])        #(1, 60000)

# print(x_train_flatten)
# print(x_test_flatten.shape)
# print(y_train_flatten)

x_train_batch = init_batch(x_train_flatten, batch_size=BATCH_SIZE) # (600, 784, 100)
x_test_batch = init_batch(x_test_flatten, batch_size=BATCH_SIZE)   # (100, 784, 100)
# print(x_train_batch.shape)

#set y set to i representation
y_train_sets = [] #10 sets for 10 models
for i in range(10):
    y_train_sets.append(np.asarray([[1 if num == i else 0 for num in y_train_flatten[0]]]))

y_test_sets = []
for i in range(10):
    y_test_sets.append(np.asarray([[1 if num == i else 0 for num in y_test_flatten[0]]]))
# print(y_train_sets[0].shape)
# print(sizeof(y_train_sets[0]))

#batch sets for 10 y sets
y_train_batch_sets = []   # (600, 1, 100) * 10
for model in range(NUM_OF_CLASSIFIERS):
    y_train_batch_sets.append(init_batch(y_train_sets[model], batch_size=BATCH_SIZE))
# print(y_train_batch_sets[0].shape)
y_test_batch_sets = []    # (100, 1, 100) * 10
for model in range(NUM_OF_CLASSIFIERS):
    y_test_batch_sets.append(init_batch(y_test_sets[model], batch_size=BATCH_SIZE))
# print(y_test_batch_sets[0].shape)

'''Training
'''
classifiers = []
for i in range(NUM_OF_CLASSIFIERS): # We need to train 10 models for each i
    parameter = init_parameters([784, 1]) # 1 * (1, 784) list of ndarray
    # print(parameter.shape)
    for epoch in range(EPOCH):      # num of epochs we want, 1 in our case
        for batch, X_t in enumerate(x_train_batch):
            output = sigmoid(X_t, parameter[0])
            cost, dw = corss_entropy_cost(X_t, output, y_train_batch_sets[i][batch])
            print(dw.shape)