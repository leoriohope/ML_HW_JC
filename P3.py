import numpy as np 
import math
from keras.datasets import mnist
from utils import *
from data import *

'''Hyper parameters
'''
EPOCH = 1
BATCH_SIZE = 100
LEARNINT_RATE = 0.000001
NUM_OF_CLASSIFIERS = 10

'''Training
'''
parameter = init_parameters([784, 10])
bias = 0
for epoch in range(EPOCH):
    for batch, X_t in enumerate(x_train_batch[:1]):
        z = np.dot(parameter, X_t) + bias
        cost = multiclass_cross_entropy(z, y_train_batch_sets[i][batch])
        print(cost)
        # cost, dw = mean_square_cost(X_t, output, y_train_batch_sets[i][batch])
        # parameter[0] -= LEARNINT_RATE * dw.T #updata parameter
        # if not batch % 100:
        #     print("model %d cost: " % i + str(cost))

# ''' Predicting
# '''
# def predict(classifiers, X): #prediction function using argmax for 10 models
#     outputs = []
#     for parameter in classifiers:
#         outputs.append(sigmoid(X, parameter[0]))
#     return np.argmax(outputs, axis=0)

# print("First ten test label: ")
# print((y_test_flatten[0][:40]))
# print("First ten predict label: ")
# print(predict(classifiers, x_test_flatten[:, :40]))

# test_output = predict(classifiers, x_test_flatten)
# accuracy = np.sum(np.equal(y_test_flatten[0], test_output[0])) / len(y_test_flatten[0])
# print("Accuracy: " + str(accuracy))