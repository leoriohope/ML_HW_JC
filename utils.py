import numpy as np 
import math
from keras.datasets import mnist

# #Hyper parameters
# EPOCH = 1
# BATCH_SIZE = 100
# LEARNINT_RATE = 0.01


# '''Data preparation
# '''
# (x_train, y_train), (x_test, y_test) = mnist.load_data()  #x_train(60000, 28, 28), y_train(60000), x_test(10000, 28, 28), y_test(10000,)  datatype: ndarrays
# print(x_train.shape)
# print(y_train)
# # print(x_test.shape)
# # print(y_test.shape)

# #reshaping training ans test examples
# x_train_flatten = x_train.reshape(x_train.shape[0], -1).T   #(784, 60000)
# y_train_flatten = y_train.reshape(-1, y_train.shape[0])     #(1, 60000)
# x_test_flatten = x_test.reshape(x_test.shape[0], -1).T      #(784, 10000)
# y_test_flatten = y_test.reshape(-1, y_test.shape[0])        #(1, 60000)

# # print(x_train_flatten)
# print(x_test_flatten.shape)
# print(y_train_flatten.shape)

'''Helper functions
'''
#convert the input data into batchs
def init_batch(inputs, batch_size):
    num_full_batch = len(inputs[0]) // batch_size
    batchs = []
    for i in range(num_full_batch):
        batchs.append(inputs[:,i * batch_size : (i+1) * batch_size])
    if len(inputs[0]) % batch_size:
        batchs.append(inputs[:, num_full_batch * batch_size :])
    return np.asarray(batchs)
# print(init_batch(x_train_flatten, BATCH_SIZE))

#Initialize parameters
def init_parameters(layers):
    ''' Initilize the parameters of each layer
        In this case, we only have input layer(28*28) and output layer(1) 
    '''
    parameters = []
    for layer in range(1, len(layers)):
        parameters.append(np.random.randn(layers[layer], layers[layer - 1]))
    return parameters

# parameters = init_parameters(layers)
# print(len(parameters))

#forward propagation using only logistic regression function
def sigmoid(X, w):
    z = np.dot(w, X)
    # print(z)
    res = np.divide(1, 1 + np.exp(-z))
    return res 

# print(sigmoid(x_train_flatten[:, 1], parameters[0]))

def mean_square_cost(X, A, Y):
    m = Y.shape[1]
    cost = (1 / m) * np.sum(np.square(A - Y))
    dw = (1 / m) * np.dot(X, (2 * (A - Y) * A * (1 - A)).T)
    return cost, dw
    
def corss_entropy_cost(X, A, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) 
    dw = (1 / m) * np.dot(X, (A-Y).T)
    return cost, dw

def update_parameters(w, X, Y, num_iterations, learning_rate, batch_size):
    pass


# #Problem 1
# #data prepare
# y_sets = []
# for i in range(10):
#     y_sets.append([1 if num == i else 0 for num in y_train])



# models = []
# layers = [28 * 28, 1]
# parameters = init_parameters(layers) #[[1, 28 * 28]]

# #train network with mean square error
# for i in range(10): #number of models for different digit i
#     for epoch in range(EPOCH): #Number of epochs we train, 1 in our case
#         pass
        
        









    




