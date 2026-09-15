import numpy as np
import torch 
import torch.nn as nn

#1 dataset

X = np.array([
    [2.0,3.0],
    [1.0,1.0],
    [3.0,2.0],
    [0.0,1.0]
],dtype = np.float32)

y = np.array([
    [1.0],
    [0.0],
    [1.0],
    [0.0]
],dtype = np.float32)


#activation functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der_frm_activation(a):
    """
    If a = sigmoid(z), then 
        sigmoid'(z) = a(1-a)
    """
    return a*(1-a)

#manual neural network 
#following conventions were used

#X : (number of examples,input_features)
#W1 : (input_features, hidden_neurons)
#b1 : (1,hidden_neurons)
#W2 : (hidden_neurons,output_neurons)
#b2 : (1,output_neurons)

W1 = np.array([
    [0.10,0.20],
    [0.30,0.40]
], dtype = np.float32)

b1 = np.array([
    [0.10, 0.10]
],dtype = np.float32)

W2 = np.array([
    [0.50],
    [0.60]
],dtype = np.float32)

b2 = np.array([
    [0.10]
],dtype = np.float32)


#manual forward pass
def manual_forward(X,W1,b1,W2,b2):

    #First layer
    z1 = X @ W1 + b1 

    #hidden activation
    A1 = sigmoid(z1)

    #second layer
    z2 = A1 @ W2 + b2

    #output activation
    y_hat = sigmoid(z2)

    return z1,A1,z2,y_hat

