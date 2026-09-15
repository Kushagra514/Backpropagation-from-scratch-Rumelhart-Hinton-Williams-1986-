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

#manual loss
def manual_loss(y,y_hat):
    return 0.5 * np.sum((y - y_hat) ** 2)

#manual backpropagation
def manual_backward(X,y,z1,A1,z2,y_hat,w2):
    #s1 dl/dy_hat
    #l = 0.5 * ((y_hat - y) ** 2)
    #dl/dy_hat = y_hat - y
    dL_dyhat = y_hat - y

    #s2 dy_hat/dz2
    #y_hat = sigmoid(z2)
    #derivative = y_hat(1-y_hat)

    dyhat_dz2 = sigmoid_der_frm_activation(y_hat)

    #s3 chain rule
    #dl/dz2 = dl/dy_hat * dy_hat/dz2
    dZ2 = dL_dyhat * dyhat_dz2

    #s4 z2 = a1 @ w2 + b2
    #therefore dl/dw2 = A1.T @ dl/dz2

    dW2 = A1.T @ dZ2

    #s5 dl/db2 every example contributes to the same bias therfore we sum over all the examples
    db2 = np.sum(dZ2,axis = 0,keepdims=True)

    #s6 = backpropoagte into a1
    #z2 = A1 @ w2 + b2
    #dl/A1 = dl/dz2 @ w2.T

    dA1 = dZ2 @ W2.T

    #s7
    #a1 = sigmoid(z1)
    #da1/dz1 = a1(1-a1)

    dA1_dZ1 = sigmoid_der_frm_activation(A1)

    #s8 dl/dz1
    dZ1 = dA1 * dA1_dZ1

    #s9 z1 = X @ w1 + b1
    #dl/dw1 = X.T @ dZ1 

    dW1 = X.T @ dZ1

    #s10
    # dL/db1 
    db1 = np.sum(dZ1,axis = 0, keepdims = True)    

    return dW1,db1,dW2,db2


#manual forward + backward
Z1, A1 , Z2 , manual_prediction = manual_forward(
    X,W1,b1,W2,b2
)

manual_loss_value = manual_loss(manual_prediction,y)
manual_dw1, manual_db1, manual_dw2,manual_db2 = manual_backward(
        X,
        y,
        Z1,
        A1,
        Z2,
        manual_prediction,
        W2
)

