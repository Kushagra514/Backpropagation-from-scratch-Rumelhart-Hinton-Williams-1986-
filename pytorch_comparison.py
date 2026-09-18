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

#pytorch model
x_torch = torch.tensor(X)

y_torch = torch.tensor(y)

model = nn.Sequential(
    nn.Linear(2,2),
    nn.Sigmoid(),
    nn.Linear(2,1),
    nn.Sigmoid()
)

#copying the same initial parameters

with torch.no_grad():
    #pytorch linear stores weights as:
    #(output_features,input_features)
    #therefore transpose our manual W1.

    model[0].weight.copy_(
        torch.tensor(W1.T)
    )

    model[0].bias.copy(
        torch.tensor(b1[0])
    )

    #same w2    
    #manual w2 shape: (2,1)
    #pytorch linear weight:
    #(1,2)

    model[2].weight.copy_(
        torch.tensor(W2.T)
    )

    model[2].bias.copy_(
        torch.tensor(b2[0])
    )

    #pytorch loss

    def torch_loss(y_hat,y):
        return 0.5 * torch.sum(
            (y_hat - y) ** 2
        )

    #pytorch forward and backward

    torch_prediction = model(x_torch)
    torch_loss_value = torch_loss(
        torch_prediction,
        y_torch
    )

    torch_loss_value.backward()

    #print losses
    print("=== Loss Comparision ===")

    print(
        "Manual Loss: ",
        manual_loss_value
    )

    print(
        "Pytorch loss:",
        torch_loss_value.item()
    )

    #print Gradients

    print("===Manual Gradients===")
    print("\ndw1:")
    print(manual_dw1)

    print("\ndb1:")
    print(manual_db2)

    print("\ndw2:")
    print(manual_dw2)

    print("\ndb2:")
    print(manual_db2)

    print("\n===Pytorch gradients===")
    print("\n1dw1:")
    print(model[0].weight.grad)

    print("\ndb1:")
    print(model[0].bias.grad)

    print("\ndw2:")
    print(model[2].weight.grad)

    print("\ndb2:")
    print(model[2].bias.grad)


    #gradient comparison
    #pytorch stores linear weights transposed relative to our manual convention,so have to transpose them back

    torch_dw1 = model[0].weight.grad.detach().numpy().T
    torch_db1 = model[0].bias.grad.detach().numpy().reshape(1,2)

    torch_dw2 = model[2].weight.grad.detach().numpy().T
    torch_db2 = model[2].bias.grad.detach().numpy().reshape(1,1)

    #print("\n Gradient differences")
    dw1_difference = np.max(
        np.abs(manual_dw1 - torch_dw1)
    )

    db1_difference = np.max(
        np.abs(manual_db1 - torch_db1)
    )

    dw2_difference = np.max(
        np.abs(manual_dw2 - torch_dw2)
    )

    db2_difference = np.max(
        np.abs(manual_db2 - torch_db2)
    )

    print("dW1 max difference:", dw1_difference)
    print("db1 max difference:", db1_difference)
    print("dW2 max difference:", dw2_difference)
    print("db2 max difference:", db2_difference)

    #final verification

    tolerance = 1e-6

    all_gradients_match = (
        dw1_difference < tolerance
        and db1_difference < tolerance
        and dw2_difference < tolerance
        and db2_difference < tolerance
    )

    print("\n========== FINAL RESULT ==========")

    if all_gradients_match:
        print("PASS: Manual gradients match PyTorch gradients.")
    else:
        print("FAIL: Manual gradients do not match PyTorch gradients.")

                                                                                                                                                                                                                                                  
