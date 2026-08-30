import math 

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# initial forward pass values 
x = 2
y = 1
w = 0.5
b = 0.1
learning_rate = 0.1

for iteration in range(110000):
    z = w * x * b
    a = sigmoid(z)

    loss = 0.5 * (a - y) ** 2
    print(f"Iteration{iteration}: loss = {loss:.6f}")
    # backward pass accounting for the gradient 
    dL_da = a - y
    da_dz = a * (1 - a)
    dz_dw = x
    dz_db = 1

    dL_dw = dL_da * da_dz * dz_dw
    dL_db = dL_da * da_dz * dz_db
    #gradient descent in action 
    w -= learning_rate * dL_dw
    b -= learning_rate * dL_db