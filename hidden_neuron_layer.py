import math 

#parameters

w1 = [
    [0.5, 0.2],
    [0.1, 0.4]
]

b1 = [
    [0.1],
    [0.2]
]

w2 = [
    [0.3, 0.6]
]

b2 = [
    [0.1]
]

#input and target

x = [
    [2],
    [3]
]

y = [
    [1]
]

#activation 

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

#training

learning_rate = 0.1

for iteration in range(10):
    #forward pass
    #z1 = w1 @ x + b1

    z1 = [
        [0],
        [0]
    ]

    for i in range(len(w1)):
        total = 0
        for j in range(len(x)):
            total += w1[i][j] * x[j][0]

        z1[i][0] = total + b1[i][0]

    #h = sigmoid(z1)

    h = [
        [0],
        [0]
    ]

    for i in range(len(z1)):
        h[i][0] = sigmoid(z1[i][0])

    #z2 = w2 @ h + b2

    z2 = [
        [0]
    ]

    total = 0

    for j in range(len(h)):
        total += w2[0][j] * h[j][0]

    z2[0][0] = total + b2[0][0]

    #y_hat = sigmoid(z2)

    y_hat = [
        [sigmoid(z2[0][0])]
    ]

    #Loss

    loss = 0.5 * (y_hat[0][0] - y[0][0]) ** 2
    print(f"Iteration {iteration}: loss = {loss: .6f}")

    #backward pass
    #output layer  -> delta2 = (y_hat0 - y) * y_hat * (1 - y_hat)

    delta2 = [
        [0]
    ]

    delta2[0][0] = (
        (y_hat[0][0] - y[0][0]) * y_hat[0][0] * (1 - y_hat[0][0])
    )

    #dw2 = delta2 @ h.T

    dw2 = [
        [0,0]
    ]

    for j in range(len(h)):
        dw2[0][j] = delta2[0][0] * h[j][0]

    #db2 = delta2

    db2 = [
        [delta2[0][0]]
    ]

    #propogate into hidden layer 
    #dl/dh = w2.T @ delta2

    dL_dh = [
        [0],
        [0]
    ]

    for i in range(len(h)):
        dL_dh[i][0] = w2[0][i] * delta2[0][0]

    #hidden layer delta

    delta1 = [
        [0],
        [0]
    ]

    for i in range(len(h)):
        delta1[i][0] = dL_dh[i][0] * h[i][0] * (1 - h[i][0])

    dw1 = [
        [0, 0],
        [0, 0]
    ]

    db1 = [
        [delta1[0][0]],
        [delta1[1][0]]
    ]
    
    for i in range(len(delta1)):
        for j in range(len(x)):
            dw1[i][j] = delta1[i][0] * x[j][0]


    #parameter updates

    for i in range(len(w1)):
        for j in range(len(dw1[0])):
            w1[i][j] -= learning_rate * dw1[i][j]

    for i in range(len(b1)):
        b1[i][0] -= learning_rate * db1[i][0]

    for i in range(len(w2)):
        for j in range(len(dw2[0])):
            w2[i][j] -= learning_rate * dw2[i][j]

    for i in range(len(b2)):
        b2[i][0] -= learning_rate * db2[i][0]