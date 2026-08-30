import math

w = [[0.5,0.2],[0.1,0.4]]
x = [[2],[3]]
b = [[0.1],[0.2]]

z = [[0],[0]]
a = [[0],[0]]

y = [[1],[0]]

def sigmoid(z):
    return 1 /(1 + math.exp(-z))

for iter in range(10):
    loss = 0
    for i in range(len(w)):
        for j in range(len(x[0])):
            total = 0
            for k in range(len(x)):
                total += w[i][k] * x[k][j]
            z[i][j] = total + b[i][0]
    #add bias
    for i in range(len(z)):
        for j in range(len(z[0])):
            a[i][j] = sigmoid(z[i][j])

    #calculating loss
    for i in range(len(z)):
            loss += 0.5 * (a[i][0] - y[i][0]) ** 2
    print(f"Iteration{iter}: loss = {loss:.6f}")

    delta = [[0], [0]]

    #going to backward pass

    #calculate delta
    for i in range(len(a)):
        delta[i][0] = (
            (a[i][0] - y[i][0])
            * a[i][0]
            * (1 - a[i][0])
        )

    print("delta =", delta)
    #calculate dW now which is equal to delta * x^T
    dW = [[0, 0],
        [0, 0]]

    for i in range(len(w)):
        for j in range(len(w[0])):
            dW[i][j] = delta[i][0] * x[j][0]

    print("dW =", dW)
    #calculate db now
    db = [[0],
        [0]]

    for i in range(len(b)):
        db[i][0] = delta[i][0]

    print("db =", db)

    learning_rate = 0.1

    for i in range(len(w)):
        for j in range(len(w[0])):
            w[i][j] -= learning_rate * dW[i][j]

    for i in range(len(b)):
        b[i][0] -= learning_rate * db[i][0]