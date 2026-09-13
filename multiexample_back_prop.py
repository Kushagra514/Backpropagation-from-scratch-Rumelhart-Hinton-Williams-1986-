import math

# parameters
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

# dataset
X = [
    [2, 1, 3, 0],
    [3, 1, 2, 1]
]

Y = [
    [1, 0, 1, 0]
]

N = 4
learning_rate = 0.1


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# --------------------------------------------------
# FORWARD PASS + LOSS ONLY
# --------------------------------------------------

def compute_loss(w1, b1, w2, b2, X, Y, N):

    # Z1
    z1 = [
        [0 for _ in range(N)],
        [0 for _ in range(N)]
    ]

    for i in range(len(w1)):
        for k in range(N):

            total = 0

            for j in range(len(X)):
                total += w1[i][j] * X[j][k]

            z1[i][k] = total + b1[i][0]

    # H
    H = [
        [0 for _ in range(N)],
        [0 for _ in range(N)]
    ]

    for i in range(len(z1)):
        for k in range(N):
            H[i][k] = sigmoid(z1[i][k])

    # Z2
    z2 = [
        [0 for _ in range(N)]
    ]

    for k in range(N):

        total = 0

        for j in range(len(H)):
            total += w2[0][j] * H[j][k]

        z2[0][k] = total + b2[0][0]

    # predictions
    y_hat = [
        [0 for _ in range(N)]
    ]

    for k in range(N):
        y_hat[0][k] = sigmoid(z2[0][k])

    # average loss
    loss = 0

    for k in range(N):
        loss += 0.5 * (y_hat[0][k] - Y[0][k]) ** 2

    loss /= N

    return loss


# --------------------------------------------------
# FORWARD + BACKPROPAGATION
# --------------------------------------------------

def compute_gradients(w1, b1, w2, b2, X, Y, N):

    # Z1
    z1 = [
        [0 for _ in range(N)],
        [0 for _ in range(N)]
    ]

    for i in range(len(w1)):
        for k in range(N):

            total = 0

            for j in range(len(X)):
                total += w1[i][j] * X[j][k]

            z1[i][k] = total + b1[i][0]

    # H
    H = [
        [0 for _ in range(N)],
        [0 for _ in range(N)]
    ]

    for i in range(len(z1)):
        for k in range(N):
            H[i][k] = sigmoid(z1[i][k])

    # Z2
    z2 = [
        [0 for _ in range(N)]
    ]

    for k in range(N):

        total = 0

        for j in range(len(H)):
            total += w2[0][j] * H[j][k]

        z2[0][k] = total + b2[0][0]

    # predictions
    y_hat = [
        [0 for _ in range(N)]
    ]

    for k in range(N):
        y_hat[0][k] = sigmoid(z2[0][k])

    # loss
    loss = 0

    for k in range(N):
        loss += 0.5 * (y_hat[0][k] - Y[0][k]) ** 2

    loss /= N

    # --------------------------------------------------
    # OUTPUT LAYER
    # --------------------------------------------------

    delta2 = [
        [0 for _ in range(N)]
    ]

    for k in range(N):

        delta2[0][k] = (
            (y_hat[0][k] - Y[0][k])
            * y_hat[0][k]
            * (1 - y_hat[0][k])
            / N
        )

    # dW2
    dw2 = [
        [0, 0]
    ]

    for j in range(len(H)):
        for k in range(N):
            dw2[0][j] += delta2[0][k] * H[j][k]

    # db2
    db2 = [
        [0]
    ]

    for k in range(N):
        db2[0][0] += delta2[0][k]

    # --------------------------------------------------
    # HIDDEN LAYER
    # --------------------------------------------------

    # dL/dH
    dl_dh = [
        [0 for _ in range(N)],
        [0 for _ in range(N)]
    ]

    for i in range(len(H)):
        for k in range(N):
            dl_dh[i][k] = w2[0][i] * delta2[0][k]

    # delta1
    delta1 = [
        [0 for _ in range(N)],
        [0 for _ in range(N)]
    ]

    for i in range(len(H)):
        for k in range(N):

            delta1[i][k] = (
                dl_dh[i][k]
                * H[i][k]
                * (1 - H[i][k])
            )

    # dW1
    dw1 = [
        [0, 0],
        [0, 0]
    ]

    for i in range(len(delta1)):
        for j in range(len(X)):
            for k in range(N):
                dw1[i][j] += delta1[i][k] * X[j][k]

    # db1
    db1 = [
        [0],
        [0]
    ]

    for i in range(len(delta1)):
        for k in range(N):
            db1[i][0] += delta1[i][k]

    return loss, dw1, db1, dw2, db2


# --------------------------------------------------
# NUMERICAL GRADIENT
# --------------------------------------------------

def numerical_gradient(parameter, i, j):

    epsilon = 1e-6

    original_value = parameter[i][j]

    # w + epsilon
    parameter[i][j] = original_value + epsilon

    loss_plus = compute_loss(
        w1, b1, w2, b2, X, Y, N
    )

    # w - epsilon
    parameter[i][j] = original_value - epsilon

    loss_minus = compute_loss(
        w1, b1, w2, b2, X, Y, N
    )

    # restore
    parameter[i][j] = original_value

    return (loss_plus - loss_minus) / (2 * epsilon)


# --------------------------------------------------
# GRADIENT CHECK
# --------------------------------------------------

loss, dw1, db1, dw2, db2 = compute_gradients(
    w1, b1, w2, b2, X, Y, N
)

print("\nGradient checking:")

print("w1[0][0] analytical:", dw1[0][0])
print("w1[0][0] numerical  :", numerical_gradient(w1, 0, 0))

print("w1[0][1] analytical:", dw1[0][1])
print("w1[0][1] numerical  :", numerical_gradient(w1, 0, 1))

print("w1[1][0] analytical:", dw1[1][0])
print("w1[1][0] numerical  :", numerical_gradient(w1, 1, 0))

print("w1[1][1] analytical:", dw1[1][1])
print("w1[1][1] numerical  :", numerical_gradient(w1, 1, 1))

print("b1[0][0] analytical:", db1[0][0])
print("b1[0][0] numerical  :", numerical_gradient(b1, 0, 0))

print("b1[1][0] analytical:", db1[1][0])
print("b1[1][0] numerical  :", numerical_gradient(b1, 1, 0))

print("w2[0][0] analytical:", dw2[0][0])
print("w2[0][0] numerical  :", numerical_gradient(w2, 0, 0))

print("w2[0][1] analytical:", dw2[0][1])
print("w2[0][1] numerical  :", numerical_gradient(w2, 0, 1))

print("b2[0][0] analytical:", db2[0][0])
print("b2[0][0] numerical  :", numerical_gradient(b2, 0, 0))


# --------------------------------------------------
# NORMAL TRAINING
# --------------------------------------------------

for iteration in range(1000):

    loss, dw1, db1, dw2, db2 = compute_gradients(
        w1, b1, w2, b2, X, Y, N
    )

    # update W1
    for i in range(len(w1)):
        for j in range(len(w1[0])):
            w1[i][j] -= learning_rate * dw1[i][j]

    # update b1
    for i in range(len(b1)):
        b1[i][0] -= learning_rate * db1[i][0]

    # update W2
    for i in range(len(w2)):
        for j in range(len(w2[0])):
            w2[i][j] -= learning_rate * dw2[i][j]

    # update b2
    for i in range(len(b2)):
        b2[i][0] -= learning_rate * db2[i][0]

    if iteration % 100 == 0:
        print(f"Iteration {iteration}: loss = {loss:.6f}")