# Day 1: Backpropagation Basics

This project is a simple implementation of the backpropagation idea from the paper *Learning representations by back-propagating errors* by Rumelhart, Hinton, and Williams.

The code in [1neuron.py](1neuron.py) shows how a single neuron learns by:

- doing a forward pass,
- computing loss,
- applying the chain rule backward,
- updating weights and bias with gradient descent.

## Example

```python
z = w * x * b
a = sigmoid(z)
loss = 0.5 * (a - y) ** 2
```

The key gradient steps are:

```python
dL_da = a - y
da_dz = a * (1 - a)
dL_dw = dL_da * da_dz * x
dL_db = dL_da * da_dz
w -= learning_rate * dL_dw
b -= learning_rate * dL_db
```

This is the core idea behind backpropagation: compute the error, propagate it backward, and adjust the parameters to reduce loss.

## Run it

```bash
python 1neuron.py
```

The script prints the loss over many iterations, showing the model improving over time.

---

# Day 2: Two-Neuron Layer

This section extends the same idea to a small layer with two neurons. The file [2neuron_layer.py](2neuron_layer.py) builds a simple matrix-based forward pass and computes the gradients needed for learning.

The layer uses:

- weights `w`
- inputs `x`
- bias `b`
- output activations `a`
- target values `y`

## Forward pass

```python
z = w * x + b
a = sigmoid(z)
```

The loss is computed as the sum of squared errors across the neurons:

```python
loss += 0.5 * (a[i][0] - y[i][0]) ** 2
```

## Backward pass

The derivative for each output is:

```python
delta[i][0] = (a[i][0] - y[i][0]) * a[i][0] * (1 - a[i][0])
```

Then the weight gradient is computed using the input values:

```python
dW[i][j] = delta[i][0] * x[j][0]
```

and the bias gradient is simply:

```python
db[i][0] = delta[i][0]
```

This shows how the same backpropagation rule scales from one neuron to a small neural network layer.

## Run it

```bash
python 2neuron_layer.py
```

The script prints the loss and gradient values at each iteration, helping visualize how the model updates over time.

---

# Day 3: Hidden Neuron Layer

Day 3 introduces a two-layer neural network in [hidden_neuron_layer.py](hidden_neuron_layer.py). The network takes two input values, passes them through a hidden layer containing two neurons, and produces one output value.

## Network structure

```text
2 inputs -> 2 hidden neurons -> 1 output neuron
```

The forward pass calculates the hidden layer first:

```python
z1 = w1 @ x + b1
h = sigmoid(z1)
```

The hidden activations are then used by the output neuron:

```python
z2 = w2 @ h + b2
y_hat = sigmoid(z2)
```

The model uses squared error loss:

```python
loss = 0.5 * (y_hat - y) ** 2
```

During the backward pass, the output error is propagated back through the output weights and the hidden-layer sigmoid derivatives:

```python
delta2 = (y_hat - y) * y_hat * (1 - y_hat)
dL_dh = w2.T @ delta2
delta1 = dL_dh * h * (1 - h)
```

Gradients are calculated for `w1`, `b1`, `w2`, and `b2`, then all parameters are updated with gradient descent. This demonstrates how backpropagation learns through a hidden layer instead of updating only the output neuron.

## Run it

```bash
python hidden_neuron_layer.py
```

The script prints the loss for each of the 10 training iterations.

---

# Day 4: Multiple Examples with a Hidden Layer

[multiexample_back_prop.py](multiexample_back_prop.py) extends the hidden-layer example to train on four examples at once. It uses a network with two input features, two hidden neurons, and one output neuron.

## Data and parameters

The input matrix stores features by row and examples by column:

```python
X = [
	[2, 1, 3, 0],
	[3, 1, 2, 1],
]
Y = [[1, 0, 1, 0]]
```

The script trains for `1000` iterations with a learning rate of `0.1`. The weights and biases are initialized explicitly so each step of the calculation remains visible.

## Forward pass

For every iteration, the script computes the hidden pre-activations and applies the sigmoid function to each example:

```python
z1 = w1 @ X + b1
H = sigmoid(z1)
```

The output neuron then combines the hidden activations and produces one prediction per example:

```python
z2 = w2 @ H + b2
y_hat = sigmoid(z2)
```

The loss is the average of the per-example squared errors:

```python
loss = (1 / N) * sum(0.5 * (y_hat[0][k] - Y[0][k]) ** 2 for k in range(N))
```

## Backward pass

The output delta applies the squared-error derivative, the sigmoid derivative, and the average over the four examples:

```python
delta2 = (y_hat - Y) * y_hat * (1 - y_hat) / N
```

The script uses this delta to calculate gradients for the output weights and bias. It then propagates the error through `w2` and the hidden sigmoid activations:

```python
dL_dh = w2.T @ delta2
delta1 = dL_dh * H * (1 - H)
```

Finally, it accumulates `dw1`, `db1`, `dw2`, and `db2` across all examples and updates every parameter with gradient descent.

## Run it

```bash
python multiexample_back_prop.py
```

The script prints the loss before each parameter update and reports the iteration number. With the current initialization, the loss decreases to approximately `0.093039` by iteration `999`, showing the network learning from all four examples in each batch update.

---

# Day 5: Modular Multi-Example Backpropagation

Day 5 updates [multiexample_back_prop.py](multiexample_back_prop.py) to make the multi-example network easier to verify and understand. The architecture is still a `2 -> 2 -> 1` sigmoid network, but its loss calculation, gradient calculation, gradient checking, and training loop are separated into explicit stages.

## Architecture

The script keeps the model parameters and dataset at module scope:

```text
X (2 features x 4 examples)
	|
	v
W1, b1 -> Z1 -> sigmoid -> H (2 hidden neurons x 4 examples)
	|
	v
W2, b2 -> Z2 -> sigmoid -> y_hat (1 output x 4 examples)
```

Each column represents one training example. `w1` and `b1` connect the two input features to the hidden layer, while `w2` and `b2` connect the hidden layer to the single output neuron.

## Separated loss and gradient functions

`compute_loss(...)` performs only the forward pass and returns the average squared-error loss. `compute_gradients(...)` repeats the forward pass, computes the output and hidden-layer deltas, and returns:

```text
loss, dw1, db1, dw2, db2
```

Keeping these responsibilities explicit lets the same loss function be reused by numerical gradient checking while the training loop consumes the analytical gradients.

## Numerical gradient checking

Before normal training begins, `numerical_gradient(...)` checks selected parameters using the central-difference approximation:

```python
gradient = (loss_plus - loss_minus) / (2 * epsilon)
```

For each checked weight or bias, the function temporarily evaluates the loss at `parameter + epsilon` and `parameter - epsilon`, then restores the original value. The script prints the analytical and numerical gradients for `w1`, `b1`, `w2`, and `b2` so the chain-rule implementation can be compared against an independent estimate.

## Training and verification

After gradient checking, the normal training loop calls `compute_gradients(...)` 1,000 times and updates every weight and bias with learning rate `0.1`:

```python
parameter -= learning_rate * gradient
```

The script logs the loss every 100 iterations. Run the Day 5 implementation with:

```bash
python multiexample_back_prop.py
```

The output first shows analytical-versus-numerical gradients, then reports the training loss at iterations `0`, `100`, and so on through `900`. This makes the file both an implementation of multi-example backpropagation and a small, inspectable test of its gradient calculations.
