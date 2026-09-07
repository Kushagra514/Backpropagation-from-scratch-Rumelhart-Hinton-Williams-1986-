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

This section extends the same idea to a small layer with two neurons. The file [2neruon_layer.py](2neruon_layer.py) builds a simple matrix-based forward pass and computes the gradients needed for learning.

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
python 2neruon_layer.py
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
