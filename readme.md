# Backpropagation From Scratch

Small, explicit Python implementations of backpropagation, built as a progression from one trainable neuron to a multi-example neural network. The final example verifies the hand-derived gradients against PyTorch autograd.

The implementations intentionally keep the matrix operations and chain rule visible. They use sigmoid activations and squared-error loss rather than hiding the learning process behind a training framework.

## Learning progression

| File | Model | Main idea |
| --- | --- | --- |
| [1neuron.py](1neuron.py) | One neuron | Forward pass, loss, derivatives, and gradient descent |
| [2neuron_layer.py](2neuron_layer.py) | Two-neuron layer | Vector-style weights, per-neuron deltas, and updates |
| [hidden_neuron_layer.py](hidden_neuron_layer.py) | `2 -> 2 -> 1` network | Backpropagation through a hidden layer |
| [multiexample_back_prop.py](multiexample_back_prop.py) | `2 -> 2 -> 1` network | Batch loss, analytical gradients, numerical gradient checking, and training |
| [pytorch_comparison.py](pytorch_comparison.py) | `2 -> 2 -> 1` network | Manual gradients compared with PyTorch autograd |

## Core equations

For a sigmoid unit:

```text
sigmoid(z) = 1 / (1 + exp(-z))
sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
```

The examples use the per-example squared-error loss:

```text
L = 0.5 * (y_hat - y)^2
```

For the output unit, the error signal is:

```text
delta2 = (y_hat - y) * y_hat * (1 - y_hat)
```

The hidden-layer signal applies the chain rule through the output weights and hidden sigmoid:

```text
dL/dH = W2.T @ delta2
delta1 = dL/dH * H * (1 - H)
```

For the multi-example implementation, gradients are accumulated across the four examples and divided by `N`, because its reported loss is the average batch loss.

## Running the examples

The first four scripts use only the Python standard library. The PyTorch comparison requires NumPy and PyTorch.

From the project directory, create and configure the virtual environment used by this workspace:

```bash
python -m venv .venv
.venv/bin/python -m pip install numpy torch
```

Run an individual example with:

```bash
.venv/bin/python 1neuron.py
.venv/bin/python 2neuron_layer.py
.venv/bin/python hidden_neuron_layer.py
.venv/bin/python multiexample_back_prop.py
.venv/bin/python pytorch_comparison.py
```

## What each script demonstrates

### `1neuron.py`

Trains one sigmoid neuron for 110,000 iterations. It prints the loss on every iteration. With the current initialization, the final printed loss is approximately `0.000002`.

### `2neuron_layer.py`

Implements a two-neuron layer with nested Python lists. Each iteration prints the loss, output deltas, weight gradients, and bias gradients. After the ten iterations currently configured, the final printed loss is approximately `0.119257`.

### `hidden_neuron_layer.py`

Builds a single-example `2 -> 2 -> 1` sigmoid network. It explicitly computes the output delta, propagates it into the hidden layer, and updates `w1`, `b1`, `w2`, and `b2`. After ten iterations, the final printed loss is approximately `0.036872`.

### `multiexample_back_prop.py`

Extends the same architecture to four examples. Its data is stored as features by rows and examples by columns:

```text
X = [[2, 1, 3, 0],
     [3, 1, 2, 1]]
Y = [[1, 0, 1, 0]]
```

The script separates the workflow into:

1. `compute_loss(...)` for the forward pass and average loss.
2. `compute_gradients(...)` for analytical backpropagation.
3. `numerical_gradient(...)` for central-difference checking.
4. A 1,000-iteration training loop.

The numerical check uses:

```text
(loss(parameter + epsilon) - loss(parameter - epsilon)) / (2 * epsilon)
```

The current run reports close agreement between analytical and numerical gradients. Training logs show the average loss decreasing from `0.133016` at iteration `0` to `0.097047` at iteration `900`.

### `pytorch_comparison.py`

Recreates the manual network with:

```text
4 examples -> 2 input features -> 2 hidden neurons -> 1 output
```

Unlike `multiexample_back_prop.py`, this file stores one example per row in `X`, matching the batch convention used by `torch.nn.Linear`. It initializes the PyTorch weights from the manual parameters, computes the same summed squared-error loss, and compares every gradient.

The final verification metric is the maximum absolute difference for each parameter gradient:

```text
max(abs(manual_gradient - pytorch_gradient)) < 1e-6
```

With the current parameters, the comparison produces:

```text
Manual loss:  0.5490019
PyTorch loss: 0.5490018725
dW1 max difference: 2.79e-09
db1 max difference: 1.86e-09
dW2 max difference: 1.49e-08
db2 max difference: 1.49e-08
```

The script ends with:

```text
PASS: Manual gradients match PyTorch gradients.
```

This is the strongest verification in the project: the independently derived NumPy gradients agree with automatic differentiation to floating-point precision.

## Notes on conventions

- `multiexample_back_prop.py` uses shape `(features, examples)` for `X`.
- `pytorch_comparison.py` uses shape `(examples, features)` for `X`.
- Manual weight matrices follow `input_features x output_features` for the first layer and `hidden_neurons x output_neurons` for the second layer.
- PyTorch stores `nn.Linear` weights as `output_features x input_features`, so the comparison transposes them before checking the gradients.
- All examples use sigmoid activations and squared-error loss to keep the chain rule easy to inspect.
