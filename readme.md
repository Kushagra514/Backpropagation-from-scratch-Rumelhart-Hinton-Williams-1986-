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
