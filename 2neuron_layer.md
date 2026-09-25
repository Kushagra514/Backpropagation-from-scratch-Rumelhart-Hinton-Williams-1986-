# `2neuron_layer.py`

## 1. What this file demonstrates

How the scalar chain rule from a single neuron **generalises to a layer of neurons** using matrix notation. The key insight is that `dW = δ xᵀ` — the outer product of the error vector and the input vector — naturally emerges when you write all neurons' gradient equations at once.

---

## 2. Network / Setup

| Property | Value |
|---|---|
| Architecture | 2 inputs → 2 output neurons (no hidden layer) |
| Input | `x = [[2], [3]]` — shape `(2, 1)` |
| Weights | `w = [[0.5, 0.2], [0.1, 0.4]]` — shape `(2, 2)` |
| Biases | `b = [[0.1], [0.2]]` — shape `(2, 1)` |
| Targets | `y = [[1], [0]]` — shape `(2, 1)` |
| Activation | Sigmoid (per neuron) |
| Loss | Sum of per-neuron squared errors: `L = Σᵢ 0.5(aᵢ − yᵢ)²` |
| Learning rate | `η = 0.1` |
| Iterations | 10 |
| Dependencies | `math` (stdlib only) |

Weight matrix layout: `w[i][j]` = weight from input `j` to output neuron `i`.

---

## 3. Forward Pass

The code uses three nested loops to compute `z = W @ x + b`:

```python
for i in range(len(w)):           # output neuron index
    for j in range(len(x[0])):    # example index (only 1 example here)
        total = 0
        for k in range(len(x)):   # input feature index
            total += w[i][k] * x[k][j]
        z[i][j] = total + b[i][0]
```

Mathematically:

```
zᵢ = Σⱼ wᵢⱼ xⱼ + bᵢ     for each output neuron i
```

Then sigmoid is applied per element:

```
aᵢ = σ(zᵢ) = 1 / (1 + e^−zᵢ)
```

Loss is summed across both output neurons:

```
L = 0.5(a₀ − y₀)² + 0.5(a₁ − y₁)²
```

---

## 4. Backward Pass

### Error signal per neuron

Each output neuron `i` has its own scalar δ:

```
δᵢ = (aᵢ − yᵢ) · aᵢ(1 − aᵢ)
```

In code:

```python
delta[i][0] = (a[i][0] - y[i][0]) * a[i][0] * (1 - a[i][0])
```

This is identical to the single-neuron derivation — each neuron computes its own `∂L/∂zᵢ` independently.

### Weight gradients — the outer product

For weight `wᵢⱼ` (output neuron `i`, input `j`):

```
∂L/∂wᵢⱼ = ∂L/∂zᵢ · ∂zᵢ/∂wᵢⱼ = δᵢ · xⱼ
```

Writing this for all `i` and `j` at once:

```
dW = δ xᵀ
```

This is the **outer product** of the delta vector `δ` (shape `2×1`) and the input vector `xᵀ` (shape `1×2`), yielding `dW` of shape `(2, 2)` — the same shape as `W`.

In code:

```python
dW[i][j] = delta[i][0] * x[j][0]
```

The loops implement exactly `dWᵢⱼ = δᵢ · xⱼ`.

### Bias gradients

Since `∂zᵢ/∂bᵢ = 1`:

```
∂L/∂bᵢ = δᵢ
```

```python
db[i][0] = delta[i][0]
```

---

## 5. Parameter Update

```python
w[i][j] -= learning_rate * dW[i][j]    # W ← W − η · dW
b[i][0] -= learning_rate * db[i][0]    # b ← b − η · db
```

Standard gradient descent applied element-wise.

---

## 6. Important Implementation Details

- **Pure Python nested lists, no NumPy.** All matrix operations are explicit loops. This is intentional — it makes `dWᵢⱼ = δᵢ · xⱼ` visible as a loop rather than a library call.
- **Shape mental model:**
  - `w` is `(output_neurons × input_features)` = `(2 × 2)`
  - `x` is `(input_features × 1)` = `(2 × 1)`
  - `delta` is `(output_neurons × 1)` = `(2 × 1)`
  - `dW` is `(output_neurons × input_features)` = `(2 × 2)`, same as `w`
  - `db` is `(output_neurons × 1)` = `(2 × 1)`, same as `b`
- **Only 10 iterations.** Loss decreases but does not converge — this file illustrates the mechanics, not full training.
- **No hidden layer.** This is a direct linear-then-sigmoid layer. Backprop does not need to recurse further than the input.
- **Each output neuron's δ is independent.** The two output neurons do not interact during the backward pass because there is no shared hidden layer.

---

## 7. Interview Questions

**Q1: What is `dW = δ xᵀ` and why is it an outer product?**
For each pair `(i, j)`, the gradient `∂L/∂wᵢⱼ = δᵢ · xⱼ`. Arranging all these products into a matrix gives the outer product `δ xᵀ`. Its shape matches `W`: `(output_neurons × input_features)`.

**Q2: Why does each output neuron compute its own δ independently?**
Because there is no weight shared between the two output neurons. Neuron `i`'s loss contribution `0.5(aᵢ − yᵢ)²` only depends on `zᵢ = Σⱼ wᵢⱼ xⱼ + bᵢ`. The gradients for neuron 0 and neuron 1 don't interact.

**Q3: Why is `∂zᵢ/∂wᵢⱼ = xⱼ`?**
Because `zᵢ = wᵢ₀x₀ + wᵢ₁x₁ + bᵢ`. Differentiating with respect to `wᵢⱼ` picks out `xⱼ`.

**Q4: Why does `db[i] = delta[i]`?**
Because `∂zᵢ/∂bᵢ = 1`, so `∂L/∂bᵢ = δᵢ · 1 = δᵢ`.

**Q5: What shape is `dW`, and how does it relate to `W`?**
`dW` is `(2, 2)` — same shape as `W`. This is required for the element-wise subtraction `W -= η * dW` to work correctly.

**Q6: How does this file bridge the single-neuron case to the hidden-layer case?**
It replaces a single scalar `δ` with a vector of deltas (one per output neuron) and a scalar `dL/dw = δ · x` with a matrix `dW = δ xᵀ`. The hidden-layer file then adds the question: how do we compute the input layer's δ when we only know the output layer's δ?

**Q7: What would change if there were 3 input features instead of 2?**
`W` would be `(2 × 3)`, `x` would be `(3 × 1)`, `dW` would be `(2 × 3)`, and the inner loop over inputs would run 3 times. The delta vector stays `(2 × 1)` — it only depends on the output neurons.
