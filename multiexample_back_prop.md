# `multiexample_back_prop.py`

## 1. What this file demonstrates

How to extend the single-example `2 → 2 → 1` network to **multiple training examples** processed together. It introduces:

- batch forward pass with a data matrix `X`
- average loss across `N` examples
- batch gradient accumulation
- **numerical gradient checking** to independently verify the analytical derivatives

---

## 2. Network / Setup

| Property | Value |
|---|---|
| Architecture | 2 inputs → 2 hidden neurons → 1 output |
| Dataset `X` | `[[2, 1, 3, 0], [3, 1, 2, 1]]` — shape `(2, 4)` |
| Targets `Y` | `[[1, 0, 1, 0]]` — shape `(1, 4)` |
| Examples | `N = 4` |
| Hidden weights `w1` | `[[0.5, 0.2], [0.1, 0.4]]` — shape `(2, 2)` |
| Hidden biases `b1` | `[[0.1], [0.2]]` — shape `(2, 1)` |
| Output weights `w2` | `[[0.3, 0.6]]` — shape `(1, 2)` |
| Output biases `b2` | `[[0.1]]` — shape `(1, 1)` |
| Activation | Sigmoid at both layers |
| Loss | Average squared error: `L = (1/N) Σ 0.5(ŷₖ − yₖ)²` |
| Learning rate | `η = 0.1` |
| Training iterations | 1,000 (after gradient check) |
| Dependencies | `math` (stdlib only) |

---

## 3. Forward Pass

### Data convention: `(features, examples)`

`X` has shape `(2, 4)` — **one column per example**:

```
X = [[x₀₀, x₀₁, x₀₂, x₀₃],     ← feature 0 for all examples
     [x₁₀, x₁₁, x₁₂, x₁₃]]     ← feature 1 for all examples
```

Column `k` is example `k`. This is the **column-per-example** layout standard in mathematical notation (not the row-per-example layout used by PyTorch).

### Batch hidden pre-activation

```
Z1ᵢₖ = Σⱼ w1ᵢⱼ · X[j][k] + b1ᵢ     for i ∈ {0,1}, k ∈ {0,…,3}
```

Result: `Z1` has shape `(2, 4)` — 2 hidden neurons, 4 examples.

```python
for i in range(len(w1)):
    for k in range(N):
        total = 0
        for j in range(len(X)):
            total += w1[i][j] * X[j][k]
        z1[i][k] = total + b1[i][0]
```

### Batch hidden activation

```
H[i][k] = σ(z1[i][k])
```

`H` has shape `(2, 4)`.

### Batch output pre-activation

```
Z2[0][k] = Σⱼ w2[0][j] · H[j][k] + b2
```

`Z2` has shape `(1, 4)`.

### Batch predictions

```
y_hat[0][k] = σ(Z2[0][k])
```

### Average loss

```
L = (1/N) Σₖ 0.5(y_hat[0][k] − Y[0][k])²
```

`compute_loss()` returns only this scalar. `compute_gradients()` recomputes the same forward pass and then continues into the backward pass.

---

## 4. Backward Pass

### Where `1/N` is introduced

The `1/N` normalisation is **folded into `delta2`** — not applied as a separate step:

```
delta2[0][k] = (y_hat[0][k] − Y[0][k]) · y_hat[0][k] · (1 − y_hat[0][k]) / N
```

```python
delta2[0][k] = (
    (y_hat[0][k] - Y[0][k])
    * y_hat[0][k]
    * (1 - y_hat[0][k])
    / N
)
```

This is correct: the average loss is `L = (1/N) Σₖ 0.5(...)²`, so `∂L/∂z2[k] = (1/N) · (ŷₖ−yₖ) · ŷₖ(1−ŷₖ)`.

### Output weight gradient (accumulated across examples)

```
dw2[0][j] = Σₖ delta2[0][k] · H[j][k]
```

This is the `(0, j)` entry of the matrix product `δ² Hᵀ`, summed over all examples:

```python
for j in range(len(H)):
    for k in range(N):
        dw2[0][j] += delta2[0][k] * H[j][k]
```

### Output bias gradient (summed across examples)

```
db2[0][0] = Σₖ delta2[0][k]
```

### Propagating error to the hidden layer

```
dL_dh[i][k] = w2[0][i] · delta2[0][k]
```

This is `W²ᵀ @ δ²` column by column — for each example `k` independently:

```python
for i in range(len(H)):
    for k in range(N):
        dl_dh[i][k] = w2[0][i] * delta2[0][k]
```

### Hidden layer delta (per example)

```
delta1[i][k] = dL_dh[i][k] · H[i][k] · (1 − H[i][k])
```

### Hidden weight gradient (accumulated)

```
dw1[i][j] = Σₖ delta1[i][k] · X[j][k]
```

This is `δ¹ Xᵀ` summed over examples:

```python
for i in range(len(delta1)):
    for j in range(len(X)):
        for k in range(N):
            dw1[i][j] += delta1[i][k] * X[j][k]
```

### Hidden bias gradient

```
db1[i][0] = Σₖ delta1[i][k]
```

---

## 5. Numerical Gradient Checking

### Central finite difference

```
∂L/∂θ ≈ [L(θ + ε) − L(θ − ε)] / 2ε     with ε = 1e-6
```

### Implementation

```python
def numerical_gradient(parameter, i, j):
    epsilon = 1e-6
    original_value = parameter[i][j]

    parameter[i][j] = original_value + epsilon
    loss_plus = compute_loss(w1, b1, w2, b2, X, Y, N)

    parameter[i][j] = original_value - epsilon
    loss_minus = compute_loss(w1, b1, w2, b2, X, Y, N)

    parameter[i][j] = original_value   # restore
    return (loss_plus - loss_minus) / (2 * epsilon)
```

The function perturbs one scalar in-place, calls the forward-pass-only `compute_loss()` twice, then restores the original value.

### What the check verifies

If `numerical_gradient(w1, i, j)` ≈ `dw1[i][j]`, it confirms that the chain-rule derivation for that weight is correct. The check is applied to every entry of `w1`, `b1`, `w2`, `b2`.

### What the check does not prove

- Correctness for other architectures or loss functions
- That the code is numerically stable
- That training will converge

Numerical gradients are only an approximation (error `O(ε²)`), so tiny differences between analytical and numerical values are expected.

---

## 6. Parameter Update

```python
for iteration in range(1000):
    loss, dw1, db1, dw2, db2 = compute_gradients(w1, b1, w2, b2, X, Y, N)

    w1[i][j] -= learning_rate * dw1[i][j]
    b1[i][0] -= learning_rate * db1[i][0]
    w2[i][j] -= learning_rate * dw2[i][j]
    b2[i][0] -= learning_rate * db2[i][0]
```

Loss is printed every 100 iterations.

---

## 7. Important Implementation Details

- **Three separate functions.** `compute_loss()` is used exclusively by the numerical gradient checker (it perturbs a parameter and needs only the loss, not gradients). `compute_gradients()` does both. `numerical_gradient()` calls `compute_loss()` twice per parameter entry.
- **`1/N` lives in `delta2`, not applied afterwards.** This is a design choice — the normalisation is applied at the first gradient computation point so that all downstream gradients (`dw2`, `db2`, `delta1`, `dw1`, `db1`) are automatically normalised.
- **Gradient accumulation via `+=`.** `dw1`, `dw2`, `db1`, `db2` are initialised to zero and summed across the `N` examples. This is equivalent to the matrix products `δ² Hᵀ` and `δ¹ Xᵀ`.
- **`X` shape is `(features, examples)`, not `(examples, features)`.** One column = one example. This differs from `pytorch_comparison.py`, which uses rows.
- **No NumPy.** All matrix products are triple-nested Python loops.
- **Gradient check runs before training.** The script calls `compute_gradients()` once, checks every gradient numerically, then runs the 1,000-iteration training loop.
- **`b1` is `(2, 1)` — broadcast across examples.** The same bias applies to every example; it is added once per neuron per example, and its gradient is the sum of `delta1[i][k]` over all `k`.

### Shape summary

| Variable | Shape | Description |
|---|---|---|
| `X` | `(2, 4)` | Input matrix — 2 features, 4 examples |
| `Y` | `(1, 4)` | Targets |
| `z1`, `H` | `(2, 4)` | Hidden pre-act / act (2 neurons, 4 examples) |
| `z2`, `y_hat` | `(1, 4)` | Output pre-act / predictions |
| `delta2` | `(1, 4)` | Output error signals (with `1/N` baked in) |
| `dw2` | `(1, 2)` | Output weight gradient |
| `db2` | `(1, 1)` | Output bias gradient |
| `dl_dh` | `(2, 4)` | Loss gradient w.r.t. `H` |
| `delta1` | `(2, 4)` | Hidden error signals |
| `dw1` | `(2, 2)` | Hidden weight gradient |
| `db1` | `(2, 1)` | Hidden bias gradient |

---

## 8. Insights

**Q1: Why is `1/N` placed inside `delta2` rather than applied to the final gradients?**
Because `∂L/∂z2[k] = (1/N)(ŷₖ−yₖ)ŷₖ(1−ŷₖ)` — the `1/N` is genuinely part of the gradient of the average loss. Folding it in early means all downstream gradients (`dw2`, `delta1`, `dw1`, etc.) are automatically scaled, without needing a separate normalisation step.

**Q2: What does `dw1[i][j] = Σₖ delta1[i][k] · X[j][k]` compute?**
This is the `(i, j)` entry of `δ¹ Xᵀ`. It accumulates the gradient contribution of weight `w1[i][j]` across all `N` training examples. Each example contributes `delta1[i][k] · X[j][k]`.

**Q3: Why does `compute_loss()` exist separately from `compute_gradients()`?**
The numerical gradient checker needs to evaluate only the forward pass (no backward needed). Using a separate, lighter function avoids redundant backward-pass work during the perturbation loop.

**Q4: Why is central finite difference more accurate than forward difference?**
Central: error is `O(ε²)`. Forward `[L(θ+ε) − L(θ)] / ε`: error is `O(ε)`. Central difference cancels the odd-order Taylor terms, giving a second-order accurate approximation for the same computational cost.

**Q5: What does passing the gradient check prove?**
That the analytical (chain-rule) gradient matches the numerical (finite-difference) gradient to `O(ε²)` precision, for these specific parameter values and this specific architecture. It does not prove correctness in general.

**Q6: Why are bias gradients `db1[i][0] = Σₖ delta1[i][k]` summed rather than averaged separately?**
Because the `1/N` was already folded into `delta2`. Every delta downstream already carries the `1/N` factor, so summing over examples gives the correct gradient of the *average* loss.

**Q7: What would break if `X` were transposed to `(examples, features)` without changing the code?**
The inner-loop index order `w1[i][j] * X[j][k]` would be wrong — `X[j][k]` with `j` as the feature index and `k` as the example index would access row `j`, column `k`, but in transposed `X` the roles are swapped. The resulting `z1` would be computed from wrong combinations of features.

**Q8: How does `db2[0][0] = Σₖ delta2[0][k]` relate to the gradient of the average loss?**
`b2` applies the same scalar offset to all examples, so `∂L_avg/∂b2 = (1/N) Σₖ (∂Lₖ/∂z2[k])`. With `1/N` already inside `delta2[0][k]`, summing gives exactly `∂L_avg/∂b2`.
