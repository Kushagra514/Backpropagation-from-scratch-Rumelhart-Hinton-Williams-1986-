# `pytorch_comparison.py`

## 1. What this file demonstrates

That **hand-derived NumPy gradients match PyTorch's automatic differentiation** for the same `2 → 2 → 1` sigmoid network. It also documents a key layout difference: the manual convention stores weight matrices as `(in_features, out_features)`, while `torch.nn.Linear` stores them as `(out_features, in_features)`.

---

## 2. Network / Setup

| Property | Value |
|---|---|
| Architecture | 2 inputs → 2 hidden neurons → 1 output |
| Dataset `X` | Shape `(4, 2)` — **one row per example** (4 examples, 2 features) |
| Targets `y` | Shape `(4, 1)` |
| Activation | Sigmoid at both layers |
| Loss | `0.5 × sum((ŷ − y)²)` — **summed**, not averaged |
| Dependencies | `numpy`, `torch` |

> **Layout note:** This file uses `(examples, features)` = `(4, 2)` for `X`. This is the **opposite** of `multiexample_back_prop.py`'s `(features, examples)` = `(2, 4)` convention. The difference is intentional: `pytorch_comparison.py` matches `torch.nn.Linear`'s row-per-example expectation.

### Initial parameters

**Manual convention — `(in_features, out_features)` / `(hidden, output)`:**

```python
W1 = [[0.10, 0.20],   # shape (2, 2): (input_features, hidden_neurons)
      [0.30, 0.40]]

b1 = [[0.10, 0.10]]   # shape (1, 2): one bias per hidden neuron

W2 = [[0.50],         # shape (2, 1): (hidden_neurons, output_neurons)
      [0.60]]

b2 = [[0.10]]         # shape (1, 1)
```

---

## 3. Manual Forward Pass

With `X` of shape `(4, 2)` and `W1` of shape `(2, 2)`:

```python
z1 = X @ W1 + b1      # (4,2) @ (2,2) + (1,2) → (4,2)
A1 = sigmoid(z1)      # (4,2)
z2 = A1 @ W2 + b2     # (4,2) @ (2,1) + (1,1) → (4,1)
y_hat = sigmoid(z2)   # (4,1)
```

NumPy broadcasts `b1` of shape `(1,2)` across the 4 examples and `b2` of shape `(1,1)` across 4 rows.

### Manual loss

```python
manual_loss = 0.5 * np.sum((y - y_hat) ** 2)    # scalar
```

This is a **sum** (not average) over all 4 examples. PyTorch uses the same definition for an apples-to-apples comparison.

---

## 4. Manual Backward Pass

The backward pass follows the chain rule in reverse layer order.

### Output layer

```python
dL_dyhat = y_hat - y                        # (4,1)  ∂L/∂ŷ
dyhat_dz2 = sigmoid_der_frm_activation(y_hat)  # (4,1)  ŷ(1−ŷ)
dZ2 = dL_dyhat * dyhat_dz2                  # (4,1)  δ² = ∂L/∂z2
```

```python
dW2 = A1.T @ dZ2     # (2,4) @ (4,1) → (2,1)   ← A1ᵀ δ²
db2 = np.sum(dZ2, axis=0, keepdims=True)   # (1,1)  sum over examples
```

### Hidden layer

```python
dA1 = dZ2 @ W2.T     # (4,1) @ (1,2) → (4,2)   ← δ² W2ᵀ
dA1_dZ1 = sigmoid_der_frm_activation(A1)   # (4,2)  A1(1−A1)
dZ1 = dA1 * dA1_dZ1  # (4,2)  δ¹ = (δ² W2ᵀ) ⊙ σ′(z1)
```

```python
dW1 = X.T @ dZ1      # (2,4) @ (4,2) → (2,2)   ← Xᵀ δ¹
db1 = np.sum(dZ1, axis=0, keepdims=True)   # (1,2)
```

### Why `axis=0` for bias sums?

`X` is `(examples, features)` so axis 0 is the example axis. Summing `dZ2` (shape `(4,1)`) along axis 0 gives shape `(1,1)` — the gradient of the single output bias. Summing `dZ1` (shape `(4,2)`) gives `(1,2)` — one gradient per hidden bias.

---

## 5. PyTorch Implementation

```python
model = nn.Sequential(
    nn.Linear(2, 2),    # layer 0: 2 inputs → 2 hidden
    nn.Sigmoid(),
    nn.Linear(2, 1),    # layer 2: 2 hidden → 1 output
    nn.Sigmoid()
)
```

### Copying manual parameters into PyTorch

`nn.Linear` stores its weight as shape `(out_features, in_features)` — the **transpose** of the manual convention.

```python
model[0].weight.copy_(torch.tensor(W1.T))   # W1 is (2,2); W1.T is (2,2)
model[0].bias.copy_(torch.tensor(b1[0]))    # b1[0] is (2,)

model[2].weight.copy_(torch.tensor(W2.T))   # W2 is (2,1); W2.T is (1,2)
model[2].bias.copy_(torch.tensor(b2[0]))    # b2[0] is (1,)
```

### PyTorch forward and backward

```python
torch_prediction = model(x_torch)
torch_loss = 0.5 * torch.sum((torch_prediction - y_torch) ** 2)
torch_loss.backward()
```

### Gradient comparison — transpose back

PyTorch gradients are stored in the transposed layout:

```python
torch_dw1 = model[0].weight.grad.detach().numpy().T    # (2,2) → (2,2)
torch_db1 = model[0].bias.grad.detach().numpy().reshape(1, 2)

torch_dw2 = model[2].weight.grad.detach().numpy().T    # (1,2) → (2,1)
torch_db2 = model[2].bias.grad.detach().numpy().reshape(1, 1)
```

`.T` is applied to `.weight.grad` because `nn.Linear.weight.grad` is `(out, in)` — the transpose of the manual `dW` which is `(in, out)`.

---

## 6. Verification

```python
tolerance = 1e-6

all_match = (
    np.max(np.abs(manual_dw1 - torch_dw1)) < tolerance
    and np.max(np.abs(manual_db1 - torch_db1)) < tolerance
    and np.max(np.abs(manual_dw2 - torch_dw2)) < tolerance
    and np.max(np.abs(manual_db2 - torch_db2)) < tolerance
)
```

If all four conditions pass:

```
PASS: Manual gradients match PyTorch gradients.
```

The small residual difference (if any) is floating-point rounding — single-precision arithmetic (`float32`) accumulates error in a different order between NumPy and PyTorch.

---

## 7. Important Implementation Details

- **`sigmoid_der_frm_activation(a)` takes the activation, not `z`.** Since `σ′(z) = a(1−a)`, the derivative is computed from the already-cached forward pass output.
- **Loss is `0.5 * sum(...)`, not `0.5 * mean(...)`.** Both implementations use the same definition. Using `sum` rather than `mean` removes any `1/N` ambiguity.
- **Weight-layout difference is the main subtlety:**
  - Manual: `W1` is `(input, hidden)` = `(2, 2)`; `W2` is `(hidden, output)` = `(2, 1)`
  - PyTorch: `model[0].weight` is `(hidden, input)` = `(2, 2)`; `model[2].weight` is `(output, hidden)` = `(1, 2)`
  - When the matrices happen to be square (`W1` is `2×2`), the transpose is its own shape — this makes the discrepancy invisible unless you check values, not just shapes.
- **All code runs inside `with torch.no_grad():`** except `torch_loss.backward()`. This prevents PyTorch from tracking unnecessary computation.
- **`torch.set_grad_enabled(True)` is called** inside the `no_grad` block to allow `.backward()` to work on `torch_loss`.

### Shape summary

| Variable | Shape | Convention |
|---|---|---|
| `X` | `(4, 2)` | (examples, features) — row per example |
| `y` | `(4, 1)` | |
| `W1` | `(2, 2)` | Manual: (in, hidden) |
| `b1` | `(1, 2)` | Manual: (1, hidden) |
| `z1`, `A1` | `(4, 2)` | (examples, hidden) |
| `W2` | `(2, 1)` | Manual: (hidden, out) |
| `b2` | `(1, 1)` | |
| `z2`, `y_hat` | `(4, 1)` | (examples, out) |
| `dZ2` | `(4, 1)` | δ² |
| `dW2` | `(2, 1)` | Manual gradient, same shape as `W2` |
| `db2` | `(1, 1)` | |
| `dA1` | `(4, 2)` | ∂L/∂A1 |
| `dZ1` | `(4, 2)` | δ¹ |
| `dW1` | `(2, 2)` | Manual gradient, same shape as `W1` |
| `db1` | `(1, 2)` | |

---

## 8. Interview Questions

**Q1: Why does PyTorch store `nn.Linear.weight` as `(out_features, in_features)` instead of `(in_features, out_features)`?**
PyTorch's `nn.Linear` computes `y = xWᵀ + b` (row-vector input times transposed weight). Storing weights as `(out, in)` lets the forward pass be `x @ weight.T + bias` without an explicit transpose, which is efficient for row-major memory layouts with row-per-example data.

**Q2: Why must `.weight.grad` also be transposed when comparing with the manual gradient?**
`model[0].weight.grad` has the same shape as `model[0].weight` — which is `(out, in)`. The manual `dW1` is `(in, out)`. To compare element-by-element, you need `.weight.grad.T`.

**Q3: Why is `dW2 = A1.T @ dZ2` and not `dZ2 @ A1.T`?**
`∂L/∂W2 = A1ᵀ δ²`. `A1` is `(4, 2)` and `dZ2` is `(4, 1)`, so `A1.T @ dZ2` gives `(2, 1)` — matching `W2`. The outer product for each example is `A1[k].T @ dZ2[k]`; summing across examples is `A1.T @ dZ2`.

**Q4: Why is `dA1 = dZ2 @ W2.T`?**
`z2 = A1 @ W2 + b2`, so `∂z2/∂A1 = W2`. The chain rule gives `∂L/∂A1 = δ² Wᵀ` per example. In matrix form across all examples: `dA1 = dZ2 @ W2.T`.

**Q5: What is `sigmoid_der_frm_activation` and why does it take `a` instead of `z`?**
It computes `σ′(z) = σ(z)(1 − σ(z))`. Since `a = σ(z)` is already computed, the derivative is `a(1 − a)` — reusing the cached activation instead of recomputing the sigmoid.

**Q6: Why does the comparison use `max |manual − pytorch|` rather than checking equality?**
Floating-point arithmetic is not associative. NumPy and PyTorch accumulate rounding errors in different orders, so even correct implementations will differ by `~1e-8` in `float32`. A tolerance of `1e-6` accepts these rounding differences while catching genuine errors.

**Q7: What does the PASS result prove?**
That the manually derived chain-rule gradients agree with PyTorch's automatic differentiation to within `1e-6` for this specific architecture (`2→2→1`), these specific parameters, and this specific loss function (`0.5 × sum`). It is a correctness check for this implementation — not a universal proof.

**Q8: Why is the loss `0.5 * sum(...)` instead of the mean?**
To match the definition used in `multiexample_back_prop.py`'s single-example case and to avoid any ambiguity about how `1/N` is applied. Both the manual and PyTorch implementations use the same `sum` definition, so any difference must come from gradient computation, not from loss scaling.
