# `hidden_neuron_layer.py`

## 1. What this file demonstrates

**Backpropagation through a hidden layer** for a `2 → 2 → 1` network. This is the first file in the project where the error signal must be propagated *backward through weights* to reach an earlier layer — introducing the `W²ᵀ δ²` operation that is the core of backpropagation.

---

## 2. Network / Setup

| Property | Value |
|---|---|
| Architecture | 2 inputs → 2 hidden neurons → 1 output |
| Input | `x = [[2], [3]]` — shape `(2, 1)` |
| Target | `y = [[1]]` — shape `(1, 1)` |
| Hidden weights | `w1 = [[0.5, 0.2], [0.1, 0.4]]` — shape `(2, 2)` |
| Hidden biases | `b1 = [[0.1], [0.2]]` — shape `(2, 1)` |
| Output weights | `w2 = [[0.3, 0.6]]` — shape `(1, 2)` |
| Output biases | `b2 = [[0.1]]` — shape `(1, 1)` |
| Activation | Sigmoid at both layers |
| Loss | `L = 0.5(ŷ − y)²` (single output) |
| Learning rate | `η = 0.1` |
| Iterations | 10 |
| Dependencies | `math` (stdlib only) |

---

## 3. Forward Pass

The full computation graph, step by step:

```
x  →  z1  →  h  →  z2  →  ŷ  →  L
```

**Hidden layer pre-activation:**
```
z1ᵢ = Σⱼ w1ᵢⱼ · xⱼ + b1ᵢ     for i ∈ {0, 1}
```

In code:
```python
for i in range(len(w1)):
    total = 0
    for j in range(len(x)):
        total += w1[i][j] * x[j][0]
    z1[i][0] = total + b1[i][0]
```

**Hidden activation:**
```
hᵢ = σ(z1ᵢ)
```

**Output layer pre-activation:**
```
z2 = Σⱼ w2[0][j] · hⱼ + b2
```

**Output activation:**
```
ŷ = σ(z2)
```

**Loss:**
```
L = 0.5 · (ŷ − y)²
```

---

## 4. Backward Pass

### Output layer delta

The output neuron is identical to the single-neuron case:

```
δ² = ∂L/∂z2 = (ŷ − y) · ŷ(1 − ŷ)
```

```python
delta2[0][0] = (y_hat[0][0] - y[0][0]) * y_hat[0][0] * (1 - y_hat[0][0])
```

### Output weight and bias gradients

`z2 = w2 @ h + b2`, so `∂z2/∂w2[0][j] = hⱼ` and `∂z2/∂b2 = 1`:

```
dw2[0][j] = δ² · hⱼ     → dW² = δ² hᵀ
db2 = δ²
```

```python
for j in range(len(h)):
    dw2[0][j] = delta2[0][0] * h[j][0]
db2 = [[delta2[0][0]]]
```

### Propagating error to the hidden layer

The hidden activations `h` feed into `z2 = w2 @ h + b2`. By the chain rule:

```
∂L/∂hᵢ = ∂L/∂z2 · ∂z2/∂hᵢ = δ² · w2[0][i]
```

Across all hidden neurons this is `W²ᵀ δ²`:

```
∂L/∂h = W²ᵀ δ²
```

**Why the transpose?**
`z2 = W² h`, so `∂z2/∂h = W²` (each column of `W²` corresponds to one hidden unit). To route the scalar error `δ²` *back* to each hidden unit `i`, you multiply by the `i`-th column of `W²`, which is the `i`-th row of `W²ᵀ`. The transpose is the algebraic consequence of reversing the matrix-vector product.

In code (explicit loop implementing `W²ᵀ @ δ²`):

```python
for i in range(len(h)):
    dL_dh[i][0] = w2[0][i] * delta2[0][0]
```

`w2[0][i]` selects column `i` of `w2` (row `i` of `w2ᵀ`), scaled by the single scalar `delta2[0][0]`.

### Hidden layer delta

`hᵢ = σ(z1ᵢ)`, so `∂hᵢ/∂z1ᵢ = hᵢ(1 − hᵢ)`. The chain rule gives:

```
δ¹ᵢ = ∂L/∂z1ᵢ = (∂L/∂hᵢ) · (∂hᵢ/∂z1ᵢ) = (∂L/∂hᵢ) · hᵢ(1 − hᵢ)
```

In vector form:

```
δ¹ = (W²ᵀ δ²) ⊙ h(1 − h)
```

```python
for i in range(len(h)):
    delta1[i][0] = dL_dh[i][0] * h[i][0] * (1 - h[i][0])
```

### Hidden weight and bias gradients

Identical pattern to the output layer:

```
dw1[i][j] = δ¹ᵢ · xⱼ     → dW¹ = δ¹ xᵀ
db1ᵢ = δ¹ᵢ
```

```python
for i in range(len(delta1)):
    for j in range(len(x)):
        dw1[i][j] = delta1[i][0] * x[j][0]
db1 = [[delta1[0][0]], [delta1[1][0]]]
```

### Summary of the backward chain

```
L
→ δ² = (ŷ−y)·ŷ(1−ŷ)            output error signal
→ dW² = δ² hᵀ                    output weight gradient
→ db² = δ²                        output bias gradient
→ ∂L/∂h = W²ᵀ δ²                 propagate error through W²
→ δ¹ = (∂L/∂h) ⊙ h(1−h)         hidden error signal
→ dW¹ = δ¹ xᵀ                    hidden weight gradient
→ db¹ = δ¹                        hidden bias gradient
```

---

## 5. Parameter Update

```python
# update W1, b1
w1[i][j] -= learning_rate * dw1[i][j]
b1[i][0] -= learning_rate * db1[i][0]

# update W2, b2
w2[i][j] -= learning_rate * dw2[i][j]
b2[i][0] -= learning_rate * db2[i][0]
```

All four parameter groups updated with `η = 0.1` per iteration.

---

## 6. Important Implementation Details

- **Pure Python nested lists, no NumPy.** Every matrix operation is a manual loop. This keeps the shape arithmetic visible.
- **Shapes at a glance:**

  | Variable | Shape | Meaning |
  |---|---|---|
  | `x` | `(2, 1)` | Input |
  | `w1` | `(2, 2)` | Hidden weights |
  | `b1` | `(2, 1)` | Hidden biases |
  | `z1`, `h` | `(2, 1)` | Hidden pre-act / act |
  | `w2` | `(1, 2)` | Output weights |
  | `b2` | `(1, 1)` | Output bias |
  | `z2`, `y_hat` | `(1, 1)` | Output pre-act / prediction |
  | `delta2` | `(1, 1)` | Output error signal |
  | `dL_dh` | `(2, 1)` | Loss gradient w.r.t. hidden activations |
  | `delta1` | `(2, 1)` | Hidden error signals |
  | `dw2` | `(1, 2)` | Output weight gradient |
  | `dw1` | `(2, 2)` | Hidden weight gradient |

- **`dL_dh` is computed as an intermediate step.** The code separates `W²ᵀ δ²` from the elementwise sigmoid multiplication. This makes the two parts of δ¹ — the backpropagated error and the local gate — visually distinct.
- **Single training example only.** All matrices are `(n, 1)` columns, not batches. Multi-example generalisation is in `multiexample_back_prop.py`.
- **Only 10 iterations.** Loss decreases but does not converge — the file demonstrates mechanics.

---

## 7. Interview Questions

**Q1: Why does `W²ᵀ` appear when computing the hidden layer's error signal?**
Because `z2 = W² h`. Differentiating: `∂z2/∂h = W²`. The chain rule gives `∂L/∂h = (∂L/∂z2)(∂z2/∂h) = δ² W²`. Written as a matrix-vector product with the error vector on the right: `∂L/∂h = W²ᵀ δ²`. The transpose routes each output neuron's error back to the hidden units proportionally to the connecting weights.

**Q2: What does `δ¹ = (W²ᵀ δ²) ⊙ h(1−h)` compute?**
`W²ᵀ δ²` gives how much each hidden activation contributed to the output error. The elementwise product `⊙ h(1−h)` converts this into the gradient at the hidden pre-activation `z1`, by gating through the local sigmoid derivative.

**Q3: Why is `⊙` elementwise rather than a full matrix product?**
Because `hᵢ = σ(z1ᵢ)` is a scalar function applied independently to each `z1ᵢ`. Its derivative `∂hᵢ/∂z1ᵢ = hᵢ(1−hᵢ)` is a diagonal matrix — the off-diagonal entries are zero. Multiplying by a diagonal matrix is the same as elementwise multiplication.

**Q4: What is the general form of backpropagation this file demonstrates?**
`δˡ = (Wˡ⁺¹ᵀ δˡ⁺¹) ⊙ σ′(zˡ)`. Error signals propagate from output to input, weighted by the transposed weight matrix and gated by the local activation derivative.

**Q5: Why do both `dW¹ = δ¹ xᵀ` and `dW² = δ² hᵀ` have the same form?**
Because the gradient rule `dW = δ aᵀ` (outer product of error signal and layer input) is universal — it applies to every layer. The input to the first layer is `x`; the input to the second layer is `h`.

**Q6: What would change if the hidden layer had 4 neurons instead of 2?**
`w1` becomes `(4, 2)`, `b1` becomes `(4, 1)`, `h` becomes `(4, 1)`, `w2` becomes `(1, 4)`, `dL_dh` becomes `(4, 1)`, `delta1` becomes `(4, 1)`, `dw2` becomes `(1, 4)`, `dw1` becomes `(4, 2)`. The structure of the computation is identical.

**Q7: Where does the sigmoid derivative `h(1−h)` come from in `delta1`?**
`hᵢ = σ(z1ᵢ)` was computed during the forward pass and cached in `h`. Since `σ′(z) = σ(z)(1−σ(z)) = a(1−a)`, the derivative at `z1ᵢ` is `hᵢ(1−hᵢ)` — reused directly from the forward pass without recomputing `z1`.
