# Backpropagation From Scratch

Manual implementation of backpropagation in Python, built step-by-step from a single sigmoid neuron to a two-layer network trained on multiple examples. Gradients are derived and coded by hand using the chain rule — no autograd. The final script verifies every gradient against PyTorch's automatic differentiation.

Inspired by Rumelhart, Hinton & Williams (1986), *"Learning representations by back-propagating errors."* This is an educational re-derivation, not a reproduction of the paper's experiments.

---

## Learning Progression

| File | Architecture | Main idea |
|---|---|---|
| [`1neuron.py`](./1neuron.py) | Single sigmoid neuron | Scalar chain rule, gradient descent |
| [`2neuron_layer.py`](./2neuron_layer.py) | 2-input → 2-output layer | Per-neuron δ, `dW = δxᵀ`, bias update |
| [`hidden_neuron_layer.py`](./hidden_neuron_layer.py) | 2 → 2 → 1 network | Backprop through a hidden layer, `Wᵀδ` |
| [`multiexample_back_prop.py`](./multiexample_back_prop.py) | 2 → 2 → 1, 4 examples | Batch gradients, averaged loss, numerical gradient checking |
| [`pytorch_comparison.py`](./pytorch_comparison.py) | 2 → 2 → 1, 4 examples | NumPy manual gradients vs. PyTorch autograd |

---

## Core Mathematics

### Forward Propagation

For a single layer, the forward pass computes:

```
z = Wx + b      ← pre-activation (linear combination)
a = σ(z)        ← activation (squashed through sigmoid)
```

The loss for a single example is:

```
L = 0.5 × (a − y)²     ← squared-error loss; the 0.5 cancels the exponent cleanly
```

### Sigmoid and Its Derivative

```
σ(z) = 1 / (1 + e^−z)
```

Differentiating with respect to `z`:

```
σ′(z) = σ(z) · (1 − σ(z))
```

Crucially, if `a = σ(z)` is already computed during the forward pass, then `σ′(z) = a(1 − a)`. This means **the backward pass needs no extra computation** — the derivative is free from the cached activation.

### Chain Rule — Scalar Derivation

Starting from `L = 0.5(a − y)²`, `a = σ(z)`, `z = wx + b`:

```
∂L/∂a = a − y
∂a/∂z = a(1 − a)         ← sigmoid derivative reused from forward pass
∂z/∂w = x
∂z/∂b = 1
```

Defining the **error signal** δ as the loss sensitivity at the pre-activation:

```
δ = ∂L/∂z = (∂L/∂a)(∂a/∂z) = (a − y) · a(1 − a)
```

The parameter gradients then follow directly:

```
∂L/∂w = δ · x      ← chain rule: ∂L/∂z × ∂z/∂w
∂L/∂b = δ          ← because ∂z/∂b = 1
```

`δ` is the central quantity in backpropagation. Computed once at a neuron's pre-activation, it is reused to derive every weight gradient touching that neuron and to carry the error signal to earlier layers.

### Gradient Descent

After computing gradients, parameters are updated opposite the gradient:

```
w ← w − η · ∂L/∂w
b ← b − η · ∂L/∂b
```

where `η` is the learning rate (`0.1` throughout this project).

---

## From Scalar to Matrix: One Layer with Multiple Neurons

When a layer has multiple output neurons, the scalar rule extends naturally to matrix form. For weight matrix `W`, input vector `x`, and output `a = σ(Wx + b)`:

- Each output neuron `i` has its own scalar error signal: `δᵢ = (aᵢ − yᵢ) · aᵢ(1 − aᵢ)`
- Each weight `wᵢⱼ` has gradient: `∂L/∂wᵢⱼ = δᵢ · xⱼ`

Stacking all neurons, this is exactly the **outer product**:

```
δ = [δ₁, δ₂, ...]ᵀ     (column vector, one entry per output neuron)

dW = δ xᵀ              ← outer product; (output_neurons × input_features)
db = δ                  ← one entry per neuron, since ∂z/∂b = 1
```

The scalar rule `dL/dw = δ · x` and the matrix rule `dW = δxᵀ` are the same equation — the matrix form is just all neurons at once.

---

## Hidden-Layer Backpropagation

Architecture used in [`hidden_neuron_layer.py`](./hidden_neuron_layer.py):

```
x → [W¹, b¹] → z¹ → σ → h → [W², b²] → z² → σ → ŷ → L
```

**Output layer** (same as the single-layer case):

```
δ² = (ŷ − y) · ŷ(1 − ŷ)

dW² = δ² hᵀ
db² = δ²
```

**Propagating error to the hidden layer:**

The hidden activations `h` contribute to the loss through `z² = W²h + b²`, so:

```
∂L/∂h = W²ᵀ δ²
```

`W²ᵀ` appears because `∂z²/∂h = W²`, and the chain rule requires its transpose to route the scalar output error back to each hidden unit that produced it.

**Hidden-layer error signal:**

```
δ¹ = (W²ᵀ δ²) ⊙ σ′(z¹)     ← ⊙ is elementwise multiplication
   = (W²ᵀ δ²) ⊙ h(1 − h)   ← sigmoid derivative reused from forward pass

dW¹ = δ¹ xᵀ
db¹ = δ¹
```

**General recursive form** (backpropagation across any number of layers):

```
δˡ = (Wˡ⁺¹ᵀ δˡ⁺¹) ⊙ σ′(zˡ)
```

Error signals flow from output to input: each layer receives the next layer's error weighted by the connecting weights transposed, then gates it through the local activation derivative.

---

## Multiple Examples / Batch Backpropagation

[`multiexample_back_prop.py`](./multiexample_back_prop.py) uses the convention **`(features, examples)`**:

```
X shape = (2, 4)    — 2 features, 4 examples; one column = one example
Y shape = (1, 4)
```

The average loss over `N` examples is:

```
L = (1/N) Σₖ 0.5(ŷₖ − yₖ)²
```

The `1/N` factor is baked into `delta2` per example:

```
delta2[k] = (ŷₖ − yₖ) · ŷₖ(1 − ŷₖ) / N
```

Gradients are accumulated across all examples (equivalent to summing outer products):

```
dW² = δ² Hᵀ          (sums over N examples)
dW¹ = δ¹ Xᵀ          (sums over N examples)
```

---

## Numerical Gradient Checking

The central finite difference approximation:

```
∂L/∂θ ≈ [L(θ + ε) − L(θ − ε)] / 2ε
```

[`multiexample_back_prop.py`](./multiexample_back_prop.py) uses `ε = 1e-6`. Each weight and bias is perturbed individually; the resulting loss change is compared against the analytical gradient from backpropagation. Close agreement provides an independent check that the hand-derived chain-rule expressions are correct.

---

## PyTorch Verification

[`pytorch_comparison.py`](./pytorch_comparison.py) runs the same forward and backward pass two ways:

| | Implementation |
|---|---|
| **Manual** | NumPy + hand-derived gradients |
| **Reference** | PyTorch `Linear → Sigmoid → Linear → Sigmoid` |

**Weight-layout difference:**

| | W1 shape | W2 shape |
|---|---|---|
| Manual | `(2, 2)` — (input\_features, hidden\_neurons) | `(2, 1)` — (hidden\_neurons, output\_neurons) |
| PyTorch `nn.Linear` | `(2, 2)` — (out\_features, in\_features) | `(1, 2)` — (out\_features, in\_features) |

PyTorch stores weights **transposed** relative to the manual convention. The script seeds PyTorch with `W1.T` and `W2.T`, and transposes `.weight.grad` back when comparing.

The verification checks:

```
max |manual_gradient − pytorch_gradient| < 1e-6
```

for `dW1`, `db1`, `dW2`, and `db2`. This confirms the manual implementation is correct for this specific architecture, loss function, and parameter values — it does not prove backpropagation universally.

> **Note:** This file uses `X` shape `(examples, features)` = `(4, 2)`, matching PyTorch's row-per-example convention. This differs from `multiexample_back_prop.py` intentionally.

---

## File-by-File Summary

### [`1neuron.py`](./1neuron.py)

- Single sigmoid neuron; `x = 2`, `y = 1`, `w = 0.5`, `b = 0.1`
- Forward pass: `z = w * x * b`, `a = σ(z)`, `L = 0.5(a − y)²`
- Backward pass explicitly names each partial derivative: `dL_da`, `da_dz`, `dz_dw`, `dz_db`
- Gradient descent: `w -= lr * dL_dw`, `b -= lr * dL_db`; `lr = 0.1`, 110,000 iterations
- Establishes the core pattern: **forward → loss → chain rule → δ → update**

> **Note on `z`:** `1neuron.py` computes `z = w * x * b` (multiplication). The general form throughout the project is `z = wx + b` (addition); the scalar file treats `b` as a multiplicative parameter rather than an additive bias.

### [`2neuron_layer.py`](./2neuron_layer.py)

- 2-input, 2-output layer; pure Python nested lists, no NumPy
- Forward pass via explicit loops computing `z = W @ x + b` (matrix-multiply semantics)
- Backward pass: per-neuron `δᵢ = (aᵢ − yᵢ) · aᵢ(1 − aᵢ)`, then `dW[i][j] = δᵢ · xⱼ`
- Makes the transition from scalar `dL/dw = δ · x` to matrix `dW = δxᵀ` explicit
- 10 iterations, `lr = 0.1`

### [`hidden_neuron_layer.py`](./hidden_neuron_layer.py)

- 2 → 2 → 1 network; single training example
- Output-layer delta: `δ² = (ŷ − y) · ŷ(1 − ŷ)`, then `dW² = δ² hᵀ`, `db² = δ²`
- Hidden-layer: `dL/dh = W²ᵀ δ²`, then `δ¹ = dL/dh ⊙ h(1 − h)`, `dW¹ = δ¹ xᵀ`
- First file to demonstrate the recursive `Wᵀδ` pattern for propagating error backward
- 10 iterations, `lr = 0.1`

### [`multiexample_back_prop.py`](./multiexample_back_prop.py)

- Same 2 → 2 → 1 architecture with 4 training examples; `X` is `(2, 4)`, `Y` is `(1, 4)`
- Separates `compute_loss()`, `compute_gradients()`, and `numerical_gradient()` into distinct functions
- Average batch loss `L = (1/N) Σ 0.5(ŷ − y)²`; the `1/N` factor lives inside `delta2`
- Gradient check compares analytical vs. central-difference (`ε = 1e-6`) for every parameter
- 1,000-iteration training loop with `lr = 0.1`

### [`pytorch_comparison.py`](./pytorch_comparison.py)

- Reimplements the 2 → 2 → 1 network in NumPy (manual) and PyTorch side-by-side
- `X` is `(4, 2)` — one row per example — matching PyTorch's `nn.Linear` convention
- Loss is `0.5 × sum(...)` (not averaged) in both implementations
- Seeds PyTorch weights from manual parameters with transpose adjustments
- Checks `max |manual_grad − pytorch_grad| < 1e-6` for all four parameter tensors

---

## Shapes and Conventions

### `multiexample_back_prop.py` — `(features, examples)` layout

| Symbol | Shape | Description |
|---|---|---|
| `X` | `(2, 4)` | Input — 2 features, 4 examples |
| `W1` | `(2, 2)` | Hidden-layer weights |
| `b1` | `(2, 1)` | Hidden-layer biases |
| `Z1` / `H` | `(2, 4)` | Hidden pre-activations / activations |
| `W2` | `(1, 2)` | Output-layer weights |
| `b2` | `(1, 1)` | Output-layer bias |
| `Z2` / `Ŷ` | `(1, 4)` | Output pre-activations / predictions |

### `pytorch_comparison.py` — `(examples, features)` layout

| Symbol | Shape | Description |
|---|---|---|
| `X` | `(4, 2)` | Input — 4 examples, 2 features |
| `W1` | `(2, 2)` | (input\_features, hidden\_neurons) |
| `b1` | `(1, 2)` | |
| `W2` | `(2, 1)` | (hidden\_neurons, output\_neurons) |
| `b2` | `(1, 1)` | |

> The `(features, examples)` vs. `(examples, features)` difference is **intentional**. `multiexample_back_prop.py` follows the column-per-example convention common in mathematical notation; `pytorch_comparison.py` follows the row-per-example layout required by `torch.nn.Linear`.

---

## What Is Implemented

**Implemented:**
- Sigmoid activation and its derivative `σ′(z) = σ(z)(1 − σ(z))`
- Squared-error loss (per-example and average batch)
- Forward propagation
- Manual backpropagation via the chain rule
- Error signals `δ = ∂L/∂z` for output and hidden layers
- Weight and bias gradients (`dW = δ aᵀ`, `db = δ`)
- Gradient descent parameter updates
- Batch gradients accumulated over multiple examples
- Numerical gradient checking (central finite difference, `ε = 1e-6`)
- PyTorch gradient verification

**Not implemented:**
- Activations other than sigmoid (ReLU, softmax, etc.)
- Loss functions other than squared error
- Optimisers other than vanilla gradient descent
- More than two layers
- Mini-batching (training always uses the full dataset per step)
- Regularisation

---

## Why This Project Exists

The goal is to understand what frameworks like PyTorch perform automatically. By deriving and coding the forward pass, loss, chain rule, δ signals, gradient accumulation, and parameter update by hand, every step of learning becomes concrete and inspectable. The PyTorch comparison at the end closes the loop: manually derived gradients must agree with automatic differentiation to floating-point precision.

---

## Core Mental Model

1. **Forward pass** — compute `z = Wx + b`, then `a = σ(z)`, layer by layer.
2. **Loss** — `L = 0.5(ŷ − y)²` measures prediction error.
3. **Backpropagation** — apply the chain rule from output to input.
4. **`δ = ∂L/∂z`** — the local error signal at a neuron's pre-activation.
5. **`dW = δ aᵀ`** — outer product of the error signal and the layer's input activation.
6. **`db = δ`** — because `∂z/∂b = 1`, the bias gradient is just the error signal.
7. **Propagate backward** — `∂L/∂h = Wᵀ δ`, then gate through `σ′` to get the next δ.
8. **Gradient descent** — `θ ← θ − η ∂L/∂θ`; move opposite the gradient.

### Insigts at a glance

- Why is backpropagation just the chain rule?
- What exactly does `δ` represent, and why is it defined at `z` rather than at `a`?
- Why is `dW = δxᵀ`? (Outer product — each `wᵢⱼ` is scaled by `δᵢ` and `xⱼ`)
- Why is `db = δ`? (Because `∂z/∂b = 1`)
- Why does `Wᵀ` appear when propagating error backward?
- Why is the sigmoid derivative `σ(1 − σ)`? (Falls out of the quotient rule)
- Why is `1/N` used for the batch loss, and where does it appear in the code?
- How does numerical gradient checking work, and what does it verify?
- Why do manual and PyTorch gradients differ slightly even when both are correct?
- Why does PyTorch store `Linear.weight` transposed relative to the manual convention?

---

## Running the Project

The first four scripts use **only the Python standard library** (`math`). [`pytorch_comparison.py`](./pytorch_comparison.py) requires **NumPy** and **PyTorch**.

```bash
# Standard-library files — no dependencies
python 1neuron.py
python 2neuron_layer.py
python hidden_neuron_layer.py
python multiexample_back_prop.py

# Requires numpy and torch
python pytorch_comparison.py
```

To install the required packages:

```bash
pip install numpy torch
```

---

## Reference

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).
Learning representations by back-propagating errors.
*Nature*, 323, 533–536.
https://doi.org/10.1038/323533a0
