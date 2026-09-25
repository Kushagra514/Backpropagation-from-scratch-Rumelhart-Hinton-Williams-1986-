# `1neuron.py`

## 1. What this file demonstrates

The complete learning loop for a **single sigmoid neuron** using only scalar arithmetic. Every derivative is written out as a named Python variable so the chain rule is visible step by step — nothing is hidden behind a framework.

---

## 2. Network / Setup

| Property | Value |
|---|---|
| Input | `x = 2` (scalar) |
| Target | `y = 1` (scalar) |
| Initial weight | `w = 0.5` |
| Initial bias | `b = 0.1` |
| Activation | Sigmoid: `σ(z) = 1 / (1 + e^−z)` |
| Loss | Squared error: `L = 0.5(a − y)²` |
| Learning rate | `η = 0.1` |
| Iterations | 110,000 |
| Dependencies | `math` (stdlib only) |

---

## 3. Forward Pass

The code computes:

```
z = w * x * b       ← pre-activation
a = σ(z)            ← activation
L = 0.5 * (a − y)² ← loss
```

> ⚠️ **Implementation note — `z = w * x * b`, not `z = wx + b`**
>
> The standard neuron equation is `z = wx + b` (weight times input, *plus* bias).
> This file computes `z = w * x * b` — it **multiplies** by `b` instead of adding it.
>
> Consequences:
> - `b` here behaves as a **scale factor** on the input, not an additive offset.
> - The true partial derivative `∂z/∂b` for `z = wxb` is `wx`, **not** `1`.
> - However, the code sets `dz_db = 1` (as if `z = wx + b`).
> - This means `dL/db` is computed incorrectly relative to the actual forward pass.
>
> The gradient descent still runs and the loss still decreases (because `w` is updated correctly and `b` is small), but the bias gradient is analytically wrong for `z = wxb`. This is worth noting for interview discussion: a mismatch between the forward pass and the gradient is a common bug.

---

## 4. Backward Pass

The code names each partial derivative explicitly:

```python
dL_da = a - y           # ∂L/∂a
da_dz = a * (1 - a)     # ∂a/∂z  (sigmoid derivative)
dz_dw = x               # ∂z/∂w  (correct for z=wxb: ∂/∂w(wxb) = xb, but code uses x)
dz_db = 1               # ∂z/∂b  (correct for z=wx+b, but z=wxb here)
```

Then applies the chain rule:

```python
dL_dw = dL_da * da_dz * dz_dw    # ∂L/∂w = (a−y) · a(1−a) · x
dL_db = dL_da * da_dz * dz_db    # ∂L/∂b = (a−y) · a(1−a) · 1
```

### Step-by-step derivation (as-coded)

```
∂L/∂a = a − y                     ← derivative of 0.5(a−y)²
∂a/∂z = a(1 − a)                  ← sigmoid derivative; reuses forward-pass value a
∂z/∂w = x                         ← as coded (= xb only if z=wxb, but code uses x)
∂z/∂b = 1                         ← as coded (= wx only if z=wxb, but code uses 1)

δ = ∂L/∂z = (a − y) · a(1 − a)   ← error signal at the pre-activation

∂L/∂w = δ · x
∂L/∂b = δ · 1 = δ
```

The quantity `δ = ∂L/∂z` is the central idea of backpropagation. It combines the loss gradient and the activation derivative into a single reusable scalar.

---

## 5. Parameter Update

```python
w -= learning_rate * dL_dw    # w ← w − η · ∂L/∂w
b -= learning_rate * dL_db    # b ← b − η · ∂L/∂b
```

Both parameters move opposite their gradient. `learning_rate = 0.1`.

---

## 6. Important Implementation Details

- **Pure Python scalar arithmetic.** Only `import math` is used; no NumPy.
- **No matrix operations.** Everything is a Python float. This is intentional — it keeps each derivative term visible as a named variable.
- **`dL_da`, `da_dz`, `dz_dw`, `dz_db` are all named.** In later files (and in frameworks), these are collapsed into `δ`. Here they are left separate to make the chain rule explicit.
- **Sigmoid is defined from scratch** using `math.exp`.
- **110,000 iterations** are used. The loss decreases but may not reach zero because the target `y = 1` is the asymptote of sigmoid — the neuron can approach but never exactly produce `a = 1`.
- **`b` is initialised to `0.1`.** Since `z = w * x * b` and `b` starts near zero, `z` is initially near zero (`0.5 * 2 * 0.1 = 0.1`), and the gradient update for `b` is incorrect as noted above.

---

## 7. Insights 

**Q1: What is δ and why is it useful?**
`δ = ∂L/∂z = (a − y) · a(1 − a)`. It is the loss sensitivity at the pre-activation `z`. Computing it once lets you derive `∂L/∂w = δ · x` and `∂L/∂b = δ` without repeating work. In deeper networks, it is also used to propagate error backward.

**Q2: Why is the sigmoid derivative `a(1 − a)` instead of something involving `z`?**
Because `σ′(z) = σ(z)(1 − σ(z))`, and `a = σ(z)` is already cached from the forward pass. Reusing `a` avoids re-evaluating `σ(z)`.

**Q3: What is the bug in this file's forward pass?**
`z = w * x * b` multiplies by `b`, while the standard neuron uses `z = wx + b`. The backward pass then applies `∂z/∂b = 1` (correct for addition, wrong for multiplication). The correct derivative for `z = wxb` would be `∂z/∂b = wx`.

**Q4: Why is `∂L/∂b = δ` in the standard neuron?**
Because `z = wx + b`, so `∂z/∂b = 1`. The chain rule gives `∂L/∂b = ∂L/∂z · ∂z/∂b = δ · 1 = δ`.

**Q5: Why does the loss use a factor of `0.5`?**
So that `∂L/∂a = (a − y)` (the 2 from the exponent cancels the `0.5`), keeping the gradient expression clean.

**Q6: Why does the loss never reach exactly zero?**
The target `y = 1` is the positive asymptote of sigmoid. Gradient descent can push `a` arbitrarily close to 1, but `z` must go to `+∞` to achieve `a = 1`.

**Q7: What would change if `z = wx + b` were used instead?**
The forward pass would compute a different `z` and `a`. The derivative `∂z/∂w` would be `x` (same) and `∂z/∂b` would be `1` (same as coded), so the backward pass would actually become *correct*. The learning dynamic would also change since `b` would act as a proper offset.
