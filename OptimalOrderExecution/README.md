# Literature Review — Optimal Execution Models

## Almgren–Chriss Framework

One of the most fundamental models I understand is the **Almgren–Chriss framework**.

### Stock Price Dynamics

The stock price evolves as an arithmetic geometric model (AGM) with zero drift:

$$
S_{k+1} = S_k + g(v_k) + \sigma \sqrt{\tau} \epsilon_k
$$

where:
- $g(v)$ — permanent market impact  
- $\sigma$ — volatility  
- $\tau$ — interval length  

---

### Execution Price

The effective execution price per trade is:

$$
\tilde{S}_k = S_k + h(v_k)
$$

where:
- $h(v)$ — temporary market impact

---

### Trading Revenue

$$
\text{Trading Revenue} = \text{Initial Market Value of the Position} + \text{Volatility Impact (Noise)} - \text{Permanent Market Impact} - \text{Temporary Market Impact}
$$

---

### Total Cost of Trading

$$
C(x) = \text{(Initial Value)} - \text{Trading Revenue}
$$

Expected cost:

$$
E[C(x)] = \ldots
$$

Variance of cost:

$$
V[C(x)] = \ldots
$$

---

## Efficient Frontier

The goal of a trader is to **minimize expected cost for a given level of variance**.  
Thus, we want to find optimal strategies that minimize the following unconstrained optimization problem:

**Objective Function:**

$$
\min_x \; E[C(x)] + \lambda V[C(x)]
$$

In discrete form, under **linear temporary** and **permanent impacts**, this can be written as a **quadratic cost function** of $x$:

$$
\min_x \ \frac{1}{2}x^T Q x + q^T x
$$

subject to the **total execution constraint**:

$$
\sum_{k=1}^{N} x_k = X_0
$$

where:
- $x_k$ — quantity traded in period $k$  
- $Q$ — encodes deterministic temporary and permanent impact cost terms  
- $\lambda$ — risk aversion parameter  

---

## Code Implementation

With the convention:

$$
x = \text{trade schedule}
$$

I implemented the above optimization by transforming it into a **KKT (Karush–Kuhn–Tucker) linear system**, solved using Python.

```python
# Example: Solving the Almgren-Chriss model using KKT system
import numpy as np

# parameters
lambda_ = 0.1
gamma = 0.1   # permanent impact
eta = 0.01    # temporary impact
sigma = 0.02  # volatility
tau = 1.0

# system setup
N = 10
Q = eta * (2*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1))
q = gamma * np.ones(N)

# KKT solution
A = np.ones((1, N))
b = np.array([1])
KKT_matrix = np.block([[Q, A.T], [A, np.zeros((1,1))]])
rhs = np.concatenate([-q, b])

solution = np.linalg.solve(KKT_matrix, rhs)
x_optimal = solution[:-1]

print("Optimal trade schedule:", x_optimal)
