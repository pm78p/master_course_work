# Summary of Work: Direct Methods for the Solution of Linear Systems and Floating Point Arithmetic

## Project Tasks

### Direct Methods for Linear Systems

#### Given Task:
- Compute the right-hand side `b = Ax`.
- Calculate the condition number of matrix `A` in 2-norm and ∞-norm.
- Solve the system `Ax = b` using `np.linalg.solve()`.
- Compute the relative error.

#### Plot:
- Relative errors as a function of `n`.
- Condition numbers (2-norm and ∞-norm) as a function of `n`.

#### Matrices Tested:
- **Random matrix** (sizes: `n = 10, 20, ..., 100`).
- **Vandermonde matrix** (sizes: `n = 10, 20, ..., 100`).
- **Hilbert matrix** (sizes: `n = 10, 20, ..., 100`).

#### Observations:

**Random Matrix:**
- Condition numbers grow with `n`, leading to ill-conditioning for `n > 50`.
- Relative errors increase significantly beyond this threshold.

**Vandermonde Matrix:**
- Extremely ill-conditioned for `n > 10`.
- Relative errors increase exponentially, aligning with the condition number behavior.

**Hilbert Matrix:**
- Severely ill-conditioned beyond `n = 15`.
- Relative errors grow drastically with `n`, reflecting the matrix's intrinsic properties.

### Key Takeaways:
- Ill-conditioning depends on the matrix, not the norm used for measurement.
- Condition number and relative error exhibit an increasing relationship.
- Norms (e.g., 2-norm vs ∞-norm) show similar trends due to bounding properties.

---

# Floating Point Arithmetic

## Machine Epsilon

Computed as the smallest `ε` such that `1.0 + ε > 1.0`:

```python
def calculate_machine_epsilon():
    epsilon = 1.0
    while 1.0 + epsilon > 1.0:
        epsilon /= 2.0
    return epsilon * 2.0
```
## Result

- **Machine Epsilon**: `ε = 2.220446049250313e-16`

---

## Euler Constant Approximation

Approximated `e` and compared with `np.exp(1)`:

- Errors converge as `n -> ∞`, but large `n` introduces floating-point inaccuracies.
- **Observed behavior**: Convergence stabilizes for `n > 20`.

---

## Matrix Properties

### Computed Results:

- **Rank**:
    - `A1`: Full-rank.
    - `A2`: Not full-rank.
  
- **Eigenvalues**:
    - `A1`: All non-zero.
    - `A2`: Contains zero eigenvalues.

### Deduction:

- Full-rank matrices have non-zero eigenvalues.
- Rank deficiency corresponds to zero eigenvalues.

---

## Code Highlights

### Random Matrix Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

def linear_system_analysis(n_values):
    rel_errors = []
    cond_2_norm = []
    cond_inf_norm = []

    for n in n_values:
        A = np.random.rand(n, n)
        x_true = np.ones(n)
        b = A @ x_true

        # Condition numbers
        cond_2 = np.linalg.cond(A, 2)
        cond_inf = np.linalg.cond(A, np.inf)
        cond_2_norm.append(cond_2)
        cond_inf_norm.append(cond_inf)

        # Solve and calculate error
        x = np.linalg.solve(A, b)
        rel_error = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        rel_errors.append(rel_error)

    # Plotting
    plt.figure()
    plt.plot(n_values, rel_errors, label='Relative Error')
    plt.title('Relative Error vs n')
    plt.xlabel('n')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(n_values, cond_2_norm, label='2-Norm Condition Number')
    plt.plot(n_values, cond_inf_norm, label='∞-Norm Condition Number')
    plt.title('Condition Numbers vs n')
    plt.xlabel('n')
    plt.ylabel('Condition Number')
    plt.legend()
    plt.show()

n_values = range(10, 101, 10)
linear_system_analysis(n_values)
```

## Additional Observations

- **Norm Equivalence**: All norms are bounded by scalar multiples of each other, making them consistent in analyzing matrix behavior.
- **Practical Implication**: Ill-conditioning amplifies numerical errors, especially in larger or special matrices (e.g., Vandermonde, Hilbert).

