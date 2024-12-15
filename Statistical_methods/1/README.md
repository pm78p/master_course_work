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

### Floating Point Arithmetic

#### Machine Epsilon:
Computed as the smallest `ε` such that:

```python
def calculate_machine_epsilon():
    epsilon = 1.0
    while 1.0 + epsilon > 1.0:
        epsilon /= 2.0
    return epsilon * 2.0
