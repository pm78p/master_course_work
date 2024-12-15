# Summary of Work: Optimization via Gradient Descent and Stochastic Gradient Descent

## Project Tasks

### Part 1: Gradient Descent (GD)

#### Task Overview

**Objective**: Implement the Gradient Descent (GD) algorithm for solving optimization problems.

**Features**:
- Support for fixed step size and backtracking line search.
- Capability to compute key metrics (gradient norm, error, and function values) at each iteration.

#### Methods

##### Vanilla Gradient Descent

- Iterative algorithm.
- Convergence based on tolerance criteria or threshold.

##### Backtracking Line Search

- Dynamically determines step size to ensure sufficient descent.
- Backtracking condition: ensures the sufficient decrease in function value.

#### Test Functions

- **Optimum**: The optimal solution.
- **Observation**: Backtracking converges more efficiently and accurately to the optimum.

- **Optimum**: Solution for the ill-conditioned problem.
- **Observation**: Without backtracking, the algorithm stagnates due to high condition number.

- **Setup**: Vandermonde matrix setup, `A`.
- **Observation**: Backtracking ensures robust convergence even for ill-conditioned matrices.

- **Setup**: Regularization parameter `l`.
- **Observation**: Increasing `l` penalizes large values of `x`, improving stability.

- **Observation**: Backtracking consistently finds the global minimum, while a fixed step size often gets trapped in local minima.

#### Visualizations and Insights

- **Gradient Norms**: Plotting gradient norm over iterations highlights faster convergence with backtracking.
  
- **Error Analysis**: Comparison of errors shows that backtracking achieves lower errors with fewer iterations.

- **Contour Plots**: Visualized GD paths for functions 1 and 2 confirm smoother trajectories with backtracking.

### Part 2: Stochastic Gradient Descent (SGD)

#### Task Overview

**Objective**: Implement SGD for optimization problems with large datasets.

**Features**:
- Processes data in mini-batches.
- Supports logistic regression for binary classification.

#### Methods

##### Stochastic Gradient Descent

- Iterative algorithm.
- Batch size determines mini-batch processing.

##### Logistic Regression

- Model: Logistic regression model for binary classification.
- Gradient: The gradient used in the optimization.

#### Test Cases

- **Dataset**: MNIST handwritten digits.
- **Tasks**: Binary classification for digit pairs (e.g., 0 vs. 1, 2 vs. 3).
- **Training Set Sizes**: 100, 500, 1000, 2000.

#### Observations

- **Accuracy Trends**: Larger training sets improve accuracy for both GD and SGD.
- **SGD Performance**: SGD performs comparably to GD and often converges faster.

- **Digit Pair Analysis**:
  - **0 vs 1**: Higher accuracy with larger datasets due to improved feature discrimination.
  - **2 vs 3**: Consistently high accuracy, indicating distinct features for these digits.

- **Efficiency**: SGD processes mini-batches, making it computationally efficient for large datasets.

#### Insights

- **Training Set Size**: Both GD and SGD benefit from larger datasets, achieving higher accuracy.
- **SGD vs GD**: SGD is more efficient and less prone to overfitting.
- **Logistic Regression Effectiveness**: High accuracy across digit pairs demonstrates the model’s suitability for binary classification. SGD’s stochastic nature helps escape local minima, enhancing generalization.

---

## Code Highlights

### Gradient Descent

```python
def gradient_descent(f, grad_f, x0, kmax, tolf, tolx, alpha, A, b, l):
    x, f_val, grads = [x0], [f(x0, A, b, l)], [grad_f(x0, A, b, l)]
    for k in range(kmax):
        gradient = grad_f(x[-1], A, b, l)
        x_new = x[-1] - alpha * gradient
        x.append(x_new)
        f_val.append(f(x_new, A, b, l))
        grads.append(gradient)
        if np.linalg.norm(gradient) < tolf * np.linalg.norm(grads[0]) or np.linalg.norm(x_new - x[-1]) < tolx:
            break
    return x, f_val, grads
```

### Backtracking Line Search

```python
def backtracking_line_search(f, grad_f, x, alpha=1.0, rho=0.7, c=1e-4, A=0, b=0, l=0):
    while f(x - alpha * grad_f(x, A, b, l), A, b, l) > f(x, A, b, l) - c * alpha * np.linalg.norm(grad_f(x, A, b, l))**2:
        alpha *= rho
    return alpha

```

### Logistic Regression with SGD

```python
def stochastic_gradient_descent(grad_func, X, y, w_init, alpha, epochs, batch_size):
    w = w_init
    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        X_shuffled, y_shuffled = X[indices], y[indices]
        for i in range(0, len(X), batch_size):
            X_batch, y_batch = X_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size]
            w -= alpha * grad_func(w, X_batch, y_batch)
    return w
```

