# Maximum Likelihood Estimation (MLE) and Maximum a Posteriori (MAP) Analysis

## Project Scope

This project compares two optimization approaches, MLE and MAP, applied to a polynomial regression problem, emphasizing the impact of model complexity (degree **K**) and regularization (**λ**). Using synthetic data, various tasks explore error trends, overfitting, underfitting, and the effectiveness of regularization.

### 1. Data Generation and Initialization

#### Synthetic Data:

- **N** data points are generated in the range \([a, b]\).
- Target: 
  \[
  Y = \Phi(X)\theta_{\text{true}} + e
  \]
  where \( e \) is Gaussian noise with variance \( \sigma^2 \).
- Vandermonde matrix \( \Phi(X) \) represents polynomial basis functions.

#### Key Observations:
- The noise variance \( \sigma^2 \) affects the quality of \( Y \) and the regression model's ability to generalize.
- Model performance is strongly influenced by **K**, the polynomial degree, and data density (**N**).

### 2. MLE and MAP Solutions

#### MLE Solution:
- Finds \( \theta_{\text{MLE}} \) by minimizing the negative log-likelihood.
- Equivalent to solving the least squares problem.

#### MAP Solution:
- Includes a prior distribution:
  \[
  \theta \sim N(0, \sigma_{\theta}^2 I)
  \]
  to introduce regularization.
- Balances data likelihood and prior information, controlled by **λ**.

#### Implementation:
- Vandermonde matrix \( \Phi(X) \) construction.
- Solutions computed via Normal Equations, Gradient Descent (GD), and Stochastic Gradient Descent (SGD).

### 3. Training vs. Test Errors

#### Key Experiments:
- For varying **K**, compute training and test errors for MLE and MAP.
- Visualize the learned regression \( f_{\theta}(x) \) on training and test datasets.

#### Findings:
- Low **K**: Underfitting occurs, failing to capture data patterns.
- High **K**: Overfitting leads to low training error but high test error.
- MAP Effectiveness: Regularization via **λ** mitigates overfitting, stabilizing the model for high **K**.

### 4. Degree **K** and Regularization **λ**

#### Comparison of Errors:
- MLE and MAP errors are compared for increasing **K**.
- MAP performs better for large **K**, thanks to regularization.

#### Relative Error Analysis:
- Compute:
  \[
  \text{Err}(\theta) = \frac{\|\theta - \theta_{\text{true}}\|_2}{\|\theta_{\text{true}}\|_2}
  \]
- MAP exhibits lower relative errors for high **λ**, suggesting robustness to overfitting.

### 5. Performance Across Methods

#### Normal Equations vs. GD and SGD:
- Normal Equations deliver the most precise solutions.
- GD and SGD exhibit acceptable but less precise results.

#### Efficiency:
- SGD converges faster for larger datasets, processing batches rather than the full dataset.

### 6. Visualizations

#### Regression Models:
- Plot \( f_{\theta}(x) \) for MLE and MAP with varying **K** and **λ**.
- Training and test datasets visualized with regression curves.

#### Error Trends:
- Training and test errors plotted against **K**.
- Relative errors highlight the trade-off between model complexity and regularization.

### 7. Key Observations

#### Effect of **λ**:
- Acts as a control for overfitting in MAP.
- Optimal **λ** values yield models that generalize better, especially for noisy data or excessive complexity.

#### Impact of Dataset Size (**N**):
- Larger datasets reduce noise impact, improving model accuracy.
- Errors decrease for both MLE and MAP with more data points.

#### Algorithm Comparison:
- Normal Equations outperform iterative methods in precision but are computationally expensive.
- GD and SGD are more efficient for large datasets, with SGD leveraging mini-batches.

### 8. Insights on MLE vs. MAP

- MLE fits data without considering prior information, making it prone to overfitting for high **K**.
- MAP incorporates prior knowledge, balancing data fit and parameter stability.

## Final Notes

The project illustrates the interplay between model complexity, regularization, and dataset characteristics. It highlights the advantages of MAP over MLE, particularly for uncertain **K** or high noise scenarios. Combining insights from theoretical derivations and experimental results, the analysis underscores the importance of regularization in practical applications.
