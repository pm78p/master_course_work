# Summary of Work: Visualizing Dyad and Classification of MNIST Digits with SVD Decomposition

## Part 1: Visualizing Dyad

### Task Overview

#### Image Decomposition:
- Load an image into memory and compute its SVD.
- Visualize individual dyads of the SVD decomposition.
- Plot singular values of the matrix and observe their distribution.
- Visualize rank-`k` approximations for different values of `k` and observe the effects.
- Compute and plot the approximation error `||A - A_k||` as `k` increases.
- Plot the compression factor `m*k/n` and determine the approximation error when `k = m`.

#### Observations:
- Singular values decrease rapidly, indicating that only a few terms significantly contribute to the image reconstruction.
- Approximation error decreases with increasing `k`, stabilizing after a certain threshold.
- Compression factor demonstrates diminishing returns in the trade-off between reduced data storage and image quality.

---

## Part 2: Classification of MNIST Digits with SVD Decomposition

### Binary Classification of Digits (3 vs 4)

#### Task Overview:
- Load the MNIST dataset and split it into training and testing sets.
- Extract the subsets corresponding to digits 3 (`3`) and 4 (`4`).
- Compute the SVD decomposition of `X_3` and `X_4`:
  - Use `U_3` and `U_4` for projection.
- Classify unknown digits by comparing distances:
  - Assign `y` to 3 if `||y - U_3 * (U_3^T * y)|| < ||y - U_4 * (U_4^T * y)||`, otherwise assign to 4.
- Evaluate the misclassification rate over the test set.
- Analyze the relationship between digit similarity, data imbalance, and classification accuracy.

#### Key Results:
- **Case (3 vs 4)**:
  - High classification accuracy due to the distinct appearance of digits.
- **Imbalance and Similarity Effects**:
  - Imbalanced datasets (e.g., 0 vs 8) favor the class with more data.
  - Similar-looking digits (e.g., 5 vs 6) lead to reduced accuracy, even with balanced datasets.

#### Insights:
- Performance is significantly influenced by both visual similarity and dataset balance.
- Increasing `k` does not guarantee monotonic improvement in accuracy, as results depend on digit permutations.

### Multiclass Classification (3, 4, 5)

#### Task Overview:
- Extend the binary classifier to handle three classes (3, 4, 5).
- Compute the SVD decompositions `U_3`, `U_4`, and `U_5`.
- Classify `y` based on the smallest distance:
  - Assign `y` to the class that minimizes `||y - U_i * (U_i^T * y)||`, where `i ∈ {3, 4, 5}`.
- Evaluate accuracy for each class and overall performance.

#### Key Results:
- **Accuracy**:
  - Classes 3 and 4 exhibit high accuracy.
  - Class 5 has a ~20% drop in accuracy due to its visual similarity to other digits.

#### Insights:
- Multiclass performance depends on class-specific visual features.
- Imbalanced datasets amplify classification errors, especially for visually similar digits.

---

## Final Observations

### Binary Classification:
- Visual similarity and imbalance are critical factors affecting performance.
- Distinct pairs (e.g., 3 vs 4) yield better results than similar pairs (e.g., 0 vs 8).

### Multiclass Classification:
- Adding more classes reduces accuracy for visually similar digits.
- Dataset balance is crucial to mitigate bias in classification results.

---

## Code Highlights

### Binary Classification

```python
from scipy.io import loadmat
import numpy as np

# Load data
mnist_data = loadmat('MNIST.mat')
X = mnist_data['X']
Y = mnist_data['I'].flatten()

# Split data
def split_data(X, Y, Ntrain):
    idx = np.arange(len(Y))
    np.random.shuffle(idx)
    train_idx, test_idx = idx[:Ntrain], idx[Ntrain:]
    return (X[:, train_idx], Y[train_idx]), (X[:, test_idx], Y[test_idx])

(Xtrain, Ytrain), (Xtest, Ytest) = split_data(X, Y, int(0.7 * len(Y)))

# SVD and classification
U1, _, _ = np.linalg.svd(Xtrain[:, Ytrain == 3], full_matrices=False)
U2, _, _ = np.linalg.svd(Xtrain[:, Ytrain == 4], full_matrices=False)

def classifier(U1, U2, y):
    y1_proj = U1 @ (U1.T @ y)
    y2_proj = U2 @ (U2.T @ y)
    return 3 if np.linalg.norm(y - y1_proj) < np.linalg.norm(y - y2_proj) else 4
# Extend the classifier to handle multiple classes

### Multiclass Classification
def multiclass_classifier(U3, U4, U5, y):
    y3_proj = U3 @ (U3.T @ y)
    y4_proj = U4 @ (U4.T @ y)
    y5_proj = U5 @ (U5.T @ y)
    
    distances = [np.linalg.norm(y - y3_proj), np.linalg.norm(y - y4_proj), np.linalg.norm(y - y5_proj)]
    return np.argmin(distances) + 3
'''
asd










