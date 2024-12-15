# Summary of Work: Visualizing Dyad and Image Compression via SVD

## Project Tasks

### Visualizing Dyad and Compression

**Given Task:**

- Load an image into memory and compute its SVD.
- Visualize some dyads of the decomposition.
- Plot the singular values of the image.
- Visualize the k-rank approximation of the image for different values of k.
- Compute and plot the approximation error for increasing k.
- Plot the compression factor for increasing k.
- Determine the approximation error when the compressed image retains the same information as the uncompressed one (error = 0).

**Observations:**

1. **Dyad Visualization:**
   - Each dyad contributes specific information to the image reconstruction.
   - Low k values show coarse approximations, while higher k values add more details.
   - For k = 1, the intensity is uniform, while for higher k values, details become clearer.

2. **Singular Values:**
   - Singular values decline sharply after ~25, suggesting minimal information contribution beyond this rank.
   - This aligns with the rank of the image matrix and its inherent structure.

3. **k-Rank Approximation:**
   - Increasing k improves image quality.
   - Beyond a certain k value, visual improvements are negligible.

4. **Approximation Error:**
   - Error decreases rapidly with increasing k.
   - Error reduction correlates with the steep drop in singular values.

5. **Compression Factor:**
   - Compression factor decreases with increasing k.
   - When k reaches a specific value, approximation error is minimal, and the image quality matches the uncompressed version.

## Code Highlights

### Dyad Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Load an image and compute SVD
image = data.camera()
U, S, Vh = np.linalg.svd(image, full_matrices=False)

# Function to reconstruct image from k singular values
def reconstruct_image(U, S, Vh, k):
    return np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vh[:k, :]))

# Visualize dyads
k_visualize = 5
fig, axs = plt.subplots(1, k_visualize, figsize=(15, 5))
for i in range(k_visualize):
    axs[i].imshow(reconstruct_image(U, S, Vh, i+1), cmap='gray')
    axs[i].set_title(f'k = {i+1}')
plt.show()```

### Singular Value Plot
```python
# Plot singular values
plt.figure()
plt.plot(S, label='Camera')
plt.title('Singular Values')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.legend()
plt.show()
```

### k-Rank Approximation
```python
k_values = [5, 20, 50, 100, 300]
errors = []
compression_factors = []

for k in k_values:
    approx_image = reconstruct_image(U, S, Vh, k)
    error = np.linalg.norm(image - approx_image, 'fro')
    errors.append(error)
    compression_factor = 1 - (k * (image.shape[0] + image.shape[1] + 1)) / (image.shape[0] * image.shape[1])
    compression_factors.append(compression_factor)

    # Visualize approximation
    plt.figure()
    plt.imshow(approx_image, cmap='gray')
    plt.title(f'k-Rank Approximation with k = {k}')
    plt.show()
```

### Approximation Error and Compression Factor
```python
# Plot approximation error
plt.figure()
plt.plot(k_values, errors, label='Camera')
plt.title('Approximation Error')
plt.xlabel('k')
plt.ylabel('Error')
plt.legend()
plt.show()

# Plot compression factor
plt.figure()
plt.plot(k_values, compression_factors, label='Camera')
plt.title('Compression Factor')
plt.xlabel('k')
plt.ylabel('Compression Factor')
plt.legend()
plt.show()
```

### Approximation Error and Compression Factor
```python
def calculate_compression_factor(k, m, n):
    return 1 - (k * (m + n + 1)) / (m * n)

k_values = list(range(1, min(image.shape)))
compression_factors = [calculate_compression_factor(k, image.shape[0], image.shape[1]) for k in k_values]

plt.figure()
plt.plot(k_values, compression_factors, label='Camera')
plt.axhline(y=0, color='black', linestyle='--', label='y=0')
plt.title('Compression Factor')
plt.xlabel('k')
plt.ylabel('Compression Factor')
plt.legend()
plt.show()

# Value of k when compression factor is 0
ck0 = np.argmax(np.array(compression_factors) <= 0) + 1
print(f"Value of k when compression factor is 0: {ck0}")
```


# Summary of Work: Classification of MNIST Digits with SVD and PCA Clustering

## Project Tasks

### Task 1: Binary and Multi-Class Classification Using SVD

#### 1. Binary Classification of Digits 3 and 4

**Task Overview:**

Classify digits 3 and 4 using SVD decomposition.

Use projection onto column spaces defined by matrices for classification.

**Methodology:**

- Split data into training and testing sets.
- Compute \( U \) and \( V \) matrices from SVD of training data for classes 3 and 4.
- Project test vectors \( y \) onto \( U \) and \( V \), compute distances \( d_1 \) and \( d_2 \), and classify based on the smaller distance.

**Key Observations:**

- **Imbalance and Similarity Impact:**
  - Imbalance in data, coupled with visual similarity, skews classification in favor of the class with more training examples. For example:
    - (0,8): Imbalanced and visually similar, results show accuracy skewed heavily towards the digit with more data.
    - (8,9): Balanced despite visual similarity, accuracy remains high.
- **Robust Cases:**
  - Balanced datasets (e.g., (3,4)) lead to consistent results despite visual differences.
  - Even with imbalanced datasets, visually distinct digits (e.g., (5,6)) show acceptable accuracy, as similarity in appearance doesn’t exacerbate errors.

#### 2. Multi-Class Classification with Digits 3, 4, and 5

**Methodology:**

- Extend binary classification to three classes: 3, 4, and 5.
- Compute \( U \) matrices for each class and classify based on the smallest distance for \( y \).

**Results and Analysis:**

- Accuracy for the third class (digit 5) drops by ~20% due to its inherent visual similarity to other digits.
- Increasing \( k \) does not always improve results, as the dataset characteristics (balance and similarity) play a more significant role.

#### 3. Specific Test Cases

**Example: (0,8)**

- **Training Data:**
  - Digit 0: \( C_0 \)
  - Digit 8: \( C_8 \)
- **Test Data:**
  - Digit 0: \( t_0 \)
  - Digit 8: \( t_8 \)

**Results:**
- Digit 0: 100% accuracy.
- Digit 8: 0% accuracy.

**Conclusion:** Imbalance favors classification towards the more represented digit.

**Example: (5,6)**

- **Training Data:**
  - Digit 5: \( C_5 \)
  - Digit 6: \( C_6 \)

**Results:** Overall accuracy of 97.3% despite imbalance.

**Conclusion:** Distinct visual features mitigate imbalance effects.

**General Conclusion:**

- Cases with imbalance and visual similarity lead to poor classification (e.g., (0,8)).
- Balanced datasets or distinct digit features yield better accuracy.

### Task 2: Clustering with PCA

#### 1. Dimensionality Reduction and Visualization

**Task Overview:**

- Reduce dimensionality of MNIST digits to 2 or 3 dimensions using PCA.
- Visualize data clusters and compute centroids for each digit.

**Methodology:**

- Perform PCA on \( X \) with fixed \( k \).
- Compute centroids for clusters in the reduced space.
- Visualize data and centroids.

#### 2. Distance-Based Clustering Analysis

**Observations:**

- Average distances from centroids reveal tighter clusters for visually distinct digits.
- Test set distances are larger, indicating some variability in unseen data.

#### 3. Classification Using Centroids

**Methodology:**

- Classify test samples based on proximity to centroids in PCA-reduced space.
- Compute accuracy for varying \( k \) values.

**Results:**

- Increasing \( k \) beyond 3 dimensions does not consistently improve accuracy.
- For \( k = 2 \), accuracy is highest; beyond this, it decreases back to levels observed at \( k = 2 \).

**Key Observations:**

- PCA is effective in clustering visually distinct digits.
- Accuracy depends on the choice of \( k \), digit similarity, and dataset balance.

## Code Highlights

### Task 1: SVD Classification

#### Binary Classification (Digits 3 and 4)

```python
# Compute SVD for training data
U1, _, _ = np.linalg.svd(C1train, full_matrices=False)
U2, _, _ = np.linalg.svd(C2train, full_matrices=False)

# Classifier function
def classifier(U1, U2, y):
    y1_proj = U1 @ (U1.T @ y)
    y2_proj = U2 @ (U2.T @ y)
    d1 = np.linalg.norm(y - y1_proj, ord=2)
    d2 = np.linalg.norm(y - y2_proj, ord=2)
    return 3 if d1 < d2 else 4
```

#### Multi-Class Classification (Digits 3, 4, and 5)

```python
# Compute SVD for training data
U1, _, _ = np.linalg.svd(C1train, full_matrices=False)
U2, _, _ = np.linalg.svd(C2train, full_matrices=False)
U3, _, _ = np.linalg.svd(C3train, full_matrices=False)

# Classifier function
def classifier(U1, U2, U3, y):
    d1 = np.linalg.norm(y - (U1 @ (U1.T @ y)), ord=2)
    d2 = np.linalg.norm(y - (U2 @ (U2.T @ y)), ord=2)
    d3 = np.linalg.norm(y - (U3 @ (U3.T @ y)), ord=2)
    return 3 if d1 < d2 and d1 < d3 else (4 if d2 < d3 else 5)

```

### Task 2: PCA Clustering
#### PCA Implementation

```python
def perform_pca(X, k):
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_centered = X - X_mean
    U, S, Vt = np.linalg.svd(X_centered)
    return U[:, :k].T @ X_centered

# Perform PCA
X_train_pca = perform_pca(X_train, k=2)
X_test_pca = perform_pca(X_test, k=2)
```

#### Distance-Based Classification

```python
def classify_using_centroids(x, centroids):
    distances = {digit: np.linalg.norm(x - centroid) for digit, centroid in centroids.items()}
    return min(distances, key=distances.get)

# Classify test set
predictions = np.array([classify_using_centroids(X_test_pca[:, i], centroids) for i in range(X_test_pca.shape[1])])
accuracy = np.mean(predictions == Y_test)
print(f'Classification Accuracy: {accuracy:.2f}')

```









