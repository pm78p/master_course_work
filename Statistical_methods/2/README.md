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



