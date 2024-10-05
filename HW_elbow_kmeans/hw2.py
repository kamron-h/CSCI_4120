from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn.datasets import make_blobs
# Initialize a synthetic dataset
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)


# TODO determine the best k for k-means
# Step 2: Fit K-Means for different k values and calculate inertia (WCSS)
wcss = []
k_range = range(1, 11)  # Try k values from 1 to 10
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Step 3: Plot Inertia vs. k
plt.figure(figsize=(8,5))
plt.plot(k_range, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.grid(True)
plt.show()

# TODO calculate accuracy for best K
# TODO draw a confusion matrix