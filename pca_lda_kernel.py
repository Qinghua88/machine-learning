import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.decomposition import PCA, KernelPCA
# for kernel discrimination analysis we can use code from https://github.com/daviddiazvico/scikit-kda

def plot_scatter(X, y, str):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='x', alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
    plt.text(0, 0, str)
    plt.tight_layout()
    plt.show()


X, y = make_moons(n_samples=200, random_state=0)
plot_scatter(X, y, 'original')

# We will see linear PCA works bad here
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plot_scatter(X_pca, y, 'linear PCA')

# Now we will try kernel PCA
for i in range(1, 30, 3): # We need to try different gamma to get the best result
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=i)
    X_kpca = kpca.fit_transform(X)
    plot_scatter(X_kpca, y, 'gamma={}'.format(i))

###### below is for another dataset

X, y = make_circles(n_samples=200, random_state=0, noise=0.1, factor=0.2)
# plot_scatter(X, y, 'original')
# We will see linear PCA works bad here
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# plot_scatter(X_pca, y, 'linear PCA')

# Now we will try kernel PCA
'''for i in range(1, 30, 3):
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=i)
    X_kpca = kpca.fit_transform(X)
    plot_scatter(X_kpca, y)
    plt.text(0, 0, 'gamma={}'.format(i))'''


kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X)

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='x', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
plt.title('original data')

plt.subplot(1, 3, 2)
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='red', marker='x', alpha=0.5)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='blue', marker='o', alpha=0.5)
plt.title('linear PCA')

plt.subplot(1, 3, 3)
plt.scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='red', marker='x', alpha=0.5)
plt.scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5)
plt.title('kernel PCA')

plt.show()
