import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import matplotlib.pyplot as plt

wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                   'machine-learning-databases/wine/wine.data',
                   header=None)
X, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values
# print(np.unique(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
SC = StandardScaler()
X_train_std = SC.fit_transform(X_train)
X_test_std = SC.transform(X_test)
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
var_exp = pca.explained_variance_ratio_


def plot_var_exp(var_exp):
    cum_var_exp = np.cumsum(var_exp)
    plt.figure()
    plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='individual explained variance')
    plt.step(range(1, 14), cum_var_exp, where='mid', label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


plot_var_exp(var_exp)


def plot_scatter(c1, c2, y):
    plt.figure()
    plt.scatter(c1[y == 3], c2[y == 3], color='red')
    plt.scatter(c1[y == 1], c2[y == 1], color='blue')
    plt.scatter(c1[y == 2], c2[y == 2], color='green')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


plot_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)

from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    # plt.savefig('images.png', dpi=300)
    plt.show()


plot_decision_regions(X_train_pca, y_train, classifier=lr)

plot_decision_regions(X_test_pca, y_test, classifier=lr)

# below code shows how we use LDA to reduce dimension and build a LR classifier
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plot_decision_regions(X_test_lda, y_test, classifier=lr)