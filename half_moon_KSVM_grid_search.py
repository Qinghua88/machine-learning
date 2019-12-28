import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np

def plot_scatter(X, y, str):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(str)
    plt.tight_layout()
    # plt.show()

X, y = make_moons(n_samples=200, random_state=0, noise=0.1)
plot_scatter(X, y, 'half moon')

svc = GridSearchCV(SVC(kernel='rbf', gamma=0.1),
                   param_grid={
                       "C": [1e0, 1e1, 1e2, 1e3],
                       "gamma": np.logspace(-2, 2, 5)
                   }, scoring='accuracy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
svc.fit(X_train, y_train)

means = svc.cv_results_['mean_test_score']
stds = svc.cv_results_['std_test_score']
print('mean and std of scores for validation set in grid search')
for mean, std, params in zip(means, stds, svc.cv_results_['params']):
    print('{} : {:.3f} +/- {:.3f}'.format(params, mean, std))

svc = SVC(kernel='rbf', gamma=10, C=10)
svc.fit(X_train, y_train)
y_test_pred = svc.predict(X_test)
print(classification_report(y_test, y_test_pred))
X_support = svc.support_vectors_

plt.scatter(X_support[:, 0], X_support[:, 1], marker='o', facecolors='none', edgecolors='r')
plt.text(-1.2, -0.75, 'Red circles are support vectors when choose gamma = 10 for rbf kernel')
plt.tight_layout()
plt.show()