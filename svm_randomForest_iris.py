# -*- coding: utf-8 -*-
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import sem
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris



import matplotlib.pyplot as plt

def show_data(x1, x2, y):
    plt.figure()
    plt.scatter(x1[y == 0], x2[y == 0], color='red')
    plt.scatter(x1[y == 1], x2[y == 1], color='blue')
    plt.scatter(x1[y == 2], x2[y == 2], color='green')

def evaluate_cross_validation(svc, X, y, K):
    cv = KFold(K, shuffle=True, random_state=0)
    scores = cross_val_score(svc, X, y, cv=cv)
    print("Mean score: {:.3f} (+/- {:.3f}))".format(np.mean(scores), sem(scores)))

def main():
    X, y = load_iris(return_X_y=True)
    '''print(X.shape)
    print(np.unique(y))
    show_data(X[:, 0], X[:, 1], y)
    show_data(X[:, 0], X[:, 2], y)
    show_data(X[:, 1], X[:, 2], y)
    '''
    
    # for SVC
    svc_1 = SVC(kernel='linear')
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)
    evaluate_cross_validation(svc_1, X_train, y_train, 5)
    svc_1.fit(X_train, y_train)
    y_pred = svc_1.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    rf = RandomForestClassifier(n_estimators=73, max_depth=3, random_state=10)
    evaluate_cross_validation(rf, X_train, y_train, 5)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    '''
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    # for random forest
    rf_model_list = []
    rf_mse_list = []
    for train_index, test_index in kf.split(X_train):
        rf = RandomForestClassifier(n_estimators=73, max_depth=3, random_state=10)
        rf_model_list.append(rf)
        rf.fit(X_train[train_index], y_train[train_index])
        actuals = y_train[test_index]
        predictions = rf.predict(X_train[test_index])
        mse = mean_squared_error(actuals, predictions)
        rf_mse_list.append(mse)
    
    print('rf mse list:{}'.format(rf_mse_list))
    print('rf mse mean:{}'.format(np.mean(rf_mse_list)))

    min_mse = min(rf_mse_list)
    ind = rf_mse_list.index(min_mse)
    best_estimator = rf_model_list[ind]
    y_pred = best_estimator.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    '''
    # we found that in this case svm beats random forest if n_estimators < 73 when tree depth = 3
    
if __name__ == "__main__":
    main()