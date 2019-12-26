import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

def exp1(): # we use the original xgboost
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 3,
        'gamma': 0.1,
        'max_depth': 4,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'silent': 0,
        'eta': 0.1,
        'seed': 1000
    }

    par = params.items()
    dtrain = xgb.DMatrix(X_train, y_train)
    num_rounds = 500
    model = xgb.train(par, dtrain, num_rounds)

    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy:{:.2f}".format(accuracy))

    plot_importance(model)
    plt.show()


def exp2(): # We use sklearn style xgb, which is easier to configure
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    model =  xgb.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=1500, silent=True,
                               objective='multi:softmax', gamma=0.1, random_state=1000, min_child_weight=3,
                               colsample_bytree=0.7, subsample=0.7, reg_lambda=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy:{:.2f}".format(accuracy))

    plot_importance(model)
    plt.show()

# Next we will do regression
def exp3():
    boston = load_boston()
    X, y = boston.data, boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    params = {
        'booster': 'gbtree',
        'objective': 'reg:gamma',
        'gamma': 0.1,
        'max_depth': 4,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'silent': 0,
        'eta': 0.1,
        'seed': 1000
    }
    par = params.items()
    dtrain = xgb.DMatrix(X_train, y_train)
    num_rounds = 500
    model = xgb.train(par, dtrain, num_rounds)

    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    mse = mean_squared_error(y_test, y_pred)
    print('mse: {:.2f}'.format(mse))
    plot_importance(model)
    plt.show()

def exp4(): # We will use sklearn style xgb to do regression
    boston = load_boston()
    X, y = boston.data, boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    model = xgb.XGBRegressor(max_depth=4, learning_rate=0.1, n_estimators=1500, silent=True,
                              objective='reg:gamma', gamma=0.1, random_state=1000, min_child_weight=3,
                              colsample_bytree=0.7, subsample=0.7, reg_lambda=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('mse: {:.2f}'.format(mse))
    plot_importance(model)
    plt.show()

def main():
    exp1()
    exp2()
    exp3()
    exp4()

if __name__ == "__main__":
    main()