import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

breast = datasets.load_breast_cancer()
df = pd.DataFrame(breast.data, columns=[x.replace(' ', '_') for x in breast.feature_names])
df['label'] = breast.target
df['mean_radius'] = df['mean_radius'].apply(lambda x: int(x))
df['mean_texture'] = df['mean_texture'].apply(lambda x: int(x))
print(df)
dftrain, dftest = train_test_split(df)
features = ['mean_radius','mean_texture']
lgbm_train = lgbm.Dataset(dftrain.drop(['label'], axis=1), label=dftrain['label'], categorical_feature=features)
lgbm_test = lgbm.Dataset(dftest.drop(['label'], axis=1), label=dftest['label'], categorical_feature=features, reference=lgbm_train)


boost_round = 50
early_stop_rounds = 10
par = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['auc'],
    'num_leaves': 15,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

results = {}
gbm = lgbm.train(par,
                 lgbm_train,
                 num_boost_round=boost_round,
                 valid_sets=(lgbm_test, lgbm_train),
                 valid_names=('validate', 'train'),
                 early_stopping_rounds=early_stop_rounds,
                 evals_result=results)

y_pred_train = gbm.predict(dftrain.drop('label', axis=1), num_iteration=gbm.best_iteration)
y_pred_test = gbm.predict(dftest.drop('label', axis=1), num_iteration=gbm.best_iteration)

print('train accuracy:{:.3f}'.format(accuracy_score(dftrain['label'], y_pred_train>0.5)))
print('valid accuracy:{:.3f}'.format(accuracy_score(dftest['label'], y_pred_test>0.5)))

lgbm.plot_metric(results)
plt.show()
lgbm.plot_importance(gbm, importance_type='gain')
plt.show()

plt.scatter(df['worst_area'][df['label'] == 0], df['mean_concave_points'][df['label'] == 0], color='red')
plt.scatter(df['worst_area'][df['label'] == 1], df['mean_concave_points'][df['label'] == 1], color='blue')
plt.show()
print('end')