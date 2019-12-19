import pandas as pd
import numpy as np
import time
import pickle

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from transformers import ColumnSelector, TypeSelector
from sklearn.preprocessing import OneHotEncoder


from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sbopt.space.space import Real, Integer, Categorical
from sbopt import BayesSearchCV

from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

# load in data
train = pd.read_csv('allstate-claims-severity/train.csv')
test = pd.read_csv('allstate-claims-severity/test.csv')

# transform
train['loss'] = np.log(train['loss'])

# train test split
X_train, X_val, y_train, y_val = train_test_split(train.loc[:, ~train.columns.isin(['loss'])],
                                                    train.loc[:, train.columns.isin(['loss'])],
                                                    test_size=0.33, random_state=42)

# features
features = [col for col in X_train.columns if col != 'id']

# categorical pipeline
categorical_pipeline = Pipeline([('select_categorical', TypeSelector(dtype = 'object')),
                        ('dummy', OneHotEncoder(sparse=False, handle_unknown='ignore'))])
# numerical pipeline
numeric_pipeline = Pipeline([('select_numeric', TypeSelector(dtype = 'number'))])

# processing pipeline
processing = FeatureUnion([('categorical', categorical_pipeline),
             ('numerical', numeric_pipeline)])

# complete pipleine
estimator_pipeline = Pipeline([('Select_Cols', ColumnSelector(columns = features)),
                     ('Processing', processing),
                     ('estimator', XGBRegressor(random_state=42,
                                                    learning_rate=.01,
                                                    verbosity=3))])
# # param space
# search_space = {
#     "estimator__learning_rate": Real(0.001, 0.01),
#     "estimator__max_depth": Integer(3, 10),
#     "estimator__n_estimators": Integer(100, 1000),
#     "estimator__subsample": Real(0.5, 1),
# }
#
# predictor_params = {
#     "estimator__learning_rate": [0.001, .01, .1],
#     "estimator__max_depth": [3, 5, 8, 10],
#     "estimator__n_estimators": [100, 500, 600, 1000],
#     "estimator__subsample": [0.5, 0.8, 1],
# }

predictor_params = {
    "estimator__learning_rate": [.01],
    "estimator__max_depth": [8],
    "estimator__n_estimators": [10],
    "estimator__subsample": [0.5],
}

# metric
metric = make_scorer(
    mean_absolute_error, greater_is_better=False, needs_proba=False)

# cv
kfold_cv = KFold(n_splits = 2, shuffle=True, random_state=42)

starting_time = time.time()
# bayes_tuned_pipeline = BayesSearchCV(
#     estimator=estimator_pipeline,
#     search_spaces=search_space,
#     n_iter=1,
#     scoring=metric,
#     cv=kfold_cv,
#     verbose=12,
#     n_jobs=-1,
#     refit=True
# )

tuned_pipeline = RandomizedSearchCV(
    estimator=estimator_pipeline,
    param_distributions=predictor_params,
    n_iter=1,
    scoring=metric,
    cv=kfold_cv,
    refit=True,
    return_train_score=True,
    verbose=12,
    random_state=42,
    n_jobs=-1,
)

tuned_pipeline.fit(X_train, y_train)
# bayes_tuned_pipeline.fit(X_train, y_train)
print(time.time() - starting_time)

# results
# y_pred = bayes_tuned_pipeline.predict(X_val)
y_pred = tuned_pipeline.predict(X_val)
print(mean_absolute_error(np.exp(y_val), np.exp(y_pred)))



# Saving model using pickle
pickle.dump(tuned_pipeline, open('allstate_model.pkl','wb'))
