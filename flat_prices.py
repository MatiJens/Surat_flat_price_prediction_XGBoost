import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, mean_squared_error, \
    mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_preprocessing import data_preprocessing
import pandas as pd
import xgboost as xgb

pd.set_option('display.max_columns', None)
csv_path = "data/flat_data.csv"

features, target = data_preprocessing(csv_path)

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# HIPERPARAMETERS
N_ESTIMATORS = 1000
MAX_DEPTH = 4
LEARNING_RATE = 0.05
GAMMA = 0.1
COLSAMPLE_BYTREE = 1.0
SUBSAMPLE = 0.8
EARLY_STOPPING_ROUNDS = 50
N_JOBS = 2
"""

# Searching for best parameters
param_grid = {
    'max_depth' : [3, 4, 5, 6, 7],
    'learning_rate' : [0.01, 0.05, 0.1],
    'subsample' : [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma' : [0, 0.1, 0.2]
}

model = xgb.XGBRegressor(
    n_estimators=N_ESTIMATORS,
    #max_depth=MAX_DEPTH,
    #learning_rate=LEARNING_RATE,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    objective='reg:squarederror',
    n_jobs=2
)

random_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    verbose=1
)

random_search.fit(
    x_train,
    y_train,
    eval_set=[(x_test,y_test)],
    verbose=False
)

print("Best parameters:")
print(random_search.best_params_)

print("Best R^2 score:")
print(random_search.best_score_)

#Best parameters:
#{'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 4, 'subsample': 0.8}
#Best R^2 score:
#0.6454801440238953

"""

model = xgb.XGBRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    colsample_bytree=COLSAMPLE_BYTREE,
    subsample=SUBSAMPLE,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    objective='reg:squarederror',
    n_jobs=2
)


model.fit(x_train,
          y_train,
          eval_set=[(x_test, y_test)],
          verbose=False)

y_pred = model.predict(x_test)

y_test_INR = np.expm1(y_test)
y_pred_INR = np.expm1(y_pred)

mae = mean_absolute_error(y_test_INR, y_pred_INR)
mse = mean_squared_error(y_test_INR, y_pred_INR)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test_INR, y_pred_INR)
r2 = r2_score(y_test_INR, y_pred_INR)

print(f"MAE: {mae:,.2f} ₹")
print(f"RMSE: {rmse:,.2f} ₹")
print(f"MAPE: {mape:.2f}")
print(f"R^2 Score: {r2:.4f}")

model.save_model("real_estate_model_r2_0825.json")

# Send data to CSV to analise it
analysis_df = x_test.copy()
analysis_df['true price'] = y_test_INR.round(0)
analysis_df['predicted price'] = y_pred_INR.round(0)
analysis_df['absolute error'] = np.abs(analysis_df['true price'] - analysis_df['predicted price'])
analysis_df['percentage error'] = (analysis_df['true price'] / analysis_df['predicted price']) * 100

analysis_df = analysis_df.sort_values(by='percentage error', ascending=False)

analysis_df.to_csv("error_analise.csv")