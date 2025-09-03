#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor


from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge

from sklearn.feature_selection import RFE


from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 


# Databse
dataset = pd.read_csv("deltaloi_structure.csv")
#dataset = pd.read_csv("final_substructure_dataset_tensile.csv")
#dataset = pd.read_csv("final_substructure_dataset_deltaphrr.csv")
#dataset = pd.read_csv("Tg_DSC.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset_generated_structures = pd.read_csv("ourresult1.csv")
X_generated_structures = dataset_generated_structures

# split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)
# Delta_tensile, X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
# Delta_PHRR, X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=8)
# Delta_Tg, X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=83)


# Data processing
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_generated_structures_scaler = scaler.transform(X_generated_structures)

# Feature Selection
estimator = Ridge(alpha=1, random_state=1, max_iter=100000)
# Delta_tensile, estimator = Ridge(alpha=1, random_state=1, max_iter=100000)
# Delta_PHRR, estimator = Ridge(alpha=1, random_state=1, max_iter=100000)
# Delta_Tg, estimator = Ridge(alpha=1, random_state=1, max_iter=100000)
selector = RFE(estimator, n_features_to_select = 104)
# Delta_tensile, selector = RFE(estimator, n_features_to_select=104)
# Delta_PHRR, selector = RFE(estimator, n_features_to_select =104)
# Delta_Tg, selector = RFE(estimator, n_features_to_select =30)


selector.fit(X_train, y_train)
X_train_selector = selector.transform(X_train)
X_test_selector =selector.transform(X_test) 
X_generated_structures_selector = selector.transform(X_generated_structures_scaler)
                 
# Output Selected Feature
feature_selected = selector.get_support()
logf = open("logfile.log", "a+")
np.set_printoptions(threshold=np.inf)
print(f"{feature_selected}", file=logf, flush=True)

# Regression
regressor = GradientBoostingRegressor(n_estimators=980, max_depth=3, learning_rate=0.17,random_state=3)
# Delta_tensile, regressor = GradientBoostingRegressor(n_estimators=1200, max_depth=2, learning_rate=0.15, random_state=1)
# Delta_PHRR, regressor = GradientBoostingRegressor(n_estimators=1200, max_depth=2, learning_rate=0.15, random_state=8)
# Delta_Tg, regressor = GradientBoostingRegressor(n_estimators=550, max_depth=3, learning_rate=0.17, random_state=1)


 
regressor.fit(X_train_selector, y_train)

# Output feature importance
feature_importances_ = regressor.feature_importances_

# Model Performance

y_train_predict = regressor.predict(X_train_selector)
y_predict = regressor.predict(X_test_selector)
mean_squared_error = mean_squared_error(y_test, y_predict)
root_mean_squard_error = mean_squared_error**0.5
mean_absolute_error = mean_absolute_error(y_test, y_predict)

                                                                        
print(f"train R2: {regressor.score(X_train_selector, y_train):.3f}")
print(f"test R2: {regressor.score(X_test_selector, y_test):.3f}")

# Predict EP/BDOPO
y_generated_structures_predict = regressor.predict(X_generated_structures_selector)

