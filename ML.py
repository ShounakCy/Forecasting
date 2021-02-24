#!/usr/bin/env python
# coding: utf-8

"""
Created on Thu Feb 24 01:28:47 2021

@author: Shounak
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns


sales_df=pd.read_csv('sales.csv')

sales_df.head()


datetimes = pd.to_datetime(sales_df['Date'])

# assign your new columns
sales_df['day'] = datetimes.dt.month
sales_df['month'] = datetimes.dt.day
sales_df['year'] = datetimes.dt.year
sales_df.head()


sales_df.dtypes


sales_df=sales_df.drop(['Date'],axis=1)



sales_df.head()


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


X=sales_df.drop(['StainlessSteelPrice'],axis=1)
y=sales_df.StainlessSteelPrice



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

model=SelectFromModel(Lasso(alpha=0.008,random_state=0))

model.fit(X_train,y_train)


model.get_support()


selected_features=X_train.columns[(model.get_support())]

print(selected_features)


X_train=X_train.drop(['year','CoalAustralia_Global_USD','BCI_China','BCI_Europe', 'BCI_US','CLI_US','day'],axis=1)

X_test=X_test.drop(['year','CoalAustralia_Global_USD','BCI_China','BCI_Europe', 'BCI_US','CLI_US','day'],axis=1)



from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 
# search across 50 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 50, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)

y_pred=rf_random.predict(X_test)


mape = np.mean(abs(y_test-y_pred)/y_test)*100
mape



