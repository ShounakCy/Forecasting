# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 01:28:47 2021

@author: Shounak
"""

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

df = pd.read_csv("sales.csv", index_col=['Date'], parse_dates=['Date'])

cols =['Steel_Inventory_M_USD','Steel_Orders_M_USD','IronOre_Global_USD',
       'CoalAustralia_Global_USD','Energy_Index','Nickel_Global_USD',
       'Zinc_Global_USD','Freight_Transport_Index','WTI_Crude_Global_USD',	
       'Baltic_Dry_Index','BCI_China','BCI_Europe','BCI_US','CLI_China',
       'CLI_Europe','CLI_US','PMI_US_Manufacturing','Copper_Global_USD'] 

df.drop(cols, axis=1, inplace=True)
df=df.sort_values('Date')
#print(df.head())

df.isnull().sum()

#Grouping by order date
df=df.groupby('Date')['StainlessSteelPrice'].sum().reset_index()

#Indexing with time series data
df=df.set_index('Date')
#print(df.index)


y=df['StainlessSteelPrice'].resample('MS').mean()
y = y.fillna(y.bfill())

y.plot(figsize=(15, 7))
plt.show()

##Seasonal ARIMA
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
#print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16, 8))
plt.show()

pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2013-07-01':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(15, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('StainlessSteel Price')
plt.legend()
plt.show()

y_forecasted = pred.predicted_mean
print(y_forecasted)
y_truth = y['2019-06-01':]
print("y_truth = ",y_truth)
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(mse))

print('The Root Mean Squared Error of our forecasts is {}'.format(np.sqrt(mse)))

mape = np.mean(np.abs((y_truth - y_forecasted)/y_truth))*100
print('The Mean Average Percentage Error forecasts is {}'.format(mape))

pred_uc = results.get_forecast(steps=10)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('StainlessSteel Sales')
plt.legend()
plt.show()