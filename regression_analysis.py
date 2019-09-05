import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

stock_data = pd.read_csv('NEE.csv', header=0)
stock_data.set_index('Date',inplace=True)
#print(stock_data.head())
'''
It would not make sense to use the whole dataset for prediction due to growth of the stock over time.
eg, NEE started out in the 70s at $0.5 and now its at $219. Therefore the best method is one which uses x days of training data
from the recent past to predict the next day
'''
days_back = 30
training_size = len(stock_data)-days_back
# convert the single column of Adj Close to a 2 D matrix with a defined window size
slices = np.arange(training_size).astype(np.int)[:, None] + np.arange(days_back + 1).astype(np.int)
training_data = stock_data['Adj Close'].values[slices]
X = training_data[:,:-1]
y = training_data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
# Fit Model
models ={"Linear":LinearRegression(), "ridge":Ridge(),"GBR":GradientBoostingRegressor()}

for model in models:
    clf = models[model].fit(X_train,y_train)
    predict = clf.predict(X_test)
    print(r2_score(y_test,predict))
    print(model)

