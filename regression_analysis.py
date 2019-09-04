import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

stock_data = pd.read_csv('NEE.csv', header=0)
stock_data.set_index('Date',inplace=True)
#print(stock_data.head())
'''
It would not make sense to use the whole dataset for prediction due to growth of the stock over time.
eg, NEE started out in the 70s at $0.5 and now its at $219. Therefore the best method is one which uses x days of training data
from the recent past to predict the next day
'''
days_back = 100
training_size = len(stock_data)-days_back
slices = np.arange(training_size).astype(np.int)[:, None] + np.arange(days_back + 1).astype(np.int)
training_data = stock_data['Adj Close'].values[slices]
X = training_data[:,:-1]
y = training_data[:,-1]
print(X)
print(slices)
print(training_data)
print(y)
stock_data['Adj Close'].plot(label='NEE', figsize=(16,8), title='Adjusted Closing Price', grid=True)
#plt.show()
