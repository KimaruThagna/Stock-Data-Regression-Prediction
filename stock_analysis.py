import pandas as pd
import datetime, math
import pandas_datareader.data as web
from pandas import Series, DataFrame
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
# define starting and ending dates of the time window you want

start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.now()

df = web.DataReader("AAPL", 'yahoo', start, end)
df.reset_index(inplace=True) # allow date to be a column
print(df.tail())

'''
Using rolling mean helps in determining general trend.
This is achieved by cutting out sharp spikes in data(noise)
'''
# moving average
true_stock_price = df['Adj Close']
mavg = true_stock_price.rolling(window=100).mean() # mean of every batch of 100 values. in this case, the Adj close for 100 days

# visualize mavg
fig, ax = plt.subplots(figsize=(12,8))

sns.lineplot(data=df, x=df['Date'], y=true_stock_price, ax=ax)
sns.lineplot(data=df, x=df['Date'], y=mavg, ax=ax)
plt.legend(['APPL', 'MAVG'])
plt.title("SMOOTHING EFFECT OF MOVING AVERAGE")
plt.show()
# calculating expected return. Pt/(Pt-1)-1 where Pt is the stoc value at day t
returns = true_stock_price/true_stock_price.shift(1)-1 # technique when you want to perform operation between items within same columns same as pct_change
# returns = true_stock_price.pct_change() Same thing
sns.lineplot(x=df['Date'], y=returns)
plt.title("Expected Returns RANGE(-1 to 1) for APPL")
plt.show()

# analyze potential competitors
competitors = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']

print(f'Competitors Data Table {competitors}')

# is there a correlation among competitors?
retscomp = competitors.pct_change() # formula is newStock-oldStock/oldStock
correlation_map = retscomp.corr()
print(f'Correlation values between competing stocks{correlation_map}')
# Visualize correlation among stocks
sns.heatmap(correlation_map, cmap="Greens", annot=True)
plt.title("Correlation Matrix among Stocks")
plt.show()

# scatter plot between 2 sample stocks
sns.scatterplot(x=competitors.AAPL, y=competitors.GOOG)
plt.title("Return Distribution between 2 Stocks")
plt.xlabel("Returns APPL")
plt.ylabel("Returns GOOG")
plt.show()

# STOCK RETURN RATE(average rate of return AND RISK(standard deviation of returns)
'''
We can plot a boundary line which determines whether to buy or sell stocks.
This boundary line is dictated by an individuals risk appetite. Higher risk appetite allows the investor to buy
higher risk stocks with higher returns.
The boundary line can be initialized by doing a regression line of best fit. This line can be further varied by tuning
one's risk appetite. Higher risk appetite allows one to buy higher risk stocks with great returns
'''

figure, ax = plt.subplots(1)
sns.regplot(retscomp.mean(),retscomp.std(), ax=ax,fit_reg=True)
plt.xlabel("Return Rate")
plt.ylabel("Risk")
plt.title("Buy Sell Decision Chart")
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
     plt.annotate(
         label,
         xy=(x, y), xytext=(20, -20),
         textcoords = 'offset points', ha = 'right', va = 'bottom',
         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

ax.text(0.001, 0.012, 'BUY', style='italic',
        bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})

ax.text(0.001, 0.024, 'SELL', style='italic',
        bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
plt.show()

# # feature engineering for machine learning
# dfreg = df.loc[:,["Adj Close","Volume"]]
# dfreg["HL_PCT"] = (df["High"]-df["Low"]) / df["Close"] * 100.0
# dfreg["PCT_change"] = (df["Close"]-df["Open"]) / df["Open"] * 100.0
# print(dfreg.head())
#
# # data preprocessing
# # Drop missing value
# dfreg.fillna(value=-99999, inplace=True)
#
# # We want to separate 1 percent of the data to forecast
# forecast_out = int(math.ceil(0.01 * len(dfreg)))
# # Separating the label here, we want to predict the AdjClose
# forecast_col = 'Adj Close'
# dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
# X = np.array(dfreg.drop(['label'], 1))
# # Scale the X so that everyone can have the same distribution for linear regression
# X = preprocessing.scale(X)
# # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
# X_lately = X[-forecast_out:]
# X = X[:-forecast_out]
# # Separate label and identify it as y
# y = np.array(dfreg['label'])
# y = y[:-forecast_out]
#
# X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# # Linear regression
# clfreg = LinearRegression(n_jobs=-1)
# clfreg.fit(X_train, y_train)
# # Quadratic Regression 2
# clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
# clfpoly2.fit(X_train, y_train)
#
# # Quadratic Regression 3
# clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
# clfpoly3.fit(X_train, y_train)
#
# # KNN Regression
# clfknn = KNeighborsRegressor(n_neighbors=2)
# clfknn.fit(X_train, y_train)
#
# # confidence/accuracy
# confidencereg = clfreg.score(x_test, y_test)
# confidencepoly2 = clfpoly2.score(x_test,y_test)
# confidencepoly3 = clfpoly3.score(x_test,y_test)
# confidenceknn = clfknn.score(x_test, y_test)
#
# # print results
# print('The linear regression confidence is ', confidencereg)
# print('The quadratic regression 2 confidence is ', confidencepoly2)
# print('The quadratic regression 3 confidence is ', confidencepoly3)
# print('The knn regression confidence is ', confidenceknn)
#
# # plot the forecast
#
# forecast_set = clfpoly3.predict(X_lately)
# dfreg['Forecast'] = np.nan
#
# last_date = dfreg.iloc[-1].name
# last_unix = last_date
# next_unix = last_unix + datetime.timedelta(days=1)
#
# for i in forecast_set:
#     next_date = next_unix
#     next_unix += datetime.timedelta(days=1)
#     dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
# dfreg['Adj Close'].tail(500).plot()
# dfreg['Forecast'].tail(500).plot()
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()