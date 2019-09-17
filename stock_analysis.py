import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import seaborn as sns
from matplotlib import pyplot as plt
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
#plt.show()
# calculting expected return. Pt/(Pt-1)-1 where Pt is the stoc value at day t
print(true_stock_price.head())
print(true_stock_price.shift(1).head())
returns = true_stock_price/true_stock_price.shift(1)-1 # technique when you want to perform operation between items within same columns same as pct_change
# returns = true_stock_price.pct_change() Same thing
sns.lineplot(x=df['Date'], y=returns)
plt.title("Expected Returns RANGE(-1 to 1")
#plt.show()

# analyze potential competitors
competitors = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']
print(competitors)

# is there a correlation among competitors?
retscomp = competitors.pct_change()
correlation_map = retscomp.corr()
print(correlation_map)
# Visualize correlation among stocks
sns.heatmap(correlation_map, cmap="Greens")
plt.title("Correlation Matrix among Stocks")
plt.show()

# scatter plot between 2 sample stocks
sns.scatterplot(x=competitors.AAPL, y=competitors.GOOG)
plt.title("Return Distribution between 2 Stocks")
plt.xlabel("Returns APPL")
plt.ylabel("Returns GOOG")
plt.show()