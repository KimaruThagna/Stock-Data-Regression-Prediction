import pandas as pd
stock_data = pd.read_csv('NEE.csv', header=0)
relevant_data = stock_data[['Date', 'Adj Close']]
relevant_data['Date'] = pd.to_datetime(relevant_data['Date'])
print(relevant_data.head())
print(relevant_data.loc[relevant_data.Date == '1973-02-21'])
