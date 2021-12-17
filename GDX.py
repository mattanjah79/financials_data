import numpy as np
import pandas as pd
from pylab import mpl, plt

raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()

data = pd.DataFrame(raw['GDX'])
data.rename(columns={'GDX': 'price'}, inplace=True)
data['returns'] = np.log(data['price'] / data['price'].shift(1))
SMA = 25
data['SMA'] = data['price'].rolling(SMA).mean()
threshold = 3.5
data['distance'] = data['price'] - data['SMA']
data['distance'].dropna().plot(figsize=(10,6), legend=True)
plt.axhline(threshold, color='r')
plt.axhline(-threshold, color='r')
plt.axhline(0, color='r')
plt.show()

data['position'] = np.where(data['distance'] > threshold, -1, np.nan)
data['position'] = np.where(data['distance'] < threshold, 1, data['position'])
data['position'] = np.where(data['distance'] * data['distance'].shift(1) < 0,0, data['position'])
data['position'] = data['position'].ffill().fillna(0)
data['position'].iloc[SMA:].plot(ylim=[-1.1, 1.1], figsize=(10,6))
plt.show()

data['strategy'] = data['position'].shift(1) * data['returns']
data[['returns', 'strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()





