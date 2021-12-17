import os
import random
import numpy as np
from pylab import mpl, plt

plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
os.environ['PYTHONHASHSEED'] = '0'

x = np.linspace(0, 10)

def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
set_seeds()

y = x + np.random.standard_normal(len(x))
reg = np.polyfit(x, y, deg=1)

print(reg)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo', label='data')
plt.plot(x, np.polyval(reg, x), 'r', lw=2.5, label='linear regression')
plt.legend(loc=0)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo', label='data')
xn = np.linspace(0, 20)
plt.plot(xn, np.polyval(reg, xn), 'r', lw=2.5, label='linear regression')
plt.legend(loc=0)
plt.show()

x = np.arange(12)

lags = 3

m = np.zeros((lags + 1, len(x) - lags))

m[lags] = x[lags:]
for i in range(lags):
    m[i] = x[i:i - lags]

print(m.T)

reg = np.linalg.lstsq(m[:lags].T, m[lags], rcond=None)[0]

print(reg)

print(np.dot(m[:lags].T, reg))

import pandas as pd

raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv', index_col=0, parse_dates=True).dropna()

print(raw.info())

symbol = 'GLD'

data = pd.DataFrame(raw[symbol])

data.rename(columns={symbol: 'price'}, inplace=True)

lags = 5

cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    data[col] = data['price'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

reg2 = np.linalg.lstsq(data[cols], data['price'], rcond=None)[0]
print(reg2)

data['prediction'] = np.dot(data[cols], reg2)
data[['price', 'prediction']].plot(figsize=(10, 6))
plt.show()

data[['price', 'prediction']].loc['2019-10-1':].plot(figsize=(10, 6))
plt.show()

data['return'] = np.log(data['price'] / data['price'].shift(1))

data.dropna(inplace=True)

cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    data[col] = data['return'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

reg3 = np.linalg.lstsq(data[cols], data['return'], rcond=None)[0]

print(f'reg3={reg3}')

data['prediction'] = np.dot(data[cols], reg3)

data[['return', 'prediction']].iloc[lags:].plot(figsize=(10, 6))
plt.show()
data['prediction'].value_counts()
hits = np.sign(data['return'] * data['prediction']).value_counts()
print(hits)
print(hits.values[0]/sum(hits))

'''
data['prediction'] = np.sign(np.dot(data[cols], reg3))
data['prediction'].value_counts()
hits2 = np.sign(data['return'] * data['prediction']).value.counts()
print(hits2)
print(hits2.values[0] / sum(hits2))
'''

print(data.head())
data['strategy'] = data['prediction'] * data['return']

print(data[['return', 'strategy']].sum().apply(np.exp))

data[['return', 'strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()
