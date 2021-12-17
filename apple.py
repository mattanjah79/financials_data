import numpy as np
import pandas as pd
from pylab import mpl, plt

fn = 'AAPL_1min_05052020.csv'
data = pd.read_csv(fn, index_col=0, parse_dates=True)

data['returns'] = np.log(data['CLOSE'] / data['CLOSE'].shift(1))
to_plot = ['returns']

for m in [1, 3, 5, 7, 9]:
    data['position_%d' % m] = np.sign(data['returns'].rolling(m).mean())
    data['strategy_%d' % m] = (data['position_%d' % m].shift(1) * data['returns'])
    to_plot.append('strategy_%d' % m)

data[to_plot].dropna().cumsum().apply(np.exp).plot(title='AAPL intraday 05. May 2020',
                                                   figsize=(10, 6), style=['-', '--', '--', '--', '--', '--'])
plt.show()