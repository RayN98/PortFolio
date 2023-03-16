import pickle
import pandas as pd

import yfinance as yf
from pathlib import Path

class poggers:
    def __init__(self, ticker):
        self.ticker = ticker
        try:
            self.uwu = yf.download(tickers=ticker, period="10y", interval="1d", ignore_tz=True, prepost=False, show_errors=True)
            if 'Empty DataFrame' in str(self.uwu):
                raise Exception('Empty DataFrame - might be caused by an invalid symbol')
                self.state = ''
            else:
                filepath = Path("C:/Users/braya/Downloads/Stocks/" + str(self.ticker) + '.csv')
                filepath.parent.mkdir(parents=True, exist_ok=True)
                self.uwu.to_csv(filepath)
                self.state = ticker
            print(self.state)
        except Exception as e:
            self.state = ''

# list_1 = ['AAPL', 'NVDA', 'TSLA', 'MSFT']

with open("C:/Users/braya/Downloads/Stocks/sp500.pickle", 'rb') as f:
    list_1 = pickle.load(f)

real_list = []

for ticker in list_1:
    sniff = poggers(ticker)
    if sniff.state != '':
        real_list.append(ticker)

with open("C:/Users/braya/Downloads/Stocks/tickersTrue.pickle", "wb") as f:
    pickle.dump(real_list, f)







