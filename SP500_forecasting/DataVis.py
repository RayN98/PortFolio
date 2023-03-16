import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
sns.set_style('darkgrid')

df_main = pd.read_csv("C:/Users/braya/Downloads/Stocks/main_df.csv", index_col=0)
n_stocks = 10

target = 'Adj Close'
forecast_out = 22
predictor = 'pred_RFR_ma15'

list_1 = df_main.iloc[:n_stocks].index.to_list()
# list_1 = ['AAPL', 'NVDA', 'TSLA', 'MSFT']
# list_1 = ['AAPL']

for el1 in list_1:
    with open("C:/Users/braya/Downloads/Stocks/df_" + el1 + ".pickle", 'rb') as f:
        df = pickle.load(f)
    with open("C:/Users/braya/Downloads/Stocks/df_minmax_" + el1 + ".pickle", 'rb') as g:
        df_minmax = pickle.load(g)

    df_max_p = df_minmax.iloc[:2].copy()
    df_min_p = df_minmax.iloc[2:].copy()
    pl_bool = 1
    if pl_bool == 1:
        # data_val_limit = [df.iloc[-1 - forecast_out * 1].name, df.iloc[-1].name]
        data_val_limit = [df.iloc[-forecast_out * 2].name, df.iloc[-1].name]
        fig, ax = plt.subplots(figsize=[12, 5])
        ax = sns.lineplot(data=df['Adj Close'], alpha=0.6, color='k')
        ax = sns.lineplot(data=df['pred_RFR'], alpha=0.7)
        ax = sns.lineplot(data=df['pred_RFR_ma15'], alpha=1)
        ax = plt.axvline(x=df.iloc[-forecast_out].name, color='purple', alpha=0.8)

        ax = sns.lineplot(data=df_max_p[predictor], color='g')
        ax = sns.lineplot(data=df_min_p[predictor], color='r')

        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xlim([data_val_limit[0], df.index[-1]])
        plt.ylim([df['Adj Close'].iloc[-forecast_out * 2:].min() * 0.95,
                  df['pred_RFR'].iloc[-forecast_out * 2:].max() * 1.05])
        plt.show()
