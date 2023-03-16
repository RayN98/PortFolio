import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import datetime
import warnings
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

sns.set_style('whitegrid')

class forecast:
    def __init__(self, ticker):
        def pred(df):
            pd.set_option('display.max_columns', None)
            df.drop(columns=['Close'], inplace=True)
            # New features
            df['HL_PCT'] = (df['High'] - df['Adj Close']) / df['Adj Close'] * 100.0
            df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0
            df['ma15'] = df['Adj Close'].rolling(window=15, min_periods=8, center=True).mean()
            df['ma30'] = df['Adj Close'].rolling(window=30, min_periods=15, center=True).mean()
            df['ma50'] = df['Adj Close'].rolling(window=50, min_periods=25, center=True).mean()
            # Setting the target and prediction range
            target = 'Adj Close'
            forecast_out = 22

            # Shifting the data set to asign the close price value to the prediction of the Adj Close of the next forecast days
            df['label'] = df[target].shift(-forecast_out)
            # Setting the parameters of the features and target columns
            y = np.array(df['label'].iloc[:-forecast_out])
            X = np.array(df.iloc[:-forecast_out].drop(columns=['label'])) ### solo usando hasta -forecast
            X_forecast = np.array(df.iloc[-forecast_out:].drop(columns=['label']))

            # Create a scaler object
            scaler = StandardScaler()
            # Fit the inputs (calculate the mean and standard deviation feature-wise)
            scaler.fit(X)

            X = scaler.transform(X)
            X_forecast = scaler.transform(X_forecast)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
            reg_RFR = RandomForestRegressor(n_jobs=10, random_state=42)
            reg_RFR.fit(X_train, y_train)

            self.score_RFR = reg_RFR.score(X_test, y_test)
            df['pred_RFR'] = np.nan
            df['pred_RFR'].iloc[forecast_out:] = reg_RFR.predict(X)

            ## Real forecast of next bussines days
            n_time = pd.bdate_range(start=df.iloc[-1].name, periods=forecast_out, freq='B')
            df = df.append(pd.DataFrame(index=n_time[1:]))
            df['pred_RFR'].iloc[-forecast_out:] = reg_RFR.predict(X_forecast)
            df['pred_RFR_ma15'] = df['pred_RFR'].rolling(window=7, min_periods=4, center=True).mean()

            # Find possible income or loss
            predictor = 'pred_RFR_ma15'
            df_fore = df.iloc[-forecast_out:]
            df_fore['time'] = np.arange(0, len(df_fore), 1)
            fore_dmatrix = np.atleast_2d(df_fore[predictor].to_numpy()) - np.atleast_2d(df_fore[predictor].to_numpy()).T
            fore_dmatrix = np.triu(fore_dmatrix)

            max_p = [np.where(fore_dmatrix == fore_dmatrix.max())[0][0], np.where(fore_dmatrix == fore_dmatrix.max())[1][0]]
            min_p = [np.where(fore_dmatrix == fore_dmatrix.min())[0][0], np.where(fore_dmatrix == fore_dmatrix.min())[1][0]]
            df_max_p = df_fore.iloc[max_p]
            df_min_p = df_fore.iloc[min_p]

            df_minmax = pd.concat([df_max_p, df_min_p])
            with open("C:/Users/braya/Downloads/Stocks/df_minmax_" + el1 + ".pickle", 'wb') as f:
                pickle.dump(df_minmax, f)


            self.loss_rate = abs(fore_dmatrix[0].min()/df['Adj Close'].iloc[-forecast_out])
            self.gain_rate = abs(fore_dmatrix[0].max()/df['Adj Close'].iloc[-forecast_out])
            self.total_rate = self.gain_rate - self.loss_rate

            self.buy_gain = df_max_p[predictor].iloc[1] - df_max_p[predictor].iloc[0]
            self.buy_gain_pct = self.buy_gain / df_max_p[predictor].iloc[1]

            self.start = df_max_p.iloc[0].name
            self.end = df_max_p.iloc[1].name

            self.entry = df_max_p[predictor].iloc[0]
            self.sell = df_max_p[predictor].iloc[1]

            if len(df_max_p) == 2:
                self.buy_gain_time = df_max_p['time'][1] - df_max_p['time'][0]
                self.buy_gain_wait_time = df_max_p['time'][1]
            else:
                self.buy_gain_time = 0
                self.buy_gain_wait_time = 0
            with open("C:/Users/braya/Downloads/Stocks/df_" + el1 + ".pickle", 'wb') as f:
                pickle.dump(df, f)
            # Plot
            pl_bool = 0
            if pl_bool == 1:
                # data_val_limit = [df.iloc[-1 - forecast_out * 1].name, df.iloc[-1].name]
                data_val_limit = [df.iloc[-forecast_out*2].name, df.iloc[-1].name]
                fig, ax = plt.subplots(figsize=[12, 5])
                ax = sns.lineplot(data=df['Adj Close'], alpha=0.8)
                ax = sns.lineplot(data=df['pred_RFR'], alpha=0.8)
                ax = sns.lineplot(data=df['pred_RFR_ma15'], alpha=0.8)
                ax = plt.axvline(x=df.iloc[-forecast_out].name, color='purple', alpha=0.8)

                ax = sns.lineplot(data=df_max_p[predictor])
                ax = sns.lineplot(data=df_min_p[predictor])

                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.xlim([data_val_limit[0], df.index[-1]])
                plt.ylim([df['Adj Close'].iloc[-forecast_out*2:].min() * 0.95, df['pred_RFR'].iloc[-forecast_out*2:].max() * 1.05])
                plt.show()

        self.ticker = ticker
        self.df = pd.read_csv("C:/Users/braya/Downloads/Stocks/" + str(self.ticker) + ".csv")
        self.df['Date'] = pd.to_datetime(self.df['Date'], dayfirst=True)
        self.df.set_index('Date', drop=True, inplace=True)
        self.df.dropna(inplace=True)
        pred(self.df)

# list_1 = ['AAPL', 'NVDA', 'TSLA', 'MSFT']
# list_1 = ['AAPL']
# list_1 = ['NVDA']

with open("C:/Users/braya/Downloads/Stocks/tickersTrue.pickle", 'rb') as f:
    list_1 = pickle.load(f)

# list_1 = list_1[:10]

df_resume = pd.DataFrame(columns=['score', 'buy_gain', 'buy_gain_pct', 'buy_gain_time', 'buy_gain_wait_time', 'start', 'end', 'entry_price', 'sell_price'], index=list_1)

count_i = 0
for el1 in list_1:
    uwu = forecast(el1)
    df_resume['buy_gain'].loc[el1] = uwu.buy_gain
    df_resume['buy_gain_pct'].loc[el1] = uwu.buy_gain_pct
    df_resume['buy_gain_time'].loc[el1] = uwu.buy_gain_time
    df_resume['buy_gain_wait_time'].loc[el1] = uwu.buy_gain_wait_time
    df_resume['score'].loc[el1] = uwu.score_RFR
    df_resume['start'].loc[el1] = uwu.start
    df_resume['end'].loc[el1] = uwu.end
    df_resume['entry_price'].loc[el1] = uwu.entry
    df_resume['sell_price'].loc[el1] = uwu.sell
    count_i += 1
    print(count_i)
df_resume.sort_values(by='buy_gain_pct', inplace=True, ascending=False)

filepath = Path('C:/Users/braya/Downloads/Stocks/main_df.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
df_resume.to_csv(filepath)
print(df_resume.head(10))