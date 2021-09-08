from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import yfinance
import pandas as pd


# Simple Moving Average
def SMA(data, ndays):
    SMA = pd.Series(data['adjClose'].rolling(ndays).mean(), name='SMA')
    return SMA


# Exponentially-weighted Moving Average
def EWMA(data, ndays):
    EMA = pd.Series(data['adjClose'].ewm(span=ndays,
                                      min_periods=ndays - 1).mean(),
                    name='EWMA_' + str(ndays))
    return EMA


# Compute the Bollinger Bands
def BBANDS(data, window=50):
    MA = data.adjClose.rolling(window).mean()
    SD = data.adjClose.rolling(window).std()
    data['UpperBB'] = MA + (2 * SD)
    data['LowerBB'] = MA - (2 * SD)
    return data


# Compute the relative strenght index (RSI)
def rsi(df, periods=14, ema=True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df['adjClose'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema == True:
        # Use exponential moving average
        ma_up = up.ewm(com=periods - 1, adjust=True,
                       min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True,
                           min_periods=periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window=periods, adjust=False).mean()
        ma_down = down.rolling(window=periods, adjust=False).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi


# volatility is the 14 days standard deviation
def volatility(df, n):
    volatil = df.adjClose.rolling(n).std(ddof=0)
    return volatil

# Moving average convergence divergence (MACD)
def macd(df):
    exp1 = df.adjClose.ewm(span=12, adjust=False).mean()
    exp2 = df.adjClose.ewm(span=26, adjust=False).mean()
    exp3 = df.adjClose.ewm(span=9, adjust=False).mean()
    macd = exp1 - exp2
    return macd

def all_tech_analysis(data):
    EMA200D = EWMA(data, 200)
    EMA20D = EWMA(data, 20)
    BBANDS_var = BBANDS(data,50)
    rsi_var = rsi(data)
    volatility_var = volatility(data,14)
    macd_var = macd(data)
    data['EMA200D'] = EMA200D
    data['EMA20D'] = EMA20D
    # data['LowerBB']
    # data['BBANDS'] = BBANDS_var
    data['rsi'] = rsi_var
    data['volatility'] = volatility_var
    data['macd'] = macd_var
    return data


# if __name__ == '___main__':
#     EMA200D = EWMA(data, 200)
#     EMA20D = EWMA(data, 20)
#     BBANDS = BBANDS(data,50)
#     rsi = rsi(data)
#     volatility = volatility(data,14)
#     macd = macd(data)
