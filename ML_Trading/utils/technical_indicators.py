import pandas as pd

def expo_MA(prices):
    ma = []
    alpha = 2/(len(prices) + 1)
    ma.append(prices[0])
    for i in range(1,len(prices)-1,1) :
        ma.append(ma[i-1] +alpha*(prices[i]-ma[i-1]))
    return ma 


def Momentum(values, n):
    """
    calculates n day diffrence
    """
    return pd.Series(values).diff(periods=n)


def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()


def BBANDS(data, n_lookback, n_std):
    """Bollinger bands indicator"""
    hlc3 = (data.High + data.Low + data.Close) / 3
    mean, std = hlc3.rolling(n_lookback).mean(), hlc3.rolling(n_lookback).std()
    upper = mean + n_std*std
    lower = mean - n_std*std
    return upper, lower

