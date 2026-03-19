import pandas as pd
import numpy as np


def to_monthly_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    return daily_prices.resample("ME").last()


def compute_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    return monthly_prices.pct_change()


def compute_momentum_signal(
    monthly_prices: pd.DataFrame,
    lookback: int = 6,
    skip: int = 1
) -> pd.DataFrame:
    """
    Momentum signal at time t:
    return from t-lookback-skip+1 ... t-skip

    Implementation:
    1) shift prices by skip months
    2) compute lookback return on shifted prices
    """
    shifted_prices = monthly_prices.shift(skip)
    signal = shifted_prices / shifted_prices.shift(lookback) - 1
    return signal


def compute_forward_returns(
    monthly_prices: pd.DataFrame,
    holding: int = 1
) -> pd.DataFrame:
    """
    Forward return from t to t+holding.
    For holding=1, this is next month's return.
    """
    forward_returns = monthly_prices.shift(-holding) / monthly_prices - 1
    return forward_returns


def compute_rolling_volatility(
    monthly_returns: pd.DataFrame,
    window: int = 12
) -> pd.DataFrame:
    """
    Rolling monthly volatility.
    """
    return monthly_returns.rolling(window).std()


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)
