import pandas as pd
import numpy as np


def form_momentum_portfolios(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    rolling_vol: pd.DataFrame | None = None,
    top_quantile: float = 0.1,
    min_stocks: int = 30,
    min_price: float = 5.0,
    max_vol: float | None = None
) -> pd.DataFrame:
    """
    Realistic cross-sectional momentum portfolio formation.

    Filters:
    - minimum current price
    - optional max volatility filter
    - minimum number of eligible stocks

    Portfolios:
    - equal-weight winners = top quantile by signal
    - equal-weight losers  = bottom quantile by signal
    - spread = winners - losers
    """
    results = []

    for date in signal.index:
        sig = signal.loc[date].dropna()
        fwd = forward_returns.loc[date].dropna()
        px = monthly_prices.loc[date].dropna()

        common = sig.index.intersection(fwd.index).intersection(px.index)

        if rolling_vol is not None:
            vol = rolling_vol.loc[date].dropna()
            common = common.intersection(vol.index)
            vol = vol.loc[common]
        else:
            vol = None

        sig = sig.loc[common]
        fwd = fwd.loc[common]
        px = px.loc[common]

        # Price filter
        eligible = px[px >= min_price].index
        sig = sig.loc[eligible]
        fwd = fwd.loc[eligible]
        px = px.loc[eligible]

        if vol is not None:
            vol = vol.loc[eligible]
            if max_vol is not None:
                eligible_vol = vol[vol <= max_vol].index
                sig = sig.loc[eligible_vol]
                fwd = fwd.loc[eligible_vol]
                px = px.loc[eligible_vol]
                vol = vol.loc[eligible_vol]

        if len(sig) < min_stocks:
            continue

        n_bucket = max(1, int(len(sig) * top_quantile))

        ranked = sig.sort_values()
        losers = ranked.index[:n_bucket]
        winners = ranked.index[-n_bucket:]

        winner_ret = fwd.loc[winners].mean()
        loser_ret = fwd.loc[losers].mean()
        spread_ret = winner_ret - loser_ret

        results.append({
            "date": date,
            "winner_return": winner_ret,
            "loser_return": loser_ret,
            "spread_return": spread_ret,
            "n_stocks": len(sig),
            "n_bucket": n_bucket,
            "winner_avg_signal": sig.loc[winners].mean(),
            "loser_avg_signal": sig.loc[losers].mean(),
            "winner_avg_price": px.loc[winners].mean(),
            "loser_avg_price": px.loc[losers].mean()
        })

    if not results:
        raise ValueError("No portfolio periods formed. Try relaxing filters.")

    return pd.DataFrame(results).set_index("date")
