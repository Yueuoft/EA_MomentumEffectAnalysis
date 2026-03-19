import yfinance as yf
import pandas as pd
from typing import List


def download_prices_batch(
    tickers: List[str],
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    batch_size: int = 50
) -> pd.DataFrame:
    all_prices = []
    total_batches = (len(tickers) - 1) // batch_size + 1

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Downloading batch {i // batch_size + 1} / {total_batches} ...")

        try:
            data = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="column"
            )
        except Exception as e:
            print(f"Batch failed: {e}")
            continue

        if data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                batch_prices = data["Close"]
            else:
                batch_prices = data
        else:
            if "Close" in data.columns:
                batch_prices = data[["Close"]].copy()
                if len(batch) == 1:
                    batch_prices.columns = batch
            else:
                batch_prices = data.copy()

        all_prices.append(batch_prices)

    if not all_prices:
        raise ValueError("No price data downloaded.")

    prices = pd.concat(all_prices, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.dropna(axis=1, how="all").sort_index()

    return prices
