!pip install yfinance pandas numpy matplotlib scipy lxml html5lib

import matplotlib.pyplot as plt
import pandas as pd

# from google.colab import files
# uploaded = files.upload()

# !ls

from data_loader import download_prices_batch
from features import (
    to_monthly_prices,
    compute_monthly_returns,
    compute_momentum_signal,
    compute_forward_returns,
    compute_rolling_volatility,
)
from portfolio import form_momentum_portfolios
from evaluation import summarize_results, cumulative_returns


def get_large_cap_tickers():
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK-B", "UNH",
        "XOM", "JNJ", "JPM", "V", "PG", "AVGO", "MA", "HD", "CVX", "LLY",
        "MRK", "ABBV", "PEP", "COST", "KO", "BAC", "ADBE", "WMT", "CRM", "NFLX",
        "MCD", "AMD", "TMO", "CSCO", "ACN", "ABT", "DHR", "LIN", "PFE", "CMCSA",
        "VZ", "DIS", "TXN", "INTC", "QCOM", "HON", "AMGN", "LOW", "UNP", "IBM",
        "CAT", "SPGI", "GE", "INTU", "GS", "RTX", "BKNG", "ISRG", "BLK", "AXP",
        "NOW", "DE", "MDT", "SYK", "PLD", "TJX", "ADP", "MMC", "AMT", "LMT",
        "MO", "GILD", "C", "SCHW", "CB", "CI", "TMUS", "SO", "DUK", "ZTS",
        "EOG", "BSX", "USB", "PNC", "CL", "TGT", "APD", "BDX", "FIS", "EQIX",
        "NSC", "ITW", "REGN", "SLB", "MU", "VRTX", "ELV", "CME", "AON", "SHW",
        "ICE", "ETN", "PYPL", "MPC", "KLAC", "EW", "GD", "EMR", "MAR", "ORLY",
        "FDX", "GM", "F", "ROP", "ADI", "HCA", "PSA", "MET", "SNPS", "AEP",
        "OXY", "MCK", "D", "TRV", "SRE", "KMB", "NOC", "AFL", "ALL", "WMB",
        "ROST", "AZO", "JCI", "GIS", "AIG", "KMI", "LHX", "CTAS", "MSI", "ADM",
        "PAYX", "IDXX", "TT", "PH", "CMI", "A", "DOW", "YUM", "STZ", "MS",
        "EXC", "PRU", "PCAR", "RSG", "CHTR", "ODFL", "MNST", "DVN", "HAL", "BIIB"
    ]


def run_strategy(
    monthly_prices: pd.DataFrame,
    lookback: int,
    skip: int,
    holding: int,
    top_quantile: float,
    min_price: float,
    max_vol: float | None
):
    monthly_returns = compute_monthly_returns(monthly_prices)
    signal = compute_momentum_signal(monthly_prices, lookback=lookback, skip=skip)
    forward_returns = compute_forward_returns(monthly_prices, holding=holding)
    rolling_vol = compute_rolling_volatility(monthly_returns, window=12)

    results = form_momentum_portfolios(
        signal=signal,
        forward_returns=forward_returns,
        monthly_prices=monthly_prices,
        rolling_vol=rolling_vol,
        top_quantile=top_quantile,
        min_stocks=30,
        min_price=min_price,
        max_vol=max_vol
    )

    summary = summarize_results(results)
    return results, summary


def main():
    tickers = get_large_cap_tickers()
    print(f"Number of tickers requested: {len(tickers)}")

    prices = download_prices_batch(
        tickers=tickers,
        start="2015-01-01",
        end="2025-01-01",
        batch_size=50
    )

    print(f"Downloaded usable price series for {prices.shape[1]} stocks.")

    # Optional: drop names with too little history
    min_obs = 60
    keep_cols = prices.columns[prices.notna().sum() >= min_obs]
    prices = prices[keep_cols]
    print(f"Stocks remaining after requiring >= {min_obs} daily observations: {prices.shape[1]}")

    monthly_prices = to_monthly_prices(prices)

    configs = [
        {"name": "J6_skip1_K1",  "lookback": 6,  "skip": 1, "holding": 1},
        {"name": "J9_skip1_K1",  "lookback": 9,  "skip": 1, "holding": 1},
        {"name": "J12_skip1_K1", "lookback": 12, "skip": 1, "holding": 1},
    ]

    top_quantile = 0.10
    min_price = 5.0
    max_vol = None   # try 0.20 or 0.15 later if you want a volatility filter

    all_results = {}
    all_summaries = {}

    for cfg in configs:
        print(f"\nRunning {cfg['name']} ...")

        results, summary = run_strategy(
            monthly_prices=monthly_prices,
            lookback=cfg["lookback"],
            skip=cfg["skip"],
            holding=cfg["holding"],
            top_quantile=top_quantile,
            min_price=min_price,
            max_vol=max_vol
        )

        all_results[cfg["name"]] = results
        all_summaries[cfg["name"]] = summary

    summary_df = pd.DataFrame(all_summaries).T
    print("\n=== Quant Momentum Summary ===")
    print(summary_df.round(4))

    best_name = summary_df.sort_values(
        by=["spread_sharpe", "spread_mean_monthly"],
        ascending=False
    ).index[0]

    best_results = all_results[best_name]

    print(f"\nBest strategy selected: {best_name}")
    print("\n=== Best Strategy Summary ===")
    print(summary_df.loc[best_name].round(4))

    winner_curve = cumulative_returns(best_results["winner_return"])
    loser_curve = cumulative_returns(best_results["loser_return"])
    spread_curve = cumulative_returns(best_results["spread_return"])

    plt.figure(figsize=(10, 6))
    plt.plot(winner_curve, label="Winners")
    plt.plot(loser_curve, label="Losers")
    plt.plot(spread_curve, label="Winner-Loser Spread")
    plt.title(f"Quant Momentum Strategy Cumulative Returns ({best_name})")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
