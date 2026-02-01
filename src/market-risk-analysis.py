"""
Market Risk Analysis

Outputs:
- data/prices_wide.csv
- data/returns_wide.csv
- data/metrics_summary.csv
- outputs/equity_curves.png
- outputs/drawdowns.png
- outputs/rolling_correlation_vs_sp500.png
- outputs/<TICKER>_rolling_vol.png
- outputs/<TICKER>_rolling_sharpe.png
- outputs/<TICKER>_return_hist.png
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

# Force non-GUI backend (prevents Tkinter issues on Windows)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yfinance as yf


# ==========================================================
# CONFIGURATION
# ==========================================================

TICKERS = ["^GSPC", "AAPL", "TSLA"]
START_DATE = "2019-01-01"

ROLLING_WINDOW = 20
TRADING_DAYS = 252

RISK_FREE_RATE_ANNUAL = 0.03
VAR_LEVEL = 0.95


# ==========================================================
# LOGGING
# ==========================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ==========================================================
# FILE / OUTPUT HYGIENE
# ==========================================================

def ensure_dirs() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


def clean_old_outputs() -> None:
    """
    Remove legacy plots that are no longer generated.
    Prevents confusion from stale files.
    """
    legacy_files = [
        "outputs/correlation_heatmap.png",
    ]

    for f in legacy_files:
        if os.path.exists(f):
            os.remove(f)
            logging.info(f"Removed legacy file: {f}")


def safe_name(ticker: str) -> str:
    return (
        ticker.replace("^", "")
              .replace("/", "_")
              .replace("\\", "_")
              .replace(":", "_")
              .replace("*", "_")
              .replace("?", "_")
              .replace('"', "_")
              .replace("<", "_")
              .replace(">", "_")
              .replace("|", "_")
              .strip()
    )


def to_series(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def annual_to_daily_rate(r_annual: float) -> float:
    return (1 + r_annual) ** (1 / TRADING_DAYS) - 1


# ==========================================================
# DATA DOWNLOAD
# ==========================================================

def download_prices(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df = df.reset_index()
    close = to_series(df["Close"])

    out = pd.DataFrame({
        "date": df["Date"],
        "close": pd.to_numeric(close, errors="coerce")
    })

    return out.dropna()


# ==========================================================
# METRICS
# ==========================================================

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df["daily_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    return df


def drawdown_series(prices: pd.Series) -> pd.Series:
    prices = to_series(prices).dropna()
    running_max = prices.cummax()
    return (prices / running_max) - 1.0


def max_drawdown(prices: pd.Series) -> float:
    dd = drawdown_series(prices)
    return float(dd.min()) if not dd.empty else np.nan


def cagr(prices: pd.Series) -> float:
    prices = to_series(prices).dropna()
    years = len(prices) / TRADING_DAYS
    return float((prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1)


def annualised_sharpe(r: pd.Series, rf_annual: float) -> float:
    r = to_series(r).dropna()
    rf_daily = annual_to_daily_rate(rf_annual)
    excess = r - rf_daily
    return float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS))


def annualised_sortino(r: pd.Series, rf_annual: float) -> float:
    r = to_series(r).dropna()
    rf_daily = annual_to_daily_rate(rf_annual)
    excess = r - rf_daily
    downside = excess[excess < 0]
    return float((excess.mean() / downside.std()) * np.sqrt(TRADING_DAYS))


def historical_var_cvar(r: pd.Series, level: float) -> Dict[str, float]:
    r = to_series(r).dropna()
    alpha = 1 - level
    var = np.quantile(r, alpha)
    cvar = r[r <= var].mean()
    return {"var": float(var), "cvar": float(cvar)}


def rolling_vol_annualised(r: pd.Series, window: int) -> pd.Series:
    return r.rolling(window).std() * np.sqrt(TRADING_DAYS)


def rolling_sharpe(r: pd.Series, window: int, rf_annual: float) -> pd.Series:
    rf_daily = annual_to_daily_rate(rf_annual)

    def _s(x):
        ex = x - rf_daily
        return (ex.mean() / ex.std()) * np.sqrt(TRADING_DAYS)

    return r.rolling(window).apply(_s, raw=False)


# ==========================================================
# PLOTS
# ==========================================================

def save_histogram(r: pd.Series, ticker: str) -> None:
    plt.figure()
    plt.hist(r.dropna(), bins=60)
    plt.title(f"{ticker} Daily Return Distribution")
    plt.tight_layout()
    plt.savefig(f"outputs/{safe_name(ticker)}_return_hist.png", dpi=200)
    plt.close()


def save_rolling_plots(df: pd.DataFrame, ticker: str) -> None:
    t = safe_name(ticker)

    plt.figure()
    plt.plot(df.index, df["rolling_vol_annualised"])
    plt.title(f"{ticker} Rolling Volatility")
    plt.tight_layout()
    plt.savefig(f"outputs/{t}_rolling_vol.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df.index, df["rolling_sharpe"])
    plt.title(f"{ticker} Rolling Sharpe")
    plt.tight_layout()
    plt.savefig(f"outputs/{t}_rolling_sharpe.png", dpi=200)
    plt.close()


def save_equity_curves(prices: pd.DataFrame) -> None:
    plt.figure()
    for col in prices:
        equity = prices[col] / prices[col].iloc[0]
        plt.plot(equity, label=col)
    plt.legend()
    plt.title("Equity Curves (Growth of $1)")
    plt.tight_layout()
    plt.savefig("outputs/equity_curves.png", dpi=200)
    plt.close()


def save_drawdowns(prices: pd.DataFrame) -> None:
    plt.figure()
    for col in prices:
        plt.plot(drawdown_series(prices[col]), label=col)
    plt.legend()
    plt.title("Drawdowns")
    plt.tight_layout()
    plt.savefig("outputs/drawdowns.png", dpi=200)
    plt.close()


def plot_rolling_correlation_vs_sp500(returns: pd.DataFrame) -> None:
    if "^GSPC" not in returns.columns:
        return

    sp = returns["^GSPC"]
    plt.figure()
    for col in returns.columns:
        if col != "^GSPC":
            plt.plot(returns[col].rolling(ROLLING_WINDOW).corr(sp), label=col)

    plt.axhline(0, linestyle="--")
    plt.legend()
    plt.title("Rolling Correlation vs S&P 500")
    plt.tight_layout()
    plt.savefig("outputs/rolling_correlation_vs_sp500.png", dpi=200)
    plt.close()


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main() -> int:
    setup_logging()
    ensure_dirs()
    clean_old_outputs()

    price_series = {}
    returns_series = []
    metrics = []

    for t in TICKERS:
        df = compute_returns(download_prices(t, START_DATE)).set_index("date")

        df["rolling_vol_annualised"] = rolling_vol_annualised(df["daily_return"], ROLLING_WINDOW)
        df["rolling_sharpe"] = rolling_sharpe(df["daily_return"], ROLLING_WINDOW, RISK_FREE_RATE_ANNUAL)

        save_histogram(df["daily_return"], t)
        save_rolling_plots(df, t)

        price_series[t] = df["close"]
        returns_series.append(df["daily_return"].rename(t))

        stats = historical_var_cvar(df["daily_return"], VAR_LEVEL)

        metrics.append({
            "ticker": t,
            "cagr": cagr(df["close"]),
            "volatility": df["daily_return"].std() * np.sqrt(TRADING_DAYS),
            "max_drawdown": max_drawdown(df["close"]),
            "sharpe": annualised_sharpe(df["daily_return"], RISK_FREE_RATE_ANNUAL),
            "sortino": annualised_sortino(df["daily_return"], RISK_FREE_RATE_ANNUAL),
            "var": stats["var"],
            "cvar": stats["cvar"],
        })

    prices_wide = pd.DataFrame(price_series)
    returns_wide = pd.concat(returns_series, axis=1)

    prices_wide.to_csv("data/prices_wide.csv")
    returns_wide.to_csv("data/returns_wide.csv")
    pd.DataFrame(metrics).to_csv("data/metrics_summary.csv", index=False)

    save_equity_curves(prices_wide)
    save_drawdowns(prices_wide)
    plot_rolling_correlation_vs_sp500(returns_wide)

    logging.info("✅ Analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())