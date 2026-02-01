# Market Risk Analysis Toolkit

A professional-grade Python project for analysing **market risk and performance** of financial assets using historical price data.

This toolkit computes key **risk, return, and drawdown metrics**, produces high-quality visualisations, and exports clean datasets suitable for further analysis in Excel, Power BI, or Tableau.

---

## Features

- Multi-asset analysis (stocks, indices)
- Daily & log returns
- Rolling volatility (annualised)
- Rolling Sharpe ratio
- Maximum drawdown
- CAGR (Compound Annual Growth Rate)
- Historical Value-at-Risk (VaR) and Conditional VaR (CVaR)
- Rolling correlation vs S&P 500
- Clean CSV outputs for further analysis
- Publication-ready charts

---

## Assets Analysed

- **S&P 500** (`^GSPC`)
- **Apple Inc.** (`AAPL`)
- **Tesla Inc.** (`TSLA`)

You can easily add more tickers inside the script.

---

## Outputs

### Data
- `data/prices_wide.csv` – aligned price series
- `data/returns_wide.csv` – aligned daily returns
- `data/metrics_summary.csv` – key risk & performance metrics

### Visualisations
- Equity curves (growth of $1)
- Drawdowns (peak-to-trough losses)
- Rolling volatility (20-day)
- Rolling Sharpe ratio
- Rolling correlation vs S&P 500
- Return distributions

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/market-risk-analysis.git
cd market-risk-analysis