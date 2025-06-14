import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from scipy.optimize import minimize
import os

plt.style.use('seaborn-v0_8-darkgrid')

API_KEY = os.getenv('API_KEY_FMP')
BASE_URL = 'https://financialmodelingprep.com/api/v3'
rf = 0.03

def safe_float(value):
    return float(value) if value not in [None, ''] else 0

def max_drawdown(series):
    peak = series.cummax()
    drawdown = (series - peak) / peak
    return drawdown.min()

def get_ratios(ticker):
    url = f'{BASE_URL}/ratios-ttm/{ticker}?apikey={API_KEY}'
    r = requests.get(url)
    if r.status_code == 200 and r.json():
        return r.json()[0]
    return None

def get_zscore(ticker):
    url = f'{BASE_URL}/ratios/{ticker}?limit=1&apikey={API_KEY}'
    r = requests.get(url)
    if r.status_code == 200 and r.json():
        return r.json()[0].get("altmanZScore", None)
    return None

def analizar_fundamentales(ticker):
    data = get_ratios(ticker)
    if not data:
        return None

    try:
        roic = safe_float(data.get("returnOnCapitalEmployedTTM"))
        ebit_margin = safe_float(data.get("operatingProfitMarginTTM"))
        zscore = safe_float(get_zscore(ticker))
        quick_ratio = safe_float(data.get("quickRatioTTM"))
        interest_coverage = safe_float(data.get("interestCoverageTTM"))
        debt_to_equity = safe_float(data.get("debtEquityRatioTTM"))
    except:
        return None

    criterios = {
        'ROIC > 8%': roic > 0.08,
        'EBIT Margin > 15%': ebit_margin > 0.15,
        'Altman Z > 3': zscore > 3,
        'Quick Ratio > 1': quick_ratio > 1,
        'Interest Coverage > 3': interest_coverage > 3,
        'Debt to Equity < 1.5': debt_to_equity < 1.5
    }

    score = sum(criterios.values())
    signal = 'BUY' if score >= 5 else 'HOLD' if score >= 3 else 'SELL'

    return {
        'ticker': ticker,
        'ROIC': roic,
        'EBIT Margin': ebit_margin,
        'Z-Score': zscore,
        'Quick Ratio': quick_ratio,
        'Interest Coverage': interest_coverage,
        'Debt/Equity': debt_to_equity,
        'Score': score,
        'Signal': signal,
        'Criterios': criterios
    }

def run_analysis(tickers_str: str, pesos_str: str):
    tickers = [t.strip().upper() for t in tickers_str.split(',')]
    pesos = np.array([float(p) for p in pesos_str.split(',')]) / 100

    benchmark = "^GSPC"
    data = yf.download(tickers + [benchmark], start="2020-01-01")["Close"].dropna()
    returns = data.pct_change().dropna()
    returns_annual = returns.mean() * 252
    vol_annual = returns.std() * np.sqrt(252)
    sharpe_ratios = (returns_annual - rf) / vol_annual
    drawdowns = {ticker: max_drawdown(data[ticker]) for ticker in tickers}

    benchmark_returns = returns[benchmark]
    betas = {
        ticker: np.cov(returns[ticker], benchmark_returns)[0][1] / np.var(benchmark_returns)
        for ticker in tickers
    }

    port_return = np.dot(returns_annual[tickers], pesos)
    port_vol = np.sqrt(np.dot(pesos.T, np.dot(returns[tickers].cov() * 252, pesos)))
    port_sharpe = (port_return - rf) / port_vol

    def negative_sharpe(weights, mean_returns, cov_matrix):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(ret - rf) / vol

    bounds = tuple((0, 1) for _ in range(len(tickers)))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    initial_weights = np.array([1 / len(tickers)] * len(tickers))
    opt_result = minimize(
        negative_sharpe,
        initial_weights,
        args=(returns_annual[tickers], returns[tickers].cov() * 252),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = opt_result.x

    fundamentales = [analizar_fundamentales(t) for t in tickers]
    resumen = pd.DataFrame({
        "Ticker": tickers,
        "Ret Anual %": returns_annual[tickers] * 100,
        "Vol Anual %": vol_annual[tickers] * 100,
        "Sharpe Ratio": sharpe_ratios[tickers],
        "Max Drawdown %": [drawdowns[t] * 100 for t in tickers],
        "Beta vs S&P500": [betas[t] for t in tickers],
        "Peso Óptimo": optimal_weights,
        "Señal": [f['Signal'] if f else 'N/A' for f in fundamentales]
    })

    return resumen, fundamentales
