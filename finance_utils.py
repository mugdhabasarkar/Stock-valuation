import yfinance as yf
import pandas as pd

# Fetches financial data for a given stock ticker
def fetch_financial_data(ticker: str) -> pd.Series:
    stock = yf.Ticker(ticker)
    cashflow_df = stock.cashflow
    key_options = ['Total Cash From Operating Activities', 'Operating Cash Flow']
    
    # Extract cash flow data if available
    for key in key_options:
        if key in cashflow_df.index:
            return cashflow_df.loc[key][::-1].astype(float)
    return pd.Series(dtype=float)

# Calculates intrinsic value using Discounted Cash Flow (DCF) method
def dcf_valuation(cashflows: pd.Series, discount_rate: float = 0.1, growth_rate: float = 0.03, terminal_growth: float = 0.02, years: int = 10) -> float:
    if cashflows.empty:
        return 0
    
    last_cashflow = cashflows.iloc[-1]
    
    # Project future cash flows and discount them to present value
    projected_cashflows = [last_cashflow * (1 + growth_rate) ** i / (1 + discount_rate) ** i for i in range(1, years + 1)]
    
    # Compute and discount the terminal value
    terminal_value = (projected_cashflows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    
    return sum(projected_cashflows) + terminal_value / (1 + discount_rate) ** years

# Computes expected return using CAPM model
def capm_model(beta: float, risk_free_rate: float = 0.03, market_return: float = 0.08) -> float:
    return risk_free_rate + beta * (market_return - risk_free_rate)

# Incorporates Modigliani-Miller Theorem to analyze leverage impact
def modigliani_miller_adjustment(market_cap: float, debt: float, cost_of_debt: float, tax_rate: float = 0.21) -> float:
    return market_cap + debt * (1 - tax_rate) - (cost_of_debt * debt)

# Fetches stock market capitalization, beta value, current stock price, and total debt
def fetch_stock_info(ticker: str):
    stock = yf.Ticker(ticker)
    market_cap = stock.info.get('marketCap', 0)
    beta = stock.info.get('beta', 1)
    current_price = stock.info.get('currentPrice', 0)
    total_debt = stock.info.get('totalDebt', 0)
    cost_of_debt = stock.info.get('yield', 0.05)
    return market_cap, beta, current_price, total_debt, cost_of_debt
