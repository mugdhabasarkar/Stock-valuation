import yfinance as yf
import pandas as pd
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fetches financial data for a given stock ticker
def fetch_financial_data(ticker: str) -> pd.Series:
    try:
        stock = yf.Ticker(ticker)
        cashflow_df = stock.cashflow
        key_options = ['Total Cash From Operating Activities', 'Operating Cash Flow']
        
        # Extract cash flow data if available
        for key in key_options:
            if key in cashflow_df.index:
                return cashflow_df.loc[key][::-1].astype(float)
        
        logger.warning(f"No cash flow data found for ticker {ticker}")
        return pd.Series(dtype=float)
    except Exception as e:
        logger.error(f"Error fetching financial data for {ticker}: {e}")
        return pd.Series(dtype=float)

# Calculates intrinsic value using Discounted Cash Flow (DCF) method
def dcf_valuation(
    cashflows: pd.Series, 
    discount_rate: float = 0.1, 
    growth_rate: float = 0.03, 
    terminal_growth: float = 0.02, 
    years: int = 10
) -> Optional[float]:
    try:
        if cashflows.empty:
            logger.warning("Empty cashflow series provided")
            return None
        
        if terminal_growth >= discount_rate:
            logger.error("Terminal growth rate must be less than discount rate")
            return None
        
        last_cashflow = cashflows.iloc[-1]
        
        # Project future cash flows and discount them to present value
        projected_cashflows = [
            last_cashflow * (1 + growth_rate) ** i / (1 + discount_rate) ** i 
            for i in range(1, years + 1)
        ]
        
        # Compute and discount the terminal value
        terminal_value = (projected_cashflows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        
        return sum(projected_cashflows) + terminal_value / (1 + discount_rate) ** years
    except Exception as e:
        logger.error(f"Error in DCF valuation: {e}")
        return None

# Computes expected return using CAPM model
def capm_model(
    beta: float, 
    risk_free_rate: float = 0.03, 
    market_return: float = 0.08
) -> float:
    try:
        return risk_free_rate + beta * (market_return - risk_free_rate)
    except Exception as e:
        logger.error(f"Error in CAPM model calculation: {e}")
        return risk_free_rate

# Incorporates Modigliani-Miller Theorem to analyze leverage impact
def modigliani_miller_adjustment(
    market_cap: float, 
    debt: float, 
    cost_of_debt: float, 
    tax_rate: float = 0.21
) -> float:
    try:
        # Added validation for potential negative inputs
        if market_cap < 0 or debt < 0 or cost_of_debt < 0:
            logger.warning("Negative values detected in Modigliani-Miller calculation")
            return market_cap
        
        return market_cap + debt * (1 - tax_rate) - (cost_of_debt * debt)
    except Exception as e:
        logger.error(f"Error in Modigliani-Miller adjustment: {e}")
        return market_cap

# Fetches stock market capitalization, beta value, current stock price, and total debt
def fetch_stock_info(ticker: str) -> Tuple[float, float, float, float, float]:
    try:
        stock = yf.Ticker(ticker)
        
        # More robust info fetching with error handling
        market_cap = stock.info.get('marketCap', 0)
        beta = stock.info.get('beta', 1)
        current_price = stock.info.get('currentPrice', 0)
        total_debt = stock.info.get('totalDebt', 0)
        cost_of_debt = stock.info.get('yield', 0.05)
        
        # Additional validation
        if market_cap == 0:
            logger.warning(f"Market cap not found for {ticker}")
        
        return market_cap, beta, current_price, total_debt, cost_of_debt
    except Exception as e:
        logger.error(f"Error fetching stock info for {ticker}: {e}")
        # Return default values if fetching fails
        return 0, 1, 0, 0, 0.05
