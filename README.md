import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go

# Fetches financial data for a given stock ticker
def fetch_financial_data(ticker: str) -> pd.Series:
    stock = yf.Ticker(ticker)
    cashflow_df = stock.cashflow
    key_options = ['Total Cash From Operating Activities', 'Operating Cash Flow']
    
    for key in key_options:
        if key in cashflow_df.index:
            return cashflow_df.loc[key][::-1].astype(float)
    return pd.Series(dtype=float)

# Calculates intrinsic value using Discounted Cash Flow (DCF) method
def dcf_valuation(cashflows: pd.Series, discount_rate: float = 0.1, growth_rate: float = 0.03, terminal_growth: float = 0.02, years: int = 10) -> float:
    if cashflows.empty:
        return 0
    
    last_cashflow = cashflows.iloc[-1]
    projected_cashflows = [last_cashflow * (1 + growth_rate) ** i / (1 + discount_rate) ** i for i in range(1, years + 1)]
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

# Generates stock price visualization
def generate_stock_graph(stock_data: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Closing Price'))
    fig.update_layout(title=f'{ticker} Stock Price', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_dark')
    return fig

# Initialize Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Stock Valuation Dashboard", style={'textAlign': 'center', 'color': '#ffffff'}),
    html.Div([
        dcc.Input(id='ticker-input', type='text', placeholder='Enter Stock Ticker', debounce=True, style={'marginRight': '10px'}),
        html.Button('Submit', id='submit-button', n_clicks=0, style={'backgroundColor': '#4CAF50', 'color': 'white'})
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
    html.Div(id='valuation-output', style={'textAlign': 'center', 'whiteSpace': 'pre-line', 'color': '#ffffff'}),
    dcc.Graph(id='stock-graph')
], style={'backgroundColor': '#1e1e1e', 'padding': '20px'})

# Callback function to update dashboard based on user input
@app.callback(
    [Output('valuation-output', 'children'), Output('stock-graph', 'figure')],
    [Input('submit-button', 'n_clicks')],
    [dash.State('ticker-input', 'value')]
)
def update_dashboard(n_clicks: int, ticker: str):
    if not ticker:
        return "Enter a valid stock ticker.", go.Figure()
    
    stock_data = yf.Ticker(ticker).history(period='5y')
    
    if stock_data.empty:
        return "Invalid ticker or no data available.", go.Figure()
    
    cashflows = fetch_financial_data(ticker)
    intrinsic_value = dcf_valuation(cashflows)
    market_cap, beta, current_price, total_debt, cost_of_debt = fetch_stock_info(ticker)
    expected_return = capm_model(beta)
    adjusted_value = modigliani_miller_adjustment(market_cap, total_debt, cost_of_debt)
    
    advice = "Consider Investing!" if adjusted_value > market_cap else "Be Cautious!"
    
    valuation_text = f"""
    Stock Valuation Summary
    -----------------------------------
    📈 Intrinsic Value (DCF): ${intrinsic_value:,.2f}
    💰 Market Cap: ${market_cap:,.2f}
    📊 Expected Return (CAPM): {expected_return:.2%}
    🏦 Adjusted Market Value (Modigliani-Miller): ${adjusted_value:,.2f}
    ✅ Investment Advice: {advice}
    """
    
    return valuation_text, generate_stock_graph(stock_data, ticker)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
