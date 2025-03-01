import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import traceback
from dash.exceptions import PreventUpdate
import time
from datetime import datetime, timedelta

# Enhanced error handling wrapper
def safe_execution(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            print(traceback.format_exc())
            return None
    return wrapper

# Improved financial data fetching with caching
@safe_execution
def fetch_financial_data(ticker: str, cache={}):
    # Simple caching mechanism to avoid redundant API calls
    cache_key = f"{ticker}_financial_data"
    cache_timestamp_key = f"{ticker}_timestamp"
    
    # Return cached data if available and recent (less than 60 minutes old)
    current_time = time.time()
    if cache_key in cache and cache_timestamp_key in cache:
        if current_time - cache[cache_timestamp_key] < 3600:  # 1 hour cache
            return cache[cache_key]
    
    # Fetch new data
    stock = yf.Ticker(ticker)
    cashflow_df = stock.cashflow
    balance_sheet = stock.balance_sheet
    income_stmt = stock.income_stmt
    
    # More robust cash flow retrieval
    key_options = ['Total Cash From Operating Activities', 'Operating Cash Flow', 'Free Cash Flow']
    cashflow_series = None
    
    for key in key_options:
        if key in cashflow_df.index:
            cashflow_series = cashflow_df.loc[key][::-1].astype(float)
            break
    
    # If no cash flow data is found, estimate from net income and depreciation
    if cashflow_series is None or cashflow_series.empty:
        if 'Net Income' in income_stmt.index and 'Depreciation' in income_stmt.index:
            net_income = income_stmt.loc['Net Income'][::-1].astype(float)
            depreciation = income_stmt.loc['Depreciation'][::-1].astype(float)
            cashflow_series = net_income + depreciation
    
    # If still no data, return empty series
    if cashflow_series is None:
        cashflow_series = pd.Series(dtype=float)
    
    # Store in cache
    cache[cache_key] = cashflow_series
    cache[cache_timestamp_key] = current_time
    
    return cashflow_series

# Improved DCF valuation with sensitivity analysis
@safe_execution
def dcf_valuation(cashflows: pd.Series, discount_rate: float = 0.1, growth_rate: float = 0.03, 
                  terminal_growth: float = 0.02, years: int = 10, outstanding_shares: int = None):
    if cashflows.empty or len(cashflows) < 2:
        return 0, {}
    
    # Calculate historical growth rate if we have enough data points
    if len(cashflows) >= 3:
        growth_rates = []
        for i in range(1, len(cashflows)):
            if cashflows.iloc[i-1] > 0 and cashflows.iloc[i] > 0:  # Avoid division by zero or negative values
                annual_growth = (cashflows.iloc[i] / cashflows.iloc[i-1]) - 1
                growth_rates.append(annual_growth)
        
        if growth_rates:
            # Use average of historical growth rates, capped to be reasonable
            historical_growth = min(max(np.mean(growth_rates), 0.01), 0.15)
            growth_rate = min(historical_growth, 0.15)  # Cap at 15% to be conservative
    
    # Get the most recent positive cash flow
    last_cashflow = None
    for cf in reversed(cashflows):
        if cf > 0:
            last_cashflow = cf
            break
    
    if last_cashflow is None:
        return 0, {}
    
    # Project future cash flows
    projected_cashflows = [last_cashflow * (1 + growth_rate) ** i for i in range(1, years + 1)]
    discounted_cashflows = [cf / (1 + discount_rate) ** i for i, cf in enumerate(projected_cashflows, 1)]
    
    # Calculate terminal value
    terminal_value = (projected_cashflows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    discounted_terminal_value = terminal_value / (1 + discount_rate) ** years
    
    # Total intrinsic value
    total_value = sum(discounted_cashflows) + discounted_terminal_value
    
    # Get per share value if we have outstanding shares
    per_share_value = total_value / outstanding_shares if outstanding_shares else None
    
    # Return sensitivity analysis as well
    sensitivity = {}
    for dr in [discount_rate - 0.02, discount_rate, discount_rate + 0.02]:
        for gr in [growth_rate - 0.01, growth_rate, growth_rate + 0.01]:
            disc_cfs = [last_cashflow * (1 + gr) ** i / (1 + dr) ** i for i in range(1, years + 1)]
            term_val = (disc_cfs[-1] * (1 + terminal_growth)) / (dr - terminal_growth) / (1 + dr) ** years
            sensitivity[f"DR:{dr:.1%},GR:{gr:.1%}"] = sum(disc_cfs) + term_val
    
    return total_value, sensitivity

# Enhanced CAPM model with multiple market indices
@safe_execution
def capm_model(ticker: str, beta: float = None, risk_free_rate: float = None, 
               market_premium: float = None, lookback_years: int = 5):
    # If parameters are provided, use them; otherwise, calculate from market data
    if beta is not None and risk_free_rate is not None and market_premium is not None:
        return risk_free_rate + beta * market_premium
    
    # Get stock data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)
    stock_data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    
    # Get market index data (S&P 500)
    market_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
    
    # Get risk-free rate (10-year Treasury yield)
    try:
        treasury = yf.download('^TNX', start=end_date - timedelta(days=30), end=end_date)['Adj Close']
        current_risk_free = treasury.iloc[-1] / 100  # Convert from percentage to decimal
    except:
        current_risk_free = 0.03  # Default if unable to fetch
    
    # Align dates
    aligned_data = pd.DataFrame({
        'stock': stock_data,
        'market': market_data
    }).dropna()
    
    # Calculate returns
    returns = aligned_data.pct_change().dropna()
    
    # Calculate beta using regression
    X = sm.add_constant(returns['market'])
    model = sm.OLS(returns['stock'], X).fit()
    calculated_beta = model.params[1]
    
    # Calculate market premium (historical average)
    historical_market_return = returns['market'].mean() * 252  # Annualized
    market_premium = historical_market_return - current_risk_free
    
    # Calculate expected return
    expected_return = current_risk_free + calculated_beta * market_premium
    
    return expected_return, calculated_beta, current_risk_free, market_premium

# Improved Modigliani-Miller model
@safe_execution
def modigliani_miller_analysis(market_cap: float, debt: float, cost_of_debt: float, 
                              cost_of_equity: float, tax_rate: float = 0.21):
    # Calculate enterprise value
    enterprise_value = market_cap + debt
    
    # Calculate WACC (Weighted Average Cost of Capital)
    if enterprise_value > 0:
        equity_weight = market_cap / enterprise_value
        debt_weight = debt / enterprise_value
    else:
        equity_weight = 1
        debt_weight = 0
    
    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)
    
    # Calculate values with different capital structures
    mm_values = {}
    
    # Current structure value
    mm_values['current'] = enterprise_value
    
    # Value with no debt (unlevered)
    mm_values['unlevered'] = enterprise_value
    
    # Value with optimal debt (assuming tax shield benefits)
    # Simple assumption: optimal debt is around 30% of enterprise value for many industries
    optimal_debt = enterprise_value * 0.3
    optimal_equity = enterprise_value - optimal_debt
    
    # Assuming cost of equity increases with leverage according to MM Proposition II
    if market_cap > 0 and debt > 0:
        current_debt_equity_ratio = debt / market_cap
        optimal_debt_equity_ratio = optimal_debt / optimal_equity
        
        # Adjust cost of equity for optimal structure using MM Proposition II
        unlevered_cost = cost_of_equity - (debt / market_cap) * (cost_of_equity - cost_of_debt) * (1 - tax_rate)
        optimal_cost_of_equity = unlevered_cost + (optimal_debt / optimal_equity) * (unlevered_cost - cost_of_debt) * (1 - tax_rate)
        
        # Calculate optimal WACC
        optimal_wacc = (optimal_equity / enterprise_value) * optimal_cost_of_equity + \
                      (optimal_debt / enterprise_value) * cost_of_debt * (1 - tax_rate)
        
        # Value with tax shield benefits
        mm_values['optimal'] = enterprise_value * (wacc / optimal_wacc)
    else:
        mm_values['optimal'] = enterprise_value
    
    return mm_values, wacc

# Enhanced stock info fetching
@safe_execution
def fetch_stock_info(ticker: str):
    stock = yf.Ticker(ticker)
    
    # Get basic info with fallbacks
    try:
        market_cap = stock.info.get('marketCap', 0)
        if market_cap == 0:
            # Try calculating from share price and outstanding shares
            current_price = stock.info.get('currentPrice', stock.info.get('previousClose', 0))
            shares_outstanding = stock.info.get('sharesOutstanding', 0)
            market_cap = current_price * shares_outstanding
    except:
        market_cap = 0
    
    try:
        beta = stock.info.get('beta', 1.0)
        # Handle nonsensical beta values
        if beta is None or beta <= 0 or beta > 5:
            beta = 1.0
    except:
        beta = 1.0
    
    try:
        current_price = stock.info.get('currentPrice', 0)
        if current_price == 0:
            current_price = stock.info.get('previousClose', 0)
    except:
        current_price = 0
    
    # Try to get debt from multiple sources
    try:
        balance_sheet = stock.balance_sheet
        total_debt = 0
        
        debt_items = ['Total Debt', 'Long Term Debt', 'Short Long Term Debt']
        for item in debt_items:
            if item in balance_sheet.index:
                total_debt += balance_sheet.loc[item].iloc[0]
        
        if total_debt == 0:
            # Alternative calculation
            if 'Total Liabilities' in balance_sheet.index:
                total_debt = balance_sheet.loc['Total Liabilities'].iloc[0]
    except:
        total_debt = 0
    
    # Try to get cost of debt
    try:
        # Look for interest expense and debt to calculate effective interest rate
        income_stmt = stock.income_stmt
        interest_expense = 0
        
        interest_items = ['Interest Expense', 'Interest Income', 'Interest Expense Non Operating']
        for item in interest_items:
            if item in income_stmt.index:
                interest_expense += abs(income_stmt.loc[item].iloc[0])
        
        if interest_expense > 0 and total_debt > 0:
            cost_of_debt = interest_expense / total_debt
            # Sanity check - cap between 1% and 15%
            cost_of_debt = min(max(cost_of_debt, 0.01), 0.15)
        else:
            # Default based on current environment
            cost_of_debt = 0.05
    except:
        cost_of_debt = 0.05
    
    # Get shares outstanding
    try:
        shares_outstanding = stock.info.get('sharesOutstanding', 0)
    except:
        shares_outstanding = 0
    
    return {
        'market_cap': market_cap,
        'beta': beta,
        'current_price': current_price,
        'total_debt': total_debt,
        'cost_of_debt': cost_of_debt,
        'shares_outstanding': shares_outstanding,
        'name': stock.info.get('shortName', ticker),
        'industry': stock.info.get('industry', 'Unknown'),
        'sector': stock.info.get('sector', 'Unknown')
    }

# Generate enhanced stock price visualization with technical indicators
@safe_execution
def generate_stock_visualization(ticker: str, period: str = '5y'):
    stock_data = yf.Ticker(ticker).history(period=period)
    
    if stock_data.empty:
        return go.Figure()
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=(f'{ticker} Stock Price', 'Volume'),
                        row_heights=[0.7, 0.3])
    
    # Add price trace
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Calculate and add moving averages
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
    
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data['MA50'],
            line=dict(color='orange', width=1),
            name='50-day MA'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data['MA200'],
            line=dict(color='red', width=1),
            name='200-day MA'
        ),
        row=1, col=1
    )
    
    # Add volume bar chart
    fig.add_trace(
        go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            name='Volume',
            marker=dict(color='rgba(58, 71, 80, 0.6)')
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Analysis',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

# Create Dashboard
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Advanced Stock Valuation Dashboard", style={'textAlign': 'center', 'color': '#ffffff', 'marginBottom': '20px'}),
    
    # User Input Section
    html.Div([
        html.Div([
            html.Label("Stock Ticker:", style={'color': '#ffffff', 'marginRight': '10px'}),
            dcc.Input(id='ticker-input', type='text', placeholder='e.g., AAPL', 
                    style={'marginRight': '10px', 'padding': '8px'}),
        ], style={'marginRight': '20px'}),
        
        html.Div([
            html.Label("Analysis Options:", style={'color': '#ffffff', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='analysis-period',
                options=[
                    {'label': '1 Year', 'value': '1y'},
                    {'label': '3 Years', 'value': '3y'},
                    {'label': '5 Years', 'value': '5y'},
                    {'label': '10 Years', 'value': '10y'},
                ],
                value='5y',
                style={'width': '150px', 'background-color': '#333333', 'color': 'black'}
            ),
        ], style={'marginRight': '20px'}),
        
        html.Button(
            'Analyze Stock', 
            id='submit-button', 
            n_clicks=0, 
            style={
                'backgroundColor': '#4CAF50', 
                'color': 'white',
                'border': 'none',
                'padding': '10px 20px',
                'cursor': 'pointer',
                'borderRadius': '4px'
            }
        )
    ], style={
        'display': 'flex', 
        'justifyContent': 'center', 
        'alignItems': 'center',
        'marginBottom': '30px',
        'padding': '15px',
        'backgroundColor': '#333333',
        'borderRadius': '8px'
    }),
    
    # Loading state
    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            # Status and error messages
            html.Div(id='status-message', style={
                'textAlign': 'center', 
                'color': '#ff9800', 
                'margin': '10px 0',
                'fontStyle': 'italic'
            }),
            
            # Stock Info Section
            html.Div(id='stock-info-container', style={'display': 'none'}, children=[
                html.Div([
                    html.Div(id='stock-header', style={
                        'display': 'flex',
                        'justifyContent': 'space-between',
                        'alignItems': 'center',
                        'marginBottom': '20px',
                        'backgroundColor': '#444444',
                        'padding': '15px',
                        'borderRadius': '8px'
                    }),
                    
                    # Main content container with two columns
                    html.Div([
                        # Left column for valuation metrics
                        html.Div([
                            # DCF Valuation Card
                            html.Div([
                                html.H3("DCF Valuation", style={'color': '#4CAF50', 'borderBottom': '1px solid #555555', 'paddingBottom': '10px'}),
                                html.Div(id='dcf-valuation-output', style={'padding': '10px'})
                            ], style={
                                'backgroundColor': '#333333',
                                'borderRadius': '8px',
                                'padding': '15px',
                                'marginBottom': '20px'
                            }),
                            
                            # CAPM Analysis Card
                            html.Div([
                                html.H3("CAPM Analysis", style={'color': '#2196F3', 'borderBottom': '1px solid #555555', 'paddingBottom': '10px'}),
                                html.Div(id='capm-output', style={'padding': '10px'})
                            ], style={
                                'backgroundColor': '#333333',
                                'borderRadius': '8px',
                                'padding': '15px',
                                'marginBottom': '20px'
                            }),
                            
                            # Modigliani-Miller Card
                            html.Div([
                                html.H3("Capital Structure Analysis", style={'color': '#9C27B0', 'borderBottom': '1px solid #555555', 'paddingBottom': '10px'}),
                                html.Div(id='mm-output', style={'padding': '10px'})
                            ], style={
                                'backgroundColor': '#333333',
                                'borderRadius': '8px',
                                'padding': '15px',
                                'marginBottom': '20px'
                            }),
                            
                            # Investment Recommendation Card
                            html.Div([
                                html.H3("Investment Recommendation", style={'color': '#FF9800', 'borderBottom': '1px solid #555555', 'paddingBottom': '10px'}),
                                html.Div(id='recommendation-output', style={'padding': '10px'})
                            ], style={
                                'backgroundColor': '#333333',
                                'borderRadius': '8px',
                                'padding': '15px'
                            })
                        ], style={'width': '40%', 'paddingRight': '15px'}),
                        
                        # Right column for charts
                        html.Div([
                            dcc.Graph(id='stock-graph', style={'height': '100%'})
                        ], style={'width': '60%'})
                    ], style={'display': 'flex', 'marginBottom': '20px'})
                ])
            ])
        ]
    )
], style={
    'backgroundColor': '#1e1e1e', 
    'padding': '20px',
    'fontFamily': 'Arial, sans-serif',
    'minHeight': '100vh'
})

@app.callback(
    [
        Output('status-message', 'children'),
        Output('stock-info-container', 'style'),
        Output('stock-header', 'children'),
        Output('dcf-valuation-output', 'children'),
        Output('capm-output', 'children'),
        Output('mm-output', 'children'),
        Output('recommendation-output', 'children'),
        Output('stock-graph', 'figure')
    ],
    [Input('submit-button', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('analysis-period', 'value')]
)
def update_dashboard(n_clicks, ticker, analysis_period):
    if n_clicks == 0 or not ticker:
        return "Enter a stock ticker and click 'Analyze Stock' to begin analysis.", {'display': 'none'}, [], [], [], [], [], go.Figure()
    
    ticker = ticker.upper().strip()
    
    try:
        # Fetch stock information
        stock_info = fetch_stock_info(ticker)
        
        if stock_info['current_price'] == 0:
            return f"Could not retrieve data for {ticker}. Please check the ticker symbol.", {'display': 'none'}, [], [], [], [], [], go.Figure()
        
        # Create header with stock name and basic info
        stock_header = [
            html.Div([
                html.H2(f"{stock_info['name']} ({ticker})", style={'color': '#ffffff', 'margin': '0'}),
                html.Span(f"{stock_info['sector']} | {stock_info['industry']}", style={'color': '#aaaaaa'})
            ]),
            html.Div([
                html.H2(f"${stock_info['current_price']:.2f}", style={'color': '#4CAF50', 'margin': '0'})
            ])
        ]
        
        # Fetch financial data
        cashflows = fetch_financial_data(ticker)
        
        # Calculate DCF valuation
        dcf_value, sensitivity = dcf_valuation(
            cashflows=cashflows, 
            outstanding_shares=stock_info['shares_outstanding']
        )
        
        # Calculate per share value
        per_share_dcf = dcf_value / stock_info['shares_outstanding'] if stock_info['shares_outstanding'] > 0 else 0
        
        # Format DCF output
        dcf_output = html.Div([
            html.Div([
                html.Div([
                    html.P("Enterprise Value (DCF)", style={'margin': '0', 'fontWeight': 'bold', 'color': '#eeeeee'}),
                    html.H3(f"${dcf_value/1e9:.2f} B", style={'margin': '5px 0', 'color': '#ffffff'})
                ], style={'width': '50%'}),
                html.Div([
                    html.P("Per Share Value", style={'margin': '0', 'fontWeight': 'bold', 'color': '#eeeeee'}),
                    html.H3([
                        f"${per_share_dcf:.2f}",
                        html.Span(
                            f" ({(per_share_dcf/stock_info['current_price']-1)*100:.1f}%)" if stock_info['current_price'] > 0 else "",
                            style={'color': '#4CAF50' if per_share_dcf > stock_info['current_price'] else '#F44336'}
                        )
                    ], style={'margin': '5px 0', 'color': '#ffffff'})
                ], style={'width': '50%'})
            ], style={'display': 'flex', 'marginBottom': '10px'}),
            
            html.P("Sensitivity Analysis", style={'fontWeight': 'bold', 'borderTop': '1px solid #555555', 'paddingTop': '10px', 'marginBottom': '10px'}),
            html.Div([
                html.Table(
                    [
                        html.Tr([html.Th("Scenario", style={'textAlign': 'left', 'padding': '5px 10px'})] + 
                                [html.Th(key, style={'padding': '5px 10px'}) for key in sensitivity.keys()]),
                        html.Tr([html.Td("Enterprise Value (B)", style={'padding': '5px 10px'})] + 
                                [html.Td(f"${value/1e9:.2f}", style={'padding': '5px 10px'}) for value in sensitivity.values()]),
                        html.Tr([html.Td("Per Share", style={'padding': '5px 10px'})] + 
                                [html.Td(
                                    f"${value/stock_info['shares_outstanding']:.2f}" if stock_info['shares_outstanding'] > 0 else "N/A", 
                                    style={'padding': '5px 10px'}
                                ) for value in sensitivity.values()])
                    ],
                    style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '0.9em'}
                )
            ])
        ])
        
        # Calculate CAPM expected return
        expected_return, beta, risk_free, market_premium = capm_model(ticker)
        
        # Format CAPM output
        capm_output = html.Div([
            html.Div([
                html.Div([
                    html.P("Beta", style={'margin': '0', 'fontWeight': 'bold', 'color': '#eeeeee'}),
                    html.H3(f"{beta:.2f}", style={'margin': '5px 0', 'color': '#ffffff'})
                ], style={'width': '33%'}),
                html.Div([
                    html.P("Risk-Free Rate", style={'margin': '0', 'fontWeight': 'bold', 'color': '#eeeeee'}),
                    html.H3(f"{risk_free:.2%}", style={'margin': '5px 0', 'color': '#ffffff'})
                ], style={'width': '33%'}),
                html.Div([
                    html.P("Market Premium", style={'margin': '0', 'fontWeight': 'bold', 'color': '#eeeeee'}),
                    html.H3(f"{market_premium:.2%}", style={'margin': '5px 0', 'color': '#ffffff'})
                ], style={'width': '33%'})
            ], style={'display': 'flex', 'marginBottom': '10px'}),
            
            html.Div([
                html.P("Expected Annual Return (CAPM)", style={'margin': '0', 'fontWeight': 'bold', 'color': '#eeeeee'}),
                html.H2(f"{expected_return:.2%}", style={
                    'margin': '5px 0', 
                    'color': '#4CAF50' if expected_return > risk_free + 0.03 else '#F44336'
                })
            ], style={'marginTop': '10px', 'borderTop': '1px solid #555555', 'paddingTop': '10px'})
        ])
        
        # Calculate MM analysis
        mm_values, wacc = modigliani_miller_analysis(
            market_cap=stock_info['market_cap'],
            debt=stock_info['total_debt'],
            cost_of_debt=stock_info['cost_of_debt'],
            cost_of_equity=expected_return  # Using CAPM as cost of equity
        )
        
        # Format MM output
        mm_output = html.Div([
            html.Div([
                html.Div([
                    html.P("Current Debt", style={'margin': '0', 'fontWeight': 'bold', 'color': '#eeeeee'}),
                    html.H3(f"${stock_info['total_debt']/1e9:.2f} B", style={'margin': '5px 0', 'color': '#ffffff'})
                ], style={'width': '33%'}),
                html.Div([
                    html.P("Debt/Equity Ratio", style={'margin': '0', 'fontWeight': 'bold', 'color': '#eeeeee'}),
                    html.H3(
                        f"{stock_info['total_debt']/stock_info['market_cap']:.2f}" if stock_info['market_cap'] > 0 else "N/A", 
                        style={'margin': '5px 0', 'color': '#ffffff'}
                    )