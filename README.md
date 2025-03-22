# Stock Valuation Dashboard

This project implements a stock valuation tool that uses various financial models including Discounted Cash Flow (DCF), Capital Asset Pricing Model (CAPM), and Modigliani-Miller adjustments to provide comprehensive financial analyses of publicly traded companies.

## Features

- **Real-time Financial Data Retrieval**: Fetches up-to-date financial information using Yahoo Finance API
- **Multiple Valuation Models**:
  - Discounted Cash Flow (DCF) analysis
  - Capital Asset Pricing Model (CAPM) for expected returns
  - Modigliani-Miller adjustments for leverage impact
- **Interactive Dashboard**: Built with Dash and Plotly for visualizing stock data and valuation metrics
- **Comprehensive Testing Suite**: Includes unit and integration tests for reliable functionality

## Project Structure

```
stock-valuation/
├── DCF.ipynb              # Main Jupyter notebook with dashboard implementation
├── environment.yml        # Conda environment specification
├── finance_utils.py       # Core financial functions (you need to create this)
├── test_finance_functions.py # Unit tests for financial functions
├── test_integration.py    # Integration tests for the full pipeline
└── README.md              # This documentation file
```

## Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Internet connection for fetching financial data

### Setting Up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-valuation.git
   cd stock-valuation
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate stock-valuation
   ```

3. Create the `finance_utils.py` file with the following content:
   ```python
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
   ```

## Running the Application

### Via Jupyter Notebook

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `DCF.ipynb` and run all cells.

3. The interactive dashboard will appear in the notebook output. You can also access it directly at http://127.0.0.1:8060/ in your web browser.

### As a Standalone Application

1. Create a standalone `app.py` file by extracting the code from the notebook:
   ```python
   import yfinance as yf
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import statsmodels.api as sm
   import dash
   from dash import dcc, html, Input, Output
   import plotly.graph_objs as go
   from finance_utils import (
       fetch_financial_data,
       dcf_valuation,
       capm_model,
       modigliani_miller_adjustment,
       fetch_stock_info
   )

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
       app.run_server(debug=True, port=8060)
   ```

2. Run the standalone application:
   ```bash
   python app.py
   ```

3. Access the dashboard at http://127.0.0.1:8060/ in your web browser.

## Running Tests

1. Run unit tests for financial functions:
   ```bash
   python -m unittest test_finance_functions.py
   ```

2. Run integration tests:
   ```bash
   python -m unittest test_integration.py
   ```

## Usage Instructions

1. Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOGL) in the input field.
2. Click the "Submit" button.
3. View the calculated financial metrics:
   - Intrinsic Value based on DCF analysis
   - Market Cap
   - Expected Return based on CAPM
   - Adjusted Market Value with Modigliani-Miller considerations
   - Investment recommendation
4. Examine the 5-year stock price chart.

## Financial Models Explained

### Discounted Cash Flow (DCF)
Projects future cash flows and discounts them to present value, providing an estimate of intrinsic value.

### Capital Asset Pricing Model (CAPM)
Calculates the expected return on an investment based on the risk-free rate, market return, and the stock's beta.

### Modigliani-Miller Theorem
Incorporates the impact of a company's capital structure (debt and equity financing) on its valuation.

## Troubleshooting

- **No data available error**: Some tickers may not have complete financial data on Yahoo Finance. Try well-established companies.
- **Environment issues**: Ensure all dependencies are properly installed via the conda environment.
- **API limitations**: Yahoo Finance API may occasionally have rate limits or change their data structure.

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses the [yfinance](https://github.com/ranaroussi/yfinance) library for financial data retrieval.
- Dashboard built with [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/).



# Discounted Cash Flow (DCF) Valuation

This repository contains a Python-based implementation of Discounted Cash Flow (DCF) valuation for financial analysis. The model retrieves financial data, calculates the DCF valuation, and provides a sensitivity analysis.

## Features
- Fetches financial data from an external source
- Computes DCF valuation based on cash flows
- Performs sensitivity analysis
- Implements error handling with safe execution decorators

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/your-username/DCF-valuation.git
cd DCF-valuation
```

### 2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m unittest test_finance_functions.py
```

### 4. Install conda environment
```bash
conda env create -f environment.yml
conda activate stock-valuation
```

## Usage
Run the Jupyter Notebook to perform the valuation analysis:
```bash
jupyter notebook DCF_v2.ipynb
```
