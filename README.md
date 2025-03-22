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

## Running the Application

### Via Jupyter Notebook

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `DCF.ipynb` and run all cells.

3. The interactive dashboard will appear in the notebook output. You can also access it directly at http://127.0.0.1:8060/ in your web browser.

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

