import unittest
import pandas as pd
import numpy as np
from finance_utils import (
    capm_model, 
    dcf_valuation, 
    modigliani_miller_adjustment, 
    fetch_financial_data, 
    fetch_stock_info
)

class TestFinanceFunctions(unittest.TestCase):
    
    def test_capm_model(self):
        # Standard scenarios
        test_cases = [
            (1.0, 0.03, 0.08, 0.08),      # Baseline market beta
            (1.5, 0.03, 0.08, 0.105),     # High beta
            (0.0, 0.03, 0.08, 0.03),      # Zero beta
            (-0.5, 0.03, 0.08, 0.055)     # Negative beta
        ]
        
        for beta, risk_free, market_return, expected in test_cases:
            with self.subTest(beta=beta):
                self.assertAlmostEqual(
                    capm_model(beta, risk_free, market_return), 
                    expected, 
                    places=3
                )
    
    def test_dcf_valuation(self):
        # Various cashflow scenarios
        test_cases = [
            # Standard cashflow series
            pd.Series([1000000, 1100000, 1200000, 1300000]),
            # Volatile cashflow series
            pd.Series([500000, 750000, 600000, 850000]),
            # Edge cases
            pd.Series([]),  # Empty series
            pd.Series([0, 0, 0])  # Zero cashflows
        ]
        
        for cashflows in test_cases:
            with self.subTest(cashflows=cashflows):
                result = dcf_valuation(cashflows)
                
                if not cashflows.empty and cashflows.sum() > 0:
                    self.assertIsNotNone(result)
                    self.assertGreater(result, 0)
                else:
                    self.assertIsNone(result)
    
    def test_modigliani_miller_adjustment(self):
        # Comprehensive test scenarios
        test_cases = [
            (1000000, 500000, 0.05, 0.21),  # Standard case
            (2000000, 0, 0.05, 0.21),       # No debt
            (1000000, -500000, 0.05, 0.21), # Negative debt (edge case)
            (0, 1000000, 0.05, 0.21)        # Zero market cap
        ]
        
        for market_cap, debt, cost_of_debt, tax_rate in test_cases:
            with self.subTest(market_cap=market_cap, debt=debt):
                result = modigliani_miller_adjustment(
                    market_cap, debt, cost_of_debt, tax_rate
                )
                self.assertIsInstance(result, (int, float))
    
    def test_stock_data_fetching(self):
        # Test with some well-known tickers
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        
        for ticker in test_tickers:
            with self.subTest(ticker=ticker):
                # Test financial data fetching
                cashflows = fetch_financial_data(ticker)
                self.assertFalse(cashflows.empty, f"Failed to fetch cashflows for {ticker}")
                
                # Test stock info fetching
                market_cap, beta, price, debt, cost_of_debt = fetch_stock_info(ticker)
                
                self.assertGreaterEqual(market_cap, 0, f"Invalid market cap for {ticker}")
                self.assertTrue(0 <= beta <= 3, f"Unrealistic beta for {ticker}")
                self.assertGreaterEqual(price, 0, f"Invalid price for {ticker}")
                self.assertGreaterEqual(debt, 0, f"Invalid debt for {ticker}")
                self.assertTrue(0 <= cost_of_debt <= 1, f"Unrealistic cost of debt for {ticker}")

    def test_integration_scenarios(self):
        # Complex integration test
        test_ticker = 'MSFT'
        
        # Fetch data
        cashflows = fetch_financial_data(test_ticker)
        market_cap, beta, price, debt, cost_of_debt = fetch_stock_info(test_ticker)
        
        # Perform calculations
        dcf_value = dcf_valuation(cashflows)
        capm_return = capm_model(beta)
        mm_adjustment = modigliani_miller_adjustment(market_cap, debt, cost_of_debt)
        
        # Validate results
        self.assertIsNotNone(dcf_value)
        self.assertGreater(dcf_value, 0)
        self.assertGreater(capm_return, 0)
        self.assertGreater(mm_adjustment, 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
