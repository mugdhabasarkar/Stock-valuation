import unittest
import pandas as pd
from finance_utils import fetch_financial_data, dcf_valuation, capm_model, fetch_stock_info

class TestIntegration(unittest.TestCase):
    
    def test_valuation_pipeline(self):
        """Test the full valuation pipeline with a real stock"""
        # Use a stable, well-known ticker for testing
        ticker = "MSFT"
        
        # Test data fetching
        cashflows = fetch_financial_data(ticker)
        self.assertFalse(cashflows.empty, "Failed to fetch financial data")
        
        # Test DCF valuation
        intrinsic_value = dcf_valuation(cashflows)
        self.assertGreater(intrinsic_value, 0, "DCF valuation failed")
        
        # Test stock info fetching
        market_cap, beta, price, debt, cost_of_debt = fetch_stock_info(ticker)
        self.assertGreater(market_cap, 0, "Failed to fetch market cap")
        self.assertGreater(beta, 0, "Failed to fetch beta")
        
        # Test CAPM calculation
        expected_return = capm_model(beta)
        self.assertGreater(expected_return, 0, "CAPM calculation failed")
        
        # Check if all pieces work together
        print(f"Integration test results for {ticker}:")
        print(f"Intrinsic value: ${intrinsic_value:,.2f}")
        print(f"Market cap: ${market_cap:,.2f}")
        print(f"Expected return: {expected_return:.2%}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
