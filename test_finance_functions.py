import unittest
import pandas as pd
import numpy as np
from finance_utils import capm_model, dcf_valuation, modigliani_miller_adjustment

class TestFinanceFunctions(unittest.TestCase):
    
    def test_capm_model(self):
        # Test with standard values
        self.assertAlmostEqual(capm_model(1.0, 0.03, 0.08), 0.08)
        # Test with high beta
        self.assertAlmostEqual(capm_model(1.5, 0.03, 0.08), 0.105)
        # Test with zero beta
        self.assertAlmostEqual(capm_model(0.0, 0.03, 0.08), 0.03)
    
    def test_dcf_valuation(self):
        # Create test cashflow series
        cashflows = pd.Series([1000000, 1100000, 1200000, 1300000])
        # Test with default parameters
        result = dcf_valuation(cashflows)
        # Should be positive and significant
        self.assertGreater(result, 10000000)
        # Test with empty series
        empty_series = pd.Series(dtype=float)
        self.assertEqual(dcf_valuation(empty_series), 0)
    
    def test_modigliani_miller(self):
        # Test tax shield effect
        result = modigliani_miller_adjustment(1000000, 500000, 0.05)
        self.assertGreater(result, 1000000)  # Should be greater due to tax shield

if __name__ == '__main__':
    unittest.main()
