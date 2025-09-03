#!/usr/bin/env python3
"""
Backend Test for SA-CCR Streamlit Application
Tests the core functionality of the SA-CCR application components
"""

import sys
import os
sys.path.append('/app')

from datetime import datetime
import pandas as pd
import numpy as np

class SACCRBackendTester:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        
    def run_test(self, name, test_func):
        """Run a single test"""
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            result = test_func()
            if result:
                self.tests_passed += 1
                print(f"✅ Passed - {name}")
                return True
            else:
                print(f"❌ Failed - {name}")
                return False
        except Exception as e:
            print(f"❌ Failed - {name}: {str(e)}")
            return False
    
    def test_imports(self):
        """Test all required imports"""
        try:
            # Test main application imports
            from src.engine.saccr_engine import SACCREngine
            from src.data.database_manager import DatabaseManager
            from src.config.config_manager import ConfigManager
            from src.ui.components import UIComponents
            from src.models.trade_models import Trade, NettingSet, Collateral, AssetClass, TradeType, CollateralType
            from src.utils.validators import TradeValidator
            from src.utils.progress_tracker import ProgressTracker
            
            # Test main application class
            from main import SACCRApplication
            
            print("✅ All imports successful")
            return True
        except Exception as e:
            print(f"❌ Import error: {e}")
            return False
    
    def test_application_instantiation(self):
        """Test application instantiation"""
        try:
            from main import SACCRApplication
            app = SACCRApplication()
            
            # Check if key components are initialized
            if hasattr(app, 'config_manager') and hasattr(app, 'saccr_engine'):
                print("✅ Application components initialized")
                return True
            else:
                print("❌ Application components not properly initialized")
                return False
        except Exception as e:
            print(f"❌ Application instantiation error: {e}")
            return False
    
    def test_config_manager(self):
        """Test configuration manager"""
        try:
            from src.config.config_manager import ConfigManager
            config_manager = ConfigManager()
            
            # Test config validation
            status = config_manager.validate_config()
            if isinstance(status, dict) and 'valid' in status:
                print(f"✅ Config validation: {status}")
                return True
            else:
                print("❌ Config validation failed")
                return False
        except Exception as e:
            print(f"❌ Config manager error: {e}")
            return False
    
    def test_database_manager(self):
        """Test database manager"""
        try:
            from src.data.database_manager import DatabaseManager
            db_manager = DatabaseManager()
            
            # Test basic database operations
            trade_count = db_manager.get_trade_count()
            portfolio_count = db_manager.get_portfolio_count()
            
            print(f"✅ Database stats - Trades: {trade_count}, Portfolios: {portfolio_count}")
            return True
        except Exception as e:
            print(f"❌ Database manager error: {e}")
            return False
    
    def test_saccr_engine(self):
        """Test SA-CCR calculation engine"""
        try:
            from src.config.config_manager import ConfigManager
            from src.engine.saccr_engine import SACCREngine
            
            config_manager = ConfigManager()
            saccr_engine = SACCREngine(config_manager)
            
            print("✅ SA-CCR engine initialized successfully")
            return True
        except Exception as e:
            print(f"❌ SA-CCR engine error: {e}")
            return False
    
    def test_trade_models(self):
        """Test trade model creation"""
        try:
            from src.models.trade_models import Trade, AssetClass, TradeType
            from datetime import datetime, timedelta
            
            # Create a sample trade
            trade = Trade(
                trade_id="TEST_001",
                counterparty="Test Bank",
                asset_class=AssetClass.INTEREST_RATE,
                trade_type=TradeType.SWAP,
                notional=100_000_000,
                currency="USD",
                underlying="USD-LIBOR-3M",
                maturity_date=datetime.now() + timedelta(days=1825),  # 5 years
                mtm_value=0,
                delta=1.0
            )
            
            if trade.trade_id == "TEST_001" and trade.notional == 100_000_000:
                print("✅ Trade model creation successful")
                return True
            else:
                print("❌ Trade model creation failed")
                return False
        except Exception as e:
            print(f"❌ Trade model error: {e}")
            return False
    
    def test_validators(self):
        """Test trade validators"""
        try:
            from src.utils.validators import TradeValidator
            
            validator = TradeValidator()
            
            # Test valid trade data
            valid_data = {
                'trade_id': 'TEST_001',
                'notional': 100_000_000,
                'currency': 'USD',
                'maturity_years': 5.0
            }
            
            result = validator.validate_trade_data(valid_data)
            if result.get('valid', False):
                print("✅ Trade validation successful")
                return True
            else:
                print(f"❌ Trade validation failed: {result}")
                return False
        except Exception as e:
            print(f"❌ Validator error: {e}")
            return False
    
    def test_streamlit_app_access(self):
        """Test Streamlit application accessibility"""
        try:
            import requests
            
            # Test if Streamlit app is accessible
            response = requests.get("http://localhost:8501", timeout=5)
            if response.status_code == 200:
                print("✅ Streamlit application is accessible")
                return True
            else:
                print(f"❌ Streamlit application returned status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Streamlit access error: {e}")
            return False

def main():
    """Main test execution"""
    print("🚀 Starting SA-CCR Backend Tests")
    print("=" * 50)
    
    tester = SACCRBackendTester()
    
    # Run all tests
    tests = [
        ("Module Imports", tester.test_imports),
        ("Application Instantiation", tester.test_application_instantiation),
        ("Configuration Manager", tester.test_config_manager),
        ("Database Manager", tester.test_database_manager),
        ("SA-CCR Engine", tester.test_saccr_engine),
        ("Trade Models", tester.test_trade_models),
        ("Validators", tester.test_validators),
        ("Streamlit App Access", tester.test_streamlit_app_access),
    ]
    
    for test_name, test_func in tests:
        tester.run_test(test_name, test_func)
    
    # Print results
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())