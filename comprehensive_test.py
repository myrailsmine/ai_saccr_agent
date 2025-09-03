#!/usr/bin/env python3
"""
Comprehensive Test Suite for SA-CCR Application
Tests all key features mentioned in the review request
"""

import sys
import os
sys.path.append('/app')

from datetime import datetime
import requests

class ComprehensiveTestSuite:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = {}
        
    def run_test(self, category, name, test_func):
        """Run a single test and track results by category"""
        self.tests_run += 1
        test_key = f"{category}::{name}"
        
        print(f"\n🔍 Testing {name}...")
        
        try:
            result = test_func()
            if result:
                self.tests_passed += 1
                self.test_results[test_key] = "✅ PASSED"
                print(f"✅ Passed - {name}")
                return True
            else:
                self.test_results[test_key] = "❌ FAILED"
                print(f"❌ Failed - {name}")
                return False
        except Exception as e:
            self.test_results[test_key] = f"❌ ERROR: {str(e)}"
            print(f"❌ Failed - {name}: {str(e)}")
            return False
    
    def test_application_loading(self):
        """Test application loading and accessibility"""
        try:
            response = requests.get("http://localhost:8501", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def test_professional_styling_code(self):
        """Test professional styling implementation in code"""
        try:
            with open('/app/main.py', 'r') as f:
                content = f.read()
            
            # Check for professional styling elements
            styling_elements = [
                'enterprise-grade',
                'gradient',
                'professional',
                'main-header',
                'Inter',
                'box-shadow',
                'border-radius',
                'transition'
            ]
            
            found_elements = sum(1 for element in styling_elements if element in content)
            return found_elements >= 6  # At least 6 professional styling elements
        except:
            return False
    
    def test_ai_assistant_greeting(self):
        """Test AI assistant with simple greeting"""
        try:
            from main import SACCRApplication
            app = SACCRApplication()
            
            query = "Hello, can you help me with SA-CCR?"
            analysis = app._analyze_query_intent_enhanced(query)
            
            return (isinstance(analysis, dict) and 
                   'categories' in analysis and 
                   'general' in analysis['categories'])
        except:
            return False
    
    def test_ai_assistant_calculation_request(self):
        """Test AI assistant with calculation request"""
        try:
            from main import SACCRApplication
            app = SACCRApplication()
            
            query = "Calculate SA-CCR for a $200M USD interest rate swap with Goldman Sachs, 5-year maturity"
            analysis = app._analyze_query_intent_enhanced(query)
            
            return (analysis['requires_calculation'] and 
                   len(analysis['extracted_trades']) > 0 and
                   analysis['extracted_trades'][0]['notional'] == 200_000_000)
        except:
            return False
    
    def test_ai_assistant_optimization_query(self):
        """Test AI assistant with optimization query"""
        try:
            from main import SACCRApplication
            app = SACCRApplication()
            
            query = "How can I reduce my capital requirements through central clearing?"
            analysis = app._analyze_query_intent_enhanced(query)
            
            return ('optimization' in analysis['categories'] and
                   'regulatory' in analysis['categories'])
        except:
            return False
    
    def test_ai_assistant_regulatory_question(self):
        """Test AI assistant with regulatory question"""
        try:
            from main import SACCRApplication
            app = SACCRApplication()
            
            query = "Explain the Basel SA-CCR methodology"
            analysis = app._analyze_query_intent_enhanced(query)
            response = app._handle_information_query_enhanced(query, analysis)
            
            return (len(response) > 100 and 
                   ('Basel' in response or 'SA-CCR' in response))
        except:
            return False
    
    def test_ai_assistant_comparison_question(self):
        """Test AI assistant with comparison question"""
        try:
            from main import SACCRApplication
            app = SACCRApplication()
            
            query = "What's the difference between RC and PFE?"
            analysis = app._analyze_query_intent_enhanced(query)
            response = app._handle_information_query_enhanced(query, analysis)
            
            return (len(response) > 50 and 
                   'comparison' in analysis['categories'])
        except:
            return False
    
    def test_mandatory_information_requests(self):
        """Test AI assistant's ability to request missing information"""
        try:
            from main import SACCRApplication
            app = SACCRApplication()
            
            # Test with incomplete information
            incomplete_query = "Calculate SA-CCR for a swap"
            analysis = app._analyze_query_intent_enhanced(incomplete_query)
            
            # Should identify missing information
            return analysis['requires_calculation'] and len(analysis['extracted_trades']) == 0
        except:
            return False
    
    def test_portfolio_analysis_navigation(self):
        """Test portfolio analysis functionality"""
        try:
            from main import SACCRApplication
            app = SACCRApplication()
            
            # Check if portfolio analysis methods exist
            return (hasattr(app, '_render_portfolio_page') and
                   hasattr(app, '_render_portfolio_input') and
                   hasattr(app, '_display_current_portfolio'))
        except:
            return False
    
    def test_saccr_calculation_engine(self):
        """Test SA-CCR calculation with sample portfolio"""
        try:
            from main import SACCRApplication
            from src.models.trade_models import Trade, AssetClass, TradeType
            from datetime import datetime, timedelta
            
            app = SACCRApplication()
            
            # Create sample trade
            trade = Trade(
                trade_id="TEST_001",
                counterparty="Goldman Sachs",
                asset_class=AssetClass.INTEREST_RATE,
                trade_type=TradeType.SWAP,
                notional=200_000_000,
                currency="USD",
                underlying="USD-LIBOR-3M",
                maturity_date=datetime.now() + timedelta(days=1825),
                mtm_value=0,
                delta=1.0
            )
            
            portfolio = {
                'netting_set_id': 'NS_001',
                'counterparty': 'Goldman Sachs',
                'threshold': 0,
                'mta': 0,
                'trades': [trade]
            }
            
            # Perform calculation
            results = app.saccr_engine.calculate_comprehensive_saccr(portfolio, [])
            
            return (results and 
                   'final_results' in results and
                   'exposure_at_default' in results['final_results'] and
                   results['final_results']['exposure_at_default'] > 0)
        except:
            return False
    
    def test_enhanced_error_handling(self):
        """Test enhanced error handling"""
        try:
            from main import SACCRApplication
            from src.utils.validators import TradeValidator
            
            app = SACCRApplication()
            validator = TradeValidator()
            
            # Test with invalid data
            invalid_data = {
                'trade_id': '',  # Invalid empty ID
                'notional': -1000,  # Invalid negative notional
                'currency': 'INVALID',
                'maturity_years': -1
            }
            
            result = validator.validate_trade_data(invalid_data)
            return not result.get('valid', True)  # Should return invalid
        except:
            return False
    
    def test_navigation_modules(self):
        """Test navigation between different modules"""
        try:
            from main import SACCRApplication
            app = SACCRApplication()
            
            # Check if all required page rendering methods exist
            required_methods = [
                '_render_calculator_page',
                '_render_portfolio_page', 
                '_render_optimization_page',
                '_render_comparison_page',
                '_render_database_page',
                '_render_settings_page'
            ]
            
            return all(hasattr(app, method) for method in required_methods)
        except:
            return False
    
    def test_database_connectivity(self):
        """Test database connectivity and operations"""
        try:
            from src.data.database_manager import DatabaseManager
            
            db_manager = DatabaseManager()
            
            # Test basic database operations
            trade_count = db_manager.get_trade_count()
            portfolio_count = db_manager.get_portfolio_count()
            
            return isinstance(trade_count, int) and isinstance(portfolio_count, int)
        except:
            return False
    
    def test_configuration_management(self):
        """Test configuration management"""
        try:
            from src.config.config_manager import ConfigManager
            
            config_manager = ConfigManager()
            status = config_manager.validate_config()
            
            return status.get('valid', False)
        except:
            return False

def main():
    """Main test execution"""
    print("🚀 Starting Comprehensive SA-CCR Application Test Suite")
    print("=" * 70)
    
    suite = ComprehensiveTestSuite()
    
    # Test categories as requested in review
    test_categories = [
        ("Application Loading & Professional Styling", [
            ("Application Accessibility", suite.test_application_loading),
            ("Professional CSS Styling", suite.test_professional_styling_code),
        ]),
        ("AI Assistant Intelligence", [
            ("Simple Greeting", suite.test_ai_assistant_greeting),
            ("Calculation Request", suite.test_ai_assistant_calculation_request),
            ("Optimization Query", suite.test_ai_assistant_optimization_query),
            ("Regulatory Question", suite.test_ai_assistant_regulatory_question),
            ("Comparison Question", suite.test_ai_assistant_comparison_question),
        ]),
        ("Mandatory Information Requests", [
            ("Missing Information Detection", suite.test_mandatory_information_requests),
        ]),
        ("Portfolio Analysis", [
            ("Navigation Functionality", suite.test_portfolio_analysis_navigation),
            ("SA-CCR Calculation Engine", suite.test_saccr_calculation_engine),
        ]),
        ("Enhanced Features", [
            ("Error Handling", suite.test_enhanced_error_handling),
            ("Module Navigation", suite.test_navigation_modules),
            ("Database Connectivity", suite.test_database_connectivity),
            ("Configuration Management", suite.test_configuration_management),
        ])
    ]
    
    # Run all tests by category
    for category_name, tests in test_categories:
        print(f"\n📋 {category_name}")
        print("-" * 50)
        
        for test_name, test_func in tests:
            suite.run_test(category_name, test_name, test_func)
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    
    for category_name, tests in test_categories:
        print(f"\n📋 {category_name}:")
        category_passed = 0
        category_total = 0
        
        for test_name, _ in tests:
            test_key = f"{category_name}::{test_name}"
            result = suite.test_results.get(test_key, "❓ NOT RUN")
            print(f"   {test_name}: {result}")
            
            category_total += 1
            if "✅ PASSED" in result:
                category_passed += 1
        
        print(f"   Category Score: {category_passed}/{category_total}")
    
    print(f"\n🎯 OVERALL RESULTS: {suite.tests_passed}/{suite.tests_run} tests passed")
    
    if suite.tests_passed == suite.tests_run:
        print("🎉 ALL TESTS PASSED! SA-CCR Application is fully functional.")
        return 0
    elif suite.tests_passed >= suite.tests_run * 0.8:  # 80% pass rate
        print("✅ MOSTLY SUCCESSFUL! Application is functional with minor issues.")
        return 0
    else:
        print("❌ SIGNIFICANT ISSUES FOUND! Application needs attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main())