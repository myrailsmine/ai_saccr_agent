#!/usr/bin/env python3
"""
Test script to verify the optimization and comparison functionality
"""

import sys
import os
sys.path.append('/app')

from main import SACCRApplication
import streamlit as st
from datetime import datetime, timedelta
from src.models.trade_models import Trade, AssetClass, TradeType

def test_optimization_and_comparison():
    """Test the optimization and comparison page functionality"""
    
    print("🧪 Testing SA-CCR Optimization and Comparison Functionality")
    print("=" * 60)
    
    # Initialize the application
    app = SACCRApplication()
    
    # Create a mock portfolio for testing
    mock_portfolio = {
        'netting_set_id': 'TEST_NS_001',
        'counterparty': 'Test Bank',
        'threshold': 1000000.0,
        'mta': 500000.0,
        'trades': [
            Trade(
                trade_id='TEST_TRADE_001',
                counterparty='Test Bank',
                asset_class=AssetClass.INTEREST_RATE,
                trade_type=TradeType.SWAP,
                notional=100000000.0,
                currency='USD',
                underlying='USD-LIBOR-3M',
                maturity_date=datetime.now() + timedelta(days=1825),  # 5 years
                mtm_value=1000000.0,
                delta=1.0
            )
        ]
    }
    
    # Create mock calculation results
    mock_results = {
        'final_results': {
            'exposure_at_default': 50000000.0,
            'risk_weighted_assets': 50000000.0,
            'capital_requirement': 4000000.0,
            'replacement_cost': 10000000.0,
            'potential_future_exposure': 25714285.71,
            'portfolio_summary': {
                'total_notional': 100000000.0
            }
        },
        'calculation_steps': [
            {
                'step': 15,
                'title': 'PFE Multiplier Calculation',
                'data': {
                    'multiplier': 0.85,
                    'v': 1000000.0,
                    'addon': 30000000.0
                }
            },
            {
                'step': 18,
                'title': 'Replacement Cost Calculation',
                'data': {
                    'replacement_cost': 10000000.0
                }
            },
            {
                'step': 19,
                'title': 'CEU Flag Determination',
                'data': {
                    'overall_ceu_flag': 1
                }
            }
        ]
    }
    
    # Test 1: Check if optimization methods exist and are callable
    print("1. Testing optimization analysis methods...")
    
    try:
        # Test central clearing analysis
        app._render_central_clearing_analysis(mock_results)
        print("   ✅ Central clearing analysis method works")
        
        # Test netting optimization analysis
        app._render_netting_optimization_analysis(mock_results)
        print("   ✅ Netting optimization analysis method works")
        
        # Test collateral management analysis
        app._render_collateral_management_analysis(mock_results)
        print("   ✅ Collateral management analysis method works")
        
        # Test portfolio restructuring analysis
        # Set up session state for this test
        if not hasattr(st, 'session_state'):
            class MockSessionState:
                def __init__(self):
                    self.current_portfolio = mock_portfolio
            st.session_state = MockSessionState()
        else:
            st.session_state.current_portfolio = mock_portfolio
            
        app._render_portfolio_restructuring_analysis(mock_results)
        print("   ✅ Portfolio restructuring analysis method works")
        
    except Exception as e:
        print(f"   ❌ Error in optimization analysis: {e}")
        return False
    
    # Test 2: Check scenario calculation functionality
    print("\n2. Testing scenario calculation methods...")
    
    try:
        # Test central clearing scenario
        scenario_params = {'alpha_scenario': 0.5}
        result = app._calculate_scenario('Central Clearing', mock_results, scenario_params)
        
        expected_keys = ['exposure_at_default', 'risk_weighted_assets', 'capital_requirement', 
                        'replacement_cost', 'potential_future_exposure']
        
        if all(key in result for key in expected_keys):
            print("   ✅ Central clearing scenario calculation works")
        else:
            print("   ❌ Central clearing scenario missing required keys")
            return False
            
        # Test collateral posting scenario
        scenario_params = {'collateral_amount': 10.0}
        result = app._calculate_scenario('Collateral Posting', mock_results, scenario_params)
        
        if all(key in result for key in expected_keys):
            print("   ✅ Collateral posting scenario calculation works")
        else:
            print("   ❌ Collateral posting scenario missing required keys")
            return False
            
        # Test portfolio compression scenario
        scenario_params = {'compression_ratio': 0.8}
        result = app._calculate_scenario('Portfolio Compression', mock_results, scenario_params)
        
        if all(key in result for key in expected_keys):
            print("   ✅ Portfolio compression scenario calculation works")
        else:
            print("   ❌ Portfolio compression scenario missing required keys")
            return False
            
    except Exception as e:
        print(f"   ❌ Error in scenario calculation: {e}")
        return False
    
    # Test 3: Check scenario comparison display
    print("\n3. Testing scenario comparison display...")
    
    try:
        scenario_results = {
            'exposure_at_default': 25000000.0,  # 50% reduction
            'risk_weighted_assets': 25000000.0,
            'capital_requirement': 2000000.0,
            'replacement_cost': 5000000.0,
            'potential_future_exposure': 12857142.86
        }
        
        app._display_scenario_comparison(mock_results, scenario_results, "Test Scenario")
        print("   ✅ Scenario comparison display method works")
        
    except Exception as e:
        print(f"   ❌ Error in scenario comparison display: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! Optimization and Comparison functionality is working correctly.")
    print("\n📋 Summary of implemented features:")
    print("   • Central Clearing Impact Analysis")
    print("   • Netting Optimization Analysis") 
    print("   • Collateral Management Analysis")
    print("   • Portfolio Restructuring Analysis")
    print("   • Scenario Calculation Engine")
    print("   • Interactive Scenario Comparison")
    print("   • Visual Charts and Metrics")
    print("   • Implementation Recommendations")
    
    return True

if __name__ == "__main__":
    success = test_optimization_and_comparison()
    sys.exit(0 if success else 1)