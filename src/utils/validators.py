# src/utils/validators.py

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)

class TradeValidator:
    """Comprehensive trade data validation"""
    
    def __init__(self):
        # Supported currencies (ISO 4217)
        self.supported_currencies = {
            'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD', 
            'SEK', 'NOK', 'DKK', 'PLN', 'CZK', 'HUF', 'SGD', 'HKD',
            'CNY', 'INR', 'KRW', 'BRL', 'MXN', 'ZAR'
        }
        
        # Valid asset classes
        self.valid_asset_classes = {
            'Interest Rate', 'Foreign Exchange', 'Credit', 'Equity', 'Commodity'
        }
        
        # Valid trade types
        self.valid_trade_types = {
            'Swap', 'Forward', 'Option', 'Swaption', 'Future', 'Credit Default Swap'
        }
    
    def validate_trade_data(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive trade data validation"""
        
        errors = []
        warnings = []
        
        # Validate trade ID
        trade_id = trade_data.get('trade_id', '')
        if not trade_id:
            errors.append("Trade ID is required")
        elif not isinstance(trade_id, str):
            errors.append("Trade ID must be a string")
        elif len(trade_id) > 50:
            warnings.append("Trade ID is unusually long")
        
        # Validate notional
        notional = trade_data.get('notional', 0)
        try:
            notional_decimal = Decimal(str(notional))
            if notional_decimal == 0:
                errors.append("Notional amount cannot be zero")
            elif abs(notional_decimal) < Decimal('0.01'):
                warnings.append("Notional amount is very small")
            elif abs(notional_decimal) > Decimal('1000000000000'):  # 1 trillion
                warnings.append("Notional amount is very large")
        except (InvalidOperation, ValueError):
            errors.append("Notional must be a valid number")
        
        # Validate currency
        currency = trade_data.get('currency', '')
        if not currency:
            errors.append("Currency is required")
        elif currency not in self.supported_currencies:
            warnings.append(f"Currency '{currency}' may not be fully supported")
        
        # Validate maturity
        maturity_years = trade_data.get('maturity_years', 0)
        if maturity_years <= 0:
            errors.append("Maturity must be positive")
        elif maturity_years > 50:
            warnings.append("Maturity exceeds 50 years - please verify")
        elif maturity_years < 1/365:  # Less than 1 day
            warnings.append("Very short maturity detected")
        
        # Validate MTM value (if provided)
        mtm_value = trade_data.get('mtm_value')
        if mtm_value is not None:
            try:
                mtm_decimal = Decimal(str(mtm_value))
                if abs(mtm_decimal) > abs(Decimal(str(notional))) * Decimal('2'):
                    warnings.append("MTM value is unusually large relative to notional")
            except (InvalidOperation, ValueError):
                warnings.append("MTM value is not a valid number")
        
        # Validate delta (if provided)
        delta = trade_data.get('delta')
        if delta is not None:
            try:
                delta_val = float(delta)
                if abs(delta_val) > 1.1:  # Allow slight tolerance
                    warnings.append("Delta outside normal range [-1, 1]")
            except (ValueError, TypeError):
                warnings.append("Delta is not a valid number")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'message': '; '.join(errors) if errors else 'Valid'
        }
    
    def validate_counterparty_name(self, counterparty: str) -> Dict[str, Any]:
        """Validate counterparty name"""
        
        errors = []
        warnings = []
        
        if not counterparty:
            errors.append("Counterparty name is required")
        elif len(counterparty.strip()) < 2:
            errors.append("Counterparty name too short")
        elif len(counterparty) > 100:
            warnings.append("Counterparty name is very long")
        elif not re.match(r'^[a-zA-Z0-9\s\-\.\,\&\'\"]+$', counterparty):
            warnings.append("Counterparty name contains unusual characters")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def validate_netting_set_parameters(self, threshold: float, mta: float, nica: float) -> Dict[str, Any]:
        """Validate netting set collateral parameters"""
        
        errors = []
        warnings = []
        
        # Validate threshold
        if threshold < 0:
            errors.append("Threshold cannot be negative")
        elif threshold > 1e12:  # 1 trillion
            warnings.append("Threshold amount is very large")
        
        # Validate MTA
        if mta < 0:
            errors.append("MTA cannot be negative")
        elif mta > threshold and threshold > 0:
            warnings.append("MTA is larger than threshold")
        
        # Validate NICA
        if abs(nica) > 1e12:
            warnings.append("NICA amount is very large")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def validate_portfolio_structure(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate overall portfolio structure"""
        
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['netting_set_id', 'counterparty', 'trades']
        for field in required_fields:
            if field not in portfolio_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate trades list
        trades = portfolio_data.get('trades', [])
        if not trades:
            errors.append("Portfolio must contain at least one trade")
        elif len(trades) > 10000:
            warnings.append("Portfolio contains a very large number of trades")
        
        # Check for duplicate trade IDs
        trade_ids = []
        for trade in trades:
            trade_id = trade.get('trade_id') if isinstance(trade, dict) else getattr(trade, 'trade_id', None)
            if trade_id:
                if trade_id in trade_ids:
                    errors.append(f"Duplicate trade ID found: {trade_id}")
                else:
                    trade_ids.append(trade_id)
        
        # Validate counterparty consistency
        portfolio_counterparty = portfolio_data.get('counterparty')
        if portfolio_counterparty:
            for trade in trades:
                trade_counterparty = trade.get('counterparty') if isinstance(trade, dict) else getattr(trade, 'counterparty', None)
                if trade_counterparty and trade_counterparty != portfolio_counterparty:
                    warnings.append(f"Trade counterparty mismatch: {trade_counterparty} vs {portfolio_counterparty}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'trade_count': len(trades),
            'unique_trade_ids': len(trade_ids)
        }

class DataConsistencyValidator:
    """Advanced data consistency validation"""
    
    def validate_calculation_inputs(self, netting_set, collateral=None) -> Dict[str, Any]:
        """Validate inputs for SA-CCR calculation"""
        
        errors = []
        warnings = []
        
        # Validate netting set
        if not hasattr(netting_set, 'trades') or not netting_set.trades:
            errors.append("Netting set must contain trades")
        
        # Validate trade data consistency
        for i, trade in enumerate(netting_set.trades):
            # Check maturity dates
            if hasattr(trade, 'maturity_date'):
                if trade.maturity_date <= datetime.now():
                    errors.append(f"Trade {i+1}: Maturity date is in the past")
            
            # Check delta consistency with trade type
            if hasattr(trade, 'trade_type') and hasattr(trade, 'delta'):
                if hasattr(trade.trade_type, 'value'):
                    trade_type_val = trade.trade_type.value
                else:
                    trade_type_val = str(trade.trade_type)
                
                if trade_type_val in ['Option', 'Swaption']:
                    if abs(trade.delta) > 1:
                        warnings.append(f"Trade {i+1}: Option delta outside [-1, 1] range")
                else:
                    if trade.delta != 1.0 and trade.delta != -1.0:
                        warnings.append(f"Trade {i+1}: Non-option trade should have delta of ±1")
        
        # Validate collateral consistency
        if collateral:
            total_collateral = sum(c.amount for c in collateral)
            total_notional = sum(abs(t.notional) for t in netting_set.trades)
            
            if total_collateral > total_notional * 2:
                warnings.append("Collateral amount is very high relative to trade notional")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def validate_regulatory_parameters(self, alpha: float, capital_ratio: float) -> Dict[str, Any]:
        """Validate regulatory parameters"""
        
        errors = []
        warnings = []
        
        # Validate alpha
        if alpha < 0.1 or alpha > 5.0:
            errors.append("Alpha must be between 0.1 and 5.0")
        elif alpha not in [0.5, 1.4]:  # Standard Basel values
            warnings.append("Alpha value differs from standard Basel parameters (0.5 or 1.4)")
        
        # Validate capital ratio
        if capital_ratio < 0.01 or capital_ratio > 0.5:
            errors.append("Capital ratio must be between 1% and 50%")
        elif capital_ratio != 0.08:  # Standard 8%
            warnings.append("Capital ratio differs from standard 8% requirement")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

class RegulatoryValidator:
    """Validator for Basel regulatory compliance"""
    
    def __init__(self):
        # Basel supervisory factors (as decimals)
        self.supervisory_factors = {
            'Interest Rate': {
                'USD': {'<2y': 0.005, '2-5y': 0.005, '>5y': 0.015},
                'EUR': {'<2y': 0.005, '2-5y': 0.005, '>5y': 0.015},
                'JPY': {'<2y': 0.005, '2-5y': 0.005, '>5y': 0.015},
                'GBP': {'<2y': 0.005, '2-5y': 0.005, '>5y': 0.015},
                'other': {'<2y': 0.015, '2-5y': 0.015, '>5y': 0.015}
            },
            'Foreign Exchange': {'G10': 0.04, 'emerging': 0.15},
            'Credit': {'IG_single': 0.0046, 'HY_single': 0.013},
            'Equity': {'single_large': 0.32, 'single_small': 0.40},
            'Commodity': {'energy': 0.18, 'metals': 0.18, 'agriculture': 0.18}
        }
        
        # Basel supervisory correlations
        self.supervisory_correlations = {
            'Interest Rate': 0.99,
            'Foreign Exchange': 0.60,
            'Credit': 0.50,
            'Equity': 0.80,
            'Commodity': 0.40
        }
    
    def validate_supervisory_parameters(self, asset_class: str, currency: str = None, 
                                      maturity_years: float = None) -> Dict[str, Any]:
        """Validate supervisory parameters for given trade characteristics"""
        
        errors = []
        warnings = []
        
        # Validate asset class
        if asset_class not in self.supervisory_factors:
            errors.append(f"Unknown asset class: {asset_class}")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Asset class specific validations
        if asset_class == 'Interest Rate':
            if not currency:
                warnings.append("Currency not specified for Interest Rate trade")
            elif not maturity_years:
                warnings.append("Maturity not specified for Interest Rate trade")
            else:
                # Validate maturity bucket
                if maturity_years < 0.01:
                    errors.append("Maturity too short for Interest Rate trade")
                elif maturity_years > 50:
                    warnings.append("Very long maturity for Interest Rate trade")
        
        elif asset_class == 'Foreign Exchange':
            if not currency:
                errors.append("Currency required for Foreign Exchange trade")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def validate_calculation_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SA-CCR calculation results for reasonableness"""
        
        errors = []
        warnings = []
        
        if not result or 'final_results' not in result:
            errors.append("Invalid calculation result structure")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        final_results = result['final_results']
        
        # Validate key result fields exist
        required_fields = ['replacement_cost', 'potential_future_exposure', 
                          'exposure_at_default', 'risk_weighted_assets', 'capital_requirement']
        
        for field in required_fields:
            if field not in final_results:
                errors.append(f"Missing result field: {field}")
            elif final_results[field] < 0:
                errors.append(f"Negative value for {field}: {final_results[field]}")
        
        if errors:
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Reasonableness checks
        rc = final_results['replacement_cost']
        pfe = final_results['potential_future_exposure']
        ead = final_results['exposure_at_default']
        rwa = final_results['risk_weighted_assets']
        capital = final_results['capital_requirement']
        
        # EAD should equal Alpha * (RC + PFE)
        calculated_ead = rc + pfe
        if abs(ead - calculated_ead) > calculated_ead * 0.01:  # 1% tolerance
            # Check if alpha was applied
            alpha_ratios = [0.5, 1.4]  # Possible alpha values
            alpha_applied = False
            
            for alpha in alpha_ratios:
                if abs(ead - alpha * calculated_ead) < alpha * calculated_ead * 0.01:
                    alpha_applied = True
                    break
            
            if not alpha_applied:
                warnings.append("EAD calculation may be incorrect")
        
        # Capital should be approximately 8% of RWA
        expected_capital = rwa * 0.08
        if abs(capital - expected_capital) > expected_capital * 0.01:
            warnings.append("Capital requirement calculation may be incorrect")
        
        # PFE should be positive if there are trades
        portfolio_summary = final_results.get('portfolio_summary', {})
        if portfolio_summary.get('trade_count', 0) > 0 and pfe == 0:
            warnings.append("PFE is zero despite having trades")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
