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
        elif not re.match(r'^[a-zA-Z0-9\s\-\.\,\&\'\"]+, counterparty):
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
                if trade.trade_type.value in ['Option', 'Swaption']:
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
