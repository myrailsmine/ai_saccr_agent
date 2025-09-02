# src/engine/saccr_engine.py

import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from ..models.trade_models import Trade, NettingSet, Collateral, AssetClass, TradeType, CollateralType
from ..utils.validators import TradeValidator
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class SACCREngine:
    """
    Professional SA-CCR calculation engine with improved accuracy and validation
    Implements complete Basel framework with proper error handling
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.validator = TradeValidator()
        
        # Load regulatory parameters from config
        self.supervisory_factors = self._load_supervisory_factors()
        self.supervisory_correlations = self._load_correlations()
        self.collateral_haircuts = self._load_collateral_haircuts()
        
        # Calculation cache for performance
        self._calculation_cache = {}
        
    def _load_supervisory_factors(self) -> Dict:
        """Load Basel supervisory factors with proper decimal handling"""
        return {
            AssetClass.INTEREST_RATE: {
                'USD': {'<2y': 0.0050, '2-5y': 0.0050, '>5y': 0.0150},  # Decimal values
                'EUR': {'<2y': 0.0050, '2-5y': 0.0050, '>5y': 0.0150},
                'JPY': {'<2y': 0.0050, '2-5y': 0.0050, '>5y': 0.0150},
                'GBP': {'<2y': 0.0050, '2-5y': 0.0050, '>5y': 0.0150},
                'other': {'<2y': 0.0150, '2-5y': 0.0150, '>5y': 0.0150}
            },
            AssetClass.FOREIGN_EXCHANGE: {'G10': 0.04, 'emerging': 0.15},
            AssetClass.CREDIT: {
                'IG_single': 0.0046, 'HY_single': 0.0130, 
                'IG_index': 0.0038, 'HY_index': 0.0106
            },
            AssetClass.EQUITY: {
                'single_large': 0.32, 'single_small': 0.40,
                'index_developed': 0.20, 'index_emerging': 0.25
            },
            AssetClass.COMMODITY: {
                'energy': 0.18, 'metals': 0.18, 'agriculture': 0.18, 'other': 0.18
            }
        }
    
    def _load_correlations(self) -> Dict:
        """Load supervisory correlations"""
        return {
            AssetClass.INTEREST_RATE: 0.99,
            AssetClass.FOREIGN_EXCHANGE: 0.60,
            AssetClass.CREDIT: 0.50,
            AssetClass.EQUITY: 0.80,
            AssetClass.COMMODITY: 0.40
        }
    
    def _load_collateral_haircuts(self) -> Dict:
        """Load collateral haircuts (as decimals)"""
        return {
            CollateralType.CASH: 0.00,
            CollateralType.GOVERNMENT_BONDS: 0.005,
            CollateralType.CORPORATE_BONDS: 0.04,
            CollateralType.EQUITIES: 0.15,
            CollateralType.MONEY_MARKET: 0.005
        }
    
    def calculate_comprehensive_saccr(
        self, 
        portfolio_data: Dict, 
        collateral: List[Collateral] = None,
        progress_callback: Callable = None,
        **kwargs
    ) -> Dict:
        """
        Calculate complete SA-CCR with improved accuracy and validation
        
        Args:
            portfolio_data: Portfolio dictionary with trades and netting set info
            collateral: List of collateral objects
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete calculation results with all 24 steps
        """
        
        try:
            # Create netting set object
            netting_set = self._create_netting_set(portfolio_data)
            
            # Validate inputs
            validation_result = self._validate_inputs(netting_set, collateral)
            if not validation_result['valid']:
                raise ValueError(f"Input validation failed: {validation_result['errors']}")
            
            # Initialize calculation results
            calculation_steps = []
            
            # Progress tracking
            total_steps = 24
            current_step = 0
            
            def update_progress(message: str):
                nonlocal current_step
                current_step += 1
                if progress_callback:
                    progress_callback(current_step, total_steps, message)
                if 'status_text' in kwargs:
                    kwargs['status_text'].text(f"Step {current_step}/24: {message}")
                if 'progress_bar' in kwargs:
                    kwargs['progress_bar'].progress(current_step / total_steps)
            
            # Execute all 24 SA-CCR calculation steps
            
            # Step 1: Netting Set Data
            update_progress("Processing netting set data")
            step1 = self._step01_netting_set_data(netting_set)
            calculation_steps.append(step1)
            
            # Step 2: Asset Class Classification
            update_progress("Classifying asset classes")
            step2 = self._step02_asset_classification(netting_set.trades)
            calculation_steps.append(step2)
            
            # Step 3: Hedging Set
            update_progress("Creating hedging sets")
            step3 = self._step03_hedging_set(netting_set.trades)
            calculation_steps.append(step3)
            
            # Step 4: Time Parameters
            update_progress("Calculating time parameters")
            step4 = self._step04_time_parameters(netting_set.trades)
            calculation_steps.append(step4)
            
            # Step 5: Adjusted Notional
            update_progress("Computing adjusted notionals")
            step5 = self._step05_adjusted_notional(netting_set.trades)
            calculation_steps.append(step5)
            
            # Step 6: Maturity Factor
            update_progress("Applying maturity factors")
            step6 = self._step06_maturity_factor(netting_set.trades)
            calculation_steps.append(step6)
            
            # Step 7: Supervisory Delta
            update_progress("Determining supervisory deltas")
            step7 = self._step07_supervisory_delta(netting_set.trades)
            calculation_steps.append(step7)
            
            # Step 8: Supervisory Factor
            update_progress("Applying supervisory factors")
            step8 = self._step08_supervisory_factor(netting_set.trades)
            calculation_steps.append(step8)
            
            # Step 9: Adjusted Derivatives Contract Amount
            update_progress("Calculating adjusted contract amounts")
            step9 = self._step09_adjusted_derivatives_amount(netting_set.trades, step5, step6, step7, step8)
            calculation_steps.append(step9)
            
            # Step 10: Supervisory Correlation
            update_progress("Applying supervisory correlations")
            step10 = self._step10_supervisory_correlation(netting_set.trades)
            calculation_steps.append(step10)
            
            # Step 11: Hedging Set AddOn
            update_progress("Aggregating hedging set add-ons")
            step11 = self._step11_hedging_set_addon(netting_set.trades, step9, step10)
            calculation_steps.append(step11)
            
            # Step 12: Asset Class AddOn
            update_progress("Computing asset class add-ons")
            step12 = self._step12_asset_class_addon(step11)
            calculation_steps.append(step12)
            
            # Step 13: Aggregate AddOn
            update_progress("Computing aggregate add-on")
            step13 = self._step13_aggregate_addon(step12)
            calculation_steps.append(step13)
            
            # Step 14: Sum of V, C
            update_progress("Computing V and C")
            step14 = self._step14_sum_v_c(netting_set, collateral)
            calculation_steps.append(step14)
            
            # Step 15: PFE Multiplier
            update_progress("Calculating PFE multiplier")
            step15 = self._step15_pfe_multiplier(step14, step13)
            calculation_steps.append(step15)
            
            # Step 16: PFE
            update_progress("Computing PFE")
            step16 = self._step16_pfe(step15, step13)
            calculation_steps.append(step16)
            
            # Step 17: TH, MTA, NICA
            update_progress("Processing collateral parameters")
            step17 = self._step17_th_mta_nica(netting_set)
            calculation_steps.append(step17)
            
            # Step 18: RC
            update_progress("Calculating replacement cost")
            step18 = self._step18_replacement_cost(step14, step17)
            calculation_steps.append(step18)
            
            # Step 19: CEU Flag
            update_progress("Determining central clearing status")
            step19 = self._step19_ceu_flag(netting_set.trades)
            calculation_steps.append(step19)
            
            # Step 20: Alpha
            update_progress("Applying alpha multiplier")
            step20 = self._step20_alpha(step19)
            calculation_steps.append(step20)
            
            # Step 21: EAD
            update_progress("Computing EAD")
            step21 = self._step21_ead(step20, step18, step16)
            calculation_steps.append(step21)
            
            # Step 22: Counterparty Information
            update_progress("Processing counterparty data")
            step22 = self._step22_counterparty_info(netting_set.counterparty)
            calculation_steps.append(step22)
            
            # Step 23: Risk Weight
            update_progress("Determining risk weight")
            step23 = self._step23_risk_weight(step22)
            calculation_steps.append(step23)
            
            # Step 24: RWA Calculation
            update_progress("Computing RWA")
            step24 = self._step24_rwa_calculation(step21, step23)
            calculation_steps.append(step24)
            
            # Compile final results
            final_results = {
                'replacement_cost': step18['rc'],
                'potential_future_exposure': step16['pfe'],
                'exposure_at_default': step21['ead'],
                'risk_weighted_assets': step24['rwa'],
                'capital_requirement': step24['rwa'] * 0.08,
                'calculation_date': datetime.now().isoformat(),
                'portfolio_summary': {
                    'trade_count': len(netting_set.trades),
                    'total_notional': sum(abs(t.notional) for t in netting_set.trades),
                    'counterparty': netting_set.counterparty,
                    'netting_set_id': netting_set.netting_set_id
                }
            }
            
            return {
                'calculation_steps': calculation_steps,
                'final_results': final_results,
                'validation_results': validation_result,
                'calculation_metadata': {
                    'engine_version': '2.0',
                    'calculation_time': datetime.now().isoformat(),
                    'total_steps': total_steps
                }
            }
            
        except Exception as e:
            logger.error(f"SA-CCR calculation error: {e}")
            raise
    
    def _create_netting_set(self, portfolio_data: Dict) -> NettingSet:
        """Create NettingSet object from portfolio data"""
        
        trades = []
        for trade_data in portfolio_data.get('trades', []):
            if isinstance(trade_data, Trade):
                trades.append(trade_data)
            else:
                # Convert dict to Trade object if needed
                trade = Trade(
                    trade_id=trade_data['trade_id'],
                    counterparty=trade_data['counterparty'],
                    asset_class=AssetClass(trade_data['asset_class']),
                    trade_type=TradeType(trade_data['trade_type']),
                    notional=trade_data['notional'],
                    currency=trade_data['currency'],
                    underlying=trade_data['underlying'],
                    maturity_date=trade_data['maturity_date'],
                    mtm_value=trade_data.get('mtm_value', 0.0),
                    delta=trade_data.get('delta', 1.0)
                )
                trades.append(trade)
        
        return NettingSet(
            netting_set_id=portfolio_data['netting_set_id'],
            counterparty=portfolio_data['counterparty'],
            trades=trades,
            threshold=portfolio_data.get('threshold', 0.0),
            mta=portfolio_data.get('mta', 0.0),
            nica=portfolio_data.get('nica', 0.0)
        )
    
    def _validate_inputs(self, netting_set: NettingSet, collateral: List[Collateral] = None) -> Dict:
        """Comprehensive input validation"""
        
        errors = []
        warnings = []
        
        # Validate netting set
        if not netting_set.netting_set_id:
            errors.append("Missing netting set ID")
        
        if not netting_set.counterparty:
            errors.append("Missing counterparty name")
        
        if not netting_set.trades:
            errors.append("No trades in portfolio")
        
        # Validate trades
        for i, trade in enumerate(netting_set.trades):
            trade_validation = self.validator.validate_trade_data({
                'trade_id': trade.trade_id,
                'notional': trade.notional,
                'currency': trade.currency,
                'maturity_years': trade.time_to_maturity()
            })
            
            if not trade_validation['valid']:
                errors.append(f"Trade {i+1} ({trade.trade_id}): {trade_validation['message']}")
            
            # Additional validations
            if trade.notional <= 0:
                errors.append(f"Trade {i+1}: Notional must be positive")
            
            if trade.maturity_date <= datetime.now():
                errors.append(f"Trade {i+1}: Maturity date must be in future")
            
            if abs(trade.delta) > 1:
                warnings.append(f"Trade {i+1}: Delta {trade.delta} is outside normal range [-1, 1]")
        
        # Validate collateral
        if collateral:
            for i, coll in enumerate(collateral):
                if coll.amount < 0:
                    errors.append(f"Collateral {i+1}: Amount cannot be negative")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'trade_count': len(netting_set.trades),
            'validation_time': datetime.now().isoformat()
        }
    
    # Individual SA-CCR calculation steps with improved accuracy
    
    def _step01_netting_set_data(self, netting_set: NettingSet) -> Dict:
        """Step 1: Netting Set Data Collection"""
        
        total_notional = sum(abs(trade.notional) for trade in netting_set.trades)
        total_mtm = sum(trade.mtm_value for trade in netting_set.trades)
        
        return {
            'step': 1,
            'title': 'Netting Set Data',
            'description': 'Source and validate netting set data',
            'formula': 'Data validation and aggregation',
            'result': f"Netting Set: {netting_set.netting_set_id}, Trades: {len(netting_set.trades)}",
            'data': {
                'netting_set_id': netting_set.netting_set_id,
                'counterparty': netting_set.counterparty,
                'trade_count': len(netting_set.trades),
                'total_notional': total_notional,
                'total_mtm': total_mtm
            }
        }
    
    def _step02_asset_classification(self, trades: List[Trade]) -> Dict:
        """Step 2: Asset Class Classification"""
        
        classifications = []
        asset_class_summary = {}
        
        for trade in trades:
            asset_class = trade.asset_class
            classifications.append({
                'trade_id': trade.trade_id,
                'asset_class': asset_class.value,
                'trade_type': trade.trade_type.value,
                'basis_flag': getattr(trade, 'basis_flag', False),
                'volatility_flag': getattr(trade, 'volatility_flag', False)
            })
            
            # Count by asset class
            if asset_class not in asset_class_summary:
                asset_class_summary[asset_class] = 0
            asset_class_summary[asset_class] += 1
        
        return {
            'step': 2,
            'title': 'Asset Class Classification',
            'description': 'Classify trades by Basel asset classes and sub-classes',
            'formula': 'Classification per Basel regulatory mapping',
            'result': f"Classified {len(trades)} trades across {len(asset_class_summary)} asset classes",
            'data': {
                'trade_classifications': classifications,
                'asset_class_summary': {ac.value: count for ac, count in asset_class_summary.items()}
            }
        }
    
    def _step03_hedging_set(self, trades: List[Trade]) -> Dict:
        """Step 3: Hedging Set Definition"""
        
        hedging_sets = {}
        
        for trade in trades:
            # Hedging set key: asset class + currency + underlying type
            hedging_set_key = f"{trade.asset_class.value}_{trade.currency}"
            
            if hedging_set_key not in hedging_sets:
                hedging_sets[hedging_set_key] = {
                    'trades': [],
                    'total_notional': 0
                }
            
            hedging_sets[hedging_set_key]['trades'].append(trade.trade_id)
            hedging_sets[hedging_set_key]['total_notional'] += abs(trade.notional)
        
        return {
            'step': 3,
            'title': 'Hedging Set Definition',
            'description': 'Group trades into hedging sets based on risk factors',
            'formula': 'Hedging sets = {Asset Class, Currency, Underlying}',
            'result': f"Created {len(hedging_sets)} hedging sets",
            'data': {
                'hedging_sets': {k: {
                    'trade_count': len(v['trades']),
                    'total_notional': v['total_notional']
                } for k, v in hedging_sets.items()}
            }
        }
    
    def _step04_time_parameters(self, trades: List[Trade]) -> Dict:
        """Step 4: Time Parameters (S, E, M)"""
        
        settlement_date = datetime.now()
        time_params = []
        
        for trade in trades:
            end_date = trade.maturity_date
            remaining_maturity = trade.time_to_maturity()
            
            time_params.append({
                'trade_id': trade.trade_id,
                'settlement_date': settlement_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'remaining_maturity_years': remaining_maturity,
                'remaining_maturity_days': (end_date - settlement_date).days
            })
        
        avg_maturity = sum(tp['remaining_maturity_years'] for tp in time_params) / len(time_params)
        
        return {
            'step': 4,
            'title': 'Time Parameters (S, E, M)',
            'description': 'Calculate settlement date, end date, and maturity',
            'formula': 'S = Settlement, E = End Date, M = (E - S) / 365.25',
            'result': f"Average maturity: {avg_maturity:.2f} years",
            'data': {
                'time_parameters': time_params,
                'average_maturity': avg_maturity
            }
        }
    
    def _step05_adjusted_notional(self, trades: List[Trade]) -> Dict:
        """Step 5: Adjusted Notional Calculation"""
        
        adjusted_notionals = []
        total_adjusted = 0
        
        for trade in trades:
            # For most products, adjusted notional equals trade notional
            # For some products like CDS, adjustments may apply
            adjusted_notional = abs(trade.notional)
            
            adjusted_notionals.append({
                'trade_id': trade.trade_id,
                'original_notional': trade.notional,
                'adjusted_notional': adjusted_notional,
                'adjustment_factor': 1.0  # Most trades have no adjustment
            })
            
            total_adjusted += adjusted_notional
        
        return {
            'step': 5,
            'title': 'Adjusted Notional',
            'description': 'Calculate trade-level adjusted notional amounts',
            'formula': 'Adjusted Notional = Trade Notional × Adjustment Factor',
            'result': f"Total adjusted notional: ${total_adjusted:,.0f}",
            'data': {
                'adjusted_notionals': adjusted_notionals,
                'total_adjusted_notional': total_adjusted
            }
        }
    
    def _step06_maturity_factor(self, trades: List[Trade]) -> Dict:
        """Step 6: Maturity Factor Calculation"""
        
        maturity_factors = []
        
        for trade in trades:
            remaining_maturity = max(trade.time_to_maturity(), 0.01)  # Minimum 0.01 years
            
            # Basel maturity factor formula
            mf = min(1.0, 0.05 + 0.95 * math.exp(-0.05 * max(1.0, remaining_maturity)))
            
            maturity_factors.append({
                'trade_id': trade.trade_id,
                'remaining_maturity': remaining_maturity,
                'maturity_factor': mf,
                'formula_components': {
                    'exp_term': math.exp(-0.05 * max(1.0, remaining_maturity)),
                    'before_min': 0.05 + 0.95 * math.exp(-0.05 * max(1.0, remaining_maturity))
                }
            })
        
        avg_mf = sum(mf['maturity_factor'] for mf in maturity_factors) / len(maturity_factors)
        
        return {
            'step': 6,
            'title': 'Maturity Factor (MF)',
            'description': 'Apply Basel maturity factor formula',
            'formula': 'MF = min(1, 0.05 + 0.95 × exp(-0.05 × max(1, M)))',
            'result': f"Average maturity factor: {avg_mf:.4f}",
            'data': {
                'maturity_factors': maturity_factors,
                'average_maturity_factor': avg_mf
            }
        }
    
    def _step07_supervisory_delta(self, trades: List[Trade]) -> Dict:
        """Step 7: Supervisory Delta Determination"""
        
        supervisory_deltas = []
        
        for trade in trades:
            if trade.trade_type in [TradeType.OPTION, TradeType.SWAPTION]:
                # For options, use provided delta or calculate if needed
                delta = trade.delta
                delta_method = "Provided"
            else:
                # Linear products have delta = +/- 1
                if trade.notional >= 0:
                    delta = 1.0
                else:
                    delta = -1.0
                delta_method = "Linear Product"
            
            supervisory_deltas.append({
                'trade_id': trade.trade_id,
                'trade_type': trade.trade_type.value,
                'supervisory_delta': delta,
                'delta_method': delta_method,
                'abs_delta': abs(delta)
            })
        
        return {
            'step': 7,
            'title': 'Supervisory Delta',
            'description': 'Determine supervisory delta per trade type',
            'formula': 'δ = trade delta for options, ±1 for linear products',
            'result': f"Deltas calculated for {len(trades)} trades",
            'data': {
                'supervisory_deltas': supervisory_deltas
            }
        }
    
    def _step08_supervisory_factor(self, trades: List[Trade]) -> Dict:
        """Step 8: Supervisory Factor Application"""
        
        supervisory_factors = []
        
        for trade in trades:
            sf_decimal = self._get_supervisory_factor_decimal(trade)
            sf_bp = sf_decimal * 10000  # Convert to basis points for display
            
            supervisory_factors.append({
                'trade_id': trade.trade_id,
                'asset_class': trade.asset_class.value,
                'currency': trade.currency,
                'maturity_bucket': self._get_maturity_bucket(trade),
                'supervisory_factor_decimal': sf_decimal,
                'supervisory_factor_bp': sf_bp
            })
        
        return {
            'step': 8,
            'title': 'Supervisory Factor (SF)',
            'description': 'Apply regulatory supervisory factors by asset class',
            'formula': 'SF per Basel regulatory mapping tables',
            'result': f"Applied supervisory factors for {len(trades)} trades",
            'data': {
                'supervisory_factors': supervisory_factors
            }
        }
    
    def _get_supervisory_factor_decimal(self, trade: Trade) -> float:
        """Get supervisory factor as decimal (not basis points)"""
        
        if trade.asset_class == AssetClass.INTEREST_RATE:
            maturity = trade.time_to_maturity()
            currency_group = trade.currency if trade.currency in ['USD', 'EUR', 'JPY', 'GBP'] else 'other'
            
            if maturity < 2:
                return self.supervisory_factors[AssetClass.INTEREST_RATE][currency_group]['<2y']
            elif maturity <= 5:
                return self.supervisory_factors[AssetClass.INTEREST_RATE][currency_group]['2-5y']
            else:
                return self.supervisory_factors[AssetClass.INTEREST_RATE][currency_group]['>5y']
        
        elif trade.asset_class == AssetClass.FOREIGN_EXCHANGE:
            g10_currencies = ['USD', 'EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD', 'SEK', 'NOK']
            is_g10 = trade.currency in g10_currencies
            return self.supervisory_factors[AssetClass.FOREIGN_EXCHANGE]['G10' if is_g10 else 'emerging']
        
        elif trade.asset_class == AssetClass.CREDIT:
            return self.supervisory_factors[AssetClass.CREDIT]['IG_single']  # Default to IG single
        
        elif trade.asset_class == AssetClass.EQUITY:
            return self.supervisory_factors[AssetClass.EQUITY]['single_large']  # Default to large cap
        
        elif trade.asset_class == AssetClass.COMMODITY:
            return self.supervisory_factors[AssetClass.COMMODITY]['energy']  # Default to energy
        
        return 0.01  # Default fallback
    
    def _get_maturity_bucket(self, trade: Trade) -> str:
        """Get maturity bucket for display"""
        maturity = trade.time_to_maturity()
        
        if maturity < 2:
            return '<2Y'
        elif maturity <= 5:
            return '2-5Y'
        else:
            return '>5Y'
    
    def _step09_adjusted_derivatives_amount(self, trades: List[Trade], step5: Dict, 
                                          step6: Dict, step7: Dict, step8: Dict) -> Dict:
        """Step 9: Adjusted Derivatives Contract Amount"""
        
        # Extract data from previous steps
        adjusted_notionals = {item['trade_id']: item['adjusted_notional'] for item in step5['data']['adjusted_notionals']}
        maturity_factors = {item['trade_id']: item['maturity_factor'] for item in step6['data']['maturity_factors']}
        supervisory_deltas = {item['trade_id']: item['abs_delta'] for item in step7['data']['supervisory_deltas']}
        supervisory_factors = {item['trade_id']: item['supervisory_factor_decimal'] for item in step8['data']['supervisory_factors']}
        
        adjusted_amounts = []
        total_adjusted_amount = 0
        
        for trade in trades:
            trade_id = trade.trade_id
            
            # Get components
            adj_notional = adjusted_notionals.get(trade_id, abs(trade.notional))
            mf = maturity_factors.get(trade_id, 1.0)
            delta = supervisory_deltas.get(trade_id, 1.0)
            sf = supervisory_factors.get(trade_id, 0.01)
            
            # Calculate adjusted amount
            adjusted_amount = adj_notional * delta * mf * sf
            total_adjusted_amount += adjusted_amount
            
            adjusted_amounts.append({
                'trade_id': trade_id,
                'adjusted_notional': adj_notional,
                'supervisory_delta': delta,
                'maturity_factor': mf,
                'supervisory_factor': sf,
                'adjusted_amount': adjusted_amount
            })
        
        return {
            'step': 9,
            'title': 'Adjusted Derivatives Contract Amount',
            'description': 'Calculate final risk-adjusted contract amounts',
            'formula': 'Adjusted Amount = Adj_Notional × |δ| × MF × SF',
            'result': f"Total adjusted amount: ${total_adjusted_amount:,.0f}",
            'data': {
                'adjusted_amounts': adjusted_amounts,
                'total_adjusted_amount': total_adjusted_amount
            }
        }
    
    def _step10_supervisory_correlation(self, trades: List[Trade]) -> Dict:
        """Step 10: Supervisory Correlation"""
        
        correlations = []
        asset_classes = set(trade.asset_class for trade in trades)
        
        for asset_class in asset_classes:
            correlation = self.supervisory_correlations.get(asset_class, 0.5)
            correlations.append({
                'asset_class': asset_class.value,
                'supervisory_correlation': correlation
            })
        
        return {
            'step': 10,
            'title': 'Supervisory Correlation',
            'description': 'Apply supervisory correlations by asset class',
            'formula': 'ρ per Basel regulatory mapping tables',
            'result': f"Applied correlations for {len(asset_classes)} asset classes",
            'data': {
                'correlations': correlations
            }
        }
    
    def _step11_hedging_set_addon(self, trades: List[Trade], step9: Dict, step10: Dict) -> Dict:
        """Step 11: Hedging Set AddOn"""
        
        # Get adjusted amounts by trade
        adjusted_amounts = {item['trade_id']: item['adjusted_amount'] for item in step9['data']['adjusted_amounts']}
        
        # Get correlations by asset class
        correlations = {item['asset_class']: item['supervisory_correlation'] for item in step10['data']['correlations']}
        
        # Group trades by hedging set
        hedging_sets = {}
        for trade in trades:
            hedging_set_key = f"{trade.asset_class.value}_{trade.currency}"
            
            if hedging_set_key not in hedging_sets:
                hedging_sets[hedging_set_key] = {
                    'trades': [],
                    'asset_class': trade.asset_class.value,
                    'currency': trade.currency
                }
            
            hedging_sets[hedging_set_key]['trades'].append(trade)
        
        # Calculate hedging set add-ons
        hedging_set_addons = []
        total_hedging_set_addon = 0
        
        for hedging_set_key, hs_data in hedging_sets.items():
            hs_trades = hs_data['trades']
            asset_class = hs_data['asset_class']
            
            # Sum of adjusted amounts in this hedging set
            hs_adjusted_amounts = []
            for trade in hs_trades:
                amount = adjusted_amounts.get(trade.trade_id, 0)
                hs_adjusted_amounts.append(amount)
            
            sum_adjusted_amounts = sum(hs_adjusted_amounts)
            
            # Apply correlation
            correlation = correlations.get(asset_class, 0.5)
            hedging_set_addon = sum_adjusted_amounts * math.sqrt(correlation)
            
            total_hedging_set_addon += hedging_set_addon
            
            hedging_set_addons.append({
                'hedging_set': hedging_set_key,
                'asset_class': asset_class,
                'trade_count': len(hs_trades),
                'individual_amounts': hs_adjusted_amounts,
                'sum_amounts': sum_adjusted_amounts,
                'correlation': correlation,
                'hedging_set_addon': hedging_set_addon
            })
        
        return {
            'step': 11,
            'title': 'Hedging Set AddOn',
            'description': 'Aggregate trade add-ons within hedging sets',
            'formula': 'HS_AddOn = Σ(Trade_AddOns) × √ρ',
            'result': f"Total hedging set add-on: ${total_hedging_set_addon:,.0f}",
            'data': {
                'hedging_set_addons': hedging_set_addons,
                'total_hedging_set_addon': total_hedging_set_addon
            }
        }
    
    def _step12_asset_class_addon(self, step11: Dict) -> Dict:
        """Step 12: Asset Class AddOn"""
        
        hedging_set_addons = step11['data']['hedging_set_addons']
        
        # Group by asset class
        asset_class_addons = {}
        for hs_addon in hedging_set_addons:
            asset_class = hs_addon['asset_class']
            if asset_class not in asset_class_addons:
                asset_class_addons[asset_class] = []
            asset_class_addons[asset_class].append(hs_addon['hedging_set_addon'])
        
        # Sum within each asset class
        asset_class_results = []
        total_asset_class_addon = 0
        
        for asset_class, hs_addons in asset_class_addons.items():
            asset_class_addon = sum(hs_addons)
            total_asset_class_addon += asset_class_addon
            
            asset_class_results.append({
                'asset_class': asset_class,
                'hedging_set_count': len(hs_addons),
                'hedging_set_addons': hs_addons,
                'asset_class_addon': asset_class_addon
            })
        
        return {
            'step': 12,
            'title': 'Asset Class AddOn',
            'description': 'Sum hedging set add-ons by asset class',
            'formula': 'AC_AddOn = Σ(HS_AddOns within asset class)',
            'result': f"Total asset class add-on: ${total_asset_class_addon:,.0f}",
            'data': {
                'asset_class_addons': asset_class_results,
                'total_asset_class_addon': total_asset_class_addon
            }
        }
    
    def _step13_aggregate_addon(self, step12: Dict) -> Dict:
        """Step 13: Aggregate AddOn"""
        
        asset_class_addons = step12['data']['asset_class_addons']
        
        aggregate_addon = sum(ac['asset_class_addon'] for ac in asset_class_addons)
        
        return {
            'step': 13,
            'title': 'Aggregate AddOn',
            'description': 'Sum all asset class add-ons',
            'formula': 'Aggregate_AddOn = Σ(AC_AddOns)',
            'result': f"Aggregate AddOn: ${aggregate_addon:,.0f}",
            'data': {
                'asset_class_breakdown': [(ac['asset_class'], ac['asset_class_addon']) for ac in asset_class_addons],
                'aggregate_addon': aggregate_addon
            },
            'aggregate_addon': aggregate_addon  # For easy access
        }
    
    def _step14_sum_v_c(self, netting_set: NettingSet, collateral: List[Collateral] = None) -> Dict:
        """Step 14: Sum of V, C within netting set"""
        
        # Sum of mark-to-market values (V)
        sum_v = sum(trade.mtm_value for trade in netting_set.trades)
        
        # Sum of effective collateral (C)
        sum_c = 0
        collateral_details = []
        
        if collateral:
            for coll in collateral:
                haircut_rate = self.collateral_haircuts.get(coll.collateral_type, 0.15)
                effective_value = coll.amount * (1 - haircut_rate)
                sum_c += effective_value
                
                collateral_details.append({
                    'type': coll.collateral_type.value,
                    'amount': coll.amount,
                    'haircut_rate': haircut_rate,
                    'effective_value': effective_value
                })
        
        return {
            'step': 14,
            'title': 'Sum of V, C within netting set',
            'description': 'Calculate net MTM exposure and effective collateral',
            'formula': 'V = Σ(MTM_values), C = Σ(Collateral × (1 - haircut))',
            'result': f"V = ${sum_v:,.0f}, C = ${sum_c:,.0f}",
            'data': {
                'sum_v': sum_v,
                'sum_c': sum_c,
                'net_exposure': sum_v - sum_c,
                'collateral_details': collateral_details
            },
            'sum_v': sum_v,
            'sum_c': sum_c
        }
    
    def _step15_pfe_multiplier(self, step14: Dict, step13: Dict) -> Dict:
        """Step 15: PFE Multiplier"""
        
        sum_v = step14['sum_v']
        aggregate_addon = step13['aggregate_addon']
        
        if aggregate_addon > 0:
            ratio = max(0, sum_v) / aggregate_addon
            exp_term = math.exp(-0.05 * ratio)
            multiplier = min(1.0, 0.05 + 0.95 * exp_term)
        else:
            multiplier = 1.0
            ratio = 0
            exp_term = 1.0
        
        return {
            'step': 15,
            'title': 'PFE Multiplier',
            'description': 'Calculate PFE multiplier based on netting benefit',
            'formula': 'Multiplier = min(1, 0.05 + 0.95 × exp(-0.05 × max(0, V) / AddOn))',
            'result': f"PFE Multiplier: {multiplier:.6f}",
            'data': {
                'sum_v': sum_v,
                'aggregate_addon': aggregate_addon,
                'ratio': ratio,
                'exp_term': exp_term,
                'multiplier': multiplier
            },
            'multiplier': multiplier
        }
    
    def _step16_pfe(self, step15: Dict, step13: Dict) -> Dict:
        """Step 16: PFE (Potential Future Exposure)"""
        
        multiplier = step15['multiplier']
        aggregate_addon = step13['aggregate_addon']
        
        pfe = multiplier * aggregate_addon
        
        return {
            'step': 16,
            'title': 'PFE (Potential Future Exposure)',
            'description': 'Calculate final PFE using multiplier and aggregate add-on',
            'formula': 'PFE = Multiplier × Aggregate_AddOn',
            'result': f"PFE: ${pfe:,.0f}",
            'data': {
                'multiplier': multiplier,
                'aggregate_addon': aggregate_addon,
                'pfe': pfe
            },
            'pfe': pfe
        }
    
    def _step17_th_mta_nica(self, netting_set: NettingSet) -> Dict:
        """Step 17: TH, MTA, NICA"""
        
        return {
            'step': 17,
            'title': 'TH, MTA, NICA',
            'description': 'Extract collateral parameters from netting agreement',
            'formula': 'Sourced from CSA/ISDA master agreements',
            'result': f"TH: ${netting_set.threshold:,.0f}, MTA: ${netting_set.mta:,.0f}, NICA: ${netting_set.nica:,.0f}",
            'data': {
                'threshold': netting_set.threshold,
                'mta': netting_set.mta,
                'nica': netting_set.nica
            },
            'threshold': netting_set.threshold,
            'mta': netting_set.mta,
            'nica': netting_set.nica
        }
    
    def _step18_replacement_cost(self, step14: Dict, step17: Dict) -> Dict:
        """Step 18: RC (Replacement Cost)"""
        
        sum_v = step14['sum_v']
        sum_c = step14['sum_c']
        threshold = step17['threshold']
        mta = step17['mta']
        nica = step17['nica']
        
        # Determine if margined or unmargined
        is_margined = threshold > 0 or mta > 0
        
        if is_margined:
            # Margined formula: RC = max(V - C, TH + MTA - NICA, 0)
            rc = max(sum_v - sum_c, threshold + mta - nica, 0)
            formula_used = "Margined: max(V - C, TH + MTA - NICA, 0)"
        else:
            # Unmargined formula: RC = max(V - C, 0)
            rc = max(sum_v - sum_c, 0)
            formula_used = "Unmargined: max(V - C, 0)"
        
        return {
            'step': 18,
            'title': 'RC (Replacement Cost)',
            'description': 'Calculate replacement cost with netting and collateral benefits',
            'formula': formula_used,
            'result': f"RC: ${rc:,.0f}",
            'data': {
                'sum_v': sum_v,
                'sum_c': sum_c,
                'net_exposure': sum_v - sum_c,
                'threshold': threshold,
                'mta': mta,
                'nica': nica,
                'is_margined': is_margined,
                'rc': rc
            },
            'rc': rc
        }
    
    def _step19_ceu_flag(self, trades: List[Trade]) -> Dict:
        """Step 19: CEU Flag (Central Clearing Status)"""
        
        ceu_flags = []
        overall_ceu = 0  # Default to centrally cleared
        
        for trade in trades:
            # Check if trade is centrally cleared
            trade_ceu = getattr(trade, 'ceu_flag', 0)  # 0 = centrally cleared, 1 = bilateral
            ceu_flags.append({
                'trade_id': trade.trade_id,
                'ceu_flag': trade_ceu
            })
            
            # If any trade is bilateral, entire netting set is bilateral
            if trade_ceu == 1:
                overall_ceu = 1
        
        clearing_status = "Bilateral (Non-centrally cleared)" if overall_ceu == 1 else "Centrally Cleared"
        
        return {
            'step': 19,
            'title': 'CEU Flag (Central Clearing Status)',
            'description': 'Determine central clearing eligibility',
            'formula': 'CEU = 1 if any trade is bilateral, 0 if all centrally cleared',
            'result': f"Status: {clearing_status} (CEU = {overall_ceu})",
            'data': {
                'trade_ceu_flags': ceu_flags,
                'overall_ceu_flag': overall_ceu,
                'clearing_status': clearing_status
            },
            'ceu_flag': overall_ceu
        }
    
    def _step20_alpha(self, step19: Dict) -> Dict:
        """Step 20: Alpha Multiplier"""
        
        ceu_flag = step19['ceu_flag']
        alpha = 1.4 if ceu_flag == 1 else 0.5
        
        return {
            'step': 20,
            'title': 'Alpha Multiplier',
            'description': 'Regulatory multiplier based on central clearing status',
            'formula': 'Alpha = 1.4 if bilateral (CEU=1), 0.5 if centrally cleared (CEU=0)',
            'result': f"Alpha: {alpha}",
            'data': {
                'ceu_flag': ceu_flag,
                'alpha': alpha
            },
            'alpha': alpha
        }
    
    def _step21_ead(self, step20: Dict, step18: Dict, step16: Dict) -> Dict:
        """Step 21: EAD (Exposure at Default)"""
        
        alpha = step20['alpha']
        rc = step18['rc']
        pfe = step16['pfe']
        
        ead = alpha * (rc + pfe)
        
        return {
            'step': 21,
            'title': 'EAD (Exposure at Default)',
            'description': 'Calculate final regulatory exposure at default',
            'formula': 'EAD = Alpha × (RC + PFE)',
            'result': f"EAD: ${ead:,.0f}",
            'data': {
                'alpha': alpha,
                'rc': rc,
                'pfe': pfe,
                'ead': ead
            },
            'ead': ead
        }
    
    def _step22_counterparty_info(self, counterparty: str) -> Dict:
        """Step 22: Counterparty Information"""
        
        # In a real system, this would lookup counterparty details from a database
        counterparty_data = {
            'counterparty_name': counterparty,
            'legal_entity_identifier': 'TBD',
            'jurisdiction': 'US',
            'counterparty_type': 'Corporate',  # Default classification
            'credit_rating': 'BBB',  # Default rating
            'sector': 'Financial Services'
        }
        
        return {
            'step': 22,
            'title': 'Counterparty Information',
            'description': 'Source counterparty classification and rating data',
            'formula': 'Data sourced from counterparty reference data',
            'result': f"Counterparty: {counterparty}, Type: {counterparty_data['counterparty_type']}",
            'data': counterparty_data,
            'counterparty_type': counterparty_data['counterparty_type']
        }
    
    def _step23_risk_weight(self, step22: Dict) -> Dict:
        """Step 23: Standardized Risk Weight"""
        
        counterparty_type = step22['counterparty_type']
        
        # Basel standardized risk weights
        risk_weight_mapping = {
            'Corporate': 1.00,
            'Bank': 0.20,
            'Sovereign': 0.00,
            'Central Bank': 0.00,
            'Multilateral Development Bank': 0.00,
            'Public Sector Entity': 0.50
        }
        
        risk_weight = risk_weight_mapping.get(counterparty_type, 1.00)  # Default to 100%
        
        return {
            'step': 23,
            'title': 'Standardized Risk Weight',
            'description': 'Apply Basel standardized risk weight by counterparty type',
            'formula': 'Risk Weight per Basel standardized approach mapping',
            'result': f"Risk Weight: {risk_weight * 100:.0f}%",
            'data': {
                'counterparty_type': counterparty_type,
                'risk_weight_decimal': risk_weight,
                'risk_weight_percent': risk_weight * 100
            },
            'risk_weight': risk_weight
        }
    
    def _step24_rwa_calculation(self, step21: Dict, step23: Dict) -> Dict:
        """Step 24: RWA (Risk Weighted Assets) Calculation"""
        
        ead = step21['ead']
        risk_weight = step23['risk_weight']
        
        rwa = ead * risk_weight
        capital_requirement = rwa * 0.08  # 8% capital requirement
        
        return {
            'step': 24,
            'title': 'RWA Calculation',
            'description': 'Calculate final Risk Weighted Assets',
            'formula': 'RWA = EAD × Risk_Weight',
            'result': f"RWA: ${rwa:,.0f}, Capital Required: ${capital_requirement:,.0f}",
            'data': {
                'ead': ead,
                'risk_weight': risk_weight,
                'rwa': rwa,
                'capital_requirement': capital_requirement,
                'capital_ratio': 0.08
            },
            'rwa': rwa,
            'capital_requirement': capital_requirement
        }
