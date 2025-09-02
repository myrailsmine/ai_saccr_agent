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

# src/utils/progress_tracker.py

import time
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ProgressStep:
    """Individual progress step"""
    step_id: str
    description: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = 'pending'  # pending, in_progress, completed, error
    error_message: Optional[str] = None

class ProgressTracker:
    """Professional progress tracking for long-running calculations"""
    
    def __init__(self):
        self.steps: List[ProgressStep] = []
        self.current_step: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.callbacks: List[Callable] = []
    
    def reset(self):
        """Reset progress tracker for new calculation"""
        self.steps.clear()
        self.current_step = None
        self.start_time = None
        self.end_time = None
    
    def add_step(self, step_id: str, description: str):
        """Add a step to track"""
        step = ProgressStep(step_id=step_id, description=description)
        self.steps.append(step)
    
    def start_calculation(self):
        """Mark calculation start"""
        self.start_time = datetime.now()
        logger.info("SA-CCR calculation started")
    
    def start_step(self, step_id: str):
        """Mark step as started"""
        self.current_step = step_id
        
        for step in self.steps:
            if step.step_id == step_id:
                step.status = 'in_progress'
                step.start_time = datetime.now()
                break
        
        self._notify_callbacks('step_started', step_id)
    
    def complete_step(self, step_id: str):
        """Mark step as completed"""
        for step in self.steps:
            if step.step_id == step_id:
                step.status = 'completed'
                step.end_time = datetime.now()
                break
        
        self._notify_callbacks('step_completed', step_id)
    
    def error_step(self, step_id: str, error_message: str):
        """Mark step as error"""
        for step in self.steps:
            if step.step_id == step_id:
                step.status = 'error'
                step.end_time = datetime.now()
                step.error_message = error_message
                break
        
        self._notify_callbacks('step_error', step_id, error_message)
    
    def complete_calculation(self):
        """Mark calculation as completed"""
        self.end_time = datetime.now()
        
        if self.start_time:
            duration = self.end_time - self.start_time
            logger.info(f"SA-CCR calculation completed in {duration.total_seconds():.2f} seconds")
        
        self._notify_callbacks('calculation_completed')
    
    def get_progress_percentage(self) -> float:
        """Get overall progress percentage"""
        if not self.steps:
            return 0.0
        
        completed_steps = sum(1 for step in self.steps if step.status == 'completed')
        return (completed_steps / len(self.steps)) * 100
    
    def get_current_status(self) -> Dict[str, any]:
        """Get current progress status"""
        return {
            'total_steps': len(self.steps),
            'completed_steps': sum(1 for step in self.steps if step.status == 'completed'),
            'current_step': self.current_step,
            'progress_percentage': self.get_progress_percentage(),
            'start_time': self.start_time,
            'estimated_completion': self._estimate_completion_time(),
            'steps': [
                {
                    'step_id': step.step_id,
                    'description': step.description,
                    'status': step.status,
                    'duration': self._get_step_duration(step)
                }
                for step in self.steps
            ]
        }
    
    def add_callback(self, callback: Callable):
        """Add progress callback"""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, event_type: str, *args):
        """Notify all callbacks of progress events"""
        for callback in self.callbacks:
            try:
                callback(event_type, *args)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def _get_step_duration(self, step: ProgressStep) -> Optional[float]:
        """Get step duration in seconds"""
        if step.start_time and step.end_time:
            return (step.end_time - step.start_time).total_seconds()
        elif step.start_time and step.status == 'in_progress':
            return (datetime.now() - step.start_time).total_seconds()
        return None
    
    def _estimate_completion_time(self) -> Optional[datetime]:
        """Estimate calculation completion time"""
        if not self.start_time or not self.steps:
            return None
        
        completed_steps = [step for step in self.steps if step.status == 'completed']
        if not completed_steps:
            return None
        
        # Calculate average time per completed step
        total_completed_time = sum(
            (step.end_time - step.start_time).total_seconds()
            for step in completed_steps
            if step.start_time and step.end_time
        )
        
        if total_completed_time == 0:
            return None
        
        avg_time_per_step = total_completed_time / len(completed_steps)
        remaining_steps = len(self.steps) - len(completed_steps)
        
        estimated_remaining_time = remaining_steps * avg_time_per_step
        
        return datetime.now() + timedelta(seconds=estimated_remaining_time)

class PerformanceMonitor:
    """Monitor calculation performance and resource usage"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.metrics = {
            'memory_usage': [],
            'cpu_usage': [],
            'calculation_times': {}
        }
    
    def record_step_time(self, step_name: str, execution_time: float):
        """Record execution time for a step"""
        self.metrics['calculation_times'][step_name] = execution_time
    
    def get_performance_summary(self) -> Dict[str, any]:
        """Get performance summary"""
        if not self.start_time:
            return {}
        
        total_time = time.time() - self.start_time
        
        return {
            'total_execution_time': total_time,
            'step_times': self.metrics.get('calculation_times', {}),
            'slowest_steps': sorted(
                self.metrics.get('calculation_times', {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

# src/ui/components.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional

class UIComponents:
    """Reusable UI components for the SA-CCR application"""
    
    def __init__(self):
        pass
    
    def render_metric_card(self, title: str, value: str, delta: Optional[str] = None, 
                          delta_color: str = "normal"):
        """Render a professional metric card"""
        
        delta_html = ""
        if delta:
            color = "green" if delta_color == "normal" else "red"
            delta_html = f'<div style="color: {color}; font-size: 0.8rem; margin-top: 0.25rem;">{delta}</div>'
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="result-label">{title}</div>
            <div class="result-value">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    def render_progress_bar(self, progress: float, message: str = ""):
        """Render enhanced progress bar"""
        
        progress_html = f"""
        <div class="progress-container">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: 600;">Calculation Progress</span>
                <span>{progress:.1f}%</span>
            </div>
            <div style="background: #e5e7eb; border-radius: 4px; height: 8px;">
                <div style="background: #2563eb; height: 100%; border-radius: 4px; 
                           width: {progress}%; transition: width 0.3s ease;"></div>
            </div>
            {f'<div style="margin-top: 0.5rem; color: #6b7280; font-size: 0.875rem;">{message}</div>' if message else ''}
        </div>
        """
        
        st.markdown(progress_html, unsafe_allow_html=True)
    
    def render_calculation_step_visual(self, steps: List[Dict], current_step: int = 0):
        """Render visual representation of calculation steps"""
        
        fig = go.Figure()
        
        # Create step flow visualization
        x_positions = list(range(len(steps)))
        y_position = [0] * len(steps)
        
        # Color steps based on status
        colors = []
        for i, step in enumerate(steps):
            if i < current_step:
                colors.append('#10b981')  # Green for completed
            elif i == current_step:
                colors.append('#f59e0b')  # Orange for current
            else:
                colors.append('#e5e7eb')  # Gray for pending
        
        # Add step markers
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_position,
            mode='markers',
            marker=dict(
                size=20,
                color=colors,
                line=dict(width=2, color='white')
            ),
            text=[f"Step {i+1}" for i in range(len(steps))],
            textposition="middle center",
            hovertext=[step.get('title', f'Step {i+1}') for i, step in enumerate(steps)],
            hoverinfo='text',
            showlegend=False
        ))
        
        # Add connecting lines
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_position,
            mode='lines',
            line=dict(color='#d1d5db', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title="SA-CCR Calculation Progress",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
            plot_bgcolor='white',
            height=150,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
