# src/utils/progress_tracker.py

import time
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

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
    
    def initialize_saccr_steps(self):
        """Initialize all 24 SA-CCR calculation steps"""
        saccr_steps = [
            ("step_01", "Processing netting set data"),
            ("step_02", "Classifying asset classes"),
            ("step_03", "Creating hedging sets"),
            ("step_04", "Calculating time parameters"),
            ("step_05", "Computing adjusted notionals"),
            ("step_06", "Applying maturity factors"),
            ("step_07", "Determining supervisory deltas"),
            ("step_08", "Applying supervisory factors"),
            ("step_09", "Calculating adjusted contract amounts"),
            ("step_10", "Applying supervisory correlations"),
            ("step_11", "Aggregating hedging set add-ons"),
            ("step_12", "Computing asset class add-ons"),
            ("step_13", "Computing aggregate add-on"),
            ("step_14", "Computing V and C"),
            ("step_15", "Calculating PFE multiplier"),
            ("step_16", "Computing PFE"),
            ("step_17", "Processing collateral parameters"),
            ("step_18", "Calculating replacement cost"),
            ("step_19", "Determining central clearing status"),
            ("step_20", "Applying alpha multiplier"),
            ("step_21", "Computing EAD"),
            ("step_22", "Processing counterparty data"),
            ("step_23", "Determining risk weight"),
            ("step_24", "Computing RWA")
        ]
        
        for step_id, description in saccr_steps:
            self.add_step(step_id, description)
    
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
    
    def get_current_step_number(self) -> int:
        """Get current step number (1-based)"""
        if not self.current_step:
            return 0
        
        for i, step in enumerate(self.steps):
            if step.step_id == self.current_step:
                return i + 1
        
        return 0
    
    def get_current_status(self) -> Dict[str, any]:
        """Get current progress status"""
        return {
            'total_steps': len(self.steps),
            'completed_steps': sum(1 for step in self.steps if step.status == 'completed'),
            'current_step': self.current_step,
            'current_step_number': self.get_current_step_number(),
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
    
    def record_memory_usage(self, memory_mb: float):
        """Record memory usage"""
        if 'memory_usage' not in self.metrics:
            self.metrics['memory_usage'] = []
        self.metrics['memory_usage'].append({
            'timestamp': time.time(),
            'memory_mb': memory_mb
        })
    
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
            )[:5],
            'average_memory_usage': self._get_average_memory_usage(),
            'peak_memory_usage': self._get_peak_memory_usage()
        }
    
    def _get_average_memory_usage(self) -> float:
        """Calculate average memory usage"""
        memory_data = self.metrics.get('memory_usage', [])
        if not memory_data:
            return 0.0
        
        return sum(data['memory_mb'] for data in memory_data) / len(memory_data)
    
    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage"""
        memory_data = self.metrics.get('memory_usage', [])
        if not memory_data:
            return 0.0
        
        return max(data['memory_mb'] for data in memory_data)
