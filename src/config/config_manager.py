# src/config/config_manager.py

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class SACCRConfig:
    """SA-CCR calculation configuration"""
    
    # Alpha multipliers
    alpha_bilateral: float = 1.4
    alpha_centrally_cleared: float = 0.5
    
    # Capital requirements
    capital_ratio: float = 0.08
    
    # Calculation parameters
    min_maturity_years: float = 0.01
    max_maturity_years: float = 50.0
    
    # PFE multiplier parameters
    pfe_multiplier_floor: float = 0.05
    pfe_multiplier_decay: float = 0.05
    
    # Validation tolerances
    notional_tolerance: float = 0.01
    delta_tolerance: float = 0.001
    
    # Performance settings
    enable_calculation_cache: bool = True
    cache_expiry_hours: int = 24
    max_portfolio_size: int = 10000

@dataclass
class DatabaseConfig:
    """Database configuration"""
    
    db_path: str = "data/saccr.duckdb"
    backup_path: str = "backups/"
    auto_backup: bool = True
    backup_frequency_hours: int = 24
    cleanup_days: int = 90
    
    # Performance settings
    connection_pool_size: int = 5
    query_timeout_seconds: int = 30

@dataclass
class UIConfig:
    """User interface configuration"""
    
    theme: str = "professional"
    decimal_places: int = 2
    currency_format: str = "USD"
    date_format: str = "%Y-%m-%d"
    
    # Display options
    show_step_details: bool = True
    show_debug_info: bool = False
    auto_refresh: bool = False

class ConfigManager:
    """
    Professional configuration manager for SA-CCR application
    Handles loading, validation, and updating of all configuration settings
    """
    
    def __init__(self, config_file: str = "config/saccr_config.yaml"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize default configurations
        self.saccr_config = SACCRConfig()
        self.database_config = DatabaseConfig()
        self.ui_config = UIConfig()
        
        # Load configuration from file
        self._load_config()
        
        logger.info(f"Configuration manager initialized with {self.config_file}")
    
    def _load_config(self):
        """Load configuration from file"""
        
        if not self.config_file.exists():
            logger.info("Configuration file not found, creating default configuration")
            self._save_default_config()
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Load SA-CCR configuration
            if 'saccr' in config_data:
                self._update_saccr_config(config_data['saccr'])
            
            # Load database configuration
            if 'database' in config_data:
                self._update_database_config(config_data['database'])
            
            # Load UI configuration
            if 'ui' in config_data:
                self._update_ui_config(config_data['ui'])
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
    
    def _save_default_config(self):
        """Save default configuration to file"""
        
        default_config = {
            'saccr': {
                'alpha_bilateral': self.saccr_config.alpha_bilateral,
                'alpha_centrally_cleared': self.saccr_config.alpha_centrally_cleared,
                'capital_ratio': self.saccr_config.capital_ratio,
                'min_maturity_years': self.saccr_config.min_maturity_years,
                'max_maturity_years': self.saccr_config.max_maturity_years,
                'pfe_multiplier_floor': self.saccr_config.pfe_multiplier_floor,
                'pfe_multiplier_decay': self.saccr_config.pfe_multiplier_decay,
                'notional_tolerance': self.saccr_config.notional_tolerance,
                'delta_tolerance': self.saccr_config.delta_tolerance,
                'enable_calculation_cache': self.saccr_config.enable_calculation_cache,
                'cache_expiry_hours': self.saccr_config.cache_expiry_hours,
                'max_portfolio_size': self.saccr_config.max_portfolio_size
            },
            'database': {
                'db_path': self.database_config.db_path,
                'backup_path': self.database_config.backup_path,
                'auto_backup': self.database_config.auto_backup,
                'backup_frequency_hours': self.database_config.backup_frequency_hours,
                'cleanup_days': self.database_config.cleanup_days,
                'connection_pool_size': self.database_config.connection_pool_size,
                'query_timeout_seconds': self.database_config.query_timeout_seconds
            },
            'ui': {
                'theme': self.ui_config.theme,
                'decimal_places': self.ui_config.decimal_places,
                'currency_format': self.ui_config.currency_format,
                'date_format': self.ui_config.date_format,
                'show_step_details': self.ui_config.show_step_details,
                'show_debug_info': self.ui_config.show_debug_info,
                'auto_refresh': self.ui_config.auto_refresh
            }
        }
        
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
            logger.info("Default configuration saved")
            
        except Exception as e:
            logger.error(f"Error saving default configuration: {e}")
    
    def _update_saccr_config(self, config_data: Dict[str, Any]):
        """Update SA-CCR configuration from data"""
        
        for key, value in config_data.items():
            if hasattr(self.saccr_config, key):
                setattr(self.saccr_config, key, value)
    
    def _update_database_config(self, config_data: Dict[str, Any]):
        """Update database configuration from data"""
        
        for key, value in config_data.items():
            if hasattr(self.database_config, key):
                setattr(self.database_config, key, value)
    
    def _update_ui_config(self, config_data: Dict[str, Any]):
        """Update UI configuration from data"""
        
        for key, value in config_data.items():
            if hasattr(self.ui_config, key):
                setattr(self.ui_config, key, value)
    
    # Public API methods
    
    def get_calculation_config(self) -> Dict[str, Any]:
        """Get SA-CCR calculation configuration"""
        
        return {
            'alpha_bilateral': self.saccr_config.alpha_bilateral,
            'alpha_centrally_cleared': self.saccr_config.alpha_centrally_cleared,
            'capital_ratio': self.saccr_config.capital_ratio,
            'min_maturity_years': self.saccr_config.min_maturity_years,
            'max_maturity_years': self.saccr_config.max_maturity_years,
            'pfe_multiplier_floor': self.saccr_config.pfe_multiplier_floor,
            'pfe_multiplier_decay': self.saccr_config.pfe_multiplier_decay,
            'enable_calculation_cache': self.saccr_config.enable_calculation_cache,
            'cache_expiry_hours': self.saccr_config.cache_expiry_hours,
            'max_portfolio_size': self.saccr_config.max_portfolio_size
        }
    
    def update_calculation_config(self, updates: Dict[str, Any]):
        """Update SA-CCR calculation configuration"""
        
        for key, value in updates.items():
            if hasattr(self.saccr_config, key):
                setattr(self.saccr_config, key, value)
        
        self._save_config()
        logger.info("SA-CCR configuration updated")
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        
        return {
            'db_path': self.database_config.db_path,
            'backup_path': self.database_config.backup_path,
            'auto_backup': self.database_config.auto_backup,
            'backup_frequency_hours': self.database_config.backup_frequency_hours,
            'cleanup_days': self.database_config.cleanup_days,
            'connection_pool_size': self.database_config.connection_pool_size,
            'query_timeout_seconds': self.database_config.query_timeout_seconds
        }
    
    def update_database_config(self, updates: Dict[str, Any]):
        """Update database configuration"""
        
        for key, value in updates.items():
            if hasattr(self.database_config, key):
                setattr(self.database_config, key, value)
        
        self._save_config()
        logger.info("Database configuration updated")
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration"""
        
        return {
            'theme': self.ui_config.theme,
            'decimal_places': self.ui_config.decimal_places,
            'currency_format': self.ui_config.currency_format,
            'date_format': self.ui_config.date_format,
            'show_step_details': self.ui_config.show_step_details,
            'show_debug_info': self.ui_config.show_debug_info,
            'auto_refresh': self.ui_config.auto_refresh
        }
    
    def update_ui_config(self, updates: Dict[str, Any]):
        """Update UI configuration"""
        
        for key, value in updates.items():
            if hasattr(self.ui_config, key):
                setattr(self.ui_config, key, value)
        
        self._save_config()
        logger.info("UI configuration updated")
    
    def _save_config(self):
        """Save current configuration to file"""
        
        config_data = {
            'saccr': self.get_calculation_config(),
            'database': self.get_database_config(),
            'ui': self.get_ui_config()
        }
        
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration"""
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate SA-CCR configuration
        if self.saccr_config.alpha_bilateral < 0.1 or self.saccr_config.alpha_bilateral > 5.0:
            validation_results['errors'].append("Alpha bilateral must be between 0.1 and 5.0")
        
        if self.saccr_config.alpha_centrally_cleared < 0.1 or self.saccr_config.alpha_centrally_cleared > 2.0:
            validation_results['errors'].append("Alpha centrally cleared must be between 0.1 and 2.0")
        
        if self.saccr_config.capital_ratio < 0.01 or self.saccr_config.capital_ratio > 0.5:
            validation_results['errors'].append("Capital ratio must be between 1% and 50%")
        
        if self.saccr_config.min_maturity_years <= 0:
            validation_results['errors'].append("Minimum maturity must be positive")
        
        if self.saccr_config.max_maturity_years <= self.saccr_config.min_maturity_years:
            validation_results['errors'].append("Maximum maturity must be greater than minimum")
        
        # Validate database configuration
        db_path = Path(self.database_config.db_path)
        if not db_path.parent.exists():
            try:
                db_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                validation_results['errors'].append(f"Cannot create database directory: {e}")
        
        # Validate UI configuration
        if self.ui_config.decimal_places < 0 or self.ui_config.decimal_places > 10:
            validation_results['warnings'].append("Decimal places should be between 0 and 10")
        
        if len(validation_results['errors']) > 0:
            validation_results['valid'] = False
        
        return validation_results
    
    def get_supervisory_factors(self) -> Dict[str, Dict]:
        """Get regulatory supervisory factors"""
        
        # These are Basel regulatory parameters that shouldn't be modified
        return {
            'interest_rate': {
                'USD': {'<2y': 0.0050, '2-5y': 0.0050, '>5y': 0.0150},
                'EUR': {'<2y': 0.0050, '2-5y': 0.0050, '>5y': 0.0150},
                'JPY': {'<2y': 0.0050, '2-5y': 0.0050, '>5y': 0.0150},
                'GBP': {'<2y': 0.0050, '2-5y': 0.0050, '>5y': 0.0150},
                'other': {'<2y': 0.0150, '2-5y': 0.0150, '>5y': 0.0150}
            },
            'foreign_exchange': {'G10': 0.04, 'emerging': 0.15},
            'credit': {
                'IG_single': 0.0046, 'HY_single': 0.0130,
                'IG_index': 0.0038, 'HY_index': 0.0106
            },
            'equity': {
                'single_large': 0.32, 'single_small': 0.40,
                'index_developed': 0.20, 'index_emerging': 0.25
            },
            'commodity': {
                'energy': 0.18, 'metals': 0.18, 'agriculture': 0.18, 'other': 0.18
            }
        }
    
    def get_supervisory_correlations(self) -> Dict[str, float]:
        """Get regulatory supervisory correlations"""
        
        return {
            'interest_rate': 0.99,
            'foreign_exchange': 0.60,
            'credit': 0.50,
            'equity': 0.80,
            'commodity': 0.40
        }
    
    def get_collateral_haircuts(self) -> Dict[str, float]:
        """Get standard collateral haircuts"""
        
        return {
            'cash': 0.00,
            'government_bonds': 0.005,
            'corporate_bonds': 0.04,
            'equities': 0.15,
            'money_market': 0.005
        }
    
    def reset_to_defaults(self):
        """Reset all configuration to defaults"""
        
        self.saccr_config = SACCRConfig()
        self.database_config = DatabaseConfig()
        self.ui_config = UIConfig()
        
        self._save_config()
        logger.info("Configuration reset to defaults")
    
    def export_config(self, export_path: str) -> bool:
        """Export configuration to file"""
        
        try:
            config_data = {
                'saccr': self.get_calculation_config(),
                'database': self.get_database_config(),
                'ui': self.get_ui_config(),
                'regulatory': {
                    'supervisory_factors': self.get_supervisory_factors(),
                    'supervisory_correlations': self.get_supervisory_correlations(),
                    'collateral_haircuts': self.get_collateral_haircuts()
                }
            }
            
            export_file = Path(export_path)
            
            if export_file.suffix.lower() == '.yaml':
                with open(export_file, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif export_file.suffix.lower() == '.json':
                with open(export_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
            else:
                raise ValueError("Export format must be .yaml or .json")
            
            logger.info(f"Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
