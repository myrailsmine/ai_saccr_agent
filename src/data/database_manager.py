# src/data/database_manager.py

import duckdb
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import os

from ..models.trade_models import Trade, NettingSet, Collateral, Portfolio

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Professional database manager using DuckDB for SA-CCR data persistence
    """
    
    def __init__(self, db_path: str = "data/saccr.duckdb"):
        """Initialize database manager"""
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize connection
        self.conn = duckdb.connect(str(self.db_path))
        
        # Create tables
        self._create_tables()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def _create_tables(self):
        """Create all required database tables"""
        
        # Portfolios table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            portfolio_id VARCHAR PRIMARY KEY,
            portfolio_name VARCHAR NOT NULL,
            base_currency VARCHAR(3) DEFAULT 'USD',
            portfolio_type VARCHAR DEFAULT 'Trading',
            business_unit VARCHAR DEFAULT 'Default',
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Netting sets table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS netting_sets (
            netting_set_id VARCHAR PRIMARY KEY,
            portfolio_id VARCHAR,
            counterparty VARCHAR NOT NULL,
            threshold DOUBLE DEFAULT 0.0,
            mta DOUBLE DEFAULT 0.0,
            nica DOUBLE DEFAULT 0.0,
            master_agreement_type VARCHAR DEFAULT 'ISDA',
            master_agreement_date DATE,
            csa_signed BOOLEAN DEFAULT FALSE,
            csa_date DATE,
            governing_law VARCHAR(5) DEFAULT 'NY',
            netting_enforceability BOOLEAN DEFAULT TRUE,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id)
        )
        """)
        
        # Trades table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id VARCHAR PRIMARY KEY,
            netting_set_id VARCHAR NOT NULL,
            counterparty VARCHAR NOT NULL,
            asset_class VARCHAR NOT NULL,
            trade_type VARCHAR NOT NULL,
            notional DOUBLE NOT NULL,
            currency VARCHAR(3) NOT NULL,
            underlying VARCHAR NOT NULL,
            maturity_date DATE NOT NULL,
            mtm_value DOUBLE DEFAULT 0.0,
            delta DOUBLE DEFAULT 1.0,
            basis_flag BOOLEAN DEFAULT FALSE,
            volatility_flag BOOLEAN DEFAULT FALSE,
            ceu_flag INTEGER DEFAULT 1,
            credit_rating VARCHAR,
            sector VARCHAR,
            region VARCHAR,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            trade_version INTEGER DEFAULT 1,
            FOREIGN KEY (netting_set_id) REFERENCES netting_sets(netting_set_id)
        )
        """)
        
        # Collateral table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS collateral (
            collateral_id VARCHAR PRIMARY KEY,
            netting_set_id VARCHAR NOT NULL,
            collateral_type VARCHAR NOT NULL,
            currency VARCHAR(3) NOT NULL,
            amount DOUBLE NOT NULL,
            market_value DOUBLE NOT NULL,
            haircut_override DOUBLE,
            posting_date DATE DEFAULT CURRENT_DATE,
            valuation_date DATE DEFAULT CURRENT_DATE,
            FOREIGN KEY (netting_set_id) REFERENCES netting_sets(netting_set_id)
        )
        """)
        
        # Calculation results table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS calculation_results (
            calculation_id VARCHAR PRIMARY KEY,
            netting_set_id VARCHAR NOT NULL,
            calculation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            replacement_cost DOUBLE,
            potential_future_exposure DOUBLE,
            exposure_at_default DOUBLE,
            risk_weighted_assets DOUBLE,
            capital_requirement DOUBLE,
            calculation_steps JSON,
            metadata JSON,
            FOREIGN KEY (netting_set_id) REFERENCES netting_sets(netting_set_id)
        )
        """)
        
        # Counterparty reference data
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS counterparties (
            counterparty_name VARCHAR PRIMARY KEY,
            legal_entity_identifier VARCHAR,
            jurisdiction VARCHAR(2),
            counterparty_type VARCHAR,
            credit_rating VARCHAR,
            sector VARCHAR,
            is_financial BOOLEAN DEFAULT FALSE,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Configuration table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS configuration (
            config_key VARCHAR PRIMARY KEY,
            config_value JSON NOT NULL,
            config_type VARCHAR DEFAULT 'system',
            description VARCHAR,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create indexes for performance
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes for performance"""
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_trades_netting_set ON trades(netting_set_id)",
            "CREATE INDEX IF NOT EXISTS idx_trades_counterparty ON trades(counterparty)",
            "CREATE INDEX IF NOT EXISTS idx_trades_asset_class ON trades(asset_class)",
            "CREATE INDEX IF NOT EXISTS idx_trades_maturity ON trades(maturity_date)",
            "CREATE INDEX IF NOT EXISTS idx_calculation_results_date ON calculation_results(calculation_date)",
            "CREATE INDEX IF NOT EXISTS idx_netting_sets_counterparty ON netting_sets(counterparty)"
        ]
        
        for index in indexes:
            try:
                self.conn.execute(index)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
    
    # Portfolio operations
    def save_portfolio(self, portfolio: Portfolio) -> bool:
        """Save complete portfolio to database"""
        
        try:
            with self.conn.cursor() as cursor:
                # Insert/update portfolio
                cursor.execute("""
                INSERT OR REPLACE INTO portfolios 
                (portfolio_id, portfolio_name, base_currency, portfolio_type, business_unit, last_updated)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    portfolio.portfolio_id,
                    portfolio.portfolio_name,
                    portfolio.base_currency,
                    portfolio.portfolio_type,
                    portfolio.business_unit
                ))
                
                # Save netting sets and trades
                for netting_set in portfolio.netting_sets:
                    self._save_netting_set(netting_set, portfolio.portfolio_id, cursor)
                
            logger.info(f"Portfolio {portfolio.portfolio_id} saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving portfolio {portfolio.portfolio_id}: {e}")
            return False
    
    def _save_netting_set(self, netting_set: NettingSet, portfolio_id: str, cursor) -> bool:
        """Save netting set and associated trades"""
        
        try:
            # Insert/update netting set
            cursor.execute("""
            INSERT OR REPLACE INTO netting_sets 
            (netting_set_id, portfolio_id, counterparty, threshold, mta, nica,
             master_agreement_type, master_agreement_date, csa_signed, csa_date,
             governing_law, netting_enforceability, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                netting_set.netting_set_id,
                portfolio_id,
                netting_set.counterparty,
                netting_set.threshold,
                netting_set.mta,
                netting_set.nica,
                netting_set.master_agreement_type,
                netting_set.master_agreement_date,
                netting_set.csa_signed,
                netting_set.csa_date,
                netting_set.governing_law,
                netting_set.netting_enforceability
            ))
            
            # Save trades
            for trade in netting_set.trades:
                self._save_trade(trade, cursor)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving netting set {netting_set.netting_set_id}: {e}")
            return False
    
    def _save_trade(self, trade: Trade, cursor) -> bool:
        """Save individual trade"""
        
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO trades 
            (trade_id, netting_set_id, counterparty, asset_class, trade_type,
             notional, currency, underlying, maturity_date, mtm_value, delta,
             basis_flag, volatility_flag, ceu_flag, credit_rating, sector, region,
             trade_version, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                trade.trade_id,
                getattr(trade, 'netting_set_id', ''),  # May need to be set separately
                trade.counterparty,
                trade.asset_class.value,
                trade.trade_type.value,
                trade.notional,
                trade.currency,
                trade.underlying,
                trade.maturity_date.date(),
                trade.mtm_value,
                trade.delta,
                trade.basis_flag,
                trade.volatility_flag,
                trade.ceu_flag,
                trade.credit_rating,
                trade.sector,
                trade.region,
                trade.trade_version
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving trade {trade.trade_id}: {e}")
            return False
    
    def load_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Load complete portfolio from database"""
        
        try:
            # Load portfolio metadata
            portfolio_data = self.conn.execute("""
            SELECT * FROM portfolios WHERE portfolio_id = ?
            """, (portfolio_id,)).fetchone()
            
            if not portfolio_data:
                return None
            
            # Load netting sets
            netting_sets = self.load_netting_sets_by_portfolio(portfolio_id)
            
            # Create portfolio object
            portfolio = Portfolio(
                portfolio_id=portfolio_data[0],
                portfolio_name=portfolio_data[1],
                netting_sets=netting_sets,
                base_currency=portfolio_data[2],
                portfolio_type=portfolio_data[3],
                business_unit=portfolio_data[4]
            )
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Error loading portfolio {portfolio_id}: {e}")
            return None
    
    def load_netting_sets_by_portfolio(self, portfolio_id: str) -> List[NettingSet]:
        """Load all netting sets for a portfolio"""
        
        try:
            netting_set_data = self.conn.execute("""
            SELECT * FROM netting_sets WHERE portfolio_id = ?
            """, (portfolio_id,)).fetchall()
            
            netting_sets = []
            for ns_data in netting_set_data:
                trades = self.load_trades_by_netting_set(ns_data[0])
                
                netting_set = NettingSet(
                    netting_set_id=ns_data[0],
                    counterparty=ns_data[2],
                    trades=trades,
                    threshold=ns_data[3],
                    mta=ns_data[4],
                    nica=ns_data[5],
                    master_agreement_type=ns_data[6],
                    csa_signed=ns_data[8],
                    governing_law=ns_data[10],
                    netting_enforceability=ns_data[11]
                )
                
                netting_sets.append(netting_set)
            
            return netting_sets
            
        except Exception as e:
            logger.error(f"Error loading netting sets for portfolio {portfolio_id}: {e}")
            return []
    
    def load_trades_by_netting_set(self, netting_set_id: str) -> List[Trade]:
        """Load all trades for a netting set"""
        
        try:
            trade_data = self.conn.execute("""
            SELECT * FROM trades WHERE netting_set_id = ?
            ORDER BY created_date
            """, (netting_set_id,)).fetchall()
            
            trades = []
            for t_data in trade_data:
                trade = Trade(
                    trade_id=t_data[0],
                    counterparty=t_data[2],
                    asset_class=t_data[3],
                    trade_type=t_data[4],
                    notional=t_data[5],
                    currency=t_data[6],
                    underlying=t_data[7],
                    maturity_date=datetime.combine(t_data[8], datetime.min.time()),
                    mtm_value=t_data[9],
                    delta=t_data[10],
                    basis_flag=t_data[11],
                    volatility_flag=t_data[12],
                    ceu_flag=t_data[13],
                    credit_rating=t_data[14],
                    sector=t_data[15],
                    region=t_data[16],
                    trade_version=t_data[19]
                )
                
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Error loading trades for netting set {netting_set_id}: {e}")
            return []
    
    # Calculation results operations
    def save_calculation_results(self, portfolio_data: Dict, results: Dict) -> bool:
        """Save SA-CCR calculation results"""
        
        try:
            calculation_id = f"{portfolio_data['netting_set_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            final_results = results.get('final_results', {})
            
            self.conn.execute("""
            INSERT INTO calculation_results 
            (calculation_id, netting_set_id, replacement_cost, potential_future_exposure,
             exposure_at_default, risk_weighted_assets, capital_requirement,
             calculation_steps, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                calculation_id,
                portfolio_data['netting_set_id'],
                final_results.get('replacement_cost', 0),
                final_results.get('potential_future_exposure', 0),
                final_results.get('exposure_at_default', 0),
                final_results.get('risk_weighted_assets', 0),
                final_results.get('capital_requirement', 0),
                json.dumps(results.get('calculation_steps', [])),
                json.dumps(results.get('calculation_metadata', {}))
            ))
            
            logger.info(f"Calculation results saved: {calculation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calculation results: {e}")
            return False
    
    def get_calculation_history(self, netting_set_id: str, limit: int = 10) -> pd.DataFrame:
        """Get calculation history for a netting set"""
        
        try:
            return self.conn.execute("""
            SELECT calculation_id, calculation_date, replacement_cost, 
                   potential_future_exposure, exposure_at_default, 
                   risk_weighted_assets, capital_requirement
            FROM calculation_results 
            WHERE netting_set_id = ?
            ORDER BY calculation_date DESC
            LIMIT ?
            """, (netting_set_id, limit)).df()
            
        except Exception as e:
            logger.error(f"Error retrieving calculation history: {e}")
            return pd.DataFrame()
    
    # Analytics and reporting
    def get_portfolio_summary(self) -> pd.DataFrame:
        """Get portfolio summary statistics"""
        
        try:
            return self.conn.execute("""
            SELECT 
                p.portfolio_id,
                p.portfolio_name,
                COUNT(DISTINCT ns.netting_set_id) as netting_sets,
                COUNT(DISTINCT ns.counterparty) as counterparties,
                COUNT(t.trade_id) as total_trades,
                SUM(ABS(t.notional)) as total_notional,
                SUM(t.mtm_value) as total_mtm
            FROM portfolios p
            LEFT JOIN netting_sets ns ON p.portfolio_id = ns.portfolio_id
            LEFT JOIN trades t ON ns.netting_set_id = t.netting_set_id
            GROUP BY p.portfolio_id, p.portfolio_name
            ORDER BY total_notional DESC
            """).df()
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return pd.DataFrame()
    
    def get_asset_class_breakdown(self, portfolio_id: Optional[str] = None) -> pd.DataFrame:
        """Get asset class breakdown"""
        
        try:
            where_clause = "WHERE p.portfolio_id = ?" if portfolio_id else ""
            params = (portfolio_id,) if portfolio_id else ()
            
            return self.conn.execute(f"""
            SELECT 
                t.asset_class,
                COUNT(t.trade_id) as trade_count,
                SUM(ABS(t.notional)) as total_notional,
                SUM(t.mtm_value) as total_mtm,
                AVG(DATEDIFF('day', CURRENT_DATE, t.maturity_date) / 365.0) as avg_maturity_years
            FROM trades t
            JOIN netting_sets ns ON t.netting_set_id = ns.netting_set_id
            JOIN portfolios p ON ns.portfolio_id = p.portfolio_id
            {where_clause}
            GROUP BY t.asset_class
            ORDER BY total_notional DESC
            """, params).df()
            
        except Exception as e:
            logger.error(f"Error generating asset class breakdown: {e}")
            return pd.DataFrame()
    
    def get_counterparty_exposure(self, portfolio_id: Optional[str] = None) -> pd.DataFrame:
        """Get counterparty exposure summary"""
        
        try:
            where_clause = "WHERE p.portfolio_id = ?" if portfolio_id else ""
            params = (portfolio_id,) if portfolio_id else ()
            
            return self.conn.execute(f"""
            SELECT 
                ns.counterparty,
                COUNT(DISTINCT ns.netting_set_id) as netting_sets,
                COUNT(t.trade_id) as trade_count,
                SUM(ABS(t.notional)) as gross_notional,
                SUM(t.mtm_value) as net_mtm,
                MAX(cr.exposure_at_default) as latest_ead,
                MAX(cr.calculation_date) as last_calculation
            FROM netting_sets ns
            JOIN portfolios p ON ns.portfolio_id = p.portfolio_id
            LEFT JOIN trades t ON ns.netting_set_id = t.netting_set_id
            LEFT JOIN calculation_results cr ON ns.netting_set_id = cr.netting_set_id
            {where_clause}
            GROUP BY ns.counterparty
            ORDER BY gross_notional DESC
            """, params).df()
            
        except Exception as e:
            logger.error(f"Error generating counterparty exposure: {e}")
            return pd.DataFrame()
    
    # Database maintenance
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        try:
            stats = {}
            
            # Count records
            stats['portfolio_count'] = self.conn.execute("SELECT COUNT(*) FROM portfolios").fetchone()[0]
            stats['netting_set_count'] = self.conn.execute("SELECT COUNT(*) FROM netting_sets").fetchone()[0]
            stats['trade_count'] = self.conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            stats['calculation_count'] = self.conn.execute("SELECT COUNT(*) FROM calculation_results").fetchone()[0]
            
            # Database size
            stats['db_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
            
            # Recent activity
            stats['recent_trades'] = self.conn.execute("""
                SELECT COUNT(*) FROM trades 
                WHERE created_date > CURRENT_TIMESTAMP - INTERVAL 7 DAY
            """).fetchone()[0]
            
            stats['recent_calculations'] = self.conn.execute("""
                SELECT COUNT(*) FROM calculation_results 
                WHERE calculation_date > CURRENT_TIMESTAMP - INTERVAL 7 DAY
            """).fetchone()[0]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def get_recent_activity(self, limit: int = 20) -> pd.DataFrame:
        """Get recent database activity"""
        
        try:
            return self.conn.execute("""
            SELECT 
                'Trade' as activity_type,
                trade_id as item_id,
                counterparty as description,
                created_date as activity_date
            FROM trades
            UNION ALL
            SELECT 
                'Calculation' as activity_type,
                calculation_id as item_id,
                netting_set_id as description,
                calculation_date as activity_date
            FROM calculation_results
            ORDER BY activity_date DESC
            LIMIT ?
            """, (limit,)).df()
            
        except Exception as e:
            logger.error(f"Error getting recent activity: {e}")
            return pd.DataFrame()
    
    def cleanup_old_records(self, days_old: int = 90) -> int:
        """Clean up old calculation results"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            result = self.conn.execute("""
            DELETE FROM calculation_results 
            WHERE calculation_date < ?
            """, (cutoff_date,))
            
            deleted_count = result.rowcount if hasattr(result, 'rowcount') else 0
            logger.info(f"Cleaned up {deleted_count} old calculation records")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
            return 0
    
    def optimize_database(self):
        """Optimize database performance"""
        
        try:
            # Analyze tables for query optimization
            self.conn.execute("ANALYZE")
            
            # Vacuum to reclaim space
            self.conn.execute("VACUUM")
            
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
    
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        
        try:
            import shutil
            
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Close connection temporarily
            self.conn.close()
            
            # Copy database file
            shutil.copy2(self.db_path, backup_file)
            
            # Reconnect
            self.conn = duckdb.connect(str(self.db_path))
            
            logger.info(f"Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            # Ensure connection is restored
            if not hasattr(self, 'conn') or self.conn is None:
                self.conn = duckdb.connect(str(self.db_path))
            return False
    
    def export_data(self, table_name: str, export_path: str, format: str = 'csv') -> bool:
        """Export table data to file"""
        
        try:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'csv':
                self.conn.execute(f"COPY {table_name} TO '{export_path}' (FORMAT CSV, HEADER)")
            elif format.lower() == 'parquet':
                self.conn.execute(f"COPY {table_name} TO '{export_path}' (FORMAT PARQUET)")
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported {table_name} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting {table_name}: {e}")
            return False
    
    def get_trade_count(self) -> int:
        """Get total trade count"""
        try:
            return self.conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        except:
            return 0
    
    def get_portfolio_count(self) -> int:
        """Get total portfolio count"""
        try:
            return self.conn.execute("SELECT COUNT(*) FROM portfolios").fetchone()[0]
        except:
            return 0
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
