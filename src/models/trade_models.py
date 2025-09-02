# src/models/trade_models.py

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid

class AssetClass(Enum):
    """Basel asset class enumeration"""
    INTEREST_RATE = "Interest Rate"
    FOREIGN_EXCHANGE = "Foreign Exchange"
    CREDIT = "Credit"
    EQUITY = "Equity"
    COMMODITY = "Commodity"

class TradeType(Enum):
    """Trade type enumeration"""
    SWAP = "Swap"
    FORWARD = "Forward"
    OPTION = "Option"
    SWAPTION = "Swaption"
    FUTURE = "Future"
    CDS = "Credit Default Swap"

class CollateralType(Enum):
    """Collateral type enumeration"""
    CASH = "Cash"
    GOVERNMENT_BONDS = "Government Bonds"
    CORPORATE_BONDS = "Corporate Bonds"
    EQUITIES = "Equities"
    MONEY_MARKET = "Money Market Funds"

@dataclass
class Trade:
    """Enhanced trade data model with validation"""
    
    # Required fields
    trade_id: str
    counterparty: str
    asset_class: AssetClass
    trade_type: TradeType
    notional: float
    currency: str
    underlying: str
    maturity_date: datetime
    
    # Optional fields with defaults
    mtm_value: float = 0.0
    delta: float = 1.0
    basis_flag: bool = False
    volatility_flag: bool = False
    ceu_flag: int = 1  # 1 = bilateral, 0 = centrally cleared
    
    # Metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    trade_version: int = 1
    
    # Additional risk parameters
    credit_rating: Optional[str] = None
    sector: Optional[str] = None
    region: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and processing"""
        self._validate_trade_data()
        
        # Convert strings to enums if necessary
        if isinstance(self.asset_class, str):
            self.asset_class = AssetClass(self.asset_class)
        if isinstance(self.trade_type, str):
            self.trade_type = TradeType(self.trade_type)
        
        # Ensure notional is float
        self.notional = float(self.notional)
        self.mtm_value = float(self.mtm_value)
        self.delta = float(self.delta)
    
    def _validate_trade_data(self):
        """Validate trade data integrity"""
        
        if not self.trade_id or not isinstance(self.trade_id, str):
            raise ValueError("Trade ID must be a non-empty string")
        
        if self.notional == 0:
            raise ValueError("Notional cannot be zero")
        
        if not self.currency or len(self.currency) != 3:
            raise ValueError("Currency must be 3-character code")
        
        if self.maturity_date <= datetime.now():
            raise ValueError("Maturity date must be in the future")
        
        if abs(self.delta) > 1:
            raise ValueError("Delta must be between -1 and 1")
        
        if self.ceu_flag not in [0, 1]:
            raise ValueError("CEU flag must be 0 or 1")
    
    def time_to_maturity(self) -> float:
        """Calculate time to maturity in years"""
        days_remaining = (self.maturity_date - datetime.now()).days
        return max(0, days_remaining / 365.25)
    
    def is_option_type(self) -> bool:
        """Check if trade is option-like"""
        return self.trade_type in [TradeType.OPTION, TradeType.SWAPTION]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'trade_id': self.trade_id,
            'counterparty': self.counterparty,
            'asset_class': self.asset_class.value,
            'trade_type': self.trade_type.value,
            'notional': self.notional,
            'currency': self.currency,
            'underlying': self.underlying,
            'maturity_date': self.maturity_date.isoformat(),
            'mtm_value': self.mtm_value,
            'delta': self.delta,
            'basis_flag': self.basis_flag,
            'volatility_flag': self.volatility_flag,
            'ceu_flag': self.ceu_flag,
            'created_date': self.created_date.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'trade_version': self.trade_version,
            'credit_rating': self.credit_rating,
            'sector': self.sector,
            'region': self.region
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create Trade from dictionary"""
        
        # Convert date strings back to datetime
        if isinstance(data.get('maturity_date'), str):
            data['maturity_date'] = datetime.fromisoformat(data['maturity_date'])
        if isinstance(data.get('created_date'), str):
            data['created_date'] = datetime.fromisoformat(data['created_date'])
        if isinstance(data.get('last_updated'), str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        
        # Convert enum strings back to enums
        if isinstance(data.get('asset_class'), str):
            data['asset_class'] = AssetClass(data['asset_class'])
        if isinstance(data.get('trade_type'), str):
            data['trade_type'] = TradeType(data['trade_type'])
        
        return cls(**data)

@dataclass
class Collateral:
    """Collateral data model"""
    
    collateral_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    collateral_type: CollateralType = CollateralType.CASH
    currency: str = "USD"
    amount: float = 0.0
    market_value: float = 0.0
    haircut_override: Optional[float] = None
    
    # Metadata
    posting_date: datetime = field(default_factory=datetime.now)
    valuation_date: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialization validation"""
        if self.amount < 0:
            raise ValueError("Collateral amount cannot be negative")
        
        if self.market_value == 0.0:
            self.market_value = self.amount
        
        if isinstance(self.collateral_type, str):
            self.collateral_type = CollateralType(self.collateral_type)
    
    def effective_value(self, standard_haircut: float) -> float:
        """Calculate effective collateral value after haircut"""
        haircut = self.haircut_override if self.haircut_override is not None else standard_haircut
        return self.market_value * (1 - haircut)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'collateral_id': self.collateral_id,
            'collateral_type': self.collateral_type.value,
            'currency': self.currency,
            'amount': self.amount,
            'market_value': self.market_value,
            'haircut_override': self.haircut_override,
            'posting_date': self.posting_date.isoformat(),
            'valuation_date': self.valuation_date.isoformat()
        }

@dataclass
class NettingSet:
    """Enhanced netting set data model"""
    
    netting_set_id: str
    counterparty: str
    trades: List[Trade]
    
    # Collateral agreement parameters
    threshold: float = 0.0
    mta: float = 0.0  # Minimum Transfer Amount
    nica: float = 0.0  # Net Independent Collateral Amount
    
    # Agreement details
    master_agreement_type: str = "ISDA"
    master_agreement_date: Optional[datetime] = None
    csa_signed: bool = False
    csa_date: Optional[datetime] = None
    
    # Jurisdictional details
    governing_law: str = "NY"
    netting_enforceability: bool = True
    
    # Metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialization validation and processing"""
        if not self.netting_set_id:
            raise ValueError("Netting set ID is required")
        
        if not self.counterparty:
            raise ValueError("Counterparty is required")
        
        if not self.trades:
            raise ValueError("At least one trade is required")
        
        # Validate thresholds are non-negative
        if self.threshold < 0 or self.mta < 0:
            raise ValueError("Threshold and MTA must be non-negative")
    
    def total_notional(self) -> float:
        """Calculate total notional across all trades"""
        return sum(abs(trade.notional) for trade in self.trades)
    
    def total_mtm(self) -> float:
        """Calculate total mark-to-market value"""
        return sum(trade.mtm_value for trade in self.trades)
    
    def trade_count(self) -> int:
        """Get number of trades"""
        return len(self.trades)
    
    def asset_classes(self) -> List[AssetClass]:
        """Get unique asset classes in portfolio"""
        return list(set(trade.asset_class for trade in self.trades))
    
    def currencies(self) -> List[str]:
        """Get unique currencies in portfolio"""
        return list(set(trade.currency for trade in self.trades))
    
    def is_margined(self) -> bool:
        """Check if netting set is margined"""
        return self.threshold > 0 or self.mta > 0 or self.csa_signed
    
    def add_trade(self, trade: Trade):
        """Add trade to netting set"""
        if trade.counterparty != self.counterparty:
            raise ValueError("Trade counterparty must match netting set counterparty")
        
        self.trades.append(trade)
        self.last_updated = datetime.now()
    
    def remove_trade(self, trade_id: str) -> bool:
        """Remove trade by ID, return True if removed"""
        initial_count = len(self.trades)
        self.trades = [t for t in self.trades if t.trade_id != trade_id]
        
        if len(self.trades) < initial_count:
            self.last_updated = datetime.now()
            return True
        return False
    
    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get trade by ID"""
        for trade in self.trades:
            if trade.trade_id == trade_id:
                return trade
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'netting_set_id': self.netting_set_id,
            'counterparty': self.counterparty,
            'trades': [trade.to_dict() for trade in self.trades],
            'threshold': self.threshold,
            'mta': self.mta,
            'nica': self.nica,
            'master_agreement_type': self.master_agreement_type,
            'master_agreement_date': self.master_agreement_date.isoformat() if self.master_agreement_date else None,
            'csa_signed': self.csa_signed,
            'csa_date': self.csa_date.isoformat() if self.csa_date else None,
            'governing_law': self.governing_law,
            'netting_enforceability': self.netting_enforceability,
            'created_date': self.created_date.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NettingSet':
        """Create NettingSet from dictionary"""
        
        # Convert trade dictionaries to Trade objects
        trades = [Trade.from_dict(trade_data) for trade_data in data.get('trades', [])]
        data['trades'] = trades
        
        # Convert date strings back to datetime
        date_fields = ['master_agreement_date', 'csa_date', 'created_date', 'last_updated']
        for field in date_fields:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)

@dataclass
class Portfolio:
    """Portfolio container for multiple netting sets"""
    
    portfolio_id: str
    portfolio_name: str
    netting_sets: List[NettingSet]
    
    # Portfolio metadata
    base_currency: str = "USD"
    portfolio_type: str = "Trading"
    business_unit: str = "Default"
    
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialization validation"""
        if not self.portfolio_id:
            raise ValueError("Portfolio ID is required")
        
        if not self.portfolio_name:
            raise ValueError("Portfolio name is required")
    
    def total_trades(self) -> int:
        """Total trades across all netting sets"""
        return sum(ns.trade_count() for ns in self.netting_sets)
    
    def total_notional(self) -> float:
        """Total notional across all netting sets"""
        return sum(ns.total_notional() for ns in self.netting_sets)
    
    def total_counterparties(self) -> int:
        """Number of unique counterparties"""
        counterparties = set(ns.counterparty for ns in self.netting_sets)
        return len(counterparties)
    
    def add_netting_set(self, netting_set: NettingSet):
        """Add netting set to portfolio"""
        self.netting_sets.append(netting_set)
        self.last_updated = datetime.now()
    
    def get_netting_set(self, netting_set_id: str) -> Optional[NettingSet]:
        """Get netting set by ID"""
        for ns in self.netting_sets:
            if ns.netting_set_id == netting_set_id:
                return ns
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'portfolio_id': self.portfolio_id,
            'portfolio_name': self.portfolio_name,
            'netting_sets': [ns.to_dict() for ns in self.netting_sets],
            'base_currency': self.base_currency,
            'portfolio_type': self.portfolio_type,
            'business_unit': self.business_unit,
            'created_date': self.created_date.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }

# Utility functions for trade creation and validation

def create_interest_rate_swap(
    trade_id: str,
    counterparty: str,
    notional: float,
    currency: str = "USD",
    maturity_years: float = 5.0,
    fixed_rate: float = 0.05,
    mtm_value: float = 0.0
) -> Trade:
    """Helper function to create interest rate swap"""
    
    maturity_date = datetime.now() + timedelta(days=int(maturity_years * 365))
    
    return Trade(
        trade_id=trade_id,
        counterparty=counterparty,
        asset_class=AssetClass.INTEREST_RATE,
        trade_type=TradeType.SWAP,
        notional=notional,
        currency=currency,
        underlying=f"Interest Rate - {currency}",
        maturity_date=maturity_date,
        mtm_value=mtm_value,
        delta=1.0
    )

def create_fx_forward(
    trade_id: str,
    counterparty: str,
    notional: float,
    base_currency: str = "USD",
    quote_currency: str = "EUR",
    maturity_years: float = 1.0,
    forward_rate: float = 1.10,
    mtm_value: float = 0.0
) -> Trade:
    """Helper function to create FX forward"""
    
    maturity_date = datetime.now() + timedelta(days=int(maturity_years * 365))
    
    return Trade(
        trade_id=trade_id,
        counterparty=counterparty,
        asset_class=AssetClass.FOREIGN_EXCHANGE,
        trade_type=TradeType.FORWARD,
        notional=notional,
        currency=base_currency,
        underlying=f"{base_currency}/{quote_currency}",
        maturity_date=maturity_date,
        mtm_value=mtm_value,
        delta=1.0
    )

def create_equity_option(
    trade_id: str,
    counterparty: str,
    notional: float,
    currency: str = "USD",
    underlying_ticker: str = "SPY",
    maturity_years: float = 0.25,
    option_type: str = "Call",
    delta: float = 0.5,
    mtm_value: float = 0.0
) -> Trade:
    """Helper function to create equity option"""
    
    maturity_date = datetime.now() + timedelta(days=int(maturity_years * 365))
    
    return Trade(
        trade_id=trade_id,
        counterparty=counterparty,
        asset_class=AssetClass.EQUITY,
        trade_type=TradeType.OPTION,
        notional=notional,
        currency=currency,
        underlying=f"{underlying_ticker} {option_type}",
        maturity_date=maturity_date,
        mtm_value=mtm_value,
        delta=delta
    )
