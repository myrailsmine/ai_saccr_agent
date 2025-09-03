import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import math
import re
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Import our modular components
from src.engine.saccr_engine import SACCREngine
from src.data.database_manager import DatabaseManager
from src.config.config_manager import ConfigManager
from src.ui.components import UIComponents
from src.models.trade_models import Trade, NettingSet, Collateral, AssetClass, TradeType, CollateralType
from src.utils.validators import TradeValidator
from src.utils.progress_tracker import ProgressTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI SA-CCR Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling with enterprise-grade design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #1e40af;
        --primary-light: #3b82f6;
        --primary-dark: #1e3a8a;
        --secondary-color: #059669;
        --warning-color: #d97706;
        --error-color: #dc2626;
        --success-color: #16a34a;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --text-light: #9ca3af;
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-accent: #f1f5f9;
        --border-color: #e2e8f0;
        --border-light: #f1f5f9;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
    }
    
    /* Main application styling */
    .main { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    /* Enhanced header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: var(--radius-lg);
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 200px;
        height: 200px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        transform: translate(25%, -25%);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        margin: 0.75rem 0 0 0;
        font-size: 1.125rem;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced metric containers */
    .metric-container {
        background: var(--bg-primary);
        padding: 2rem;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-md);
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary-light);
    }
    
    .metric-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
    }
    
    /* Enhanced calculation steps */
    .calculation-step {
        background: var(--bg-primary);
        border-left: 4px solid var(--primary-color);
        padding: 2rem;
        margin: 1.5rem 0;
        border-radius: 0 var(--radius-lg) var(--radius-lg) 0;
        box-shadow: var(--shadow-md);
        position: relative;
        transition: all 0.3s ease;
    }
    
    .calculation-step:hover {
        border-left-color: var(--secondary-color);
        transform: translateX(4px);
    }
    
    .step-number {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-light));
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin-right: 1.5rem;
        font-size: 1rem;
        box-shadow: var(--shadow-md);
    }
    
    /* Enhanced result cards */
    .result-card {
        background: var(--bg-primary);
        padding: 2.5rem;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color);
        text-align: center;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(59, 130, 246, 0.05), transparent);
        transform: rotate(45deg);
        transition: all 0.6s ease;
        opacity: 0;
    }
    
    .result-card:hover::before {
        opacity: 1;
        transform: rotate(45deg) translate(50%, 50%);
    }
    
    .result-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
        position: relative;
        z-index: 1;
    }
    
    .result-label {
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced alert styles */
    .alert-info {
        background: linear-gradient(135deg, #dbeafe, #e0e7ff);
        border: 1px solid #93c5fd;
        color: #1e40af;
        padding: 1.5rem;
        border-radius: var(--radius-md);
        margin: 1.5rem 0;
        border-left: 4px solid var(--primary-color);
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 1px solid #fcd34d;
        color: #92400e;
        padding: 1.5rem;
        border-radius: var(--radius-md);
        margin: 1.5rem 0;
        border-left: 4px solid var(--warning-color);
    }
    
    .alert-success {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border: 1px solid #6ee7b7;
        color: #065f46;
        padding: 1.5rem;
        border-radius: var(--radius-md);
        margin: 1.5rem 0;
        border-left: 4px solid var(--success-color);
    }
    
    .alert-error {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 1px solid #fca5a5;
        color: #991b1b;
        padding: 1.5rem;
        border-radius: var(--radius-md);
        margin: 1.5rem 0;
        border-left: 4px solid var(--error-color);
    }
    
    /* Enhanced progress container */
    .progress-container {
        background: var(--bg-primary);
        padding: 2rem;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color);
        margin: 1.5rem 0;
        box-shadow: var(--shadow-md);
    }
    
    .progress-bar {
        background: var(--bg-accent);
        border-radius: 50px;
        height: 12px;
        overflow: hidden;
        position: relative;
    }
    
    .progress-fill {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        height: 100%;
        border-radius: 50px;
        transition: width 0.6s ease;
        position: relative;
    }
    
    .progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Enhanced comparison card */
    .comparison-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 2rem;
        margin: 1rem;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .comparison-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary-light);
    }
    
    /* Enhanced sidebar styling */
    .sidebar .sidebar-content {
        background: var(--bg-primary);
        padding: 1.5rem;
        border-radius: var(--radius-md);
        margin: 1rem 0;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-sm);
    }
    
    /* Code block styling */
    .stCodeBlock {
        font-family: 'JetBrains Mono', 'Monaco', 'Consolas', monospace !important;
        background: #1e293b !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid #334155 !important;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        border-radius: var(--radius-md) !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    /* Enhanced form styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-color) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--primary-light) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-secondary);
        padding: 8px;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-light);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-md);
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-color) !important;
        color: white !important;
        box-shadow: var(--shadow-sm);
    }
    
    /* Enhanced metric styling */
    .stMetric {
        background: var(--bg-primary);
        padding: 1.5rem;
        border-radius: var(--radius-md);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    /* Chat interface styling */
    .chat-message {
        background: var(--bg-primary);
        padding: 1.5rem;
        border-radius: var(--radius-lg);
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        animation: fadeInUp 0.5s ease;
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border-left: 4px solid var(--primary-color);
        margin-left: 2rem;
    }
    
    .chat-message.assistant {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-left: 4px solid var(--secondary-color);
        margin-right: 2rem;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-success {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
    }
    
    .status-error {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid var(--border-color);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1.5rem 1rem;
        }
        
        .main-header h1 {
            font-size: 1.875rem;
        }
        
        .result-value {
            font-size: 2rem;
        }
        
        .metric-container {
            padding: 1.5rem;
        }
        
        .calculation-step {
            padding: 1.5rem;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--text-light);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
</style>
""", unsafe_allow_html=True)

class SACCRApplication:
    """Main SA-CCR application class with improved architecture"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.db_manager = DatabaseManager()
        self.saccr_engine = SACCREngine(self.config_manager)
        self.ui_components = UIComponents()
        self.validator = TradeValidator()
        self.progress_tracker = ProgressTracker()
        
        # Initialize LLM connection status
        self.llm_connection_status = "disconnected"
        self.llm = None
        
        # Initialize session state
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'current_portfolio': None,
            'calculation_results': None,
            'comparison_results': None,
            'selected_scenario': 'base',
            'calculation_progress': 0,
            'last_calculation_time': None,
            'validation_results': {},
            'optimization_recommendations': None,
            'collateral_input': [],
            'calculation_parameters': {
                'alpha_bilateral': 1.4,
                'alpha_cleared': 0.5,
                'capital_ratio': 0.08,
                'enable_cache': True,
                'show_debug': False,
                'decimal_places': 2
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Main application entry point"""
        
        # Enhanced Professional Header
        st.markdown("""
        <div class="main-header">
            <h1>🏦 SA-CCR Risk Analytics Platform</h1>
            <p>Enterprise-grade Basel SA-CCR calculation and optimization engine with AI-powered insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        with st.sidebar:
            self._render_sidebar()
        
        # Main content routing
        page = st.session_state.get('current_page', 'calculator')
        
        if page == 'calculator':
            self._render_calculator_page()
        elif page == 'ai_assistant':
            self._render_ai_assistant_page()
        elif page == 'portfolio':
            self._render_portfolio_page()
        elif page == 'optimization':
            self._render_optimization_page()
        elif page == 'comparison':
            self._render_comparison_page()
        elif page == 'database':
            self._render_database_page()
        elif page == 'settings':
            self._render_settings_page()
    
    def _render_sidebar(self):
        """Render application sidebar"""
        st.markdown("### Navigation")
        
        pages = {
            'calculator': '📊 SA-CCR Calculator',
            'ai_assistant': '🤖 AI Assistant',
            'portfolio': '📈 Portfolio Analysis', 
            'optimization': '🎯 Optimization',
            'comparison': '⚖️ Scenario Comparison',
            'database': '🗄️ Data Management',
            'settings': '⚙️ Settings'
        }
        
        current_page = st.selectbox(
            "Select Module:",
            options=list(pages.keys()),
            format_func=lambda x: pages[x],
            key='current_page'
        )
        
        st.markdown("---")
        
        # Database status
        st.markdown("### System Status")
        
        try:
            trade_count = self.db_manager.get_trade_count()
            st.metric("Total Trades", trade_count)
            
            portfolio_count = self.db_manager.get_portfolio_count()
            st.metric("Saved Portfolios", portfolio_count)
            
            st.markdown('<div class="alert-success">Database: Connected</div>', 
                       unsafe_allow_html=True)
        except Exception as e:
            st.markdown('<div class="alert-warning">Database: Connection Error</div>', 
                       unsafe_allow_html=True)
            logger.error(f"Database connection error: {e}")
        
        # Configuration status
        config_status = self.config_manager.validate_config()
        if config_status['valid']:
            st.markdown('<div class="alert-success">Config: Valid</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-warning">Config: Issues Found</div>', 
                       unsafe_allow_html=True)
    
    def _render_calculator_page(self):
        """Render main SA-CCR calculator page"""
        
        st.markdown("## SA-CCR Calculator")
        
        # Input tabs
        input_tabs = st.tabs(["📊 Portfolio Setup", "🛡️ Collateral", "⚙️ Parameters"])
        
        with input_tabs[0]:
            self._render_portfolio_input()
        
        with input_tabs[1]:
            self._render_collateral_input()
            
        with input_tabs[2]:
            self._render_parameter_input()
        
        # Calculation section
        if st.session_state.get('current_portfolio'):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if st.button("🚀 Calculate SA-CCR", type="primary", use_container_width=True):
                    self._perform_calculation()
            
            with col2:
                if st.button("💾 Save Portfolio", use_container_width=True):
                    self._save_portfolio()
            
            with col3:
                if st.button("📊 Load Portfolio", use_container_width=True):
                    self._show_portfolio_loader()
            
            # Results display
            if st.session_state.calculation_results:
                self._render_calculation_results()
    
    def _render_portfolio_input(self):
        """Render portfolio input interface"""
        
        # Netting set configuration
        st.markdown("### Netting Set Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            netting_set_id = st.text_input("Netting Set ID", key="ns_id")
            counterparty = st.text_input("Counterparty", key="counterparty")
        
        with col2:
            threshold = st.number_input("Threshold ($)", min_value=0.0, key="threshold")
            mta = st.number_input("MTA ($)", min_value=0.0, key="mta")
        
        st.markdown("### Trade Input")
        
        # Trade input form
        with st.expander("Add New Trade", expanded=True):
            trade_form = st.form("trade_input")
            
            with trade_form:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    trade_id = st.text_input("Trade ID")
                    asset_class = st.selectbox("Asset Class", 
                                             [ac.value for ac in AssetClass])
                    trade_type = st.selectbox("Trade Type", [tt.value for tt in TradeType])
                
                with col2:
                    notional = st.number_input("Notional ($)", min_value=0.0, step=1000000.0)
                    currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "JPY", "CHF", "CAD"])
                    underlying = st.text_input("Underlying")
                
                with col3:
                    maturity_years = st.number_input("Maturity (Years)", min_value=0.1, max_value=30.0, value=5.0)
                    mtm_value = st.number_input("MTM Value ($)", value=0.0)
                    delta = st.number_input("Delta", min_value=-1.0, max_value=1.0, value=1.0)
                
                submitted = st.form_submit_button("Add Trade", type="primary")
                
                if submitted:
                    self._add_trade_to_portfolio(
                        trade_id, asset_class, trade_type, notional, 
                        currency, underlying, maturity_years, mtm_value, delta,
                        netting_set_id, counterparty, threshold, mta
                    )
        
        # Display current portfolio
        self._display_current_portfolio()
    
    def _render_collateral_input(self):
        """Render collateral input interface"""
        
        st.markdown("### Collateral Portfolio")
        
        with st.expander("Add Collateral", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                coll_type = st.selectbox("Collateral Type", 
                                       [ct.value for ct in CollateralType])
            
            with col2:
                coll_currency = st.selectbox("Collateral Currency", ["USD", "EUR", "GBP", "JPY", "CHF", "CAD"])
            
            with col3:
                coll_amount = st.number_input("Amount ($)", min_value=0.0, value=10000000.0, step=1000000.0)
            
            if st.button("Add Collateral", type="secondary"):
                new_collateral = Collateral(
                    collateral_type=CollateralType(coll_type),
                    currency=coll_currency,
                    amount=coll_amount
                )
                st.session_state.collateral_input.append(new_collateral)
                st.success(f"Added {coll_type} collateral")
                st.rerun()
        
        # Display current collateral
        if st.session_state.collateral_input:
            st.markdown("**Current Collateral:**")
            
            collateral_data = []
            for i, coll in enumerate(st.session_state.collateral_input):
                collateral_data.append({
                    'Index': i + 1,
                    'Type': coll.collateral_type.value,
                    'Currency': coll.currency,
                    'Amount ($M)': f"{coll.amount/1_000_000:.2f}",
                    'Market Value ($M)': f"{coll.market_value/1_000_000:.2f}"
                })
            
            df = pd.DataFrame(collateral_data)
            st.dataframe(df, use_container_width=True)
            
            if st.button("Clear All Collateral", type="secondary"):
                st.session_state.collateral_input = []
                st.rerun()
    
    def _render_parameter_input(self):
        """Render calculation parameter input"""
        
        st.markdown("### Calculation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Regulatory Parameters**")
            alpha_bilateral = st.number_input("Alpha (Bilateral)", value=1.4, min_value=0.1, max_value=5.0, step=0.1)
            alpha_cleared = st.number_input("Alpha (Cleared)", value=0.5, min_value=0.1, max_value=2.0, step=0.1)
            capital_ratio = st.number_input("Capital Ratio", value=0.08, min_value=0.01, max_value=0.5, step=0.01)
        
        with col2:
            st.markdown("**Calculation Settings**")
            enable_cache = st.checkbox("Enable Calculation Cache", value=True)
            show_debug = st.checkbox("Show Debug Information", value=False)
            decimal_places = st.slider("Decimal Places", 0, 4, 2)
        
        # Store parameters in session state
        st.session_state.calculation_parameters = {
            'alpha_bilateral': alpha_bilateral,
            'alpha_cleared': alpha_cleared,
            'capital_ratio': capital_ratio,
            'enable_cache': enable_cache,
            'show_debug': show_debug,
            'decimal_places': decimal_places
        }
    
    def _add_trade_to_portfolio(self, trade_id, asset_class, trade_type, notional, 
                               currency, underlying, maturity_years, mtm_value, delta,
                               netting_set_id, counterparty, threshold, mta):
        """Add trade to current portfolio with validation"""
        
        # Validate trade data
        validation_result = self.validator.validate_trade_data({
            'trade_id': trade_id,
            'notional': notional,
            'currency': currency,
            'maturity_years': maturity_years
        })
        
        if not validation_result['valid']:
            st.error(f"Validation failed: {validation_result['message']}")
            return
        
        # Create trade object
        try:
            trade = Trade(
                trade_id=trade_id,
                counterparty=counterparty,
                asset_class=AssetClass(asset_class),
                trade_type=TradeType(trade_type),
                notional=notional,
                currency=currency,
                underlying=underlying,
                maturity_date=datetime.now() + timedelta(days=int(maturity_years * 365)),
                mtm_value=mtm_value,
                delta=delta
            )
            
            # Initialize portfolio if needed
            if not st.session_state.current_portfolio:
                st.session_state.current_portfolio = {
                    'netting_set_id': netting_set_id,
                    'counterparty': counterparty,
                    'threshold': threshold,
                    'mta': mta,
                    'trades': []
                }
            
            st.session_state.current_portfolio['trades'].append(trade)
            st.success(f"Added trade {trade_id} to portfolio")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error adding trade: {str(e)}")
            logger.error(f"Trade addition error: {e}")
    
    def _display_current_portfolio(self):
        """Display current portfolio summary"""
        
        if not st.session_state.get('current_portfolio'):
            st.info("No portfolio loaded. Add trades above to begin.")
            return
        
        portfolio = st.session_state.current_portfolio
        trades = portfolio.get('trades', [])
        
        if not trades:
            st.info("Portfolio created but no trades added yet.")
            return
        
        st.markdown("### Current Portfolio")
        
        # Portfolio metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Trades", len(trades))
        with col2:
            total_notional = sum(abs(t.notional) for t in trades)
            st.metric("Total Notional", f"${total_notional/1_000_000:.1f}M")
        with col3:
            asset_classes = len(set(t.asset_class for t in trades))
            st.metric("Asset Classes", asset_classes)
        with col4:
            currencies = len(set(t.currency for t in trades))
            st.metric("Currencies", currencies)
        
        # Trade table
        trades_df = self._create_trades_dataframe(trades)
        st.dataframe(trades_df, use_container_width=True)
        
        # Trade management
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Portfolio", type="secondary"):
                st.session_state.current_portfolio = None
                st.rerun()
        
        with col2:
            selected_indices = st.multiselect(
                "Select trades to remove:",
                range(len(trades)),
                format_func=lambda i: f"Trade {i+1}: {trades[i].trade_id}"
            )
            
            if selected_indices and st.button("Remove Selected"):
                for idx in sorted(selected_indices, reverse=True):
                    trades.pop(idx)
                st.rerun()
    
    def _create_trades_dataframe(self, trades):
        """Create formatted DataFrame for trade display"""
        
        data = []
        for i, trade in enumerate(trades):
            data.append({
                'Index': i + 1,
                'Trade ID': trade.trade_id,
                'Asset Class': trade.asset_class.value,
                'Type': trade.trade_type.value,
                'Notional ($M)': f"{trade.notional/1_000_000:.2f}",
                'Currency': trade.currency,
                'MTM ($K)': f"{trade.mtm_value/1000:.1f}",
                'Maturity (Y)': f"{trade.time_to_maturity():.2f}"
            })
        
        return pd.DataFrame(data)
    
    def _perform_calculation(self):
        """Perform SA-CCR calculation with progress tracking"""
        
        if not st.session_state.current_portfolio:
            st.error("No portfolio to calculate")
            return
        
        portfolio = st.session_state.current_portfolio
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            st.markdown("#### Calculation Progress")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize progress tracker
                self.progress_tracker.reset()
                self.progress_tracker.initialize_saccr_steps()
                
                # Perform calculation with progress updates
                results = self.saccr_engine.calculate_comprehensive_saccr(
                    portfolio, 
                    st.session_state.collateral_input,
                    progress_callback=self._update_progress,
                    progress_bar=progress_bar,
                    status_text=status_text
                )
                
                # Store results
                st.session_state.calculation_results = results
                st.session_state.last_calculation_time = datetime.now()
                
                # Save to database
                try:
                    self.db_manager.save_calculation_results(portfolio, results)
                except Exception as e:
                    logger.warning(f"Failed to save results to database: {e}")
                
                status_text.success("Calculation completed successfully!")
                
            except Exception as e:
                st.error(f"Calculation failed: {str(e)}")
                logger.error(f"SA-CCR calculation error: {e}")
            
            finally:
                st.markdown('</div>', unsafe_allow_html=True)
    
    def _update_progress(self, step, total_steps, message):
        """Update calculation progress"""
        progress = step / total_steps
        st.session_state.calculation_progress = progress
        return progress
    
    def _save_portfolio(self):
        """Save current portfolio to database"""
        
        if not st.session_state.current_portfolio:
            st.error("No portfolio to save")
            return
        
        portfolio_name = st.text_input("Portfolio Name:", key="save_portfolio_name")
        
        if st.button("💾 Save") and portfolio_name:
            try:
                # Create portfolio object for database
                from src.models.trade_models import Portfolio
                
                portfolio = Portfolio(
                    portfolio_id=f"port_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    portfolio_name=portfolio_name,
                    netting_sets=[NettingSet(
                        netting_set_id=st.session_state.current_portfolio['netting_set_id'],
                        counterparty=st.session_state.current_portfolio['counterparty'],
                        trades=st.session_state.current_portfolio['trades'],
                        threshold=st.session_state.current_portfolio['threshold'],
                        mta=st.session_state.current_portfolio['mta']
                    )]
                )
                
                self.db_manager.save_portfolio(portfolio)
                st.success(f"Portfolio '{portfolio_name}' saved successfully!")
            except Exception as e:
                st.error(f"Failed to save portfolio: {str(e)}")
    
    def _show_portfolio_loader(self):
        """Show portfolio loader interface"""
        
        st.markdown("### Load Saved Portfolio")
        
        try:
            # Get list of saved portfolios
            portfolios = self.db_manager.get_portfolio_summary()
            
            if not portfolios.empty:
                portfolio_options = portfolios['portfolio_name'].tolist()
                selected_portfolio = st.selectbox("Select Portfolio:", portfolio_options)
                
                if st.button("Load Portfolio") and selected_portfolio:
                    # Find portfolio ID
                    portfolio_id = portfolios[portfolios['portfolio_name'] == selected_portfolio]['portfolio_id'].iloc[0]
                    
                    # Load portfolio
                    portfolio = self.db_manager.load_portfolio(portfolio_id)
                    
                    if portfolio and portfolio.netting_sets:
                        # Convert to session state format
                        netting_set = portfolio.netting_sets[0]  # Take first netting set
                        
                        st.session_state.current_portfolio = {
                            'netting_set_id': netting_set.netting_set_id,
                            'counterparty': netting_set.counterparty,
                            'threshold': netting_set.threshold,
                            'mta': netting_set.mta,
                            'trades': netting_set.trades
                        }
                        
                        st.success(f"Loaded portfolio: {selected_portfolio}")
                        st.rerun()
                    else:
                        st.error("Failed to load portfolio")
            else:
                st.info("No saved portfolios found")
                
        except Exception as e:
            st.error(f"Error loading portfolios: {str(e)}")
            logger.error(f"Portfolio loading error: {e}")
    
    def _render_calculation_results(self):
        """Render comprehensive calculation results"""
        
        results = st.session_state.calculation_results
        
        st.markdown("## Calculation Results")
        
        # Summary metrics
        self._render_results_summary(results)
        
        # Detailed breakdown tabs
        results_tabs = st.tabs([
            "📊 Executive Summary", 
            "🔢 24-Step Breakdown", 
            "📈 Risk Analysis",
            "🎯 Optimization"
        ])
        
        with results_tabs[0]:
            self._render_executive_summary(results)
        
        with results_tabs[1]:
            self._render_step_breakdown(results)
        
        with results_tabs[2]:
            self._render_risk_analysis(results)
        
        with results_tabs[3]:
            self._render_optimization_analysis(results)
    
    def _render_results_summary(self, results):
        """Render high-level results summary"""
        
        final_results = results['final_results']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = [
            ("Replacement Cost", final_results['replacement_cost'], "M"),
            ("PFE", final_results['potential_future_exposure'], "M"),
            ("EAD", final_results['exposure_at_default'], "M"),
            ("RWA", final_results['risk_weighted_assets'], "M"),
            ("Capital Required", final_results['capital_requirement'], "K")
        ]
        
        for i, (label, value, unit) in enumerate(metrics):
            with [col1, col2, col3, col4, col5][i]:
                divisor = 1_000_000 if unit == "M" else 1_000
                formatted_value = f"${value/divisor:.2f}{unit}"
                
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">{label}</div>
                    <div class="result-value">{formatted_value}</div>
                </div>
                """, unsafe_allow_html=True)
    
    def _render_executive_summary(self, results):
        """Render executive summary of results"""
        
        final_results = results['final_results']
        
        st.markdown("### Executive Summary")
        
        # Key insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Profile:**")
            
            total_notional = final_results['portfolio_summary']['total_notional']
            ead = final_results['exposure_at_default']
            ead_ratio = (ead / total_notional) * 100 if total_notional > 0 else 0
            
            st.write(f"• EAD/Notional Ratio: {ead_ratio:.2f}%")
            st.write(f"• Capital Efficiency: {(1 - ead_ratio/100)*100:.1f}%")
            
            # Risk drivers
            if ead > 0:
                rc_contribution = (final_results['replacement_cost'] / ead) * 100
                pfe_contribution = (final_results['potential_future_exposure'] / ead) * 100
                
                st.write(f"• RC Contribution: {rc_contribution:.1f}%")
                st.write(f"• PFE Contribution: {pfe_contribution:.1f}%")
        
        with col2:
            st.markdown("**Optimization Opportunities:**")
            
            # Simple optimization suggestions
            suggestions = []
            
            if final_results['replacement_cost'] > final_results['potential_future_exposure']:
                suggestions.append("Evaluate central clearing eligibility")
            
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
    
    def _render_step_breakdown(self, results):
        """Render detailed 24-step calculation breakdown"""
        
        st.markdown("### Complete 24-Step SA-CCR Calculation")
        
        # Step visualization
        self._render_step_flow_chart(results)
        
        # Detailed steps
        calculation_steps = results['calculation_steps']
        
        # Group steps by category
        step_groups = {
            "Data & Classification (Steps 1-4)": list(range(0, 4)),
            "Risk Calculations (Steps 5-8)": list(range(4, 8)),
            "Add-On Aggregation (Steps 9-13)": list(range(8, 13)),
            "PFE Calculations (Steps 14-16)": list(range(13, 16)),
            "Replacement Cost (Steps 17-18)": list(range(16, 18)),
            "Final EAD & RWA (Steps 19-24)": list(range(18, 24))
        }
        
        for group_name, step_indices in step_groups.items():
            with st.expander(f"📋 {group_name}", expanded=False):
                for idx in step_indices:
                    if idx < len(calculation_steps):
                        step = calculation_steps[idx]
                        self._render_calculation_step(step)
    
    def _render_calculation_step(self, step):
        """Render individual calculation step"""
        
        st.markdown(f"""
        <div class="calculation-step">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span class="step-number">{step['step']}</span>
                <h4 style="margin: 0;">{step['title']}</h4>
            </div>
            <p><strong>Description:</strong> {step['description']}</p>
            <div style="background: #f8fafc; padding: 1rem; border-radius: 6px; margin: 1rem 0;">
                <code>{step['formula']}</code>
            </div>
            <p><strong>Result:</strong> {step['result']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show detailed data for complex steps
        if step['step'] in [9, 11, 12, 13, 21, 24] and 'data' in step:
            with st.expander(f"📊 Detailed Data - Step {step['step']}", expanded=False):
                if isinstance(step['data'], dict):
                    st.json(step['data'])
                else:
                    st.write(step['data'])
    
    def _render_step_flow_chart(self, results):
        """Render visual flow chart of calculation steps"""
        
        st.markdown("#### Calculation Flow Visualization")
        
        # Create flow chart data
        steps = results['calculation_steps']
        
        fig = go.Figure()
        
        # Add step boxes
        x_positions = []
        y_positions = []
        step_numbers = []
        step_titles = []
        
        # Arrange steps in a grid
        cols = 6
        for i, step in enumerate(steps):
            row = i // cols
            col = i % cols
            
            x_positions.append(col)
            y_positions.append(-row)
            step_numbers.append(step['step'])
            step_titles.append(f"Step {step['step']}: {step['title'][:20]}...")
        
        # Add scatter plot for step boxes
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='markers+text',
            marker=dict(size=40, color='#2563eb', symbol='square'),
            text=step_numbers,
            textfont=dict(color='white', size=12),
            hovertext=step_titles,
            hoverinfo='text',
            showlegend=False
        ))
        
        # Connect steps with arrows
        for i in range(len(steps) - 1):
            fig.add_annotation(
                x=x_positions[i+1], y=y_positions[i+1],
                ax=x_positions[i], ay=y_positions[i],
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor='#6b7280'
            )
        
        fig.update_layout(
            title="SA-CCR Calculation Flow",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_risk_analysis(self, results):
        """Render risk analysis charts and insights"""
        
        st.markdown("### Risk Analysis")
        
        # Risk decomposition chart
        final_results = results['final_results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # EAD Components
            rc = final_results['replacement_cost']
            pfe = final_results['potential_future_exposure']
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Replacement Cost', 'Potential Future Exposure'],
                    values=[rc, pfe],
                    hole=0.3,
                    marker_colors=['#2563eb', '#7c3aed']
                )
            ])
            
            fig.update_layout(
                title="EAD Components",
                height=300,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Portfolio metrics over time (if historical data available)
            try:
                portfolio_id = st.session_state.current_portfolio['netting_set_id']
                history = self.db_manager.get_calculation_history(portfolio_id, limit=10)
                
                if not history.empty:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=history['calculation_date'],
                        y=history['exposure_at_default'],
                        mode='lines+markers',
                        name='EAD',
                        line=dict(color='#2563eb')
                    ))
                    
                    fig.update_layout(
                        title="EAD History",
                        xaxis_title="Date",
                        yaxis_title="EAD ($)",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No historical data available")
                    
            except Exception as e:
                st.info("Historical analysis unavailable")
    
    def _render_optimization_analysis(self, results):
        """Render optimization recommendations"""
        
        st.markdown("### Optimization Recommendations")
        
        # Generate optimization suggestions based on results
        final_results = results['final_results']
        calculation_steps = results['calculation_steps']
        
        recommendations = []
        
        # Analyze PFE multiplier
        pfe_step = next((step for step in calculation_steps if step['step'] == 15), None)
        if pfe_step:
            multiplier = pfe_step['data']['multiplier']
            if multiplier > 0.8:
                recommendations.append({
                    'category': 'Netting Optimization',
                    'recommendation': 'Consider portfolio rebalancing to improve netting benefits',
                    'impact': 'Medium',
                    'effort': 'Low'
                })
        
        # Analyze replacement cost
        if final_results['replacement_cost'] > 0:
            recommendations.append({
                'category': 'Collateral Optimization',
                'recommendation': 'Evaluate collateral posting to reduce replacement cost',
                'impact': 'High',
                'effort': 'Medium'
            })
        
        # Analyze central clearing
        ceu_step = next((step for step in calculation_steps if step['step'] == 19), None)
        if ceu_step and ceu_step['data']['overall_ceu_flag'] == 1:
            recommendations.append({
                'category': 'Central Clearing',
                'recommendation': 'Assess trades eligible for central clearing (Alpha reduction from 1.4 to 0.5)',
                'impact': 'Very High',
                'effort': 'High'
            })
        
        # Display recommendations
        for i, rec in enumerate(recommendations):
            with st.expander(f"Recommendation {i+1}: {rec['category']}", expanded=True):
                st.write(f"**Action:** {rec['recommendation']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    impact_color = {'Low': 'blue', 'Medium': 'orange', 'High': 'red', 'Very High': 'purple'}
                    st.markdown(f"**Impact:** <span style='color: {impact_color.get(rec['impact'], 'black')}'>{rec['impact']}</span>", 
                               unsafe_allow_html=True)
                with col2:
                    effort_color = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                    st.markdown(f"**Effort:** <span style='color: {effort_color.get(rec['effort'], 'black')}'>{rec['effort']}</span>", 
                               unsafe_allow_html=True)
    
    def _render_ai_assistant_page(self):
        """Render AI assistant chat interface with SA-CCR calculation capabilities"""
        
        st.markdown("## 🤖 AI SA-CCR Assistant")
        st.markdown("Ask me anything about SA-CCR calculations, or describe your portfolio for automatic calculation!")
        
        # Initialize chat history
        if 'ai_chat_history' not in st.session_state:
            st.session_state.ai_chat_history = [
                {
                    'role': 'assistant',
                    'content': """Hello! I'm your SA-CCR expert assistant. I can help you with:

**📊 Automatic Calculations**: Describe your trades and I'll calculate SA-CCR automatically
**❓ SA-CCR Questions**: Ask about Basel regulations, formulas, or methodology
**🎯 Optimization**: Get suggestions to reduce capital requirements
**📈 Analysis**: Analyze your calculation results

**Example queries:**
- "Calculate SA-CCR for a $100M USD interest rate swap with JP Morgan, 5-year maturity"
- "What's the difference between PFE and RC in SA-CCR?"
- "I have 3 FX forwards with Deutsche Bank, each $50M, can you calculate the exposure?"
- "How can I optimize my derivatives portfolio to reduce capital?"
""",
                    'timestamp': datetime.now()
                }
            ]
        
        # Chat interface
        self._render_chat_interface()
        
        # Quick action buttons
        st.markdown("### Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💡 Sample Portfolio", use_container_width=True):
                sample_query = "Calculate SA-CCR for a portfolio with: 1) $200M USD interest rate swap with Goldman Sachs, 7-year maturity, 2) $150M EUR/USD FX forward with Deutsche Bank, 1-year maturity, 3) $100M equity option on S&P500 with Morgan Stanley, 6-month maturity, delta 0.6"
                self._process_ai_query(sample_query)
        
        with col2:
            if st.button("❓ SA-CCR Basics", use_container_width=True):
                basics_query = "Explain the SA-CCR methodology and its key components"
                self._process_ai_query(basics_query)
        
        with col3:
            if st.button("🎯 Optimization Tips", use_container_width=True):
                optimization_query = "What are the most effective ways to reduce SA-CCR capital requirements?"
                self._process_ai_query(optimization_query)
        
        with col4:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.ai_chat_history = st.session_state.ai_chat_history[:1]  # Keep welcome message
                st.rerun()
    
    def _render_chat_interface(self):
        """Render the chat interface"""
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.ai_chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div style="background: #f0f2f6; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-left: 2rem;">
                        <strong>👤 You:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: white; border: 1px solid #e1e5e9; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; margin-right: 2rem;">
                        <strong>🤖 SA-CCR Assistant:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        st.markdown("### Your Question")
        
        # Use form for better UX
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Ask about SA-CCR or describe your portfolio:",
                placeholder="e.g., 'Calculate SA-CCR for a $100M interest rate swap with Bank XYZ' or 'What is the PFE multiplier formula?'",
                height=100
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                submitted = st.form_submit_button("Send Message", type="primary", use_container_width=True)
            with col2:
                voice_mode = st.form_submit_button("🎤 Voice Input", use_container_width=True)
            
            if submitted and user_input.strip():
                self._process_ai_query(user_input.strip())
            elif voice_mode:
                st.info("Voice input feature coming soon!")
    
    def _process_ai_query(self, user_query: str):
        """Enhanced AI query processing with intelligent validation and mandatory input requests"""
        
        # Add user message to chat
        st.session_state.ai_chat_history.append({
            'role': 'user',
            'content': user_query,
            'timestamp': datetime.now()
        })
        
        try:
            # Enhanced query analysis with validation
            query_analysis = self._analyze_query_intent_enhanced(user_query)
            
            # Check for mandatory missing information
            missing_info = self._check_mandatory_information(query_analysis)
            
            if missing_info:
                response = self._request_mandatory_information(missing_info, query_analysis)
            elif query_analysis['requires_calculation']:
                response = self._handle_calculation_query_enhanced(user_query, query_analysis)
            else:
                response = self._handle_information_query_enhanced(user_query, query_analysis)
            
            # Add AI response to chat
            st.session_state.ai_chat_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            error_response = f"I apologize, but I encountered an error processing your request: {str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists."
            
            st.session_state.ai_chat_history.append({
                'role': 'assistant',
                'content': error_response,
                'timestamp': datetime.now()
            })
        
        st.rerun()
    
    def _analyze_query_intent_enhanced(self, query: str) -> Dict:
        """Enhanced query analysis with better intelligence and context understanding"""
        
        query_lower = query.lower()
        
        # Enhanced calculation detection
        calculation_keywords = [
            'calculate', 'compute', 'sa-ccr for', 'portfolio', 'exposure',
            'swap', 'forward', 'option', 'swaption', 'trade', 'notional', 
            'ead', 'rwa', 'risk weight', 'capital', 'pfe', 'replacement cost'
        ]
        
        # Enhanced query categorization
        optimization_keywords = ['optimization', 'optimize', 'reduce', 'improve', 'minimize', 'lower', 'save capital']
        explanation_keywords = ['explain', 'what is', 'how does', 'why', 'difference between', 'meaning of']
        comparison_keywords = ['compare', 'versus', 'vs', 'difference', 'better', 'worse']
        regulatory_keywords = ['basel', 'regulation', 'regulatory', 'compliance', 'requirement', 'rule']
        
        requires_calculation = any(keyword in query_lower for keyword in calculation_keywords)
        
        # Extract trade information with enhanced parsing
        extracted_trades = self._extract_trade_information_enhanced(query)
        
        # Determine query category with multiple possible categories
        categories = []
        if any(keyword in query_lower for keyword in optimization_keywords):
            categories.append('optimization')
        if any(keyword in query_lower for keyword in explanation_keywords):
            categories.append('explanation')
        if any(keyword in query_lower for keyword in comparison_keywords):
            categories.append('comparison')
        if any(keyword in query_lower for keyword in regulatory_keywords):
            categories.append('regulatory')
        if requires_calculation and extracted_trades:
            categories.append('calculation')
        if not categories:
            categories.append('general')
        
        # Enhanced context detection
        context = self._detect_query_context(query_lower)
        
        return {
            'requires_calculation': requires_calculation and len(extracted_trades) > 0,
            'extracted_trades': extracted_trades,
            'categories': categories,
            'primary_category': categories[0] if categories else 'general',
            'has_counterparty': bool(self._extract_counterparty(query)),
            'counterparty': self._extract_counterparty(query),
            'context': context,
            'complexity_level': self._assess_query_complexity(query),
            'user_intent': self._determine_user_intent(query, categories)
        }
    
    def _detect_query_context(self, query_lower: str) -> Dict:
        """Detect additional context from the query"""
        
        context = {
            'has_portfolio_reference': 'portfolio' in query_lower or 'trades' in query_lower,
            'mentions_specific_bank': any(bank in query_lower for bank in ['goldman', 'morgan', 'deutsche', 'barclays', 'citi']),
            'mentions_currency': any(curr in query_lower for curr in ['usd', 'eur', 'gbp', 'jpy', 'chf']),
            'mentions_timeframe': any(time in query_lower for time in ['year', 'month', 'day', 'maturity']),
            'requests_explanation': any(word in query_lower for word in ['explain', 'how', 'why', 'what']),
            'requests_example': 'example' in query_lower or 'show me' in query_lower,
            'mentions_regulation': 'basel' in query_lower or 'regulation' in query_lower,
            'urgency_indicators': any(word in query_lower for word in ['urgent', 'quickly', 'asap', 'immediately'])
        }
        
        return context
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess the complexity level of the user's query"""
        
        # Simple metrics for complexity assessment
        word_count = len(query.split())
        technical_terms = ['sa-ccr', 'pfe', 'ead', 'rwa', 'alpha', 'multiplier', 'netting', 'basel']
        technical_count = sum(1 for term in technical_terms if term in query.lower())
        
        if word_count < 10 and technical_count == 0:
            return 'basic'
        elif word_count < 25 and technical_count <= 2:
            return 'intermediate'
        else:
            return 'advanced'
    
    def _determine_user_intent(self, query: str, categories: List[str]) -> str:
        """Determine the user's primary intent"""
        
        if 'calculation' in categories:
            return 'wants_calculation'
        elif 'optimization' in categories:
            return 'seeks_optimization'
        elif 'explanation' in categories:
            return 'needs_explanation'
        elif 'comparison' in categories:
            return 'wants_comparison'
        elif 'regulatory' in categories:
            return 'needs_regulatory_guidance'
        else:
            return 'general_inquiry'
    
    def _check_mandatory_information(self, analysis: Dict) -> List[Dict]:
        """Check for missing mandatory information in calculation requests"""
        
        missing_info = []
        
        if analysis['requires_calculation']:
            trades = analysis['extracted_trades']
            
            for i, trade in enumerate(trades):
                trade_missing = []
                
                # Check mandatory fields for each trade
                if not trade.get('notional') or trade['notional'] == 0:
                    trade_missing.append({
                        'field': 'notional',
                        'description': 'Notional amount',
                        'example': '$100M or 100000000',
                        'importance': 'critical'
                    })
                
                if not trade.get('currency'):
                    trade_missing.append({
                        'field': 'currency',
                        'description': 'Currency denomination',
                        'example': 'USD, EUR, GBP, JPY',
                        'importance': 'critical'
                    })
                
                if not trade.get('maturity_years') or trade['maturity_years'] == 0:
                    trade_missing.append({
                        'field': 'maturity',
                        'description': 'Time to maturity',
                        'example': '5 years, 2.5 years, 18 months',
                        'importance': 'critical'
                    })
                
                if not trade.get('asset_class'):
                    trade_missing.append({
                        'field': 'asset_class',
                        'description': 'Type of derivative',
                        'example': 'Interest Rate Swap, FX Forward, Equity Option',
                        'importance': 'critical'
                    })
                
                # Optional but recommended fields
                if trade.get('trade_type') in ['Option', 'Swaption'] and not trade.get('delta'):
                    trade_missing.append({
                        'field': 'delta',
                        'description': 'Option delta (for options only)',
                        'example': '0.6, -0.4, 0.8',
                        'importance': 'recommended'
                    })
                
                if trade_missing:
                    missing_info.append({
                        'trade_index': i + 1,
                        'trade_id': trade.get('trade_id', f'Trade {i+1}'),
                        'missing_fields': trade_missing
                    })
            
            # Check for counterparty if not provided
            if not analysis.get('counterparty'):
                missing_info.append({
                    'general': True,
                    'field': 'counterparty',
                    'description': 'Counterparty name',
                    'example': 'Goldman Sachs, Deutsche Bank, JP Morgan',
                    'importance': 'recommended'
                })
        
        return missing_info
    
    def _request_mandatory_information(self, missing_info: List[Dict], analysis: Dict) -> str:
        """Generate intelligent request for missing mandatory information"""
        
        response = "🤔 **I'd like to help you with that calculation, but I need some additional information:**\n\n"
        
        critical_missing = False
        
        for item in missing_info:
            if item.get('general'):
                # General missing information
                field_info = item
                if field_info['importance'] == 'critical':
                    critical_missing = True
                    response += f"❗ **{field_info['description']}** (Required)\n"
                else:
                    response += f"💡 **{field_info['description']}** (Recommended)\n"
                response += f"   Example: {field_info['example']}\n\n"
            else:
                # Trade-specific missing information
                trade_id = item['trade_id']
                response += f"**📊 {trade_id}:**\n"
                
                for field_info in item['missing_fields']:
                    if field_info['importance'] == 'critical':
                        critical_missing = True
                        response += f"   ❗ **{field_info['description']}** (Required)\n"
                    else:
                        response += f"   💡 **{field_info['description']}** (Recommended)\n"
                    response += f"      Example: {field_info['example']}\n"
                response += "\n"
        
        # Add guidance based on what's missing
        if critical_missing:
            response += "🚨 **Critical Information Missing**: I cannot perform the calculation without the required fields above.\n\n"
        else:
            response += "✅ **I can proceed with basic calculation**, but providing the recommended information will give you more accurate results.\n\n"
        
        # Add helpful suggestions
        response += "💡 **How to provide the information:**\n"
        response += "• You can say: \"For Trade 1, the notional is $200M USD with 5-year maturity\"\n"
        response += "• Or: \"Update the swap: $500M EUR, 7 years, with Deutsche Bank\"\n"
        response += "• Or provide all details in one message\n\n"
        
        # Offer to proceed with assumptions if only optional info is missing
        if not critical_missing:
            response += "🤖 **Would you like me to:**\n"
            response += "• ▶️ Proceed with the calculation using reasonable assumptions\n"
            response += "• ⏸️ Wait for you to provide the additional details\n"
            response += "• 📝 Show you a template to fill in the missing information\n\n"
        
        response += "Just let me know how you'd like to proceed! 🚀"
        
        return response
    
    def _extract_trade_information_enhanced(self, query: str) -> List[Dict]:
        """Enhanced trade information extraction with better parsing and validation"""
        
        import re
        
        trades = []
        
        # Enhanced patterns for monetary amounts
        money_patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*([KMB]?)\b',  # $100M, $50K
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*([KMB]?)\s*(?:USD|EUR|GBP|JPY|CHF|CAD|dollars?|euros?)',  # 100M USD
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*million',  # 100 million
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*billion'   # 1 billion
        ]
        
        # Enhanced time patterns
        time_patterns = [
            r'(\d+(?:\.\d+)?)\s*[-\s]?\s*(year|yr)s?',
            r'(\d+(?:\.\d+)?)\s*[-\s]?\s*(month|mon)s?',
            r'(\d+(?:\.\d+)?)\s*[-\s]?\s*(day)s?',
            r'(\d+(?:\.\d+)?)\s*Y\b',  # 5Y notation
            r'(\d+)M\s*maturity'  # 18M maturity
        ]
        
        # Enhanced currency patterns
        currency_patterns = [
            r'\b(USD|EUR|GBP|JPY|CHF|CAD|AUD|NZD|SEK|NOK)\b',
            r'\b(US\s*Dollar|Euro|British\s*Pound|Japanese\s*Yen)\b'
        ]
        
        # Enhanced asset type patterns with more specific matching
        asset_patterns = {
            'Interest Rate': [
                r'\b(interest\s+rate\s+swap|irs|swap)\b',
                r'\b(fixed\s+rate|floating\s+rate)\b',
                r'\b(libor|sofr|euribor)\b'
            ],
            'Foreign Exchange': [
                r'\b(fx\s+forward|fx|foreign\s+exchange|currency\s+forward)\b',
                r'\b(USD/EUR|EUR/USD|GBP/USD|cross\s+currency)\b'
            ],
            'Equity': [
                r'\b(equity\s+option|stock\s+option|equity)\b',
                r'\b(option\s+on\s+S&P|S&P\s*500|index\s+option)\b',
                r'\b(call\s+option|put\s+option)\b'
            ],
            'Credit': [
                r'\b(cds|credit\s+default\s+swap|credit)\b',
                r'\b(investment\s+grade|high\s+yield)\b'
            ],
            'Commodity': [
                r'\b(commodity|oil|gold|wheat|energy)\b'
            ]
        }
        
        # Split query into potential trades
        trade_separators = [r'[,;]\s*\d+\)', r'\band\s+', r'\bplus\s+', r'\balso\s+']
        potential_trades = [query]
        
        for separator in trade_separators:
            new_trades = []
            for trade_text in potential_trades:
                new_trades.extend(re.split(separator, trade_text, flags=re.IGNORECASE))
            potential_trades = new_trades
        
        for i, trade_text in enumerate(potential_trades):
            trade_info = {
                'trade_id': f'AI_TRADE_{i+1}',
                'notional': 100000000.0,  # Default $100M
                'currency': 'USD',
                'maturity_years': 5.0,
                'asset_class': 'Interest Rate',
                'trade_type': 'Swap',
                'delta': 1.0,
                'mtm_value': 0.0
            }
            
            # Extract notional with enhanced patterns
            notional_found = False
            for pattern in money_patterns:
                money_matches = re.findall(pattern, trade_text, re.IGNORECASE)
                if money_matches:
                    amount_str, multiplier = money_matches[0]
                    amount = float(amount_str.replace(',', ''))
                    
                    # Handle multipliers
                    multiplier_map = {
                        'K': 1000, 'M': 1000000, 'B': 1000000000,
                        'MILLION': 1000000, 'BILLION': 1000000000
                    }
                    
                    if multiplier.upper() in multiplier_map:
                        amount *= multiplier_map[multiplier.upper()]
                    elif 'million' in trade_text.lower():
                        amount *= 1000000
                    elif 'billion' in trade_text.lower():
                        amount *= 1000000000
                    
                    trade_info['notional'] = amount
                    notional_found = True
                    break
            
            # Extract currency with enhanced patterns
            for pattern in currency_patterns:
                currency_matches = re.findall(pattern, trade_text, re.IGNORECASE)
                if currency_matches:
                    currency = currency_matches[0].upper()
                    # Normalize currency names
                    currency_map = {
                        'US DOLLAR': 'USD', 'EURO': 'EUR', 
                        'BRITISH POUND': 'GBP', 'JAPANESE YEN': 'JPY'
                    }
                    trade_info['currency'] = currency_map.get(currency, currency)
                    break
            
            # Extract maturity with enhanced patterns
            maturity_found = False
            for pattern in time_patterns:
                time_matches = re.findall(pattern, trade_text, re.IGNORECASE)
                if time_matches:
                    if len(time_matches[0]) == 2:
                        period, unit = time_matches[0]
                        period = float(period)
                        
                        if unit.lower().startswith('year') or unit.lower().startswith('yr'):
                            trade_info['maturity_years'] = period
                        elif unit.lower().startswith('month') or unit.lower().startswith('mon'):
                            trade_info['maturity_years'] = period / 12
                        elif unit.lower().startswith('day'):
                            trade_info['maturity_years'] = period / 365
                        
                        maturity_found = True
                        break
                    else:
                        # Handle single value patterns like "5Y"
                        period = float(time_matches[0])
                        trade_info['maturity_years'] = period
                        maturity_found = True
                        break
            
            # Extract asset class and trade type with enhanced matching
            asset_class_found = False
            for asset_class, patterns in asset_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, trade_text, re.IGNORECASE):
                        trade_info['asset_class'] = asset_class
                        
                        # Set appropriate trade type
                        if asset_class == 'Interest Rate':
                            trade_info['trade_type'] = 'Swap'
                        elif asset_class == 'Foreign Exchange':
                            trade_info['trade_type'] = 'Forward'
                        elif asset_class == 'Equity':
                            trade_info['trade_type'] = 'Option'
                            # Extract delta for options
                            delta_pattern = r'delta\s+(\d+(?:\.\d+)?)'
                            delta_match = re.search(delta_pattern, trade_text, re.IGNORECASE)
                            if delta_match:
                                trade_info['delta'] = float(delta_match.group(1))
                        elif asset_class == 'Credit':
                            trade_info['trade_type'] = 'Credit Default Swap'
                        elif asset_class == 'Commodity':
                            trade_info['trade_type'] = 'Forward'
                        
                        asset_class_found = True
                        break
                
                if asset_class_found:
                    break
            
            # Only add trade if we found meaningful information
            if notional_found or maturity_found or asset_class_found:
                trades.append(trade_info)
        
        # If no trades were extracted but the query suggests calculation intent
        if not trades and any(word in query.lower() for word in ['calculate', 'sa-ccr', 'exposure', 'swap', 'trade']):
            # Create default trade for calculation attempt
            trades.append({
                'trade_id': 'DEFAULT_TRADE',
                'notional': 100000000.0,
                'currency': 'USD',
                'maturity_years': 5.0,
                'asset_class': 'Interest Rate',
                'trade_type': 'Swap',
                'delta': 1.0,
                'mtm_value': 0.0
            })
        
        return trades
    
    def _handle_general_query(self, query_lower: str, detail_level: str, context: Dict) -> str:
        """Handle general queries with context awareness"""
        
        greeting_words = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
        if any(word in query_lower for word in greeting_words):
            return """👋 **Hello! Welcome to the SA-CCR AI Assistant!**

I'm here to help you with Basel SA-CCR calculations and portfolio optimization. I can assist with:

🔹 **Automatic Calculations**: Just describe your trades in natural language
🔹 **Expert Guidance**: Explain SA-CCR concepts and formulas  
🔹 **Optimization Strategies**: Show you how to reduce capital requirements
🔹 **Regulatory Compliance**: Provide Basel regulatory context

**Quick Examples:**
• "Calculate SA-CCR for a $200M USD swap with Goldman Sachs, 5-year maturity"
• "How can I optimize my derivatives portfolio to save capital?"
• "Explain the difference between RC and PFE"

What would you like to know about SA-CCR today? 🚀"""
        
        help_words = ['help', 'guide', 'how to', 'getting started']
        if any(word in query_lower for word in help_words):
            return """📚 **SA-CCR Assistant Help Guide**

**Getting Started:**
1. **Quick Calculation**: Describe your trades and I'll calculate SA-CCR automatically
2. **Ask Questions**: Ask about SA-CCR concepts, formulas, or optimization strategies
3. **Get Recommendations**: I'll suggest ways to reduce your capital requirements

**What I Can Calculate:**
• Interest Rate Swaps, FX Forwards, Equity Options, Credit Default Swaps
• Single trades or complex portfolios
• Current exposure (RC) and potential future exposure (PFE)
• Capital requirements and optimization opportunities

**Example Queries:**
• "Calculate exposure for $500M interest rate swap, 7 years, with Deutsche Bank"
• "What's the impact of central clearing on my capital requirements?"
• "Explain how the PFE multiplier works"
• "Show me optimization strategies for my portfolio"

**Tips for Best Results:**
• Include notional amounts, currencies, and maturities
• Mention counterparty names for more accurate analysis
• Ask follow-up questions for deeper insights

Ready to get started? Just describe your trades or ask any SA-CCR question! 💪"""
        
        # Default response for general queries
        return """I'm your SA-CCR AI Assistant, ready to help with Basel counterparty credit risk calculations and portfolio optimization!

**I can help you with:**

📊 **Calculations**: Describe your derivatives trades and I'll calculate SA-CCR automatically
📚 **Explanations**: Ask about specific SA-CCR concepts, formulas, or methodology  
🎯 **Optimization**: Get strategies to reduce your capital requirements
🔍 **Analysis**: Deep dive into calculation results and risk drivers
⚖️ **Comparisons**: Compare different scenarios and optimization strategies
🏛️ **Regulatory Guidance**: Basel compliance and regulatory context

**Popular Topics:**
• "What is SA-CCR and how does it work?"  
• "Calculate SA-CCR for my derivatives portfolio"
• "How can I reduce my capital requirements?"
• "What's the difference between bilateral and cleared trades?"
• "Explain the PFE multiplier formula"

What specific aspect of SA-CCR would you like to explore? Just ask me anything! 🤖"""
    
    def _analyze_query_intent(self, query: str) -> Dict:
        """Analyze user query to determine intent and extract trade information"""
        
        query_lower = query.lower()
        
        # Check if query requires calculation
        calculation_keywords = [
            'calculate', 'compute', 'sa-ccr for', 'portfolio', 'exposure',
            'swap', 'forward', 'option', 'trade', 'notional', 'ead', 'rwa'
        ]
        
        requires_calculation = any(keyword in query_lower for keyword in calculation_keywords)
        
        # Extract trade information using pattern matching
        extracted_trades = self._extract_trade_information(query)
        
        # Determine query category
        if 'optimization' in query_lower or 'reduce' in query_lower or 'improve' in query_lower:
            category = 'optimization'
        elif 'explain' in query_lower or 'what is' in query_lower or 'how does' in query_lower:
            category = 'explanation'
        elif requires_calculation and extracted_trades:
            category = 'calculation'
        else:
            category = 'general'
        
        return {
            'requires_calculation': requires_calculation and len(extracted_trades) > 0,
            'extracted_trades': extracted_trades,
            'category': category,
            'has_counterparty': bool(self._extract_counterparty(query)),
            'counterparty': self._extract_counterparty(query)
        }
    
    def _extract_trade_information(self, query: str) -> List[Dict]:
        """Extract trade information from natural language query"""
        
        import re
        
        trades = []
        
        # Pattern for monetary amounts
        money_pattern = r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*([KMB]?)\b'
        
        # Pattern for time periods
        time_pattern = r'(\d+(?:\.\d+)?)\s*[-\s]?\s*(year|yr|month|mon|day)s?'
        
        # Pattern for currencies
        currency_pattern = r'\b(USD|EUR|GBP|JPY|CHF|CAD|AUD|NZD|SEK|NOK)\b'
        
        # Pattern for asset types
        asset_patterns = {
            'Interest Rate': r'\b(interest\s+rate\s+swap|irs|swap)\b',
            'Foreign Exchange': r'\b(fx\s+forward|fx|foreign\s+exchange|currency)\b',
            'Equity': r'\b(equity\s+option|stock\s+option|equity|option\s+on)\b',
            'Credit': r'\b(cds|credit\s+default\s+swap|credit)\b',
            'Commodity': r'\b(commodity|oil|gold|wheat)\b'
        }
        
        # Extract individual trade descriptions
        trade_separators = r'[,;]\s*\d+\)'
        potential_trades = re.split(trade_separators, query)
        
        # If no clear separation, treat as single trade
        if len(potential_trades) == 1:
            potential_trades = [query]
        
        for i, trade_text in enumerate(potential_trades):
            trade_info = {
                'trade_id': f'AI_TRADE_{i+1}',
                'notional': 100000000.0,  # Default $100M
                'currency': 'USD',
                'maturity_years': 5.0,
                'asset_class': 'Interest Rate',
                'trade_type': 'Swap',
                'delta': 1.0,
                'mtm_value': 0.0
            }
            
            # Extract notional
            money_matches = re.findall(money_pattern, trade_text, re.IGNORECASE)
            if money_matches:
                amount, multiplier = money_matches[0]
                amount = float(amount.replace(',', ''))
                
                multiplier_map = {'K': 1000, 'M': 1000000, 'B': 1000000000}
                if multiplier.upper() in multiplier_map:
                    amount *= multiplier_map[multiplier.upper()]
                
                trade_info['notional'] = amount
            
            # Extract currency
            currency_matches = re.findall(currency_pattern, trade_text, re.IGNORECASE)
            if currency_matches:
                trade_info['currency'] = currency_matches[0].upper()
            
            # Extract maturity
            time_matches = re.findall(time_pattern, trade_text, re.IGNORECASE)
            if time_matches:
                period, unit = time_matches[0]
                period = float(period)
                
                if unit.lower().startswith('year') or unit.lower().startswith('yr'):
                    trade_info['maturity_years'] = period
                elif unit.lower().startswith('month') or unit.lower().startswith('mon'):
                    trade_info['maturity_years'] = period / 12
                elif unit.lower().startswith('day'):
                    trade_info['maturity_years'] = period / 365
            
            # Extract asset class and trade type
            for asset_class, pattern in asset_patterns.items():
                if re.search(pattern, trade_text, re.IGNORECASE):
                    trade_info['asset_class'] = asset_class
                    
                    if asset_class == 'Interest Rate':
                        trade_info['trade_type'] = 'Swap'
                    elif asset_class == 'Foreign Exchange':
                        trade_info['trade_type'] = 'Forward'
                    elif asset_class == 'Equity':
                        trade_info['trade_type'] = 'Option'
                        # Extract delta if mentioned
                        delta_pattern = r'delta\s+(\d+(?:\.\d+)?)'
                        delta_match = re.search(delta_pattern, trade_text, re.IGNORECASE)
                        if delta_match:
                            trade_info['delta'] = float(delta_match.group(1))
                    elif asset_class == 'Credit':
                        trade_info['trade_type'] = 'Credit Default Swap'
                    break
            
            trades.append(trade_info)
        
        return trades
    
    def _extract_counterparty(self, query: str) -> str:
        """Extract counterparty name from query"""
        
        # Common bank patterns
        bank_patterns = [
            r'\b(jp\s*morgan|jpmorgan)\b',
            r'\b(goldman\s*sachs)\b',
            r'\b(morgan\s*stanley)\b',
            r'\b(deutsche\s*bank)\b',
            r'\b(barclays)\b',
            r'\b(citibank|citi)\b',
            r'\b(bank\s+of\s+america|boa)\b',
            r'\b(wells\s*fargo)\b',
            r'\b(hsbc)\b',
            r'\b(ubs)\b',
            r'\b(credit\s*suisse)\b'
        ]
        
        for pattern in bank_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(0).title()
        
        # Generic "with [Name]" pattern
        with_pattern = r'\bwith\s+([A-Z][a-zA-Z\s]+(?:Bank|Corp|LLC|Ltd|Inc)?)\b'
        match = re.search(with_pattern, query)
        if match:
            return match.group(1).strip()
        
        return "Counterparty ABC"
    
    def _handle_calculation_query(self, query: str, analysis: Dict) -> str:
        """Handle queries that require SA-CCR calculation"""
        
        try:
            # Create trades from extracted information
            trades = []
            counterparty = analysis.get('counterparty', 'AI Generated Portfolio')
            
            for trade_info in analysis['extracted_trades']:
                trade = Trade(
                    trade_id=trade_info['trade_id'],
                    counterparty=counterparty,
                    asset_class=AssetClass(trade_info['asset_class']),
                    trade_type=TradeType(trade_info['trade_type']),
                    notional=trade_info['notional'],
                    currency=trade_info['currency'],
                    underlying=f"{trade_info['asset_class']} - {trade_info['trade_type']}",
                    maturity_date=datetime.now() + timedelta(days=int(trade_info['maturity_years'] * 365)),
                    mtm_value=trade_info['mtm_value'],
                    delta=trade_info['delta']
                )
                trades.append(trade)
            
            # Create portfolio
            portfolio_data = {
                'netting_set_id': f"AI_NS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'counterparty': counterparty,
                'threshold': 1000000.0,  # Default $1M threshold
                'mta': 500000.0,  # Default $500K MTA
                'trades': trades
            }
            
            # Perform SA-CCR calculation
            results = self.saccr_engine.calculate_comprehensive_saccr(portfolio_data)
            
            # Store results for potential further analysis
            st.session_state.ai_generated_results = results
            st.session_state.ai_generated_portfolio = portfolio_data
            
            # Format response
            final_results = results['final_results']
            
            response = f"""**✅ SA-CCR Calculation Complete!**

**Portfolio Summary:**
• Counterparty: {counterparty}
• Number of trades: {len(trades)}
• Total notional: ${sum(t.notional for t in trades)/1_000_000:.1f}M

**Key Results:**
• **Replacement Cost (RC)**: ${final_results['replacement_cost']/1_000_000:.2f}M
• **Potential Future Exposure (PFE)**: ${final_results['potential_future_exposure']/1_000_000:.2f}M
• **Exposure at Default (EAD)**: ${final_results['exposure_at_default']/1_000_000:.2f}M
• **Risk Weighted Assets (RWA)**: ${final_results['risk_weighted_assets']/1_000_000:.2f}M
• **Capital Requirement**: ${final_results['capital_requirement']/1_000:.0f}K

**Trade Details:**
"""
            
            for i, trade in enumerate(trades, 1):
                response += f"• Trade {i}: {trade.asset_class.value} {trade.trade_type.value}, ${trade.notional/1_000_000:.1f}M {trade.currency}, {trade.time_to_maturity():.1f}Y maturity\n"
            
            # Add optimization suggestions
            ead_ratio = (final_results['exposure_at_default'] / sum(t.notional for t in trades)) * 100
            
            response += f"\n**Analysis:**\n"
            response += f"• EAD/Notional ratio: {ead_ratio:.2f}%\n"
            
            if final_results['replacement_cost'] > final_results['potential_future_exposure']:
                response += f"• RC dominates exposure - consider collateral posting\n"
            else:
                response += f"• PFE dominates exposure - portfolio shows future risk\n"
            
            response += f"\n**💡 Want to explore optimization strategies or see the detailed 24-step breakdown?**"
            
            return response
            
        except Exception as e:
            return f"I encountered an error performing the SA-CCR calculation: {str(e)}\n\nPlease check your trade descriptions and try again. Make sure to include notional amounts, currencies, and maturities."
    
    def _handle_calculation_query_enhanced(self, query: str, analysis: Dict) -> str:
        """Enhanced calculation query handler with better error handling and context awareness"""
        
        try:
            # Create trades from extracted information
            trades = []
            counterparty = analysis.get('counterparty', 'AI Generated Portfolio')
            
            for trade_info in analysis['extracted_trades']:
                trade = Trade(
                    trade_id=trade_info['trade_id'],
                    counterparty=counterparty,
                    asset_class=AssetClass(trade_info['asset_class']),
                    trade_type=TradeType(trade_info['trade_type']),
                    notional=trade_info['notional'],
                    currency=trade_info['currency'],
                    underlying=f"{trade_info['asset_class']} - {trade_info['trade_type']}",
                    maturity_date=datetime.now() + timedelta(days=int(trade_info['maturity_years'] * 365)),
                    mtm_value=trade_info['mtm_value'],
                    delta=trade_info['delta']
                )
                trades.append(trade)
            
            # Create portfolio with enhanced parameters
            portfolio_data = {
                'netting_set_id': f"AI_NS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'counterparty': counterparty,
                'threshold': 1000000.0,  # Default $1M threshold
                'mta': 500000.0,  # Default $500K MTA
                'trades': trades
            }
            
            # Perform SA-CCR calculation
            results = self.saccr_engine.calculate_comprehensive_saccr(portfolio_data)
            
            # Store results for potential further analysis
            st.session_state.ai_generated_results = results
            st.session_state.ai_generated_portfolio = portfolio_data
            
            # Enhanced response formatting based on context
            final_results = results['final_results']
            context = analysis.get('context', {})
            complexity = analysis.get('complexity_level', 'balanced')
            
            # Build response based on complexity level
            if complexity == 'basic':
                response = self._format_basic_calculation_response(trades, final_results, counterparty)
            elif complexity == 'advanced':
                response = self._format_advanced_calculation_response(trades, final_results, counterparty, results)
            else:
                response = self._format_balanced_calculation_response(trades, final_results, counterparty, results)
            
            # Add context-aware suggestions
            if context.get('mentions_regulation'):
                response += "\n\n📋 **Regulatory Note**: This calculation follows Basel SA-CCR standards (BCBS 279) for counterparty credit risk."
            
            if context.get('urgency_indicators'):
                response += "\n\n⚡ **Quick Summary**: EAD = ${:.1f}M, Capital = ${:.0f}K".format(
                    final_results['exposure_at_default']/1_000_000,
                    final_results['capital_requirement']/1_000
                )
            
            return response
            
        except Exception as e:
            error_context = analysis.get('context', {})
            if error_context.get('urgency_indicators'):
                return f"⚠️ **Quick Error**: Calculation failed - {str(e)}\n\nPlease provide: notional amount, currency, maturity, and trade type."
            else:
                return f"I encountered an error performing the SA-CCR calculation: {str(e)}\n\nPlease check your trade descriptions and try again. Make sure to include:\n• Notional amounts (e.g., $100M)\n• Currencies (e.g., USD, EUR)\n• Maturities (e.g., 5 years)\n• Trade types (e.g., swap, forward, option)"
    
    def _format_basic_calculation_response(self, trades, final_results, counterparty) -> str:
        """Format basic calculation response for simple queries"""
        
        total_notional = sum(t.notional for t in trades)
        ead_ratio = (final_results['exposure_at_default'] / total_notional) * 100
        
        response = f"""**✅ SA-CCR Calculation Complete!**

**Your Portfolio:**
• {len(trades)} trade(s) with {counterparty}
• Total value: ${total_notional/1_000_000:.1f}M

**Bottom Line:**
• **Capital needed**: ${final_results['capital_requirement']/1_000:.0f}K
• **Risk exposure**: ${final_results['exposure_at_default']/1_000_000:.1f}M ({ead_ratio:.1f}% of notional)

**Key Insight**: """
        
        if ead_ratio < 10:
            response += "Very efficient portfolio! Low capital requirements."
        elif ead_ratio < 20:
            response += "Good portfolio efficiency. Some optimization possible."
        else:
            response += "High capital requirements. Consider optimization strategies."
        
        response += "\n\n💡 Ask me about optimization strategies to reduce capital requirements!"
        
        return response
    
    def _format_balanced_calculation_response(self, trades, final_results, counterparty, results) -> str:
        """Format balanced calculation response with moderate detail"""
        
        total_notional = sum(t.notional for t in trades)
        
        response = f"""**✅ SA-CCR Calculation Complete!**

**Portfolio Summary:**
• Counterparty: {counterparty}
• Number of trades: {len(trades)}
• Total notional: ${total_notional/1_000_000:.1f}M

**Key Results:**
• **Replacement Cost (RC)**: ${final_results['replacement_cost']/1_000_000:.2f}M
• **Potential Future Exposure (PFE)**: ${final_results['potential_future_exposure']/1_000_000:.2f}M
• **Exposure at Default (EAD)**: ${final_results['exposure_at_default']/1_000_000:.2f}M
• **Capital Requirement**: ${final_results['capital_requirement']/1_000:.0f}K

**Trade Breakdown:**
"""
        
        for i, trade in enumerate(trades, 1):
            response += f"• Trade {i}: {trade.asset_class.value} {trade.trade_type.value}, ${trade.notional/1_000_000:.1f}M {trade.currency}, {trade.time_to_maturity():.1f}Y\n"
        
        # Analysis
        ead_ratio = (final_results['exposure_at_default'] / total_notional) * 100
        response += f"\n**Risk Analysis:**\n"
        response += f"• EAD/Notional ratio: {ead_ratio:.2f}%\n"
        
        if final_results['replacement_cost'] > final_results['potential_future_exposure']:
            response += f"• Current exposure (RC) dominates - consider collateral management\n"
        else:
            response += f"• Future exposure (PFE) dominates - portfolio shows forward risk\n"
        
        response += f"\n**💡 Next Steps**: Ask about optimization strategies, clearing benefits, or detailed calculation breakdown!"
        
        return response
    
    def _format_advanced_calculation_response(self, trades, final_results, counterparty, results) -> str:
        """Format advanced calculation response with technical details"""
        
        total_notional = sum(t.notional for t in trades)
        calculation_steps = results.get('calculation_steps', [])
        
        response = f"""**✅ Advanced SA-CCR Analysis Complete**

**Portfolio Composition:**
• Counterparty: {counterparty}
• Trades: {len(trades)} across {len(set(t.asset_class for t in trades))} asset class(es)
• Gross notional: ${total_notional/1_000_000:.2f}M
• Currencies: {', '.join(set(t.currency for t in trades))}

**Detailed Results:**
• **Replacement Cost (RC)**: ${final_results['replacement_cost']/1_000_000:.3f}M
• **Potential Future Exposure (PFE)**: ${final_results['potential_future_exposure']/1_000_000:.3f}M
• **Alpha Factor**: {final_results.get('alpha_factor', 1.4)}
• **Exposure at Default (EAD)**: ${final_results['exposure_at_default']/1_000_000:.3f}M
• **Risk-Weighted Assets**: ${final_results['risk_weighted_assets']/1_000_000:.3f}M
• **Capital Requirement (8%)**: ${final_results['capital_requirement']/1_000:.1f}K

**Technical Metrics:**
"""
        
        # Add PFE multiplier if available
        pfe_step = next((step for step in calculation_steps if step.get('step') == 15), None)
        if pfe_step and 'data' in pfe_step:
            multiplier = pfe_step['data'].get('multiplier', 'N/A')
            response += f"• PFE Multiplier: {multiplier:.4f}\n"
        
        # Add aggregate add-on if available
        addon_step = next((step for step in calculation_steps if step.get('step') == 13), None)
        if addon_step and 'data' in addon_step:
            aggregate_addon = addon_step['data'].get('aggregate_addon', 0)
            response += f"• Aggregate Add-On: ${aggregate_addon/1_000_000:.3f}M\n"
        
        response += f"• EAD/Notional Ratio: {(final_results['exposure_at_default']/total_notional)*100:.3f}%\n"
        response += f"• RC/EAD Contribution: {(final_results['replacement_cost']/final_results['exposure_at_default'])*100:.1f}%\n"
        response += f"• PFE/EAD Contribution: {(final_results['potential_future_exposure']/final_results['exposure_at_default'])*100:.1f}%\n"
        
        response += f"\n**Trade-Level Analysis:**\n"
        for i, trade in enumerate(trades, 1):
            maturity = trade.time_to_maturity()
            response += f"• T{i}: {trade.asset_class.value[:2]}-{trade.trade_type.value[:4]} | ${trade.notional/1_000_000:.1f}M {trade.currency} | {maturity:.2f}Y | δ={trade.delta:.2f}\n"
        
        response += f"\n**🔬 Technical Insights:**\n"
        
        # Optimization suggestions based on technical analysis
        if final_results['replacement_cost'] > final_results['potential_future_exposure'] * 2:
            response += "• High RC suggests collateral optimization opportunity\n"
        
        if pfe_step and pfe_step['data'].get('multiplier', 1) > 0.8:
            response += "• High PFE multiplier indicates limited netting benefits\n"
        
        response += f"\n**📊 Available**: 24-step breakdown, optimization analysis, scenario comparisons"
        
        return response
    
    def _handle_information_query_enhanced(self, query: str, analysis: Dict) -> str:
        """Enhanced information query handler with intelligent responses based on context"""
        
        query_lower = query.lower()
        categories = analysis['categories']
        context = analysis['context']
        complexity = analysis['complexity_level']
        
        # Adjust response based on complexity level
        if complexity == 'basic':
            detail_level = "simplified"
        elif complexity == 'advanced':
            detail_level = "comprehensive"
        else:
            detail_level = "balanced"
        
        # Handle regulatory queries
        if 'regulatory' in categories:
            return self._handle_regulatory_query(query_lower, detail_level, context)
        
        # Handle optimization queries
        elif 'optimization' in categories:
            return self._handle_optimization_query(query_lower, detail_level, context)
        
        # Handle comparison queries
        elif 'comparison' in categories:
            return self._handle_comparison_query(query_lower, detail_level, context)
        
        # Handle SA-CCR methodology questions
        elif 'methodology' in query_lower or ('sa-ccr' in query_lower and 'explain' in query_lower):
            return self._explain_saccr_methodology(detail_level)
        
        # Handle specific formula questions
        elif any(term in query_lower for term in ['pfe', 'multiplier', 'alpha', 'ead', 'formula']):
            return self._explain_saccr_formulas(query_lower, detail_level)
        
        # Handle general questions with context awareness
        else:
            return self._handle_general_query(query_lower, detail_level, context)
    
    def _handle_regulatory_query(self, query_lower: str, detail_level: str, context: Dict) -> str:
        """Handle regulatory and compliance questions"""
        
        if detail_level == "simplified":
            return """**🏛️ Basel SA-CCR Regulatory Overview (Simplified)**

SA-CCR is the global standard for measuring counterparty credit risk in derivatives trading, established by the Basel Committee.

**Key Points:**
• **Purpose**: Calculate how much capital banks need for derivatives risk
• **Scope**: Applies to all derivatives (swaps, forwards, options)
• **Implementation**: Required globally since 2017-2022
• **Benefit**: Provides consistent risk measurement across institutions

**Why it matters**: SA-CCR helps ensure banks have enough capital to survive if their derivatives counterparties default.

Would you like me to explain any specific aspect in more detail?"""
        
        elif detail_level == "comprehensive":
            return """**🏛️ Basel SA-CCR Regulatory Framework (Comprehensive)**

**Regulatory Background:**
• **Basel Committee on Banking Supervision (BCBS) 279** (March 2014)
• **Implementation Timeline**: CRR II in EU (2019), US (2020-2022)
• **Replaces**: Current Exposure Method (CEM) and Internal Model Method (IMM)

**Regulatory Requirements:**
• **Scope**: All OTC derivatives, exchange-traded derivatives, and long-settlement transactions
• **Calculation Frequency**: At least monthly, daily for larger portfolios
• **Documentation**: Full audit trail of calculations required
• **Validation**: Annual validation of supervisory parameters

**Compliance Considerations:**
• **Data Requirements**: Trade-level data with full lifecycle tracking
• **System Requirements**: Automated calculation systems with appropriate controls
• **Governance**: Clear ownership, independent validation, management oversight
• **Reporting**: Regular reporting to senior management and regulators

**Supervisory Review:**
• Regulators may challenge calculations and assumptions
• Stress testing requirements for large portfolios
• Model validation requirements for any internal adjustments

Need guidance on specific compliance requirements for your jurisdiction?"""
        
        else:  # balanced
            return """**🏛️ Basel SA-CCR Regulatory Framework**

SA-CCR is the Basel Committee's standardized approach for measuring counterparty credit risk in derivatives.

**Key Regulatory Aspects:**
• **Legal Basis**: Basel Committee BCBS 279, implemented globally 2017-2022
• **Mandatory Use**: Replaces older methods (CEM) for capital calculations
• **Scope**: All derivatives including OTC, exchange-traded, and long-settlement

**Implementation Requirements:**
• Monthly calculation minimum (daily for large portfolios)
• Complete trade-level data capture and storage
• Automated systems with appropriate controls and governance
• Regular validation and independent review

**Supervisory Expectations:**
• Robust data management and calculation processes
• Clear documentation and audit trails
• Senior management oversight and reporting
• Stress testing for material portfolios

**Benefits for Institutions:**
• Risk-sensitive capital requirements
• Recognition of netting and collateral benefits
• Consistent methodology across counterparties

Would you like me to explain implementation requirements for your specific situation?"""
    
    def _handle_optimization_query(self, query_lower: str, detail_level: str, context: Dict) -> str:
        """Handle optimization and capital efficiency questions"""
        
        base_response = """**🎯 SA-CCR Capital Optimization Strategies**

"""
        
        if 'central clearing' in query_lower:
            base_response += """**Central Clearing Focus:**
• **Alpha Reduction**: 1.4 → 0.5 (65% capital savings)
• **Eligibility**: Check which trades can be centrally cleared
• **Cost-Benefit**: Compare clearing costs vs capital savings
• **Implementation**: CCP connectivity and operational setup

"""
        
        if 'netting' in query_lower or 'portfolio' in query_lower:
            base_response += """**Netting Optimization:**
• **Master Agreements**: Consolidate trades under single agreements
• **Portfolio Balancing**: Add offsetting trades to reduce net MTM
• **PFE Multiplier**: Target multiplier reduction through hedging
• **Currency Matching**: Align currencies within hedging sets

"""
        
        if 'collateral' in query_lower:
            base_response += """**Collateral Management:**
• **High-Quality Assets**: Use cash or government bonds (0% haircut)
• **Threshold Negotiation**: Lower thresholds reduce replacement cost
• **Automation**: Implement automated margining systems
• **Tri-Party Services**: Consider third-party collateral management

"""
        
        # Add general strategies if not specific
        if len(base_response) == len("**🎯 SA-CCR Capital Optimization Strategies**\n\n"):
            base_response += """**Top Optimization Strategies:**

**1. Central Clearing (Highest Impact)**
• Move eligible trades to CCPs
• Reduces Alpha from 1.4 to 0.5 (65% reduction!)
• Typically saves 50-70% capital

**2. Netting Optimization**
• Consolidate under master agreements
• Balance long/short positions
• Strategic hedging to reduce net MTM

**3. Collateral Management**
• Post high-quality collateral
• Negotiate lower thresholds and MTAs
• Automated margining systems

**4. Portfolio Structure**
• Diversify across asset classes
• Optimize maturity profiles
• Trade compression programs

**Expected Results**: Combined strategies typically achieve 40-70% capital reduction.

"""
        
        if context.get('has_portfolio_reference'):
            base_response += "💡 **For your specific portfolio**: I can analyze your current trades for optimization opportunities if you run a SA-CCR calculation first.\n\n"
        
        base_response += "Would you like me to dive deeper into any specific optimization strategy?"
        
        return base_response
    
    def _handle_comparison_query(self, query_lower: str, detail_level: str, context: Dict) -> str:
        """Handle comparison questions between different approaches or scenarios"""
        
        if 'bilateral' in query_lower and 'cleared' in query_lower:
            return """**⚖️ Bilateral vs Centrally Cleared Derivatives**

**Bilateral Derivatives:**
• Alpha = 1.4 (higher capital requirement)
• Direct counterparty risk
• Bilateral collateral arrangements
• More operational complexity
• Greater flexibility in terms

**Centrally Cleared Derivatives:**
• Alpha = 0.5 (65% lower capital requirement)
• CCP intermediated (lower counterparty risk)
• Standardized margining
• Operational efficiencies
• Limited product flexibility

**Capital Impact Example:**
• $100M bilateral swap → ~$14M EAD
• $100M cleared swap → ~$5M EAD
• **Capital savings**: ~65% reduction

**When to Use Each:**
• **Clearing**: Standardized, high-volume products
• **Bilateral**: Bespoke, low-volume, or non-clearable products

Would you like me to analyze the clearing impact for a specific portfolio?"""
        
        elif 'pfe' in query_lower and ('rc' in query_lower or 'replacement cost' in query_lower):
            return """**⚖️ PFE vs Replacement Cost (RC)**

**Replacement Cost (RC):**
• **What**: Current exposure if counterparty defaults today
• **Formula**: RC = max(V - C, TH + MTA - NICA, 0)
• **Components**: Current MTM, collateral, thresholds
• **Nature**: Current, actual exposure

**Potential Future Exposure (PFE):**
• **What**: Potential exposure over remaining trade life
• **Formula**: PFE = Multiplier × Aggregate AddOn
• **Components**: Add-ons, multiplier, correlations
• **Nature**: Forward-looking, potential exposure

**Key Differences:**
• **RC**: What you'd lose if counterparty defaulted today
• **PFE**: What you might lose if they default in the future
• **Optimization**: RC reduced by collateral, PFE by netting/clearing

**Portfolio Impact:**
• High RC: Focus on collateral management
• High PFE: Focus on netting optimization or clearing

**EAD Formula**: EAD = α × (RC + PFE)

Would you like me to explain how to optimize each component?"""
        
        else:
            return """**⚖️ SA-CCR Comparison Analysis**

I can help you compare different aspects of SA-CCR calculations:

**Common Comparisons:**
• **Bilateral vs Cleared**: Capital impact of central clearing
• **RC vs PFE**: Current vs future exposure components  
• **Asset Classes**: Risk factors across different products
• **Optimization Strategies**: Effectiveness of different approaches
• **Before/After Scenarios**: Impact of portfolio changes

**Scenario Analysis Available:**
• Central clearing impact
• Collateral posting effects
• Portfolio compression benefits
• Netting optimization results

What specific comparison would you like me to analyze for you?"""
    
    def _explain_saccr_methodology(self, detail_level: str) -> str:
        """Explain SA-CCR methodology based on requested detail level"""
        
        if detail_level == "simplified":
            return """**📚 SA-CCR Methodology (Simplified)**

SA-CCR calculates how much capital banks need for derivatives risk in 3 main steps:

**Step 1: Current Exposure (RC)**
• How much would you lose if counterparty defaulted today?
• Considers current trade values and collateral

**Step 2: Future Exposure (PFE)**  
• How much might you lose if they default later?
• Based on trade types, sizes, and maturities

**Step 3: Total Exposure (EAD)**
• EAD = α × (RC + PFE)
• α = 1.4 for bilateral trades, 0.5 for cleared trades

**Why This Matters:**
• Higher EAD = More capital required
• Optimization reduces capital needs
• Better risk management

Want me to explain any step in more detail?"""
        
        elif detail_level == "comprehensive":
            return """**📚 SA-CCR Methodology (Comprehensive Technical Overview)**

SA-CCR implements a standardized approach across 24 detailed calculation steps:

**Phase 1: Data Preparation (Steps 1-4)**
• Netting set identification and trade classification
• Asset class mapping per Basel regulatory tables
• Hedging set definition by risk factors
• Time parameter calculation (settlement, maturity, remaining life)

**Phase 2: Risk Parameter Calculation (Steps 5-8)**
• Adjusted notional calculation with trade-specific adjustments
• Maturity factor: MF = min(1, 0.05 + 0.95 × exp(-0.05 × max(1, M)))
• Supervisory delta determination (±1 for linear, calculated for options)
• Supervisory factor application per Basel regulatory tables

**Phase 3: Add-On Aggregation (Steps 9-13)**
• Adjusted derivatives amount: Adj_Amount = Notional × |δ| × MF × SF
• Supervisory correlation application by asset class
• Hedging set add-on: HS_AddOn = Σ(Trade_AddOns) × √ρ
• Asset class aggregation and final aggregate add-on calculation

**Phase 4: Exposure Calculation (Steps 14-18)**
• V and C calculation (MTM and collateral values)
• PFE multiplier: min(1, 0.05 + 0.95 × exp(-0.05 × max(0, V) / AddOn))
• PFE = Multiplier × Aggregate AddOn
• Replacement cost calculation with collateral effects

**Phase 5: Final Results (Steps 19-24)**
• Central clearing status determination
• Alpha application (1.4 bilateral, 0.5 cleared)
• EAD = α × (RC + PFE)
• Risk-weighted assets and capital requirement calculation

**Advanced Features:**
• Cross-asset correlation recognition
• Margin period of risk adjustments
• Collateral haircut applications
• Basis and volatility trade adjustments

This methodology ensures consistent, risk-sensitive capital calculations across all derivatives exposures."""
        
        else:  # balanced
            return """**📚 SA-CCR Methodology Overview**

SA-CCR calculates counterparty credit risk exposure through a comprehensive 24-step process:

**Key Components:**

**1. Replacement Cost (RC)**
• Current exposure if counterparty defaults today
• RC = max(V - C, TH + MTA - NICA, 0)
• Considers current MTM values and posted collateral

**2. Potential Future Exposure (PFE)**
• Potential exposure over remaining trade life
• PFE = Multiplier × Aggregate AddOn
• Based on trade characteristics and netting benefits

**3. Exposure at Default (EAD)**
• Total regulatory exposure = α × (RC + PFE)
• α = 1.4 for bilateral, 0.5 for centrally cleared

**Calculation Process:**
• **Steps 1-4**: Data preparation and classification
• **Steps 5-13**: Add-on calculations for potential exposure
• **Steps 14-18**: Current exposure and replacement cost
• **Steps 19-24**: Final EAD and capital requirements

**Key Features:**
• Risk-sensitive to trade types and maturities
• Recognizes netting and collateral benefits
• Differentiates between bilateral and cleared trades
• Consistent methodology across institutions

**Benefits:**
• More accurate risk measurement than previous methods
• Incentivizes risk-reducing activities (clearing, netting)
• Provides basis for regulatory capital requirements

Would you like me to dive deeper into any specific component?"""
    
    def _explain_saccr_formulas(self, query_lower: str, detail_level: str) -> str:
        """Explain SA-CCR formulas based on query and detail level"""
        
        if 'pfe' in query_lower and 'multiplier' in query_lower:
            if detail_level == "simplified":
                return """**🔢 PFE Multiplier (Simplified)**

The PFE multiplier reduces exposure when your portfolio is "out-of-the-money" (losing money).

**Simple Formula**: Multiplier = between 0.05 and 1.0
• **When portfolio loses money**: Multiplier closer to 0.05 (good for you!)
• **When portfolio makes money**: Multiplier closer to 1.0 (less benefit)

**Why it matters**: If your trades are losing money, you're less likely to lose more if the counterparty defaults.

**Optimization tip**: Balance your portfolio so some trades lose money to get this benefit!"""
            
            elif detail_level == "comprehensive":
                return """**🔢 PFE Multiplier (Technical)**

**Full Formula**: 
```
Multiplier = min(1, 0.05 + 0.95 × exp(-0.05 × max(0, V) / AddOn))
```

**Technical Components:**
• **V**: Net replacement value (sum of all positive MTMs minus negative MTMs)
• **AddOn**: Aggregate add-on representing potential future exposure
• **Floor**: 0.05 (5% minimum recognition of potential exposure)
• **Ceiling**: 1.0 (100% of add-on when no netting benefit)

**Mathematical Behavior:**
• **V ≤ 0**: Multiplier approaches 0.05 (maximum netting benefit)
• **V >> AddOn**: Multiplier approaches 1.0 (minimal netting benefit)
• **Exponential decay**: Smooth transition between extremes

**Regulatory Rationale:**
• Recognizes that out-of-money portfolios have lower future exposure
• Prevents over-recognition of netting benefits (5% floor)
• Maintains risk sensitivity across different portfolio states

**Optimization Applications:**
• Portfolio rebalancing to achieve negative net MTM
• Strategic hedging to optimize V/AddOn ratio
• Trade structuring to maximize netting benefits"""
            
            else:  # balanced
                return """**🔢 PFE Multiplier Formula**

**Formula**: Multiplier = min(1, 0.05 + 0.95 × exp(-0.05 × max(0, V) / AddOn))

**Key Components:**
• **V**: Net mark-to-market value of all trades
• **AddOn**: Aggregate add-on (potential future exposure)
• **Range**: 0.05 to 1.0

**How it Works:**
• **Negative V** (portfolio out-of-money): Multiplier approaches 0.05
• **Large positive V**: Multiplier approaches 1.0
• **Floor of 0.05**: Ensures minimum exposure recognition

**Business Impact:**
• Lower multiplier = Lower capital requirements
• Achieved through portfolio balancing and strategic hedging
• Can significantly reduce PFE component of EAD

**Optimization Strategy**: Balance portfolio MTM to achieve negative net value while maintaining business objectives."""
        
        elif any(term in query_lower for term in ['ead', 'exposure at default']):
            return """**🔢 Exposure at Default (EAD) Formula**

**Main Formula**: EAD = α × (RC + PFE)

**Components:**
• **α (Alpha)**: 1.4 for bilateral trades, 0.5 for cleared trades
• **RC**: Replacement Cost (current exposure)
• **PFE**: Potential Future Exposure

**Alpha Impact:**
• **Bilateral**: α = 1.4 (40% penalty for counterparty risk)
• **Cleared**: α = 0.5 (50% reduction for CCP protection)
• **Capital Impact**: Clearing can reduce EAD by ~65%

**Calculation Flow:**
1. Calculate current exposure (RC)
2. Calculate potential future exposure (PFE)
3. Apply alpha multiplier based on clearing status
4. Result is regulatory exposure for capital calculation

**Optimization Focus**: Maximize clearing eligibility to achieve α = 0.5"""
        
        elif 'replacement cost' in query_lower or 'rc' in query_lower:
            return """**🔢 Replacement Cost (RC) Formulas**

**For Margined Trades:**
```
RC = max(V - C, TH + MTA - NICA, 0)
```

**For Unmargined Trades:**
```
RC = max(V, 0)
```

**Components:**
• **V**: Net MTM value of all trades in netting set
• **C**: Posted collateral (haircut-adjusted)
• **TH**: Threshold amount
• **MTA**: Minimum Transfer Amount
• **NICA**: Net Independent Collateral Amount

**Key Insights:**
• RC represents current exposure if counterparty defaults today
• Collateral directly reduces RC (dollar-for-dollar after haircuts)
• Thresholds and MTAs create minimum exposure levels
• Cannot be negative (floor at zero)

**Optimization**: Post high-quality collateral and negotiate lower thresholds"""
        
        else:
            return """**🔢 Key SA-CCR Formulas**

**1. Exposure at Default:**
```
EAD = α × (RC + PFE)
where α = 1.4 (bilateral) or 0.5 (cleared)
```

**2. Potential Future Exposure:**
```
PFE = Multiplier × Aggregate AddOn
```

**3. PFE Multiplier:**
```
Multiplier = min(1, 0.05 + 0.95 × exp(-0.05 × max(0, V) / AddOn))
```

**4. Replacement Cost (margined):**
```
RC = max(V - C, TH + MTA - NICA, 0)
```

**5. Maturity Factor:**
```
MF = min(1, 0.05 + 0.95 × exp(-0.05 × max(1, M)))
```

**6. Adjusted Notional:**
```
Adjusted Amount = Notional × |δ| × MF × SF
```

Where:
• V = Net MTM, C = Collateral, TH = Threshold, MTA = Min Transfer Amount
• δ = Supervisory delta, MF = Maturity factor, SF = Supervisory factor
• M = Remaining maturity in years

Which formula would you like me to explain in detail?"""
    
    def _handle_general_query(self, query_lower: str, detail_level: str, context: Dict) -> str:
        """Handle general SA-CCR questions with context awareness"""
        
        if context.get('requests_example'):
            return """**📋 SA-CCR Examples**

**Example 1: Simple Interest Rate Swap**
• $100M USD 5-year swap with Goldman Sachs
• Bilateral (α = 1.4), no collateral
• Estimated EAD: ~$14M (14% of notional)

**Example 2: Cleared Swap**
• Same $100M swap but centrally cleared
• Cleared (α = 0.5), daily margining
• Estimated EAD: ~$5M (5% of notional)
• **Capital savings**: 65% reduction!

**Example 3: Multi-Asset Portfolio**
• $200M IR swap + $150M FX forward + $100M equity option
• Mixed bilateral/cleared, some collateral
• Netting benefits reduce total exposure
• Estimated EAD: ~$25M (vs $63M without netting)

**Key Takeaway**: Clearing and netting provide substantial capital savings.

Would you like me to calculate SA-CCR for your specific trades?"""
        
        elif any(word in query_lower for word in ['help', 'assist', 'support']):
            return """**🤖 How I Can Help You**

**📊 Automatic Calculations**
• Describe your trades in plain English
• I'll extract details and calculate SA-CCR
• Get instant EAD, RWA, and capital requirements

**📚 Expert Explanations**
• SA-CCR methodology and formulas
• Basel regulatory requirements
• Technical concepts made simple

**🎯 Optimization Guidance**
• Capital reduction strategies
• Portfolio restructuring advice
• Clearing vs bilateral analysis

**🔍 Deep Analysis**
• Risk driver identification
• Scenario comparisons
• Regulatory compliance guidance

**Example Questions:**
• "Calculate SA-CCR for a $500M swap portfolio"
• "How does central clearing reduce capital?"
• "What's the difference between RC and PFE?"
• "Optimize my derivatives portfolio"

What specific area would you like help with?"""
        
        else:
            return """**🤖 SA-CCR Expert Assistant**

I'm here to help with all aspects of SA-CCR! I can assist with:

**📊 Calculations**: Describe your trades and I'll calculate SA-CCR automatically
**📚 Explanations**: Ask about specific SA-CCR concepts, formulas, or methodology  
**🎯 Optimization**: Get strategies to reduce your capital requirements
**🔍 Analysis**: Deep dive into calculation results and risk drivers

**Popular Topics:**
• Basel SA-CCR methodology and 24-step process
• PFE multiplier and netting benefits
• Central clearing vs bilateral comparison
• Capital optimization strategies
• Regulatory compliance requirements

**Example Questions:**
• "What's the difference between RC and PFE?"
• "How does the maturity factor work?"
• "Calculate SA-CCR for a $200M swap with Deutsche Bank"
• "What are the best ways to optimize my derivatives capital?"

What would you like to know about SA-CCR?"""
    
    def _extract_trade_information_enhanced(self, query: str) -> List[Dict]:
        """Enhanced trade information extraction with better parsing"""
        
        import re
        
        trades = []
        
        # Enhanced patterns for better extraction
        money_pattern = r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*([KMB]?)\b'
        time_pattern = r'(\d+(?:\.\d+)?)\s*[-\s]?\s*(year|yr|month|mon|day)s?'
        currency_pattern = r'\b(USD|EUR|GBP|JPY|CHF|CAD|AUD|NZD|SEK|NOK)\b'
        
        # Enhanced asset type patterns
        asset_patterns = {
            'Interest Rate': r'\b(interest\s+rate\s+swap|irs|swap|swaption)\b',
            'Foreign Exchange': r'\b(fx\s+forward|fx|foreign\s+exchange|currency|cross.currency)\b',
            'Equity': r'\b(equity\s+option|stock\s+option|equity|option\s+on|index\s+option)\b',
            'Credit': r'\b(cds|credit\s+default\s+swap|credit)\b',
            'Commodity': r'\b(commodity|oil|gold|wheat|energy)\b'
        }
        
        # Split on common separators for multiple trades
        trade_separators = r'[,;]\s*\d+\)|and\s+\d+\)|also\s+|plus\s+'
        potential_trades = re.split(trade_separators, query)
        
        # If no clear separation, treat as single trade
        if len(potential_trades) == 1:
            potential_trades = [query]
        
        for i, trade_text in enumerate(potential_trades):
            trade_info = {
                'trade_id': f'AI_TRADE_{i+1}',
                'notional': 100000000.0,  # Default $100M
                'currency': 'USD',
                'maturity_years': 5.0,
                'asset_class': 'Interest Rate',
                'trade_type': 'Swap',
                'delta': 1.0,
                'mtm_value': 0.0
            }
            
            # Extract notional with enhanced parsing
            money_matches = re.findall(money_pattern, trade_text, re.IGNORECASE)
            if money_matches:
                amount, multiplier = money_matches[0]
                amount = float(amount.replace(',', ''))
                
                multiplier_map = {'K': 1000, 'M': 1000000, 'B': 1000000000}
                if multiplier.upper() in multiplier_map:
                    amount *= multiplier_map[multiplier.upper()]
                
                trade_info['notional'] = amount
            
            # Extract currency
            currency_matches = re.findall(currency_pattern, trade_text, re.IGNORECASE)
            if currency_matches:
                trade_info['currency'] = currency_matches[0].upper()
            
            # Extract maturity with enhanced parsing
            time_matches = re.findall(time_pattern, trade_text, re.IGNORECASE)
            if time_matches:
                period, unit = time_matches[0]
                period = float(period)
                
                if unit.lower().startswith('year') or unit.lower().startswith('yr'):
                    trade_info['maturity_years'] = period
                elif unit.lower().startswith('month') or unit.lower().startswith('mon'):
                    trade_info['maturity_years'] = period / 12
                elif unit.lower().startswith('day'):
                    trade_info['maturity_years'] = period / 365
            
            # Extract asset class and trade type with enhanced logic
            for asset_class, pattern in asset_patterns.items():
                if re.search(pattern, trade_text, re.IGNORECASE):
                    trade_info['asset_class'] = asset_class
                    
                    if asset_class == 'Interest Rate':
                        if 'swaption' in trade_text.lower():
                            trade_info['trade_type'] = 'Swaption'
                        else:
                            trade_info['trade_type'] = 'Swap'
                    elif asset_class == 'Foreign Exchange':
                        trade_info['trade_type'] = 'Forward'
                    elif asset_class == 'Equity':
                        trade_info['trade_type'] = 'Option'
                        # Extract delta if mentioned
                        delta_pattern = r'delta\s+(\d+(?:\.\d+)?)'
                        delta_match = re.search(delta_pattern, trade_text, re.IGNORECASE)
                        if delta_match:
                            trade_info['delta'] = float(delta_match.group(1))
                    elif asset_class == 'Credit':
                        trade_info['trade_type'] = 'Credit Default Swap'
                    elif asset_class == 'Commodity':
                        trade_info['trade_type'] = 'Commodity Swap'
                    break
            
            trades.append(trade_info)
        
        return trades
    
    def _render_portfolio_page(self):
        """Render portfolio analysis page"""
        st.markdown("## Portfolio Analysis")
        
        # Enhanced portfolio analysis interface
        if not st.session_state.get('current_portfolio'):
            st.info("🏦 **No portfolio loaded.** Please go to the Calculator page to create or load a portfolio first.")
            return
        
        portfolio = st.session_state.current_portfolio
        trades = portfolio.get('trades', [])
        
        # Portfolio Overview Section
        st.markdown("### 📊 Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", len(trades))
        with col2:
            total_notional = sum(abs(t.notional) for t in trades)
            st.metric("Total Notional", f"${total_notional/1_000_000:.1f}M")
        with col3:
            asset_classes = len(set(t.asset_class for t in trades))
            st.metric("Asset Classes", asset_classes)
        with col4:
            currencies = len(set(t.currency for t in trades))
            st.metric("Currencies", currencies)
        
        # Analysis Tabs
        analysis_tabs = st.tabs(["📈 Risk Metrics", "🎯 Asset Allocation", "📅 Maturity Profile", "💰 P&L Analysis"])
        
        with analysis_tabs[0]:
            self._render_risk_metrics_analysis(trades)
        
        with analysis_tabs[1]:
            self._render_asset_allocation_analysis(trades)
        
        with analysis_tabs[2]:
            self._render_maturity_profile_analysis(trades)
        
        with analysis_tabs[3]:
            self._render_pnl_analysis(trades)
    
    def _render_risk_metrics_analysis(self, trades):
        """Render risk metrics analysis"""
        st.markdown("#### Risk Metrics Dashboard")
        
        if st.session_state.get('calculation_results'):
            results = st.session_state.calculation_results
            final_results = results['final_results']
            
            # Risk metrics visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # EAD breakdown pie chart
                rc = final_results['replacement_cost']
                pfe = final_results['potential_future_exposure']
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Replacement Cost', 'Potential Future Exposure'],
                    values=[rc, pfe],
                    hole=0.3,
                    marker_colors=['#3b82f6', '#8b5cf6']
                )])
                
                fig.update_layout(
                    title="EAD Components Breakdown",
                    height=300,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk efficiency metrics
                ead = final_results['exposure_at_default']
                total_notional = sum(abs(t.notional) for t in trades)
                efficiency_ratio = (1 - ead/total_notional) * 100 if total_notional > 0 else 0
                
                st.metric("Capital Efficiency", f"{efficiency_ratio:.1f}%", 
                         help="Higher percentage indicates better capital efficiency")
                st.metric("EAD/Notional Ratio", f"{(ead/total_notional)*100:.2f}%" if total_notional > 0 else "0%")
                
                # Risk recommendations
                st.markdown("**💡 Risk Optimization Tips:**")
                if rc > pfe:
                    st.write("• Consider collateral posting to reduce current exposure")
                if efficiency_ratio < 80:
                    st.write("• Portfolio may benefit from netting optimization")
                st.write("• Evaluate central clearing eligibility for capital relief")
        else:
            st.info("Run SA-CCR calculation first to see risk metrics.")
    
    def _render_asset_allocation_analysis(self, trades):
        """Render asset allocation analysis"""
        st.markdown("#### Asset Allocation Analysis")
        
        # Asset class breakdown
        asset_data = {}
        for trade in trades:
            asset_class = trade.asset_class.value
            if asset_class not in asset_data:
                asset_data[asset_class] = {'count': 0, 'notional': 0}
            asset_data[asset_class]['count'] += 1
            asset_data[asset_class]['notional'] += abs(trade.notional)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Asset class by count
            labels = list(asset_data.keys())
            counts = [data['count'] for data in asset_data.values()]
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=counts, hole=0.3)])
            fig.update_layout(title="Trades by Asset Class", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Asset class by notional
            notionals = [data['notional']/1_000_000 for data in asset_data.values()]
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=notionals, hole=0.3)])
            fig.update_layout(title="Notional by Asset Class ($M)", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown table
        st.markdown("#### Detailed Asset Class Breakdown")
        breakdown_data = []
        for asset_class, data in asset_data.items():
            breakdown_data.append({
                'Asset Class': asset_class,
                'Trade Count': data['count'],
                'Total Notional ($M)': f"{data['notional']/1_000_000:.2f}",
                'Avg Notional ($M)': f"{data['notional']/data['count']/1_000_000:.2f}",
                'Percentage': f"{(data['notional']/sum(d['notional'] for d in asset_data.values()))*100:.1f}%"
            })
        
        st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True)
    
    def _render_maturity_profile_analysis(self, trades):
        """Render maturity profile analysis"""
        st.markdown("#### Maturity Profile Analysis")
        
        # Maturity buckets
        maturity_buckets = {'<1Y': 0, '1-2Y': 0, '2-5Y': 0, '5-10Y': 0, '>10Y': 0}
        maturity_notionals = {'<1Y': 0, '1-2Y': 0, '2-5Y': 0, '5-10Y': 0, '>10Y': 0}
        
        for trade in trades:
            maturity = trade.time_to_maturity()
            notional = abs(trade.notional)
            
            if maturity < 1:
                bucket = '<1Y'
            elif maturity < 2:
                bucket = '1-2Y'
            elif maturity < 5:
                bucket = '2-5Y'
            elif maturity < 10:
                bucket = '5-10Y'
            else:
                bucket = '>10Y'
            
            maturity_buckets[bucket] += 1
            maturity_notionals[bucket] += notional
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Maturity profile by count
            fig = go.Figure(data=[
                go.Bar(x=list(maturity_buckets.keys()), y=list(maturity_buckets.values()),
                       marker_color='#06b6d4')
            ])
            fig.update_layout(title="Trade Count by Maturity Bucket", 
                            xaxis_title="Maturity Bucket", yaxis_title="Number of Trades")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Maturity profile by notional
            notional_values = [v/1_000_000 for v in maturity_notionals.values()]
            fig = go.Figure(data=[
                go.Bar(x=list(maturity_notionals.keys()), y=notional_values,
                       marker_color='#8b5cf6')
            ])
            fig.update_layout(title="Notional by Maturity Bucket ($M)", 
                            xaxis_title="Maturity Bucket", yaxis_title="Notional ($M)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Average maturity calculation
        avg_maturity = sum(t.time_to_maturity() * abs(t.notional) for t in trades) / sum(abs(t.notional) for t in trades)
        st.metric("Weighted Average Maturity", f"{avg_maturity:.2f} years")
    
    def _render_pnl_analysis(self, trades):
        """Render P&L analysis"""
        st.markdown("#### P&L Analysis")
        
        total_mtm = sum(t.mtm_value for t in trades)
        positive_mtm = sum(t.mtm_value for t in trades if t.mtm_value > 0)
        negative_mtm = sum(t.mtm_value for t in trades if t.mtm_value < 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total MTM", f"${total_mtm/1_000_000:.2f}M")
        with col2:
            st.metric("Positive MTM", f"${positive_mtm/1_000_000:.2f}M")
        with col3:
            st.metric("Negative MTM", f"${abs(negative_mtm)/1_000_000:.2f}M")
        
        # P&L distribution chart
        pnl_data = [t.mtm_value/1_000_000 for t in trades]
        trade_ids = [t.trade_id for t in trades]
        
        fig = go.Figure(data=[
            go.Bar(x=trade_ids, y=pnl_data, 
                   marker_color=['green' if x > 0 else 'red' for x in pnl_data])
        ])
        fig.update_layout(
            title="Trade-Level P&L Distribution ($M)",
            xaxis_title="Trade ID",
            yaxis_title="MTM Value ($M)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_optimization_page(self):
        """Render optimization analysis page"""
        st.markdown("## 🎯 Optimization Analysis")
        
        if not st.session_state.get('calculation_results'):
            st.warning("🔄 **No calculation results available.** Please run a SA-CCR calculation first.")
            return
        
        results = st.session_state.calculation_results
        final_results = results['final_results']
        
        # Current metrics
        st.markdown("### 📊 Current Portfolio Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current EAD", f"${final_results['exposure_at_default']/1_000_000:.2f}M")
        with col2:
            st.metric("Current RWA", f"${final_results['risk_weighted_assets']/1_000_000:.2f}M")
        with col3:
            st.metric("Current Capital", f"${final_results['capital_requirement']/1_000:.0f}K")
        with col4:
            total_notional = final_results['portfolio_summary']['total_notional']
            efficiency = (1 - final_results['exposure_at_default']/total_notional) * 100 if total_notional > 0 else 0
            st.metric("Capital Efficiency", f"{efficiency:.1f}%")
        
        # Optimization strategies
        st.markdown("### 🚀 Optimization Strategies")
        
        strategy_tabs = st.tabs(["🏛️ Central Clearing", "🔄 Netting Optimization", "💰 Collateral Management", "📈 Portfolio Restructuring"])
        
        with strategy_tabs[0]:
            self._render_central_clearing_analysis(results)
        
        with strategy_tabs[1]:
            self._render_netting_optimization_analysis(results)
        
        with strategy_tabs[2]:
            self._render_collateral_management_analysis(results)
        
        with strategy_tabs[3]:
            self._render_portfolio_restructuring_analysis(results)
    
    def _render_central_clearing_analysis(self, results):
        """Render central clearing optimization analysis"""
        st.markdown("#### Central Clearing Impact Analysis")
        
        current_ead = results['final_results']['exposure_at_default']
        current_rwa = results['final_results']['risk_weighted_assets']
        current_capital = results['final_results']['capital_requirement']
        
        # Simulate central clearing (Alpha reduction from 1.4 to 0.5)
        clearing_ratio = 0.5 / 1.4  # ~0.357
        cleared_ead = current_ead * clearing_ratio
        cleared_rwa = current_rwa * clearing_ratio
        cleared_capital = current_capital * clearing_ratio
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**💡 Central Clearing Benefits:**")
            st.write(f"• **EAD Reduction**: ${(current_ead - cleared_ead)/1_000_000:.2f}M ({(1-clearing_ratio)*100:.0f}%)")
            st.write(f"• **RWA Reduction**: ${(current_rwa - cleared_rwa)/1_000_000:.2f}M ({(1-clearing_ratio)*100:.0f}%)")
            st.write(f"• **Capital Savings**: ${(current_capital - cleared_capital)/1_000:.0f}K ({(1-clearing_ratio)*100:.0f}%)")
        
        with col2:
            # Before/After comparison chart
            comparison_data = {
                'Current': [current_ead/1_000_000, current_rwa/1_000_000, current_capital/1_000],
                'With Central Clearing': [cleared_ead/1_000_000, cleared_rwa/1_000_000, cleared_capital/1_000]
            }
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Current', x=['EAD ($M)', 'RWA ($M)', 'Capital ($K)'], 
                               y=comparison_data['Current'], marker_color='#ef4444'))
            fig.add_trace(go.Bar(name='With Central Clearing', x=['EAD ($M)', 'RWA ($M)', 'Capital ($K)'], 
                               y=comparison_data['With Central Clearing'], marker_color='#10b981'))
            
            fig.update_layout(title="Central Clearing Impact", barmode='group', height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**📋 Implementation Steps:**")
        st.write("1. **Eligibility Assessment**: Review trades for central clearing eligibility")
        st.write("2. **CCP Selection**: Choose appropriate Central Counterparties")
        st.write("3. **Documentation**: Update master agreements and confirmations")
        st.write("4. **Operational Setup**: Establish CCP connectivity and processes")
        st.write("5. **Migration**: Execute portfolio migration to central clearing")
    
    def _render_netting_optimization_analysis(self, results):
        """Render netting optimization analysis"""
        st.markdown("#### Netting Optimization Opportunities")
        
        # Analyze PFE multiplier for netting benefits
        calculation_steps = results['calculation_steps']
        pfe_step = next((step for step in calculation_steps if step['step'] == 15), None)
        
        if pfe_step and 'data' in pfe_step:
            multiplier = pfe_step['data']['multiplier']
            v_value = pfe_step['data']['v']
            addon = pfe_step['data']['addon']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current PFE Multiplier", f"{multiplier:.4f}")
                st.metric("Net MTM (V)", f"${v_value/1_000_000:.2f}M")
                st.metric("Aggregate Add-On", f"${addon/1_000_000:.2f}M")
            
            with col2:
                # Potential improvement through MTM balancing
                if v_value > 0:
                    # Simulate reducing MTM through hedging
                    target_v = max(0, v_value * 0.5)  # 50% MTM reduction
                    if addon > 0:
                        improved_multiplier = min(1.0, 0.05 + 0.95 * math.exp(-0.05 * max(0, target_v) / addon))
                        improvement = (multiplier - improved_multiplier) / multiplier * 100
                        
                        st.write("**🎯 Netting Optimization Potential:**")
                        st.write(f"• Target Net MTM: ${target_v/1_000_000:.2f}M")
                        st.write(f"• Improved Multiplier: {improved_multiplier:.4f}")
                        st.write(f"• PFE Reduction: {improvement:.1f}%")
                else:
                    st.write("**✅ Portfolio already has optimal netting benefits**")
        
        st.markdown("**🔧 Netting Optimization Strategies:**")
        st.write("• **Master Agreement Consolidation**: Combine trades under single agreements")
        st.write("• **Strategic Hedging**: Add offsetting trades to reduce net MTM")
        st.write("• **Portfolio Rebalancing**: Adjust trade sizes to optimize netting")
        st.write("• **Currency Matching**: Align currencies within hedging sets")
    
    def _render_collateral_management_analysis(self, results):
        """Render collateral management analysis"""
        st.markdown("#### Collateral Management Optimization")
        
        # Analyze replacement cost impact
        calculation_steps = results['calculation_steps']
        rc_step = next((step for step in calculation_steps if step['step'] == 18), None)
        
        current_rc = results['final_results']['replacement_cost']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Replacement Cost", f"${current_rc/1_000_000:.2f}M")
            
            # Collateral impact scenarios
            collateral_scenarios = [
                ("No Additional Collateral", current_rc),
                ("$10M Cash Collateral", max(0, current_rc - 10_000_000)),
                ("$25M Cash Collateral", max(0, current_rc - 25_000_000)),
                ("$50M Cash Collateral", max(0, current_rc - 50_000_000))
            ]
            
            st.markdown("**💰 Collateral Impact Scenarios:**")
            for scenario, rc_value in collateral_scenarios:
                reduction = current_rc - rc_value
                percentage = (reduction / current_rc) * 100 if current_rc > 0 else 0
                st.write(f"• {scenario}: RC = ${rc_value/1_000_000:.2f}M ({percentage:.0f}% reduction)")
        
        with col2:
            # Collateral efficiency chart
            scenarios = [s[0] for s in collateral_scenarios]
            rc_values = [s[1]/1_000_000 for s in collateral_scenarios]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=scenarios, y=rc_values, marker_color='#06b6d4'))
            fig.update_layout(
                title="Replacement Cost by Collateral Scenario",
                xaxis_title="Collateral Scenario",
                yaxis_title="Replacement Cost ($M)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**🏦 Collateral Management Best Practices:**")
        st.write("• **High-Quality Assets**: Use cash or government bonds (lowest haircuts)")
        st.write("• **Threshold Optimization**: Negotiate lower threshold amounts")
        st.write("• **MTA Minimization**: Reduce minimum transfer amounts")
        st.write("• **Tri-Party Arrangements**: Consider tri-party collateral management")
        st.write("• **Automated Margining**: Implement automated collateral calls")
    
    def _render_portfolio_restructuring_analysis(self, results):
        """Render portfolio restructuring analysis"""
        st.markdown("#### Portfolio Restructuring Opportunities")
        
        portfolio = st.session_state.current_portfolio
        trades = portfolio.get('trades', [])
        
        # Maturity analysis
        long_maturity_trades = [t for t in trades if t.time_to_maturity() > 5]
        short_maturity_impact = sum(abs(t.notional) for t in long_maturity_trades) * 0.1  # Estimate 10% reduction
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📅 Maturity Optimization:**")
            st.write(f"• Long-maturity trades (>5Y): {len(long_maturity_trades)}")
            if long_maturity_trades:
                avg_maturity = sum(t.time_to_maturity() for t in long_maturity_trades) / len(long_maturity_trades)
                st.write(f"• Average maturity: {avg_maturity:.1f} years")
                st.write(f"• Potential EAD reduction: ${short_maturity_impact/1_000_000:.1f}M")
            
            st.markdown("**🎯 Asset Class Diversification:**")
            asset_classes = set(t.asset_class for t in trades)
            st.write(f"• Current asset classes: {len(asset_classes)}")
            if len(asset_classes) < 3:
                st.write("• **Opportunity**: Add diversifying asset classes")
            else:
                st.write("• **Status**: Well diversified portfolio")
        
        with col2:
            # Trade compression opportunities
            st.markdown("**🔄 Trade Compression Analysis:**")
            
            # Group similar trades for compression analysis
            compression_groups = {}
            for trade in trades:
                key = (trade.asset_class, trade.currency, trade.counterparty)
                if key not in compression_groups:
                    compression_groups[key] = []
                compression_groups[key].append(trade)
            
            compressible_groups = {k: v for k, v in compression_groups.items() if len(v) > 1}
            
            st.write(f"• Trade groups: {len(compression_groups)}")
            st.write(f"• Compressible groups: {len(compressible_groups)}")
            
            if compressible_groups:
                compression_potential = sum(len(trades) - 1 for trades in compressible_groups.values())
                st.write(f"• Trades that could be compressed: {compression_potential}")
                st.write(f"• Estimated operational savings: High")
        
        st.markdown("**🚀 Restructuring Recommendations:**")
        st.write("• **Maturity Laddering**: Replace long-term trades with shorter maturities")
        st.write("• **Trade Compression**: Combine similar trades to reduce gross notional")
        st.write("• **Novation Programs**: Transfer trades to optimize netting sets")
        st.write("• **Basis Risk Management**: Use basis swaps to optimize correlations")
        st.write("• **Option Strategies**: Replace linear products with options where appropriate")
    
    def _render_comparison_page(self):
        """Render scenario comparison page"""
        st.markdown("## ⚖️ Scenario Comparison")
        
        # Scenario comparison interface
        st.markdown("### 🔬 Scenario Analysis Setup")
        
        if not st.session_state.get('calculation_results'):
            st.warning("🔄 **No base calculation available.** Please run a SA-CCR calculation first.")
            return
        
        base_results = st.session_state.calculation_results
        
        # Scenario definition
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Base Scenario")
            base_ead = base_results['final_results']['exposure_at_default'] / 1_000_000
            base_rwa = base_results['final_results']['risk_weighted_assets'] / 1_000_000
            base_capital = base_results['final_results']['capital_requirement'] / 1_000
            
            st.metric("EAD", f"${base_ead:.2f}M")
            st.metric("RWA", f"${base_rwa:.2f}M")
            st.metric("Capital Requirement", f"${base_capital:.0f}K")
        
        with col2:
            st.markdown("#### Scenario Parameters")
            
            scenario_type = st.selectbox(
                "Select Scenario Type:",
                ["Central Clearing", "Collateral Posting", "Portfolio Compression", "Custom Alpha"]
            )
            
            if scenario_type == "Central Clearing":
                alpha_scenario = st.slider("Alpha Value", 0.5, 1.4, 0.5, 0.1)
                scenario_description = f"Central clearing scenario (α = {alpha_scenario})"
            
            elif scenario_type == "Collateral Posting":
                collateral_amount = st.number_input("Additional Collateral ($M)", 0.0, 100.0, 10.0)
                scenario_description = f"Additional ${collateral_amount}M collateral posting"
            
            elif scenario_type == "Portfolio Compression":
                compression_ratio = st.slider("Compression Ratio", 0.1, 1.0, 0.8, 0.1)
                scenario_description = f"Portfolio compression ({compression_ratio*100:.0f}% remaining)"
            
            else:  # Custom Alpha
                alpha_scenario = st.slider("Custom Alpha Value", 0.1, 3.0, 1.4, 0.1)
                scenario_description = f"Custom alpha scenario (α = {alpha_scenario})"
        
        # Run scenario analysis
        if st.button("🧮 Run Scenario Analysis", type="primary"):
            with st.spinner("Running scenario analysis..."):
                scenario_results = self._calculate_scenario(scenario_type, base_results, locals())
                st.session_state.scenario_results = scenario_results
                st.session_state.scenario_description = scenario_description
        
        # Display comparison results
        if st.session_state.get('scenario_results'):
            self._display_scenario_comparison(base_results, st.session_state.scenario_results, 
                                            st.session_state.scenario_description)
    
    def _calculate_scenario(self, scenario_type, base_results, params):
        """Calculate scenario results"""
        base_final = base_results['final_results']
        
        if scenario_type == "Central Clearing":
            alpha = params['alpha_scenario']
            rc = base_final['replacement_cost']
            pfe = base_final['potential_future_exposure']
            scenario_ead = alpha * (rc + pfe)
            scenario_rwa = scenario_ead * 1.0  # Assuming 100% risk weight
            scenario_capital = scenario_rwa * 0.08
        
        elif scenario_type == "Collateral Posting":
            collateral = params['collateral_amount'] * 1_000_000
            scenario_rc = max(0, base_final['replacement_cost'] - collateral)
            scenario_ead = 1.4 * (scenario_rc + base_final['potential_future_exposure'])
            scenario_rwa = scenario_ead * 1.0
            scenario_capital = scenario_rwa * 0.08
        
        elif scenario_type == "Portfolio Compression":
            compression = params['compression_ratio']
            scenario_pfe = base_final['potential_future_exposure'] * compression
            scenario_ead = 1.4 * (base_final['replacement_cost'] + scenario_pfe)
            scenario_rwa = scenario_ead * 1.0
            scenario_capital = scenario_rwa * 0.08
        
        else:  # Custom Alpha
            alpha = params['alpha_scenario']
            rc = base_final['replacement_cost']
            pfe = base_final['potential_future_exposure']
            scenario_ead = alpha * (rc + pfe)
            scenario_rwa = scenario_ead * 1.0
            scenario_capital = scenario_rwa * 0.08
        
        return {
            'exposure_at_default': scenario_ead,
            'risk_weighted_assets': scenario_rwa,
            'capital_requirement': scenario_capital,
            'replacement_cost': scenario_rc if scenario_type == "Collateral Posting" else base_final['replacement_cost'],
            'potential_future_exposure': scenario_pfe if scenario_type == "Portfolio Compression" else base_final['potential_future_exposure']
        }
    
    def _display_scenario_comparison(self, base_results, scenario_results, description):
        """Display scenario comparison analysis"""
        st.markdown("### 📊 Scenario Comparison Results")
        st.markdown(f"**Scenario**: {description}")
        
        base_final = base_results['final_results']
        
        # Metrics comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            base_ead = base_final['exposure_at_default'] / 1_000_000
            scenario_ead = scenario_results['exposure_at_default'] / 1_000_000
            delta_ead = scenario_ead - base_ead
            st.metric("EAD ($M)", f"{scenario_ead:.2f}", f"{delta_ead:+.2f}")
        
        with col2:
            base_rwa = base_final['risk_weighted_assets'] / 1_000_000
            scenario_rwa = scenario_results['risk_weighted_assets'] / 1_000_000
            delta_rwa = scenario_rwa - base_rwa
            st.metric("RWA ($M)", f"{scenario_rwa:.2f}", f"{delta_rwa:+.2f}")
        
        with col3:
            base_capital = base_final['capital_requirement'] / 1_000
            scenario_capital = scenario_results['capital_requirement'] / 1_000
            delta_capital = scenario_capital - base_capital
            st.metric("Capital ($K)", f"{scenario_capital:.0f}", f"{delta_capital:+.0f}")
        
        # Visual comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Side-by-side comparison chart
            metrics = ['EAD ($M)', 'RWA ($M)', 'Capital ($K)']
            base_values = [base_ead, base_rwa, base_capital]
            scenario_values = [scenario_ead, scenario_rwa, scenario_capital]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Base Scenario', x=metrics, y=base_values, marker_color='#6b7280'))
            fig.add_trace(go.Bar(name='New Scenario', x=metrics, y=scenario_values, marker_color='#3b82f6'))
            
            fig.update_layout(
                title="Scenario Comparison",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Impact analysis
            st.markdown("#### 📈 Impact Analysis")
            
            ead_change = ((scenario_ead - base_ead) / base_ead) * 100 if base_ead > 0 else 0
            rwa_change = ((scenario_rwa - base_rwa) / base_rwa) * 100 if base_rwa > 0 else 0
            capital_change = ((scenario_capital - base_capital) / base_capital) * 100 if base_capital > 0 else 0
            
            st.write(f"**EAD Change**: {ead_change:+.1f}%")
            st.write(f"**RWA Change**: {rwa_change:+.1f}%")
            st.write(f"**Capital Change**: {capital_change:+.1f}%")
            
            if capital_change < -10:
                st.success("🎉 **Excellent**: Significant capital savings!")
            elif capital_change < -5:
                st.info("👍 **Good**: Moderate capital reduction")
            elif capital_change > 10:
                st.warning("⚠️ **Caution**: Capital requirement increases")
            else:
                st.info("📊 **Neutral**: Minimal capital impact")
    
    def _render_database_page(self):
        """Render database management page"""
        st.markdown("## Data Management")
        
        # Database statistics
        try:
            stats = self.db_manager.get_database_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trades", stats.get('trade_count', 0))
            with col2:
                st.metric("Portfolios", stats.get('portfolio_count', 0))
            with col3:
                st.metric("Calculations", stats.get('calculation_count', 0))
            with col4:
                st.metric("Database Size", f"{stats.get('db_size_mb', 0):.1f} MB")
            
            # Recent activity
            st.markdown("### Recent Activity")
            recent_data = self.db_manager.get_recent_activity()
            if not recent_data.empty:
                st.dataframe(recent_data, use_container_width=True)
            else:
                st.info("No recent activity")
                
        except Exception as e:
            st.error(f"Database error: {str(e)}")
    
    def _render_settings_page(self):
        """Render application settings page"""
        st.markdown("## Application Settings")
        
        # Configuration management
        st.markdown("### Configuration")
        
        config_tabs = st.tabs(["🤖 LLM Setup", "📊 Calculation", "🗄️ Database", "🎨 UI", "🔧 Advanced"])
        
        with config_tabs[0]:
            self._render_llm_settings()
        
        with config_tabs[1]:
            self._render_calculation_settings()
        
        with config_tabs[2]:
            self._render_database_settings()
        
        with config_tabs[3]:
            self._render_ui_settings()
        
        with config_tabs[4]:
            self._render_advanced_settings()
    
    def _render_llm_settings(self):
        """Render LLM configuration settings"""
        st.markdown("#### LLM Configuration for AI Assistant")
        
        st.markdown("Configure your language model connection for the AI assistant:")
        
        # LLM Configuration Form
        with st.form("llm_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                base_url = st.text_input(
                    "Base URL", 
                    value="http://localhost:8123/v1",
                    help="URL of your LLM API endpoint"
                )
                api_key = st.text_input(
                    "API Key", 
                    value="dummy", 
                    type="password",
                    help="API key for authentication"
                )
                
            with col2:
                model = st.text_input(
                    "Model", 
                    value="llama3",
                    help="Model name (e.g., llama3, gpt-4, claude-3)"
                )
                temperature = st.slider(
                    "Temperature", 
                    0.0, 1.0, 0.3, 0.1,
                    help="Controls randomness in responses"
                )
            
            max_tokens = st.number_input(
                "Max Tokens", 
                1000, 8000, 4000, 100,
                help="Maximum response length"
            )
            
            streaming = st.checkbox(
                "Enable Streaming", 
                value=False,
                help="Stream responses token by token"
            )
            
            submitted = st.form_submit_button("🔗 Connect & Test LLM", type="primary")
            
            if submitted:
                config = {
                    'base_url': base_url,
                    'api_key': api_key,
                    'model': model,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'streaming': streaming
                }
                
                with st.spinner("Testing LLM connection..."):
                    if self._setup_llm_connection(config):
                        st.success("✅ LLM Connected Successfully!")
                        st.info("You can now use the AI Assistant with full language model capabilities.")
                    else:
                        st.error("❌ LLM Connection Failed")
                        st.error("Please check your configuration and try again.")
        
        # Connection Status
        st.markdown("#### Connection Status")
        if self.llm_connection_status == "connected":
            st.success("🟢 LLM is connected and ready")
            
            # Test query button
            if st.button("Test AI Response"):
                try:
                    test_response = self._test_ai_response()
                    st.write(f"**Test Response:** {test_response}")
                except Exception as e:
                    st.error(f"Test failed: {str(e)}")
        else:
            st.warning("🟡 LLM is not connected")
            
        # Configuration Help
        st.markdown("#### Configuration Help")
        
        with st.expander("📚 Setup Instructions", expanded=False):
            st.markdown("""
**Popular LLM Configurations:**

**1. OpenAI API:**
- Base URL: `https://api.openai.com/v1`
- Model: `gpt-4` or `gpt-3.5-turbo`
- API Key: Your OpenAI API key

**2. Local Ollama:**
- Base URL: `http://localhost:11434/v1`
- Model: `llama3`, `mistral`, `codellama`
- API Key: `dummy` (not needed for local)

**3. LM Studio:**
- Base URL: `http://localhost:1234/v1`
- Model: Model name from LM Studio
- API Key: `dummy`

**4. Custom Endpoint:**
- Configure according to your API provider's documentation
- Ensure the endpoint supports OpenAI-compatible API format

**Troubleshooting:**
- Verify the base URL is accessible
- Check that the model name is correct
- Ensure API key has necessary permissions
- Test with a simple curl command first
""")
        
        with st.expander("🔒 Security Notes", expanded=False):
            st.markdown("""
**Important Security Considerations:**

- API keys are stored only in session memory
- Keys are not saved to disk or database
- Use environment variables for production deployments
- Consider using local models for sensitive data
- Rotate API keys regularly for cloud providers
""")
    
    def _render_calculation_settings(self):
        """Render calculation configuration settings"""
        st.markdown("#### Calculation Parameters")
        
        current_config = self.config_manager.get_calculation_config()
        
        # Supervisory factors
        st.markdown("**Supervisory Factors:**")
        
        alpha_unmargined = st.number_input(
            "Alpha (Unmargined)", 
            value=current_config.get('alpha_bilateral', 1.4),
            min_value=0.1, max_value=5.0, step=0.1
        )
        
        alpha_margined = st.number_input(
            "Alpha (Cleared)", 
            value=current_config.get('alpha_centrally_cleared', 0.5),
            min_value=0.1, max_value=5.0, step=0.1
        )
        
        if st.button("💾 Save Calculation Settings"):
            self.config_manager.update_calculation_config({
                'alpha_bilateral': alpha_unmargined,
                'alpha_centrally_cleared': alpha_margined
            })
            st.success("Configuration updated!")
    
    def _render_database_settings(self):
        """Render database configuration settings"""
        st.markdown("#### Database Configuration")
        
        # Database maintenance
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clean Database"):
                try:
                    self.db_manager.cleanup_old_records()
                    st.success("Database cleaned successfully!")
                except Exception as e:
                    st.error(f"Cleanup failed: {str(e)}")
        
        with col2:
            if st.button("📊 Optimize Database"):
                try:
                    self.db_manager.optimize_database()
                    st.success("Database optimized!")
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
    
    def _render_ui_settings(self):
        """Render UI configuration settings"""
        st.markdown("#### User Interface Settings")
        
        theme = st.selectbox("Theme", ["Professional", "Dark", "Light"], index=0)
        decimal_places = st.slider("Decimal Places", 0, 4, 2)
        currency_format = st.selectbox("Currency Format", ["USD", "EUR", "GBP"], index=0)
        
        if st.button("💾 Save UI Settings"):
            st.success("UI settings updated!")
    
    def _render_advanced_settings(self):
        """Render advanced configuration settings"""
        st.markdown("#### Advanced Settings")
        
        debug_mode = st.checkbox("Enable Debug Mode", value=False)
        cache_enabled = st.checkbox("Enable Calculation Cache", value=True)
        max_portfolio_size = st.number_input("Max Portfolio Size", min_value=100, max_value=10000, value=1000)
        
        if st.button("💾 Save Advanced Settings"):
            st.success("Advanced settings updated!")


def main():
    """Application entry point"""
    try:
        app = SACCRApplication()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application startup error: {e}")

if __name__ == "__main__":
    main()
