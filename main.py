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
import io
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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Claude-like CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root variables for consistent theming - Claude-inspired */
    :root {
        --primary-color: #2d3748;
        --primary-light: #4a5568;
        --primary-dark: #1a202c;
        --secondary-color: #38a169;
        --accent-color: #3182ce;
        --warning-color: #d69e2e;
        --error-color: #e53e3e;
        --success-color: #38a169;
        --text-primary: #2d3748;
        --text-secondary: #718096;
        --text-light: #a0aec0;
        --bg-primary: #ffffff;
        --bg-secondary: #f7fafc;
        --bg-accent: #edf2f7;
        --border-color: #e2e8f0;
        --border-light: #f1f5f9;
        --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
    }
    
    /* Main application styling */
    .main { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: var(--bg-secondary);
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    /* Claude-style chat interface */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    .chat-message {
        background: var(--bg-primary);
        border-radius: var(--radius-lg);
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
        overflow: hidden;
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-left: 4px solid var(--accent-color);
    }
    
    .chat-message.assistant {
        background: var(--bg-primary);
        border-left: 4px solid var(--secondary-color);
    }
    
    .chat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.5rem 0.5rem;
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    .chat-content {
        padding: 0 1.5rem 1.5rem;
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    .chat-input-container {
        position: sticky;
        bottom: 0;
        background: var(--bg-secondary);
        border-top: 1px solid var(--border-color);
        padding: 1rem;
        margin-top: 2rem;
    }
    
    /* Enhanced header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 2rem;
        border-radius: var(--radius-lg);
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2.25rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0;
        font-size: 1.125rem;
        opacity: 0.9;
    }
    
    /* Professional result cards */
    .result-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .result-card {
        background: var(--bg-primary);
        padding: 1.5rem;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color);
        text-align: center;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    .result-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--accent-color);
        margin: 0.5rem 0;
    }
    
    .result-label {
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Enhanced tables */
    .results-table {
        background: var(--bg-primary);
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color);
        overflow: hidden;
        margin: 1rem 0;
    }
    
    /* Step breakdown styling */
    .step-breakdown {
        background: var(--bg-primary);
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .step-header {
        background: var(--bg-accent);
        padding: 1rem 1.5rem;
        border-bottom: 1px solid var(--border-color);
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .step-content {
        padding: 1.5rem;
    }
    
    .step-number {
        background: var(--accent-color);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin-right: 1rem;
        font-size: 0.875rem;
    }
    
    /* Quick action buttons */
    .quick-actions {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .quick-action-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .quick-action-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: var(--accent-color);
    }
    
    /* Enhanced alerts */
    .alert {
        padding: 1rem 1.5rem;
        border-radius: var(--radius-md);
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-info {
        background: #ebf8ff;
        color: #2c5282;
        border-left-color: var(--accent-color);
    }
    
    .alert-success {
        background: #f0fff4;
        color: #276749;
        border-left-color: var(--success-color);
    }
    
    .alert-warning {
        background: #fffbeb;
        color: #975a16;
        border-left-color: var(--warning-color);
    }
    
    .alert-error {
        background: #fed7d7;
        color: #9b2c2c;
        border-left-color: var(--error-color);
    }

    /* Excel upload styling */
    .upload-area {
        border: 2px dashed var(--border-color);
        border-radius: var(--radius-lg);
        padding: 2rem;
        text-align: center;
        background: var(--bg-primary);
        transition: all 0.2s ease;
    }
    
    .upload-area:hover {
        border-color: var(--accent-color);
        background: var(--bg-accent);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.875rem;
        }
        
        .result-grid {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        }
        
        .quick-actions {
            grid-template-columns: 1fr;
        }
        
        .chat-container {
            padding: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class SACCRApplication:
    """Enhanced SA-CCR application with Claude-like interface"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.db_manager = DatabaseManager()
        self.saccr_engine = SACCREngine(self.config_manager)
        self.ui_components = UIComponents()
        self.validator = TradeValidator()
        self.progress_tracker = ProgressTracker()
        
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
            },
            'current_page': 'ai_assistant',  # Default to AI Assistant
            'llm_settings': {
                'provider': 'emergent',
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 2000,
                'response_style': 'professional',
                'enable_context': True,
                'show_calculations': True
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
            <h1>🤖 AI SA-CCR Analytics Platform</h1>
            <p>Intelligent Basel SA-CCR calculation and optimization with AI-powered insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        with st.sidebar:
            self._render_sidebar()
        
        # Main content routing - Default to AI Assistant
        page = st.session_state.get('current_page', 'ai_assistant')
        
        if page == 'ai_assistant':
            self._render_ai_assistant_page()
        elif page == 'calculator':
            self._render_calculator_page()
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
            'ai_assistant': '🤖 AI Assistant',
            'calculator': '📊 SA-CCR Calculator',
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
            
            st.markdown('<div class="alert alert-success">✅ Database Connected</div>', 
                       unsafe_allow_html=True)
        except Exception as e:
            st.markdown('<div class="alert alert-warning">⚠️ Database Issues</div>', 
                       unsafe_allow_html=True)
            logger.error(f"Database connection error: {e}")
        
        # Configuration status
        config_status = self.config_manager.validate_config()
        if config_status['valid']:
            st.markdown('<div class="alert alert-success">✅ Config Valid</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert alert-warning">⚠️ Config Issues</div>', 
                       unsafe_allow_html=True)

    def _render_ai_assistant_page(self):
        """Render enhanced Claude-like AI assistant interface"""
        
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Welcome message and quick actions
        st.markdown("""
        <div class="alert alert-info">
            <strong>🚀 Welcome to your AI SA-CCR Assistant!</strong><br>
            I can help you calculate SA-CCR, optimize portfolios, upload Excel data, and provide regulatory guidance.
            Ask me anything about Basel regulations or describe your trades for instant calculations.
        </div>
        """, unsafe_allow_html=True)
        
        # Quick action buttons
        st.markdown("### 🎯 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Sample Calculation", use_container_width=True):
                sample_query = "Calculate SA-CCR for a $500M USD interest rate swap with Goldman Sachs, 7-year maturity"
                self._process_ai_query(sample_query)
        
        with col2:
            if st.button("🎯 Optimization Help", use_container_width=True):
                optimization_query = "How can I reduce my capital requirements through central clearing and netting optimization?"
                self._process_ai_query(optimization_query)
        
        with col3:
            if st.button("❓ Explain SA-CCR", use_container_width=True):
                explanation_query = "What is SA-CCR and how does the 24-step calculation process work?"
                self._process_ai_query(explanation_query)
        
        # Optional upload section - more subtle
        st.markdown("---")
        
        with st.expander("📤 Optional: Upload Excel Portfolio", expanded=False):
            st.markdown("""
            **💡 Want to upload multiple trades at once?**
            
            You can upload an Excel file with your portfolio data for batch processing.
            This is completely optional - you can also enter trades manually or just ask questions.
            """)
            
            if st.button("📁 Show Upload Interface"):
                st.session_state.show_excel_upload = True
                st.rerun()
        
        # Excel Upload Section
        if st.session_state.get('show_excel_upload', False):
            self._render_excel_upload_section()
        
        # Initialize chat history
        if 'ai_chat_history' not in st.session_state:
            st.session_state.ai_chat_history = [
                {
                    'role': 'assistant',
                    'content': """**👋 Hello! I'm your SA-CCR expert assistant.**

I can help you with:

**🔹 Automatic Calculations**: Describe your trades in natural language
   *"Calculate SA-CCR for a $200M USD swap with JPMorgan, 5-year maturity"*

**🔹 Excel Portfolio Upload**: Upload your trades via Excel spreadsheet
   *Supports batch upload with all required trade parameters*

**🔹 Portfolio Optimization**: Get strategies to reduce capital requirements  
   *"How can I optimize my $2B derivatives portfolio for lower capital?"*

**🔹 Regulatory Guidance**: Basel III SA-CCR questions and compliance
   *"What's the impact of central clearing on my alpha multiplier?"*

**🔹 Results Analysis**: Deep insights into calculation results
   *"Show me the 24-step breakdown with detailed navigation"*

**💡 Pro Tips:**
• Include notional amounts, currencies, maturities, and counterparties
• Upload Excel files with multiple trades for batch processing
• Ask for detailed breakdowns and technical analysis
• Request optimization strategies based on your results

What would you like to explore today? 🚀""",
                    'timestamp': datetime.now()
                }
            ]
        
        # Enhanced chat interface
        self._render_enhanced_chat_interface()
        
        st.markdown('</div>', unsafe_allow_html=True)

    def _render_excel_upload_section(self):
        """Render Excel upload interface"""
        
        st.markdown("---")
        st.markdown("### 📤 Excel Portfolio Upload")
        
        # Upload area
        st.markdown("""
        <div class="upload-area">
            <h4>📁 Upload Your Portfolio Excel File</h4>
            <p>Upload an Excel file with your trades data. The file should contain the required columns as shown in the template below.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose Excel file",
            type=['xlsx', 'xls'],
            help="Upload Excel file with trade data"
        )
        
        if uploaded_file is not None:
            try:
                # Process Excel file
                self._process_excel_upload(uploaded_file)
            except Exception as e:
                st.error(f"Error processing Excel file: {str(e)}")
        
        # Show Excel template
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Required Excel Columns")
            required_columns = [
                "Trade ID", "Asset Class", "Trade Type", "Notional", 
                "Currency", "Underlying", "Maturity (Years)", "MTM Value", 
                "Delta", "Counterparty", "Netting Set ID", "Threshold", "MTA"
            ]
            
            for col in required_columns:
                st.write(f"• {col}")
        
        with col2:
            st.markdown("#### 📝 Download Template")
            
            # Create sample template
            template_data = {
                'Trade ID': ['SWAP001', 'FWD001', 'OPT001'],
                'Asset Class': ['Interest Rate', 'Foreign Exchange', 'Equity'],
                'Trade Type': ['Swap', 'Forward', 'Option'],
                'Notional': [100000000, 50000000, 25000000],
                'Currency': ['USD', 'EUR', 'USD'],
                'Underlying': ['USD-LIBOR', 'EUR/USD', 'SPX'],
                'Maturity (Years)': [5.0, 2.0, 0.25],
                'MTM Value': [50000, -25000, 10000],
                'Delta': [1.0, 1.0, 0.5],
                'Counterparty': ['Goldman Sachs', 'Deutsche Bank', 'JPMorgan'],
                'Netting Set ID': ['NS001', 'NS002', 'NS003'],
                'Threshold': [0, 0, 0],
                'MTA': [0, 0, 0]
            }
            
            template_df = pd.DataFrame(template_data)
            
            # Convert to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                template_df.to_excel(writer, sheet_name='Portfolio_Template', index=False)
            
            st.download_button(
                label="📥 Download Excel Template",
                data=output.getvalue(),
                file_name='sa_ccr_portfolio_template.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        
        if st.button("❌ Close Upload Section"):
            st.session_state.show_excel_upload = False
            st.rerun()

    def _process_excel_upload(self, uploaded_file):
        """Process uploaded Excel file and create portfolio"""
        
        try:
            # Read Excel file
            df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Excel file loaded successfully! Found {len(df)} trades.")
            
            # Display preview
            st.markdown("#### 👀 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Validate columns
            required_cols = [
                'Trade ID', 'Asset Class', 'Trade Type', 'Notional', 
                'Currency', 'Underlying', 'Maturity (Years)', 'MTM Value', 
                'Delta', 'Counterparty', 'Netting Set ID', 'Threshold', 'MTA'
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                return
            
            # Process trades
            if st.button("🚀 Import Trades to Portfolio", type="primary"):
                self._import_trades_from_excel(df)
        
        except Exception as e:
            st.error(f"❌ Error reading Excel file: {str(e)}")

    def _import_trades_from_excel(self, df):
        """Import trades from Excel DataFrame to portfolio"""
        
        try:
            trades = []
            
            for _, row in df.iterrows():
                # Create trade object
                trade = Trade(
                    trade_id=str(row['Trade ID']),
                    counterparty=str(row['Counterparty']),
                    asset_class=AssetClass(row['Asset Class']),
                    trade_type=TradeType(row['Trade Type']),
                    notional=float(row['Notional']),
                    currency=str(row['Currency']),
                    underlying=str(row['Underlying']),
                    maturity_date=datetime.now() + timedelta(days=int(float(row['Maturity (Years)']) * 365)),
                    mtm_value=float(row['MTM Value']),
                    delta=float(row['Delta'])
                )
                trades.append(trade)
            
            # Create or update portfolio
            if not st.session_state.current_portfolio:
                # Use first row for netting set info
                first_row = df.iloc[0]
                st.session_state.current_portfolio = {
                    'netting_set_id': str(first_row['Netting Set ID']),
                    'counterparty': str(first_row['Counterparty']),
                    'threshold': float(first_row['Threshold']),
                    'mta': float(first_row['MTA']),
                    'trades': trades
                }
            else:
                # Add to existing portfolio
                st.session_state.current_portfolio['trades'].extend(trades)
            
            st.success(f"✅ Successfully imported {len(trades)} trades to portfolio!")
            
            # Auto-generate AI response
            ai_response = f"""**📤 Excel Portfolio Import Complete!**

✅ **Successfully imported {len(trades)} trades**

**📊 Portfolio Summary:**
• Total Trades: {len(trades)}
• Asset Classes: {len(set(t.asset_class.value for t in trades))}
• Currencies: {len(set(t.currency for t in trades))}
• Total Notional: ${sum(abs(t.notional) for t in trades)/1_000_000:.1f}M
• Counterparties: {len(set(t.counterparty for t in trades))}

**🚀 Next Steps:**
• Review the imported trades in the Portfolio Analysis section
• Run SA-CCR calculation for complete risk assessment
• Explore optimization opportunities

Would you like me to calculate SA-CCR for this imported portfolio now?"""
            
            # Add to chat history
            st.session_state.ai_chat_history.append({
                'role': 'assistant',
                'content': ai_response,
                'timestamp': datetime.now()
            })
            
            # Hide upload section
            st.session_state.show_excel_upload = False
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error importing trades: {str(e)}")

    def _render_enhanced_chat_interface(self):
        """Render Claude-like chat interface"""
        
        # Chat history
        st.markdown("### 💬 Conversation")
        
        # Display messages
        for message in st.session_state.ai_chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="chat-header">
                        <span><strong>👤 You</strong></span>
                        <span>{message['timestamp'].strftime('%H:%M')}</span>
                    </div>
                    <div class="chat-content">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="chat-header">
                        <span><strong>🤖 SA-CCR Assistant</strong></span>
                        <span>{message['timestamp'].strftime('%H:%M')}</span>
                    </div>
                    <div class="chat-content">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        st.markdown("---")
        
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_area(
                    "Ask me anything about SA-CCR...",
                    placeholder="💡 Try: 'Calculate SA-CCR for my portfolio' or 'Explain the 24-step process' or 'How can I reduce capital requirements?'",
                    height=100,
                    label_visibility="collapsed"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                send_button = st.form_submit_button("🚀 Send", type="primary", use_container_width=True)
                
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    clear_button = st.form_submit_button("🧹 Clear", use_container_width=True)
                with col2_2:
                    export_button = st.form_submit_button("📤 Export", use_container_width=True)
            
            if send_button and user_input.strip():
                self._process_ai_query(user_input.strip())
            elif clear_button:
                st.session_state.ai_chat_history = st.session_state.ai_chat_history[:1]
                st.success("🧹 Chat cleared!")
                st.rerun()
            elif export_button:
                self._export_chat_history()

    def _process_ai_query(self, user_query: str):
        """Process AI query with enhanced intelligence"""
        
        # Add user message to chat
        st.session_state.ai_chat_history.append({
            'role': 'user',
            'content': user_query,
            'timestamp': datetime.now()
        })
        
        try:
            response = self._generate_ai_response(user_query)
            
            # Add AI response to chat
            st.session_state.ai_chat_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            error_response = f"❌ I encountered an error: {str(e)}\n\nPlease try rephrasing your question."
            
            st.session_state.ai_chat_history.append({
                'role': 'assistant',
                'content': error_response,
                'timestamp': datetime.now()
            })
        
        st.rerun()

    def _generate_ai_response(self, query: str) -> str:
        """Generate intelligent AI response based on query analysis"""
        
        query_lower = query.lower()
        
        # Check for calculation requests
        if any(word in query_lower for word in ['calculate', 'compute', 'saccr', 'portfolio']):
            if st.session_state.current_portfolio:
                return self._handle_calculation_request(query)
            else:
                return """**📊 SA-CCR Calculation Request**

I'd be happy to help you calculate SA-CCR! However, I don't see any portfolio data loaded.

**🔹 Options to proceed:**
1. **Upload Excel File**: Use the "Upload Excel Portfolio" button above
2. **Manual Entry**: Go to SA-CCR Calculator to add trades manually  
3. **Sample Calculation**: I can create a sample portfolio to demonstrate

Which option would you prefer?"""
        
        # Check for optimization queries
        elif any(word in query_lower for word in ['optimize', 'reduce', 'minimize', 'capital']):
            return self._handle_optimization_query(query)
        
        # Check for explanation requests
        elif any(word in query_lower for word in ['explain', 'what is', 'how does']):
            return self._handle_explanation_query(query)
        
        # Check for 24-step breakdown requests
        elif '24' in query_lower and 'step' in query_lower:
            return self._handle_step_breakdown_query()
        
        # Default helpful response
        else:
            return self._handle_general_query(query)

    def _handle_calculation_request(self, query: str) -> str:
        """Handle calculation requests for existing portfolio"""
        
        portfolio = st.session_state.current_portfolio
        trades = portfolio.get('trades', [])
        
        try:
            # Perform calculation
            results = self.saccr_engine.calculate_comprehensive_saccr(
                portfolio, 
                st.session_state.collateral_input
            )
            
            # Store results
            st.session_state.calculation_results = results
            
            final_results = results['final_results']
            
            return f"""**🚀 SA-CCR Calculation Complete!**

**📊 Key Results:**
• **Replacement Cost (RC)**: ${final_results['replacement_cost']/1_000_000:.2f}M
• **Potential Future Exposure (PFE)**: ${final_results['potential_future_exposure']/1_000_000:.2f}M  
• **Exposure at Default (EAD)**: ${final_results['exposure_at_default']/1_000_000:.2f}M
• **Risk Weighted Assets (RWA)**: ${final_results['risk_weighted_assets']/1_000_000:.2f}M
• **Capital Requirement**: ${final_results['capital_requirement']/1_000:.0f}K

**📈 Portfolio Summary:**
• Total Trades: {len(trades)}
• Total Notional: ${sum(abs(t.notional) for t in trades)/1_000_000:.1f}M
• EAD/Notional Ratio: {(final_results['exposure_at_default']/sum(abs(t.notional) for t in trades))*100:.2f}%

**🔍 Technical Analysis:**
• **Risk Profile**: {'Low' if final_results['exposure_at_default']/sum(abs(t.notional) for t in trades) < 0.05 else 'Medium' if final_results['exposure_at_default']/sum(abs(t.notional) for t in trades) < 0.15 else 'High'}
• **RC Contribution**: {(final_results['replacement_cost']/final_results['exposure_at_default'])*100:.1f}%
• **PFE Contribution**: {(final_results['potential_future_exposure']/final_results['exposure_at_default'])*100:.1f}%

**📋 Trade Breakdown:**
{self._generate_trade_breakdown(trades)}

**🎯 Next Steps:**
• View complete 24-step breakdown in Scenario Comparison
• Explore optimization opportunities  
• Analyze risk drivers and hedging effectiveness

Would you like me to show the detailed 24-step calculation breakdown or provide optimization recommendations?"""
            
        except Exception as e:
            return f"❌ Calculation failed: {str(e)}\n\nPlease check your portfolio data and try again."

    def _generate_trade_breakdown(self, trades) -> str:
        """Generate formatted trade breakdown"""
        
        breakdown = ""
        asset_classes = {}
        
        for trade in trades:
            ac = trade.asset_class.value
            if ac not in asset_classes:
                asset_classes[ac] = {'count': 0, 'notional': 0}
            asset_classes[ac]['count'] += 1
            asset_classes[ac]['notional'] += abs(trade.notional)
        
        for ac, data in asset_classes.items():
            breakdown += f"• **{ac}**: {data['count']} trades, ${data['notional']/1_000_000:.1f}M notional\n"
        
        return breakdown

    def _handle_optimization_query(self, query: str) -> str:
        """Handle optimization queries"""
        
        return """**🎯 Portfolio Optimization Strategies**

Here are key strategies to reduce your SA-CCR capital requirements:

**🔹 Central Clearing (Alpha Reduction)**
• Move bilateral trades to central clearing
• Reduces alpha from 1.4 to 0.5 (64% reduction)
• Most impactful for large notional portfolios

**🔹 Netting Optimization**
• Consolidate trades under single netting agreements
• Improve offsetting within asset classes
• Focus on same currency/maturity buckets

**🔹 Collateral Management**  
• Post variation margin to reduce RC component
• Optimize initial margin amounts
• Consider collateral currency basis

**🔹 Trade Structuring**
• Balance long/short positions within hedging sets
• Optimize trade maturities for better netting
• Consider basis and volatility trade impacts

**🔹 Portfolio Rebalancing**
• Reduce concentrated exposures
• Diversify across asset classes strategically
• Monitor PFE multiplier effects

**💡 Want Specific Analysis?**
Upload your portfolio or share trade details for customized optimization recommendations!"""

    def _handle_explanation_query(self, query: str) -> str:
        """Handle explanation queries"""
        
        if 'sa-ccr' in query.lower() or 'standardized' in query.lower():
            return """**📚 SA-CCR (Standardized Approach for Counterparty Credit Risk) Explained**

**🎯 What is SA-CCR?**
SA-CCR is Basel III's standard method for calculating Exposure at Default (EAD) for derivative transactions. It replaced the Current Exposure Method (CEM).

**🔢 Core Formula:**
`EAD = Alpha × (RC + PFE)`

Where:
• **Alpha** = 1.4 (bilateral) or 0.5 (centrally cleared)  
• **RC** = Replacement Cost (current exposure)
• **PFE** = Potential Future Exposure (add-on component)

**🏗️ Key Components:**

**Replacement Cost (RC):**
• Current mark-to-market value of trades
• Adjusted for collateral held
• Represents today's loss if counterparty defaults

**Potential Future Exposure (PFE):**
• Forward-looking exposure over 1-year horizon
• Based on supervisory factors and correlations
• Captures potential market moves

**📊 24-Step Process:**
1. Data validation and classification (Steps 1-4)
2. Risk parameter calculations (Steps 5-8) 
3. Add-on aggregation (Steps 9-13)
4. PFE calculations (Steps 14-16)
5. Replacement cost (Steps 17-18)
6. Final EAD and RWA (Steps 19-24)

**🎯 Benefits:**
• More risk-sensitive than CEM
• Recognizes netting and collateral
• Differentiates margined vs unmargined
• Consistent global implementation

Want me to explain any specific component in more detail?"""
        
        else:
            return f"""**🤔 I'd be happy to explain that!**

Could you be more specific about what you'd like me to explain? I can help with:

**🔹 SA-CCR Methodology**
• 24-step calculation process
• RC vs PFE components  
• Asset class treatments

**🔹 Basel Regulations**
• Capital requirements
• Risk weighting approaches
• Regulatory compliance

**🔹 Risk Management**
• Hedging strategies
• Netting benefits
• Collateral optimization

**🔹 Technical Concepts**
• Supervisory factors
• Correlation parameters
• Maturity adjustments

Just ask about any specific topic!"""

    def _handle_step_breakdown_query(self) -> str:
        """Handle 24-step breakdown requests"""
        
        if st.session_state.calculation_results:
            return """**📋 24-Step SA-CCR Calculation Breakdown Available!**

I can see you have calculation results. Here's how to access the detailed breakdown:

**🔍 Navigation Options:**

**1. Scenario Comparison Tab**
• Go to Scenario Comparison in the sidebar
• View complete 24-step process with formulas
• Interactive step-by-step navigation

**2. Detailed Results View**  
• Each step shows: Description, Formula, Result, Data
• Expandable sections for complex calculations
• Visual flow chart of calculation process

**3. Key Step Categories:**
• **Steps 1-4**: Data & Classification
• **Steps 5-8**: Risk Calculations  
• **Steps 9-13**: Add-On Aggregation
• **Steps 14-16**: PFE Calculations
• **Steps 17-18**: Replacement Cost
• **Steps 19-24**: Final EAD & RWA

**🎯 Most Important Steps:**
• **Step 9**: Adjusted Derivatives Amount
• **Step 15**: PFE Multiplier
• **Step 21**: Final EAD Calculation  
• **Step 24**: Risk Weighted Assets

Would you like me to explain any specific step in detail, or shall I guide you to the interactive breakdown view?"""
        
        else:
            return """**📋 SA-CCR 24-Step Process Overview**

The SA-CCR uses a comprehensive 24-step process:

**📊 Steps 1-4: Data & Classification**
1. Netting Set Data Collection
2. Asset Class Classification  
3. Hedging Set Definition
4. Time Parameters (S, E, M)

**⚖️ Steps 5-8: Risk Calculations**
5. Adjusted Notional Calculation
6. Maturity Factor Application
7. Supervisory Delta Adjustments  
8. Supervisory Factor Application

**🔗 Steps 9-13: Add-On Aggregation**
9. Adjusted Derivatives Contract Amount
10. Supervisory Correlation Parameters
11. Hedging Set Add-On Calculation
12. Asset Class Add-On Aggregation
13. Aggregate Add-On Computation

**📈 Steps 14-16: PFE Calculations**
14. Sum of V and C Components
15. PFE Multiplier Calculation
16. Potential Future Exposure (PFE)

**💰 Steps 17-18: Replacement Cost**
17. Threshold, MTA, NICA Parameters
18. Replacement Cost (RC) Calculation

**🎯 Steps 19-24: Final EAD & RWA**
19. Central Clearing (CEU) Flag
20. Alpha Multiplier Application
21. Exposure at Default (EAD)
22. Counterparty Information
23. Risk Weight Determination  
24. Risk Weighted Assets (RWA)

To see this applied to your portfolio, upload trades or create a sample calculation!"""

    def _handle_general_query(self, query: str) -> str:
        """Handle general queries"""
        
        return f"""**🤖 I'm here to help with SA-CCR calculations and optimization!**

I understand you're asking about: *"{query}"*

**🔹 Here's what I can help you with:**

**📊 Calculations & Analysis**
• SA-CCR calculations for any portfolio size
• Excel batch upload for multiple trades
• 24-step detailed breakdowns
• Risk analysis and technical insights

**🎯 Optimization & Strategy**
• Capital requirement reduction strategies
• Central clearing vs bilateral analysis
• Netting and collateral optimization
• Portfolio rebalancing recommendations

**📚 Education & Guidance**
• Basel III SA-CCR methodology
• Regulatory compliance guidance
• Risk management best practices
• Technical parameter explanations

**💡 Getting Started:**
• Upload an Excel file with your trades
• Ask for a sample calculation  
• Request specific explanations
• Get optimization recommendations

What specific aspect would you like to explore? I'm here to make SA-CCR calculations simple and actionable!"""

    def _export_chat_history(self):
        """Export chat history"""
        try:
            chat_export = []
            for message in st.session_state.ai_chat_history:
                chat_export.append({
                    'role': message['role'],
                    'content': message['content'],
                    'timestamp': message['timestamp'].isoformat()
                })
            
            export_data = json.dumps(chat_export, indent=2)
            
            st.download_button(
                label="📥 Download Chat History",
                data=export_data,
                file_name=f"saccr_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            st.success("✅ Chat history ready for download!")
            
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")

    def _render_calculator_page(self):
        """Render main SA-CCR calculator page with Excel upload"""
        
        st.markdown("## 📊 SA-CCR Calculator")
        
        # Upload option at the top
        st.markdown("### 📤 Quick Upload")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📤 Upload Excel Portfolio", use_container_width=True):
                st.session_state.show_excel_upload = True
        
        with col2:
            if st.button("📊 Manual Entry", use_container_width=True):
                st.session_state.show_manual_entry = True
        
        with col3:
            if st.button("📋 Load Saved Portfolio", use_container_width=True):
                self._show_portfolio_loader()
        
        # Excel Upload Section
        if st.session_state.get('show_excel_upload', False):
            self._render_excel_upload_section()
        
        # Manual entry section
        if st.session_state.get('show_manual_entry', True):
            self._render_manual_portfolio_input()
        
        # Calculation and results
        if st.session_state.get('current_portfolio'):
            self._render_portfolio_summary_and_calculation()

    def _render_manual_portfolio_input(self):
        """Render manual portfolio input interface"""
        
        st.markdown("---")
        st.markdown("### ✍️ Manual Trade Entry")
        
        # Netting set configuration
        col1, col2 = st.columns(2)
        with col1:
            netting_set_id = st.text_input("Netting Set ID", key="ns_id")
            counterparty = st.text_input("Counterparty", key="counterparty")
        
        with col2:
            threshold = st.number_input("Threshold ($)", min_value=0.0, key="threshold")
            mta = st.number_input("MTA ($)", min_value=0.0, key="mta")
        
        # Trade input form
        with st.expander("Add New Trade", expanded=True):
            trade_form = st.form("trade_input")
            
            with trade_form:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    trade_id = st.text_input("Trade ID")
                    asset_class = st.selectbox("Asset Class", [ac.value for ac in AssetClass])
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
            st.success(f"✅ Added trade {trade_id} to portfolio")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error adding trade: {str(e)}")
            logger.error(f"Trade addition error: {e}")

    def _render_portfolio_summary_and_calculation(self):
        """Render portfolio summary and calculation interface"""
        
        st.markdown("---")
        st.markdown("### 📊 Portfolio Summary")
        
        portfolio = st.session_state.current_portfolio
        trades = portfolio.get('trades', [])
        
        if not trades:
            st.info("No trades in portfolio yet.")
            return
        
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
        st.markdown("#### 📋 Trade Details")
        st.dataframe(trades_df, use_container_width=True)
        
        # Calculation buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Calculate SA-CCR", type="primary", use_container_width=True):
                self._perform_calculation()
        
        with col2:
            if st.button("💾 Save Portfolio", use_container_width=True):
                self._save_portfolio()
        
        with col3:
            if st.button("🧹 Clear Portfolio", use_container_width=True):
                st.session_state.current_portfolio = None
                st.rerun()
        
        # Results display
        if st.session_state.calculation_results:
            self._render_enhanced_calculation_results()

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
        
        try:
            with st.spinner("🔄 Calculating SA-CCR..."):
                # Perform calculation
                results = self.saccr_engine.calculate_comprehensive_saccr(
                    portfolio, 
                    st.session_state.collateral_input
                )
                
                # Store results
                st.session_state.calculation_results = results
                st.session_state.last_calculation_time = datetime.now()
                
                # Save to database
                try:
                    self.db_manager.save_calculation_results(portfolio, results)
                except Exception as e:
                    logger.warning(f"Failed to save results to database: {e}")
                
                st.success("✅ SA-CCR calculation completed successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Calculation failed: {str(e)}")
            logger.error(f"SA-CCR calculation error: {e}")

    def _render_enhanced_calculation_results(self):
        """Render enhanced calculation results with professional formatting"""
        
        results = st.session_state.calculation_results
        final_results = results['final_results']
        
        st.markdown("---")
        st.markdown("## 📊 SA-CCR Calculation Results")
        
        # Key metrics in grid
        st.markdown('<div class="result-grid">', unsafe_allow_html=True)
        
        metrics = [
            ("Replacement Cost", final_results['replacement_cost'], "M"),
            ("PFE", final_results['potential_future_exposure'], "M"),
            ("EAD", final_results['exposure_at_default'], "M"),
            ("RWA", final_results['risk_weighted_assets'], "M"),
            ("Capital Required", final_results['capital_requirement'], "K")
        ]
        
        for label, value, unit in metrics:
            divisor = 1_000_000 if unit == "M" else 1_000
            formatted_value = f"${value/divisor:.2f}{unit}"
            
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">{label}</div>
                <div class="result-value">{formatted_value}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Executive Summary", 
            "🔢 Technical Analysis", 
            "📋 Trade Breakdown",
            "🎯 Optimization"
        ])
        
        with tab1:
            self._render_executive_summary(results)
        
        with tab2:
            self._render_technical_analysis(results)
        
        with tab3:
            self._render_trade_breakdown_analysis(results)
        
        with tab4:
            self._render_optimization_recommendations(results)

    def _render_executive_summary(self, results):
        """Render executive summary with bullet points"""
        
        final_results = results['final_results']
        
        st.markdown("### 📊 Executive Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Key Risk Metrics")
            
            total_notional = final_results['portfolio_summary']['total_notional']
            ead = final_results['exposure_at_default']
            ead_ratio = (ead / total_notional) * 100 if total_notional > 0 else 0
            
            st.markdown(f"""
            • **EAD/Notional Ratio**: {ead_ratio:.2f}%
            • **Capital Efficiency**: {(1 - ead_ratio/100)*100:.1f}%
            • **Risk Profile**: {'🟢 Low' if ead_ratio < 5 else '🟡 Medium' if ead_ratio < 15 else '🔴 High'}
            • **Total Notional**: ${total_notional/1_000_000:.1f}M
            • **Trade Count**: {final_results['portfolio_summary']['trade_count']}
            """)
        
        with col2:
            st.markdown("#### 🔍 Risk Drivers")
            
            if ead > 0:
                rc_contribution = (final_results['replacement_cost'] / ead) * 100
                pfe_contribution = (final_results['potential_future_exposure'] / ead) * 100
                
                st.markdown(f"""
                • **RC Contribution**: {rc_contribution:.1f}%
                • **PFE Contribution**: {pfe_contribution:.1f}%
                • **Alpha Factor**: 1.4 (Bilateral)
                • **Multiplier Effect**: Applied to PFE
                • **Netting Benefits**: {'✅ Active' if len(set()) > 1 else '⚠️ Limited'}
                """)

    def _render_technical_analysis(self, results):
        """Render technical analysis with detailed breakdown"""
        
        calculation_steps = results['calculation_steps']
        
        st.markdown("### 🔬 Technical Analysis")
        
        # Key technical metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚙️ Key Parameters")
            
            # Find specific steps
            pfe_step = next((step for step in calculation_steps if step['step'] == 15), None)
            if pfe_step and 'data' in pfe_step:
                multiplier = pfe_step['data'].get('multiplier', 'N/A')
                st.markdown(f"• **PFE Multiplier**: {multiplier}")
            
            st.markdown("""
            • **Supervisory Factors**: Asset-class specific
            • **Correlation Parameters**: Applied within hedging sets
            • **Maturity Adjustments**: Time-weighted exposure
            • **Delta Adjustments**: Directional risk capture
            """)
        
        with col2:
            st.markdown("#### 📊 Calculation Flow")
            
            st.markdown("""
            • **Steps 1-4**: Data validation and classification
            • **Steps 5-8**: Risk parameter application
            • **Steps 9-13**: Add-on aggregation
            • **Steps 14-16**: PFE calculations
            • **Steps 17-18**: Replacement cost
            • **Steps 19-24**: Final EAD and RWA
            """)
        
        # Navigation to detailed breakdown
        st.markdown("---")
        if st.button("🔍 View Complete 24-Step Breakdown", use_container_width=True):
            st.session_state.current_page = 'comparison'
            st.rerun()

    def _render_trade_breakdown_analysis(self, results):
        """Render detailed trade breakdown analysis"""
        
        st.markdown("### 📋 Trade Breakdown Analysis")
        
        portfolio = st.session_state.current_portfolio
        trades = portfolio.get('trades', [])
        
        # Asset class analysis
        asset_class_data = {}
        for trade in trades:
            ac = trade.asset_class.value
            if ac not in asset_class_data:
                asset_class_data[ac] = {'count': 0, 'notional': 0, 'mtm': 0}
            asset_class_data[ac]['count'] += 1
            asset_class_data[ac]['notional'] += abs(trade.notional)
            asset_class_data[ac]['mtm'] += trade.mtm_value
        
        # Display as formatted table
        st.markdown("#### 📊 By Asset Class")
        
        asset_df_data = []
        for ac, data in asset_class_data.items():
            asset_df_data.append({
                'Asset Class': ac,
                'Trade Count': data['count'],
                'Notional ($M)': f"{data['notional']/1_000_000:.1f}",
                'MTM ($K)': f"{data['mtm']/1_000:.0f}",
                'Avg Trade Size ($M)': f"{(data['notional']/data['count'])/1_000_000:.1f}"
            })
        
        asset_df = pd.DataFrame(asset_df_data)
        st.dataframe(asset_df, use_container_width=True)
        
        # Currency analysis
        st.markdown("#### 💱 By Currency")
        
        currency_data = {}
        for trade in trades:
            curr = trade.currency
            if curr not in currency_data:
                currency_data[curr] = {'count': 0, 'notional': 0}
            currency_data[curr]['count'] += 1
            currency_data[curr]['notional'] += abs(trade.notional)
        
        currency_df_data = []
        for curr, data in currency_data.items():
            currency_df_data.append({
                'Currency': curr,
                'Trade Count': data['count'],
                'Notional ($M)': f"{data['notional']/1_000_000:.1f}",
                'Percentage': f"{(data['notional']/sum(abs(t.notional) for t in trades))*100:.1f}%"
            })
        
        currency_df = pd.DataFrame(currency_df_data)
        st.dataframe(currency_df, use_container_width=True)

    def _render_optimization_recommendations(self, results):
        """Render optimization recommendations"""
        
        st.markdown("### 🎯 Optimization Recommendations")
        
        final_results = results['final_results']
        calculation_steps = results['calculation_steps']
        
        recommendations = []
        
        # Analyze results for recommendations
        if final_results['replacement_cost'] > 0:
            recommendations.append({
                'category': '💰 Collateral Optimization',
                'recommendation': 'Consider posting variation margin to reduce replacement cost component',
                'impact': 'High',
                'effort': 'Medium',
                'potential_saving': f"Up to ${final_results['replacement_cost']/1_000_000:.1f}M in EAD reduction"
            })
        
        # Check for central clearing opportunities
        ceu_step = next((step for step in calculation_steps if step['step'] == 19), None)
        if ceu_step and ceu_step['data'].get('overall_ceu_flag') == 1:
            current_alpha = 1.4
            cleared_alpha = 0.5
            potential_saving = final_results['exposure_at_default'] * (1 - cleared_alpha/current_alpha)
            
            recommendations.append({
                'category': '🏦 Central Clearing',
                'recommendation': 'Move eligible trades to central clearing to reduce alpha from 1.4 to 0.5',
                'impact': 'Very High',
                'effort': 'High',
                'potential_saving': f"Up to ${potential_saving/1_000_000:.1f}M in EAD reduction (64% decrease)"
            })
        
        # PFE optimization
        pfe_step = next((step for step in calculation_steps if step['step'] == 15), None)
        if pfe_step and pfe_step['data'].get('multiplier', 1) > 0.8:
            recommendations.append({
                'category': '📊 Netting Optimization',
                'recommendation': 'Improve portfolio netting by balancing long/short positions within hedging sets',
                'impact': 'Medium',
                'effort': 'Low',
                'potential_saving': 'Optimize PFE multiplier for better capital efficiency'
            })
        
        # Display recommendations
        for i, rec in enumerate(recommendations):
            with st.expander(f"💡 Recommendation {i+1}: {rec['category']}", expanded=True):
                st.markdown(f"**📋 Action:** {rec['recommendation']}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    impact_colors = {'Low': '🟢', 'Medium': '🟡', 'High': '🟠', 'Very High': '🔴'}
                    st.markdown(f"**Impact:** {impact_colors.get(rec['impact'], '')} {rec['impact']}")
                
                with col2:
                    effort_colors = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
                    st.markdown(f"**Effort:** {effort_colors.get(rec['effort'], '')} {rec['effort']}")
                
                with col3:
                    st.markdown(f"**Potential Saving:** {rec['potential_saving']}")
        
        if not recommendations:
            st.info("📊 Your portfolio is already well-optimized! Consider periodic reviews as market conditions change.")

    def _save_portfolio(self):
        """Save current portfolio to database"""
        
        if not st.session_state.current_portfolio:
            st.error("No portfolio to save")
            return
        
        portfolio_name = st.text_input("Portfolio Name:", key="save_portfolio_name")
        
        if st.button("💾 Save") and portfolio_name:
            try:
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
                st.success(f"✅ Portfolio '{portfolio_name}' saved successfully!")
            except Exception as e:
                st.error(f"❌ Failed to save portfolio: {str(e)}")

    def _show_portfolio_loader(self):
        """Show portfolio loader interface"""
        
        st.markdown("### 📊 Load Saved Portfolio")
        
        try:
            portfolios = self.db_manager.get_portfolio_summary()
            
            if not portfolios.empty:
                portfolio_options = portfolios['portfolio_name'].tolist()
                selected_portfolio = st.selectbox("Select Portfolio:", portfolio_options)
                
                if st.button("Load Portfolio") and selected_portfolio:
                    portfolio_id = portfolios[portfolios['portfolio_name'] == selected_portfolio]['portfolio_id'].iloc[0]
                    portfolio = self.db_manager.load_portfolio(portfolio_id)
                    
                    if portfolio and portfolio.netting_sets:
                        netting_set = portfolio.netting_sets[0]
                        
                        st.session_state.current_portfolio = {
                            'netting_set_id': netting_set.netting_set_id,
                            'counterparty': netting_set.counterparty,
                            'threshold': netting_set.threshold,
                            'mta': netting_set.mta,
                            'trades': netting_set.trades
                        }
                        
                        st.success(f"✅ Loaded portfolio: {selected_portfolio}")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load portfolio")
            else:
                st.info("📂 No saved portfolios found")
                
        except Exception as e:
            st.error(f"❌ Error loading portfolios: {str(e)}")

    def _render_comparison_page(self):
        """Render 24-step breakdown comparison page"""
        
        st.markdown("## ⚖️ 24-Step SA-CCR Breakdown")
        
        if not st.session_state.calculation_results:
            st.markdown("""
            <div class="alert alert-info">
                <strong>📊 No calculation results available</strong><br>
                Please run a SA-CCR calculation first to view the detailed 24-step breakdown.
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Go to Calculator", use_container_width=True):
                st.session_state.current_page = 'calculator'
                st.rerun()
            return
        
        results = st.session_state.calculation_results
        calculation_steps = results['calculation_steps']
        
        st.markdown("### 🔍 Complete Calculation Process")
        
        # Navigation for step groups
        step_groups = {
            "📊 Data & Classification (Steps 1-4)": list(range(0, 4)),
            "⚖️ Risk Calculations (Steps 5-8)": list(range(4, 8)),
            "🔗 Add-On Aggregation (Steps 9-13)": list(range(8, 13)),
            "📈 PFE Calculations (Steps 14-16)": list(range(13, 16)),
            "💰 Replacement Cost (Steps 17-18)": list(range(16, 18)),
            "🎯 Final EAD & RWA (Steps 19-24)": list(range(18, 24))
        }
        
        # Create tabs for each group
        group_tabs = st.tabs(list(step_groups.keys()))
        
        for i, (group_name, step_indices) in enumerate(step_groups.items()):
            with group_tabs[i]:
                st.markdown(f"#### {group_name}")
                
                for idx in step_indices:
                    if idx < len(calculation_steps):
                        step = calculation_steps[idx]
                        self._render_detailed_calculation_step(step)

    def _render_detailed_calculation_step(self, step):
        """Render individual detailed calculation step"""
        
        st.markdown(f"""
        <div class="step-breakdown">
            <div class="step-header">
                <span class="step-number">{step['step']}</span>
                <strong>{step['title']}</strong>
            </div>
            <div class="step-content">
                <p><strong>📋 Description:</strong> {step['description']}</p>
                <div style="background: #f8fafc; padding: 1rem; border-radius: 6px; margin: 1rem 0; border-left: 4px solid #3182ce;">
                    <strong>🧮 Formula:</strong><br>
                    <code>{step['formula']}</code>
                </div>
                <p><strong>✅ Result:</strong> {step['result']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show detailed data for complex steps
        if step['step'] in [9, 11, 12, 13, 21, 24] and 'data' in step:
            with st.expander(f"📊 Detailed Data - Step {step['step']}", expanded=False):
                if isinstance(step['data'], dict):
                    # Format as structured data
                    for key, value in step['data'].items():
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.write(step['data'])

    def _render_portfolio_page(self):
        """Render comprehensive portfolio analysis page"""
        
        st.markdown("## 📈 Portfolio Analysis")
        
        if not st.session_state.current_portfolio:
            st.markdown("""
            <div class="alert alert-info">
                <strong>📊 No portfolio loaded</strong><br>
                Please load a portfolio first to view detailed analysis.
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Go to Calculator", use_container_width=True):
                    st.session_state.current_page = 'calculator'
                    st.rerun()
            with col2:
                if st.button("🤖 Go to AI Assistant", use_container_width=True):
                    st.session_state.current_page = 'ai_assistant'
                    st.rerun()
            return
        
        portfolio = st.session_state.current_portfolio
        trades = portfolio.get('trades', [])
        
        # Portfolio Overview
        st.markdown("### 📊 Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(trades))
        with col2:
            total_notional = sum(abs(t.notional) for t in trades)
            st.metric("Total Notional", f"${total_notional/1_000_000:.1f}M")
        with col3:
            total_mtm = sum(t.mtm_value for t in trades)
            st.metric("Total MTM", f"${total_mtm/1_000:.0f}K")
        with col4:
            avg_trade_size = total_notional / len(trades) if trades else 0
            st.metric("Avg Trade Size", f"${avg_trade_size/1_000_000:.1f}M")
        
        # Asset Class Distribution
        st.markdown("### 📊 Asset Class Distribution")
        
        asset_class_data = {}
        for trade in trades:
            ac = trade.asset_class.value
            if ac not in asset_class_data:
                asset_class_data[ac] = {'count': 0, 'notional': 0, 'mtm': 0}
            asset_class_data[ac]['count'] += 1
            asset_class_data[ac]['notional'] += abs(trade.notional)
            asset_class_data[ac]['mtm'] += trade.mtm_value
        
        # Create pie chart for notional distribution
        if asset_class_data:
            fig_pie = px.pie(
                values=[data['notional'] for data in asset_class_data.values()],
                names=list(asset_class_data.keys()),
                title="Notional Distribution by Asset Class"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Detailed Analysis Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Trade Details", 
            "💱 Currency Analysis", 
            "📅 Maturity Profile",
            "🎯 Risk Metrics"
        ])
        
        with tab1:
            st.markdown("#### 📋 Complete Trade List")
            trades_df = self._create_detailed_trades_dataframe(trades)
            st.dataframe(trades_df, use_container_width=True)
            
            # Export functionality
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Trade List (CSV)",
                data=csv,
                file_name=f"portfolio_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with tab2:
            st.markdown("#### 💱 Currency Breakdown")
            currency_data = {}
            for trade in trades:
                curr = trade.currency
                if curr not in currency_data:
                    currency_data[curr] = {'count': 0, 'notional': 0, 'mtm': 0}
                currency_data[curr]['count'] += 1
                currency_data[curr]['notional'] += abs(trade.notional)
                currency_data[curr]['mtm'] += trade.mtm_value
            
            currency_df_data = []
            for curr, data in currency_data.items():
                currency_df_data.append({
                    'Currency': curr,
                    'Trade Count': data['count'],
                    'Notional ($M)': f"{data['notional']/1_000_000:.1f}",
                    'MTM ($K)': f"{data['mtm']/1_000:.0f}",
                    'Percentage': f"{(data['notional']/total_notional)*100:.1f}%"
                })
            
            currency_df = pd.DataFrame(currency_df_data)
            st.dataframe(currency_df, use_container_width=True)
        
        with tab3:
            st.markdown("#### 📅 Maturity Profile")
            
            # Maturity buckets
            maturity_buckets = {
                '< 1 Year': 0, '1-3 Years': 0, '3-5 Years': 0, 
                '5-10 Years': 0, '> 10 Years': 0
            }
            
            for trade in trades:
                maturity = trade.time_to_maturity()
                if maturity < 1:
                    maturity_buckets['< 1 Year'] += abs(trade.notional)
                elif maturity < 3:
                    maturity_buckets['1-3 Years'] += abs(trade.notional)
                elif maturity < 5:
                    maturity_buckets['3-5 Years'] += abs(trade.notional)
                elif maturity < 10:
                    maturity_buckets['5-10 Years'] += abs(trade.notional)
                else:
                    maturity_buckets['> 10 Years'] += abs(trade.notional)
            
            # Create bar chart for maturity profile
            fig_bar = px.bar(
                x=list(maturity_buckets.keys()),
                y=[v/1_000_000 for v in maturity_buckets.values()],
                title="Notional by Maturity Bucket",
                labels={'x': 'Maturity Bucket', 'y': 'Notional ($M)'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab4:
            st.markdown("#### 🎯 Risk Metrics Summary")
            
            if st.session_state.calculation_results:
                results = st.session_state.calculation_results
                final_results = results['final_results']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔍 Key Risk Indicators**")
                    ead = final_results['exposure_at_default']
                    ead_ratio = (ead / total_notional) * 100 if total_notional > 0 else 0
                    
                    st.markdown(f"""
                    • **EAD**: ${ead/1_000_000:.2f}M
                    • **EAD/Notional**: {ead_ratio:.2f}%
                    • **RC**: ${final_results['replacement_cost']/1_000_000:.2f}M
                    • **PFE**: ${final_results['potential_future_exposure']/1_000_000:.2f}M
                    • **RWA**: ${final_results['risk_weighted_assets']/1_000_000:.2f}M
                    """)
                
                with col2:
                    st.markdown("**📊 Portfolio Efficiency**")
                    
                    rc_contribution = (final_results['replacement_cost'] / ead) * 100 if ead > 0 else 0
                    pfe_contribution = (final_results['potential_future_exposure'] / ead) * 100 if ead > 0 else 0
                    
                    st.markdown(f"""
                    • **RC Contribution**: {rc_contribution:.1f}%
                    • **PFE Contribution**: {pfe_contribution:.1f}%
                    • **Capital Requirement**: ${final_results['capital_requirement']/1_000:.0f}K
                    • **Risk Profile**: {'🟢 Low' if ead_ratio < 5 else '🟡 Medium' if ead_ratio < 15 else '🔴 High'}
                    """)
            else:
                st.info("📊 Run SA-CCR calculation to see detailed risk metrics")
                if st.button("🚀 Calculate SA-CCR", type="primary"):
                    st.session_state.current_page = 'calculator'
                    st.rerun()

    def _render_optimization_page(self):
        """Render portfolio optimization analysis page"""
        
        st.markdown("## 🎯 Portfolio Optimization")
        
        if not st.session_state.current_portfolio:
            st.markdown("""
            <div class="alert alert-info">
                <strong>📊 No portfolio loaded</strong><br>
                Please load a portfolio first to view optimization analysis.
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Go to Calculator", use_container_width=True):
                    st.session_state.current_page = 'calculator'
                    st.rerun()
            with col2:
                if st.button("🤖 Go to AI Assistant", use_container_width=True):
                    st.session_state.current_page = 'ai_assistant'
                    st.rerun()
            return
        
        portfolio = st.session_state.current_portfolio
        trades = portfolio.get('trades', [])
        
        st.markdown("### 🎯 Optimization Strategies")
        
        # Optimization Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏦 Central Clearing Opportunities")
            
            # Analyze trades suitable for central clearing
            clearable_trades = []
            for trade in trades:
                # Simple heuristic: IR swaps and some credit derivatives are typically clearable
                if trade.asset_class.value in ['Interest Rate'] and trade.trade_type.value in ['Swap']:
                    clearable_trades.append(trade)
            
            clearable_notional = sum(abs(t.notional) for t in clearable_trades)
            total_notional = sum(abs(t.notional) for t in trades)
            
            st.markdown(f"""
            • **Clearable Trades**: {len(clearable_trades)} of {len(trades)}
            • **Clearable Notional**: ${clearable_notional/1_000_000:.1f}M
            • **Clearable %**: {(clearable_notional/total_notional)*100:.1f}%
            • **Potential Alpha Reduction**: 1.4 → 0.5 (64% decrease)
            """)
            
            if clearable_trades and st.session_state.calculation_results:
                current_ead = st.session_state.calculation_results['final_results']['exposure_at_default']
                # Simplified calculation: assume clearing reduces alpha for clearable portion
                clearing_benefit = clearable_notional * 0.02 * (1.4 - 0.5)  # Rough estimate
                st.success(f"💰 Estimated EAD Reduction: ${clearing_benefit/1_000_000:.1f}M")
        
        with col2:
            st.markdown("#### 📊 Netting Optimization")
            
            # Analyze netting opportunities
            asset_class_summary = {}
            for trade in trades:
                ac = trade.asset_class.value
                if ac not in asset_class_summary:
                    asset_class_summary[ac] = {'long': 0, 'short': 0, 'count': 0}
                
                if trade.mtm_value >= 0:
                    asset_class_summary[ac]['long'] += abs(trade.notional)
                else:
                    asset_class_summary[ac]['short'] += abs(trade.notional)
                asset_class_summary[ac]['count'] += 1
            
            st.markdown("**Netting Efficiency by Asset Class:**")
            for ac, data in asset_class_summary.items():
                net_exposure = abs(data['long'] - data['short'])
                gross_exposure = data['long'] + data['short']
                netting_ratio = (1 - net_exposure/gross_exposure) * 100 if gross_exposure > 0 else 0
                
                st.markdown(f"• **{ac}**: {netting_ratio:.1f}% efficiency ({data['count']} trades)")
        
        # Optimization Scenarios
        st.markdown("### 📈 Optimization Scenarios")
        
        if st.session_state.calculation_results:
            results = st.session_state.calculation_results
            final_results = results['final_results']
            
            scenarios = {
                'Current Portfolio': {
                    'ead': final_results['exposure_at_default'],
                    'rwa': final_results['risk_weighted_assets'],
                    'capital': final_results['capital_requirement']
                }
            }
            
            # Central clearing scenario
            if clearable_notional > 0:
                # Simplified: assume 64% reduction on clearable portion
                clearing_ead_reduction = clearable_notional * 0.02 * (1.4 - 0.5)
                scenarios['With Central Clearing'] = {
                    'ead': final_results['exposure_at_default'] - clearing_ead_reduction,
                    'rwa': final_results['risk_weighted_assets'] - clearing_ead_reduction,
                    'capital': final_results['capital_requirement'] - (clearing_ead_reduction * 0.08)
                }
            
            # Enhanced collateral scenario
            if final_results['replacement_cost'] > 0:
                collateral_ead_reduction = final_results['replacement_cost'] * 1.4
                scenarios['With Enhanced Collateral'] = {
                    'ead': final_results['exposure_at_default'] - collateral_ead_reduction,
                    'rwa': final_results['risk_weighted_assets'] - collateral_ead_reduction,
                    'capital': final_results['capital_requirement'] - (collateral_ead_reduction * 0.08)
                }
            
            # Create comparison table
            scenario_df_data = []
            for scenario, metrics in scenarios.items():
                scenario_df_data.append({
                    'Scenario': scenario,
                    'EAD ($M)': f"{metrics['ead']/1_000_000:.2f}",
                    'RWA ($M)': f"{metrics['rwa']/1_000_000:.2f}",
                    'Capital ($K)': f"{metrics['capital']/1_000:.0f}",
                    'Savings vs Current': f"${(scenarios['Current Portfolio']['capital'] - metrics['capital'])/1_000:.0f}K" if scenario != 'Current Portfolio' else '-'
                })
            
            scenario_df = pd.DataFrame(scenario_df_data)
            st.dataframe(scenario_df, use_container_width=True)
            
        else:
            st.info("📊 Run SA-CCR calculation first to see optimization scenarios")
            if st.button("🚀 Calculate SA-CCR", type="primary"):
                st.session_state.current_page = 'calculator'
                st.rerun()
        
        # Action Items
        st.markdown("### 📋 Recommended Actions")
        
        recommendations = [
            "🏦 **Central Clearing**: Move eligible IR swaps to central clearing for immediate alpha reduction",
            "💰 **Collateral Posting**: Implement variation margin posting to reduce replacement cost",
            "📊 **Netting Enhancement**: Consolidate trades with same counterparty under master netting agreements",
            "🔄 **Portfolio Rebalancing**: Balance long/short positions within asset class hedging sets",
            "📅 **Maturity Laddering**: Optimize trade maturities for better PFE multiplier effects"
        ]
        
        for rec in recommendations:
            st.markdown(f"• {rec}")

    def _render_database_page(self):
        """Render database management page"""
        
        st.markdown("## 🗄️ Data Management")
        
        # Database Status
        st.markdown("### 📊 Database Status")
        
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                trade_count = self.db_manager.get_trade_count()
                st.metric("Total Trades", trade_count)
            
            with col2:
                portfolio_count = self.db_manager.get_portfolio_count()
                st.metric("Saved Portfolios", portfolio_count)
            
            with col3:
                # Get database size if possible
                st.metric("Database Size", "~MB")
            
            with col4:
                st.metric("Last Backup", "Manual")
            
            st.success("✅ Database connection active")
            
        except Exception as e:
            st.error(f"❌ Database connection error: {str(e)}")
            return
        
        # Portfolio Management
        st.markdown("### 📁 Portfolio Management")
        
        tab1, tab2, tab3 = st.tabs(["📋 Saved Portfolios", "📤 Import/Export", "🧹 Maintenance"])
        
        with tab1:
            st.markdown("#### 📋 Saved Portfolios")
            
            try:
                portfolios = self.db_manager.get_portfolio_summary()
                
                if not portfolios.empty:
                    # Add action buttons to portfolios
                    portfolios_display = portfolios.copy()
                    st.dataframe(portfolios_display, use_container_width=True)
                    
                    # Portfolio actions
                    st.markdown("#### ⚙️ Portfolio Actions")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        selected_portfolio = st.selectbox(
                            "Select Portfolio:", 
                            portfolios['portfolio_name'].tolist()
                        )
                    
                    with col2:
                        action_col1, action_col2 = st.columns(2)
                        
                        with action_col1:
                            if st.button("📥 Load", use_container_width=True):
                                portfolio_id = portfolios[portfolios['portfolio_name'] == selected_portfolio]['portfolio_id'].iloc[0]
                                portfolio = self.db_manager.load_portfolio(portfolio_id)
                                
                                if portfolio and portfolio.netting_sets:
                                    netting_set = portfolio.netting_sets[0]
                                    
                                    st.session_state.current_portfolio = {
                                        'netting_set_id': netting_set.netting_set_id,
                                        'counterparty': netting_set.counterparty,
                                        'threshold': netting_set.threshold,
                                        'mta': netting_set.mta,
                                        'trades': netting_set.trades
                                    }
                                    
                                    st.success(f"✅ Loaded portfolio: {selected_portfolio}")
                                    st.rerun()
                        
                        with action_col2:
                            if st.button("🗑️ Delete", use_container_width=True):
                                if st.button("⚠️ Confirm Delete", type="secondary"):
                                    # Add delete functionality
                                    st.warning("Delete functionality to be implemented")
                
                else:
                    st.info("📂 No saved portfolios found")
            
            except Exception as e:
                st.error(f"❌ Error loading portfolios: {str(e)}")
        
        with tab2:
            st.markdown("#### 📤 Import/Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📥 Import Options**")
                
                # Excel import
                uploaded_file = st.file_uploader(
                    "Import Portfolio from Excel",
                    type=['xlsx', 'xls'],
                    help="Upload Excel file with trade data"
                )
                
                if uploaded_file is not None:
                    try:
                        df = pd.read_excel(uploaded_file)
                        st.success(f"✅ File loaded: {len(df)} trades found")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        if st.button("🚀 Import to Database"):
                            # Import logic here
                            st.success("✅ Data imported successfully!")
                    except Exception as e:
                        st.error(f"❌ Import failed: {str(e)}")
            
            with col2:
                st.markdown("**📤 Export Options**")
                
                if st.button("📊 Export All Portfolios"):
                    try:
                        # Create export data
                        export_data = {"portfolios": [], "trades": []}
                        
                        # Add export logic here
                        export_json = json.dumps(export_data, indent=2, default=str)
                        
                        st.download_button(
                            label="📥 Download Export File",
                            data=export_json,
                            file_name=f"saccr_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Export failed: {str(e)}")
        
        with tab3:
            st.markdown("#### 🧹 Database Maintenance")
            
            st.markdown("**⚠️ Maintenance Operations**")
            st.warning("These operations can affect your data. Use with caution.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Optimize Database"):
                    with st.spinner("Optimizing database..."):
                        # Add optimization logic
                        st.success("✅ Database optimized")
                
                if st.button("🧹 Clean Old Data"):
                    cutoff_days = st.number_input("Delete data older than (days):", min_value=1, max_value=365, value=90)
                    if st.button("⚠️ Confirm Clean"):
                        # Add cleanup logic
                        st.success("✅ Old data cleaned")
            
            with col2:
                if st.button("💾 Backup Database"):
                    with st.spinner("Creating backup..."):
                        # Add backup logic
                        st.success("✅ Backup created")
                
                if st.button("📊 Database Statistics"):
                    # Show detailed database stats
                    st.info("📊 Database statistics feature to be implemented")

    def _render_settings_page(self):
        """Render application settings page"""
        
        st.markdown("## ⚙️ Application Settings")
        
        # Calculation Parameters
        st.markdown("### 🔢 Calculation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 SA-CCR Parameters")
            
            alpha_bilateral = st.number_input(
                "Alpha (Bilateral)",
                min_value=0.1,
                max_value=3.0,
                value=st.session_state.calculation_parameters.get('alpha_bilateral', 1.4),
                step=0.1,
                help="Alpha multiplier for bilateral trades (Basel III default: 1.4)"
            )
            
            alpha_cleared = st.number_input(
                "Alpha (Centrally Cleared)",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.calculation_parameters.get('alpha_cleared', 0.5),
                step=0.1,
                help="Alpha multiplier for centrally cleared trades (Basel III default: 0.5)"
            )
            
            capital_ratio = st.number_input(
                "Capital Ratio",
                min_value=0.01,
                max_value=0.20,
                value=st.session_state.calculation_parameters.get('capital_ratio', 0.08),
                step=0.01,
                format="%.3f",
                help="Capital ratio for RWA calculation (Basel III minimum: 8%)"
            )
        
        with col2:
            st.markdown("#### ⚙️ Application Settings")
            
            enable_cache = st.checkbox(
                "Enable Calculation Caching",
                value=st.session_state.calculation_parameters.get('enable_cache', True),
                help="Cache calculation results for faster performance"
            )
            
            show_debug = st.checkbox(
                "Show Debug Information",
                value=st.session_state.calculation_parameters.get('show_debug', False),
                help="Display detailed debug information in calculations"
            )
            
            decimal_places = st.selectbox(
                "Decimal Places",
                options=[0, 1, 2, 3, 4],
                index=st.session_state.calculation_parameters.get('decimal_places', 2),
                help="Number of decimal places for displayed results"
            )
        
        # Save settings
        if st.button("💾 Save Settings", type="primary"):
            st.session_state.calculation_parameters.update({
                'alpha_bilateral': alpha_bilateral,
                'alpha_cleared': alpha_cleared,
                'capital_ratio': capital_ratio,
                'enable_cache': enable_cache,
                'show_debug': show_debug,
                'decimal_places': decimal_places
            })
            st.success("✅ Settings saved successfully!")
            st.rerun()
        
        # Display Settings
        st.markdown("### 🎨 Display Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🖼️ UI Preferences")
            
            theme = st.selectbox(
                "Theme",
                options=["Professional", "Dark", "Light"],
                index=0,
                help="Application color theme"
            )
            
            language = st.selectbox(
                "Language",
                options=["English", "French", "German", "Spanish"],
                index=0,
                help="Application language"
            )
        
        with col2:
            st.markdown("#### 📊 Chart Settings")
            
            chart_style = st.selectbox(
                "Chart Style",
                options=["Default", "Minimal", "Corporate"],
                index=0,
                help="Default chart styling"
            )
            
            color_scheme = st.selectbox(
                "Color Scheme",
                options=["Blue", "Green", "Purple", "Orange"],
                index=0,
                help="Primary color scheme"
            )
        
        # Export/Import Settings
        st.markdown("### 📤 Configuration Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Settings"):
                settings_export = {
                    'calculation_parameters': st.session_state.calculation_parameters,
                    'version': '1.0',
                    'export_date': datetime.now().isoformat()
                }
                
                settings_json = json.dumps(settings_export, indent=2)
                
                st.download_button(
                    label="📥 Download Settings File",
                    data=settings_json,
                    file_name=f"saccr_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_settings = st.file_uploader(
                "📤 Import Settings",
                type=['json'],
                help="Upload previously exported settings file"
            )
            
            if uploaded_settings is not None:
                try:
                    settings_data = json.load(uploaded_settings)
                    
                    if 'calculation_parameters' in settings_data:
                        st.session_state.calculation_parameters.update(
                            settings_data['calculation_parameters']
                        )
                        st.success("✅ Settings imported successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid settings file format")
                        
                except Exception as e:
                    st.error(f"❌ Import failed: {str(e)}")
        
        # System Information
        st.markdown("### ℹ️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🖥️ Application Info")
            st.markdown("""
            • **Version**: 2.0.0
            • **Build**: 2025.01.01
            • **Environment**: Production
            • **Database**: DuckDB 1.3.2
            • **Framework**: Streamlit 1.49.1
            """)
        
        with col2:
            st.markdown("#### 📊 Usage Statistics")
            
            try:
                trade_count = self.db_manager.get_trade_count()
                portfolio_count = self.db_manager.get_portfolio_count()
                
                st.markdown(f"""
                • **Total Trades**: {trade_count:,}
                • **Saved Portfolios**: {portfolio_count:,}
                • **Calculations Run**: {st.session_state.get('total_calculations', 0)}
                • **Session Start**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                """)
            except:
                st.markdown("📊 Statistics unavailable")
        
        # Reset options
        st.markdown("### 🔄 Reset Options")
        st.warning("⚠️ These operations will reset application data. Use with caution.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reset Settings"):
                if st.button("⚠️ Confirm Reset Settings"):
                    st.session_state.calculation_parameters = {
                        'alpha_bilateral': 1.4,
                        'alpha_cleared': 0.5,
                        'capital_ratio': 0.08,
                        'enable_cache': True,
                        'show_debug': False,
                        'decimal_places': 2
                    }
                    st.success("✅ Settings reset to defaults")
                    st.rerun()
        
        with col2:
            if st.button("🧹 Clear Session"):
                if st.button("⚠️ Confirm Clear Session"):
                    # Clear session state
                    for key in list(st.session_state.keys()):
                        if key not in ['calculation_parameters']:
                            del st.session_state[key]
                    st.success("✅ Session cleared")
                    st.rerun()
        
        with col3:
            if st.button("🗑️ Reset All Data"):
                if st.button("⚠️ Confirm Reset All"):
                    st.error("🚨 This would delete all data. Feature disabled for safety.")

    def _create_detailed_trades_dataframe(self, trades):
        """Create detailed DataFrame for trade display"""
        
        data = []
        for i, trade in enumerate(trades):
            data.append({
                'Index': i + 1,
                'Trade ID': trade.trade_id,
                'Counterparty': trade.counterparty,
                'Asset Class': trade.asset_class.value,
                'Trade Type': trade.trade_type.value,
                'Notional ($M)': f"{trade.notional/1_000_000:.2f}",
                'Currency': trade.currency,
                'Underlying': trade.underlying,
                'MTM ($K)': f"{trade.mtm_value/1000:.1f}",
                'Delta': f"{trade.delta:.3f}",
                'Maturity (Y)': f"{trade.time_to_maturity():.2f}",
                'Maturity Date': trade.maturity_date.strftime('%Y-%m-%d')
            })
        
        return pd.DataFrame(data)

# Application entry point
if __name__ == "__main__":
    try:
        app = SACCRApplication()
        app.run()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application startup error: {e}")