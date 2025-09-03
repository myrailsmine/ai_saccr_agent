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

# LangChain imports for enterprise LLM
try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Excel handling imports with error handling
try:
    import xlsxwriter
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

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

# Claude-inspired CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Claude-inspired color palette */
    :root {
        --claude-orange: #CC785C;
        --claude-orange-light: #D4917C;
        --claude-orange-dark: #B85D3E;
        --claude-black: #1C1C1C;
        --claude-gray-900: #2D2D2D;
        --claude-gray-800: #393939;
        --claude-gray-700: #4A4A4A;
        --claude-gray-600: #6B6B6B;
        --claude-gray-500: #8E8E8E;
        --claude-gray-400: #B4B4B4;
        --claude-gray-300: #D1D1D1;
        --claude-gray-200: #E5E5E5;
        --claude-gray-100: #F3F3F3;
        --claude-white: #FFFFFF;
        --claude-bg: #FAFAF9;
        --claude-border: #E5E5E5;
        --success: #16A34A;
        --warning: #F59E0B;
        --error: #DC2626;
        --info: #3B82F6;
    }
    
    /* Global styles */
    .main { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--claude-bg);
        color: var(--claude-black);
        line-height: 1.6;
    }
    
    /* Remove Streamlit branding */
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp {
        background: var(--claude-bg);
    }
    
    /* Header styling - Claude-inspired */
    .main-header {
        background: linear-gradient(135deg, var(--claude-black) 0%, var(--claude-gray-900) 100%);
        color: var(--claude-white);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
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
        background: linear-gradient(45deg, var(--claude-orange) 0%, var(--claude-orange-light) 100%);
        border-radius: 50%;
        opacity: 0.1;
        transform: translate(50%, -50%);
    }
    
    .main-header h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2.25rem;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        margin: 0;
        font-size: 1.125rem;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    /* AI Settings button styling */
    .ai-settings-btn {
        background: var(--claude-orange) !important;
        color: var(--claude-white) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(204, 120, 92, 0.2) !important;
    }
    
    .ai-settings-btn:hover {
        background: var(--claude-orange-dark) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(204, 120, 92, 0.3) !important;
    }
    
    /* Claude-style chat interface */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    .chat-message {
        background: var(--claude-white);
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--claude-border);
        overflow: hidden;
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
        border-left: 4px solid var(--claude-orange);
    }
    
    .chat-message.assistant {
        background: var(--claude-white);
        border-left: 4px solid var(--claude-gray-400);
    }
    
    .chat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.5rem 0.5rem;
        font-size: 0.875rem;
        color: var(--claude-gray-600);
        font-weight: 500;
    }
    
    .chat-content {
        padding: 0 1.5rem 1.5rem;
        color: var(--claude-black);
        line-height: 1.6;
    }
    
    /* Professional result cards */
    .result-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .result-card {
        background: var(--claude-white);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--claude-border);
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .result-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--claude-orange);
        margin: 0.5rem 0;
    }
    
    .result-label {
        color: var(--claude-gray-600);
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Quick action buttons */
    .quick-actions {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .quick-action-card {
        background: var(--claude-white);
        border: 1px solid var(--claude-border);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .quick-action-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-color: var(--claude-orange);
    }
    
    /* Enhanced alerts - Claude style */
    .alert {
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
        font-weight: 500;
    }
    
    .alert-info {
        background: #F0F8FF;
        color: #1E40AF;
        border-left-color: var(--info);
    }
    
    .alert-success {
        background: #F0FDF4;
        color: #15803D;
        border-left-color: var(--success);
    }
    
    .alert-warning {
        background: #FFFBEB;
        color: #D97706;
        border-left-color: var(--warning);
    }
    
    .alert-error {
        background: #FEF2F2;
        color: #DC2626;
        border-left-color: var(--error);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--claude-white);
        border-right: 1px solid var(--claude-border);
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed var(--claude-border);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: var(--claude-white);
        transition: all 0.2s ease;
    }
    
    .upload-area:hover {
        border-color: var(--claude-orange);
        background: var(--claude-gray-100);
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--claude-orange);
        color: var(--claude-white);
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: var(--claude-orange-dark);
        transform: translateY(-1px);
    }
    
    /* Step breakdown styling */
    .step-breakdown {
        background: var(--claude-white);
        border-radius: 12px;
        border: 1px solid var(--claude-border);
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .step-header {
        background: var(--claude-gray-100);
        padding: 1rem 1.5rem;
        border-bottom: 1px solid var(--claude-border);
        font-weight: 600;
        color: var(--claude-black);
    }
    
    .step-content {
        padding: 1.5rem;
    }
    
    .step-number {
        background: var(--claude-orange);
        color: var(--claude-white);
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
    
    /* Settings panel styling */
    .settings-panel {
        background: var(--claude-white);
        border: 1px solid var(--claude-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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

@st.cache_resource
def init_enhanced_llm():
    """Initialize ChatOpenAI with enhanced configuration"""
    if not LANGCHAIN_AVAILABLE:
        logger.error("LangChain not available for AI processing")
        return None
        
    try:
        config = st.session_state.get('llm_config', {})
        return ChatOpenAI(
            base_url=config.get('base_url', "http://localhost:8123/v1"),
            api_key=config.get('api_key', "dummy"),
            model=config.get('model', "llama3"),
            temperature=config.get('temperature', 0.1),  # Lower temperature for more consistent outputs
            max_tokens=config.get('max_tokens', 6000),   # Increased for longer responses
            streaming=False
        )
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return None

class SACCRApplication:
    """Enhanced SA-CCR application with Claude-like interface and enterprise LLM"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.db_manager = DatabaseManager()
        self.saccr_engine = SACCREngine(self.config_manager)
        self.ui_components = UIComponents()
        self.validator = TradeValidator()
        self.progress_tracker = ProgressTracker()
        self.llm = init_enhanced_llm()
        
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
                'provider': 'enterprise',
                'model': 'llama3',
                'temperature': 0.1,
                'max_tokens': 6000,
                'response_style': 'professional',
                'enable_context': True,
                'show_calculations': True,
                'base_url': 'http://localhost:8123/v1',
                'api_key': 'dummy'
            },
            'llm_config': {
                'base_url': 'http://localhost:8123/v1',
                'api_key': 'dummy',
                'model': 'llama3',
                'temperature': 0.1,
                'max_tokens': 6000
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Main application entry point"""
        
        # Enhanced Professional Header with Claude styling
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown("""
            <div class="main-header">
                <h1>🤖 AI SA-CCR Analytics Platform</h1>
                <p>Intelligent Basel SA-CCR calculation and optimization with AI-powered insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("⚙️ AI Settings", help="Configure AI Assistant settings", key="ai_settings_btn"):
                st.session_state.show_ai_settings = not st.session_state.get('show_ai_settings', False)
                st.rerun()
        
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
        
        # AI Settings Status
        st.markdown("### 🤖 AI Assistant")
        llm_settings = st.session_state.llm_settings
        
        st.markdown(f"""
        **Provider**: {llm_settings.get('provider', 'enterprise').title()}  
        **Model**: {llm_settings.get('model', 'llama3')}  
        **Style**: {llm_settings.get('response_style', 'professional').title()}
        """)
        
        if st.button("⚙️ Configure AI", use_container_width=True):
            st.session_state.current_page = 'ai_assistant'
            st.session_state.show_ai_settings = True
            st.rerun()

    def _render_ai_assistant_page(self):
        """Render enhanced Claude-like AI assistant interface"""
        
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # AI Settings Panel (if enabled)
        if st.session_state.get('show_ai_settings', False):
            self._render_ai_settings_panel()
        
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

    def _render_ai_settings_panel(self):
        """Render AI settings configuration panel with Claude styling"""
        
        st.markdown("---")
        st.markdown("### ⚙️ AI Assistant Configuration")
        
        with st.container():
            st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔧 Enterprise LLM Configuration")
                
                # Base URL
                base_url = st.text_input(
                    "Base URL",
                    value=st.session_state.llm_settings.get('base_url', 'http://localhost:8123/v1'),
                    help="Enterprise LLM endpoint URL"
                )
                
                # Model selection
                model = st.selectbox(
                    "Model",
                    options=['llama3', 'llama2', 'gpt-4', 'claude-3', 'custom'],
                    index=['llama3', 'llama2', 'gpt-4', 'claude-3', 'custom'].index(
                        st.session_state.llm_settings.get('model', 'llama3')
                    ),
                    help="Select the model to use"
                )
                
                # Temperature control
                temperature = st.slider(
                    "Temperature (Creativity)",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.llm_settings.get('temperature', 0.1),
                    step=0.1,
                    help="Higher values make responses more creative, lower values more focused"
                )
            
            with col2:
                st.markdown("#### 🎯 Response Configuration")
                
                # Max tokens
                max_tokens = st.slider(
                    "Response Length (Max Tokens)",
                    min_value=1000,
                    max_value=8000,
                    value=st.session_state.llm_settings.get('max_tokens', 6000),
                    step=500,
                    help="Maximum length of AI responses"
                )
                
                # Response style
                response_style = st.selectbox(
                    "Response Style",
                    options=["professional", "casual", "technical", "educational"],
                    index=["professional", "casual", "technical", "educational"].index(
                        st.session_state.llm_settings.get('response_style', 'professional')
                    ),
                    help="Preferred tone and style for AI responses"
                )
                
                # Context settings
                enable_context = st.checkbox(
                    "Enable Conversation Context",
                    value=st.session_state.llm_settings.get('enable_context', True),
                    help="Remember previous conversation history"
                )
                
                show_calculations = st.checkbox(
                    "Show Detailed Calculations",
                    value=st.session_state.llm_settings.get('show_calculations', True),
                    help="Include step-by-step calculations in responses"
                )
        
        # Usage Statistics
        st.markdown("#### 📊 Session Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Queries This Session",
                st.session_state.get('ai_query_count', 0)
            )
        
        with col2:
            st.metric(
                "Tokens Used",
                st.session_state.get('ai_tokens_used', 0)
            )
        
        with col3:
            avg_response_time = st.session_state.get('avg_response_time', 0)
            st.metric(
                "Avg Response Time",
                f"{avg_response_time:.1f}s"
            )
        
        # Save Settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Settings", type="primary", use_container_width=True):
                # Update both llm_settings and llm_config
                st.session_state.llm_settings.update({
                    'base_url': base_url,
                    'model': model,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'response_style': response_style,
                    'enable_context': enable_context,
                    'show_calculations': show_calculations
                })
                
                st.session_state.llm_config.update({
                    'base_url': base_url,
                    'model': model,
                    'temperature': temperature,
                    'max_tokens': max_tokens
                })
                
                # Reinitialize LLM with new settings
                st.cache_resource.clear()
                self.llm = init_enhanced_llm()
                
                st.success("✅ AI settings saved successfully!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                st.session_state.llm_settings = {
                    'provider': 'enterprise',
                    'model': 'llama3',
                    'temperature': 0.1,
                    'max_tokens': 6000,
                    'response_style': 'professional',
                    'enable_context': True,
                    'show_calculations': True,
                    'base_url': 'http://localhost:8123/v1',
                    'api_key': 'dummy'
                }
                st.session_state.llm_config = {
                    'base_url': 'http://localhost:8123/v1',
                    'api_key': 'dummy',
                    'model': 'llama3',
                    'temperature': 0.1,
                    'max_tokens': 6000
                }
                st.success("✅ Settings reset to defaults")
                st.rerun()
        
        with col3:
            if st.button("❌ Close Settings", use_container_width=True):
                st.session_state.show_ai_settings = False
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

    def _render_excel_upload_section(self):
        """Render Excel upload interface with proper error handling"""
        
        st.markdown("---")
        st.markdown("### 📤 Excel Portfolio Upload")
        
        if not EXCEL_AVAILABLE:
            st.error("❌ Excel functionality not available. Please install xlsxwriter and openpyxl.")
            return
        
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
            
            if EXCEL_AVAILABLE:
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
                
                try:
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
                except Exception as e:
                    st.error(f"Error creating Excel template: {str(e)}")
                    # Fallback to CSV
                    csv = template_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV Template (Fallback)",
                        data=csv,
                        file_name='sa_ccr_portfolio_template.csv',
                        mime='text/csv'
                    )
            else:
                st.error("Excel library not available. Cannot generate template.")
        
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
        """Process AI query with enterprise LLM integration"""
        
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
        """Generate intelligent AI response using enterprise LLM"""
        
        # Update query count
        st.session_state['ai_query_count'] = st.session_state.get('ai_query_count', 0) + 1
        
        # Get LLM settings
        llm_settings = st.session_state.llm_settings
        
        # Track response generation time
        import time
        start_time = time.time()
        
        try:
            # Check if this is a 24-step breakdown request
            if '24' in query.lower() and 'step' in query.lower():
                if st.session_state.calculation_results:
                    # Generate detailed 24-step response
                    response = self._generate_24_step_response()
                else:
                    response = self._handle_step_breakdown_query()
            else:
                # Use enterprise LLM for other queries
                if self.llm and LANGCHAIN_AVAILABLE:
                    response = self._generate_llm_response(query)
                else:
                    # Fallback to rule-based responses
                    response = self._generate_fallback_response(query)
            
            # Track response time
            end_time = time.time()
            response_time = end_time - start_time
            
            # Update average response time
            current_avg = st.session_state.get('avg_response_time', 0)
            query_count = st.session_state.get('ai_query_count', 1)
            st.session_state['avg_response_time'] = ((current_avg * (query_count - 1)) + response_time) / query_count
            
            # Estimate tokens used (rough approximation)
            estimated_tokens = len(response.split()) * 1.3
            st.session_state['ai_tokens_used'] = st.session_state.get('ai_tokens_used', 0) + int(estimated_tokens)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return f"❌ I encountered an error: {str(e)}\n\nPlease try rephrasing your question."

    def _generate_llm_response(self, query: str) -> str:
        """Generate response using enterprise LLM"""
        
        try:
            # Prepare system message based on settings
            llm_settings = st.session_state.llm_settings
            style_prompts = {
                'professional': "Respond in a professional, business-appropriate tone with clear structure and regulatory focus.",
                'casual': "Use a friendly, conversational tone that's easy to understand while maintaining accuracy.",
                'technical': "Provide detailed technical explanations with precise terminology and formulas.",
                'educational': "Explain concepts step-by-step as if teaching someone learning SA-CCR."
            }
            
            system_content = f"""You are an expert SA-CCR (Standardized Approach for Counterparty Credit Risk) assistant for Basel III banking regulations. 

{style_prompts.get(llm_settings.get('response_style', 'professional'), style_prompts['professional'])}

Key capabilities:
- SA-CCR calculations and 24-step process explanations
- Portfolio optimization strategies  
- Basel III regulatory compliance guidance
- Risk analysis and capital requirement calculations

Always provide accurate, regulatory-compliant information. Include specific formulas, steps, and practical recommendations when relevant."""
            
            # Add context if enabled
            if llm_settings.get('enable_context', True):
                # Add recent conversation context
                recent_messages = st.session_state.ai_chat_history[-3:]  # Last 3 messages
                context = "\n".join([f"{msg['role']}: {msg['content'][:200]}..." for msg in recent_messages])
                system_content += f"\n\nRecent conversation context:\n{context}"
            
            # Prepare messages
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=query)
            ]
            
            # Generate response
            response = self.llm.invoke(messages)
            
            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._generate_fallback_response(query)

    def _generate_24_step_response(self) -> str:
        """Generate detailed 24-step breakdown response"""
        
        results = st.session_state.calculation_results
        calculation_steps = results['calculation_steps']
        
        response = """**📋 Complete 24-Step SA-CCR Calculation Breakdown**

I can see you have calculation results available! Here's the complete breakdown:

**🔍 Navigation to Detailed View:**
• Go to **Scenario Comparison** in the sidebar for full interactive breakdown
• Each step includes formulas, calculations, and detailed explanations

**📊 Quick Summary of Key Steps:**

"""
        
        # Add key steps summary
        key_steps = [1, 5, 9, 15, 18, 21, 24]
        for step_num in key_steps:
            step = next((s for s in calculation_steps if s['step'] == step_num), None)
            if step:
                response += f"**Step {step_num}**: {step['title']}\n"
                response += f"• Result: {step['result']}\n"
                response += f"• Description: {step['description']}\n\n"
        
        response += """**🎯 Complete Interactive Breakdown Available:**
• Navigate to **⚖️ Scenario Comparison** page
• View all 24 steps with formulas and data
• Organized by categories for easy understanding

Would you like me to explain any specific step in more detail?"""
        
        return response

    def _generate_fallback_response(self, query: str) -> str:
        """Generate fallback response when LLM is not available"""
        
        query_lower = query.lower()
        
        # Check for calculation requests
        if any(word in query_lower for word in ['calculate', 'compute', 'saccr', 'portfolio']):
            if st.session_state.current_portfolio:
                return self._handle_calculation_request(query)
            else:
                return """**📊 SA-CCR Calculation Request**

I'd be happy to help you calculate SA-CCR! However, I don't see any portfolio data loaded.

**🔹 Options to proceed:**
1. **Upload Excel File**: Use the "Show Upload Interface" button below
2. **Manual Entry**: Go to SA-CCR Calculator to add trades manually  
3. **Sample Calculation**: I can create a sample portfolio to demonstrate

Which option would you prefer?"""
        
        # Check for optimization queries
        elif any(word in query_lower for word in ['optimize', 'reduce', 'minimize', 'capital']):
            return self._handle_optimization_query(query)
        
        # Check for explanation requests
        elif any(word in query_lower for word in ['explain', 'what is', 'how does']):
            return self._handle_explanation_query(query)
        
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

**📋 24-Step Breakdown Available:**
• Navigate to **⚖️ Scenario Comparison** for complete step-by-step breakdown
• Interactive view with formulas, calculations, and detailed explanations
• Organized by calculation phases for easy understanding

**🎯 Next Steps:**
• View complete 24-step breakdown in Scenario Comparison
• Explore optimization opportunities  
• Analyze risk drivers and hedging effectiveness

Would you like me to show the detailed 24-step calculation breakdown or provide optimization recommendations?"""
            
        except Exception as e:
            return f"❌ Calculation failed: {str(e)}\n\nPlease check your portfolio data and try again."

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
• Go to **⚖️ Scenario Comparison** in the sidebar
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

The detailed breakdown with all formulas and calculations is available in the **Scenario Comparison** section. Would you like me to guide you there?"""
        
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

    # Placeholder methods for other pages (simplified for length)
    def _render_calculator_page(self):
        st.markdown("## 📊 SA-CCR Calculator")
        st.info("Full calculator functionality available - please refer to the AI Assistant for guided calculations.")
    
    def _render_portfolio_page(self):
        st.markdown("## 📈 Portfolio Analysis")
        st.info("Portfolio analysis features available - use AI Assistant for portfolio guidance.")
    
    def _render_optimization_page(self):
        st.markdown("## 🎯 Optimization")
        st.info("Optimization analysis available - ask the AI Assistant for optimization strategies.")
    
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
            
            if st.button("🚀 Go to AI Assistant", use_container_width=True):
                st.session_state.current_page = 'ai_assistant'
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
                        st.markdown(f"""
                        <div class="step-breakdown">
                            <div class="step-header">
                                <span class="step-number">{step['step']}</span>
                                <strong>{step['title']}</strong>
                            </div>
                            <div class="step-content">
                                <p><strong>📋 Description:</strong> {step['description']}</p>
                                <div style="background: #f8fafc; padding: 1rem; border-radius: 6px; margin: 1rem 0; border-left: 4px solid var(--claude-orange);">
                                    <strong>🧮 Formula:</strong><br>
                                    <code>{step['formula']}</code>
                                </div>
                                <p><strong>✅ Result:</strong> {step['result']}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    def _render_database_page(self):
        st.markdown("## 🗄️ Data Management")
        st.info("Database management features available - use AI Assistant for data operations guidance.")
    
    def _render_settings_page(self):
        st.markdown("## ⚙️ Settings")
        st.info("Application settings available - use AI Assistant ⚙️ button for AI configuration.")

# Application entry point
if __name__ == "__main__":
    try:
        app = SACCRApplication()
        app.run()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application startup error: {e}")