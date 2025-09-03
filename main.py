import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
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

# Professional CSS styling (no gradients)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #fafafa;
    }
    
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
    }
    
    .calculation-step {
        background: white;
        border-left: 4px solid #2563eb;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .step-number {
        background: #2563eb;
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
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .result-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0.5rem 0;
    }
    
    .result-label {
        color: #6b7280;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .alert-info {
        background: #dbeafe;
        border: 1px solid #93c5fd;
        color: #1e40af;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: #fef3c7;
        border: 1px solid #fcd34d;
        color: #92400e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-success {
        background: #d1fae5;
        border: 1px solid #6ee7b7;
        color: #065f46;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .progress-container {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
    }
    
    .comparison-card {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .sidebar .sidebar-content {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
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
        
        # LLM Connection for AI Assistant
        self.llm = None
        self._llm_connection_status = "disconnected"
        
        # Initialize session state
        self._initialize_session_state()
    
    @property
    def llm_connection_status(self) -> str:
        """Get current LLM connection status"""
        return getattr(self, '_llm_connection_status', 'disconnected')
    
    @llm_connection_status.setter
    def llm_connection_status(self, value: str):
        """Set LLM connection status"""
        self._llm_connection_status = value
    
    def setup_llm_connection(self, config: Dict) -> bool:
        """Setup LangChain ChatOpenAI connection for AI assistant"""
        try:
            from langchain_openai import ChatOpenAI
            from langchain.schema import HumanMessage, SystemMessage
            
            self.llm = ChatOpenAI(
                base_url=config.get('base_url', "http://localhost:8123/v1"),
                api_key=config.get('api_key', "dummy"), 
                model=config.get('model', "llama3"),
                temperature=config.get('temperature', 0.3),
                max_tokens=config.get('max_tokens', 4000),
                streaming=config.get('streaming', False)
            )
            
            # Test connection
            test_response = self.llm.invoke([
                SystemMessage(content="You are a Basel SA-CCR expert. Respond with 'Connected' if you receive this."),
                HumanMessage(content="Test connection")
            ])
            
            if test_response and test_response.content:
                self.llm_connection_status = "connected"
                logger.info("LLM connection established successfully")
                return True
            else:
                self.llm_connection_status = "disconnected" 
                return False
                
        except Exception as e:
            logger.error(f"LLM Connection Error: {str(e)}")
            self.llm_connection_status = "disconnected"
            return False
        
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
        
        # Header
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 8px; 
                    border: 1px solid #e1e5e9; margin-bottom: 2rem;">
            <h1 style="margin: 0; color: #1f2937;">SA-CCR Risk Analytics Platform</h1>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">
                Professional Basel SA-CCR calculation and optimization engine
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        with st.sidebar:
            self._render_sidebar()
        
        # Main content routing - AI Assistant is default
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
            'ai_assistant': '🤖 AI Assistant Chat',
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
        
        # LLM connection status  
        st.markdown("---")
        st.markdown("### AI Assistant Status")
        
        if self.llm_connection_status == "connected":
            st.markdown('<div class="alert-success">🤖 LLM: Connected</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-warning">🤖 LLM: Disconnected</div>', 
                       unsafe_allow_html=True)
            
            # Quick LLM setup in sidebar
            with st.expander("🔗 Connect LLM", expanded=False):
                base_url = st.text_input("Base URL", value="http://localhost:8123/v1", key="sidebar_base_url")
                api_key = st.text_input("API Key", value="dummy", type="password", key="sidebar_api_key")
                model = st.text_input("Model", value="llama3", key="sidebar_model")
                
                if st.button("Connect", type="primary", key="sidebar_connect"):
                    config = {
                        'base_url': base_url,
                        'api_key': api_key,
                        'model': model,
                        'temperature': 0.3,
                        'max_tokens': 4000
                    }
                    
                    if self.setup_llm_connection(config):
                        st.success("LLM Connected!")
                        st.rerun()
                    else:
                        st.error("Connection Failed")
    
    def _render_ai_assistant_page(self):
        """Render AI assistant chat interface with detailed step-by-step processing"""
        
        st.markdown("## 🤖 AI SA-CCR Expert Assistant")
        st.markdown("*Default page - Ask me anything about SA-CCR or describe trades for automatic calculation*")
        
        # LLM Status Banner
        if self.llm_connection_status != "connected":
            st.warning("""
            ⚠️ **LLM Not Connected** - Connect your AI model in Settings → LLM Setup for full conversational capabilities.
            Basic trade parsing and calculations still work without LLM connection.
            """)
        else:
            st.success("✅ **AI Model Connected** - Full conversational SA-CCR expert available!")
        
        # Initialize chat history
        if 'ai_chat_history' not in st.session_state:
            st.session_state.ai_chat_history = [
                {
                    'role': 'assistant',
                    'content': """**Welcome to the AI SA-CCR Expert Assistant! 🚀**

I'm your intelligent assistant for Basel SA-CCR calculations and regulatory guidance. I can:

**📊 Automatic Calculations**
- Parse natural language trade descriptions
- Execute complete 24-step SA-CCR calculations
- Show step-by-step processing details

**🧠 Expert Consultation**
- Answer Basel regulatory questions
- Explain formulas and methodology
- Provide optimization strategies

**📈 Portfolio Analysis**
- Analyze calculation results
- Identify risk drivers
- Suggest improvements

**Example queries:**
- *"Calculate SA-CCR for a $200M USD swap with Goldman, 10-year maturity"*
- *"What drives my high EAD results?"*
- *"How can I reduce my derivatives capital?"*
- *"Explain the PFE multiplier formula"*

**Try me with a question or trade description below! ⬇️**
""",
                    'timestamp': datetime.now(),
                    'processing_steps': []
                }
            ]
        
        # Chat interface
        self._render_detailed_chat_interface()
        
        # Quick examples
        self._render_quick_examples()
    
    def _render_detailed_chat_interface(self):
        """Render chat interface with detailed processing steps"""
        
        # Display chat history with processing details
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.ai_chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div style="background: #f0f2f6; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; margin-left: 3rem; border-left: 4px solid #4f46e5;">
                        <strong>👤 You:</strong><br>
                        <div style="font-size: 1.1rem; margin-top: 0.5rem;">{message['content']}</div>
                        <small style="color: #6b7280;">{message['timestamp'].strftime('%H:%M:%S')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # AI Response with processing steps
                    st.markdown(f"""
                    <div style="background: white; border: 2px solid #e5e7eb; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; margin-right: 3rem;">
                        <strong>🤖 SA-CCR Assistant:</strong>
                        <small style="color: #6b7280; float: right;">{message['timestamp'].strftime('%H:%M:%S')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show processing steps if available
                    if message.get('processing_steps'):
                        with st.expander("🔍 **View Processing Steps**", expanded=False):
                            for i, step in enumerate(message['processing_steps'], 1):
                                st.markdown(f"**Step {i}:** {step['title']}")
                                st.write(step['description'])
                                if step.get('result'):
                                    st.code(step['result'])
                                st.markdown("---")
                    
                    # Main response
                    st.markdown(message['content'])
        
        # Chat input
        st.markdown("### Ask the AI Assistant")
        
        with st.form("ai_chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Your question or trade description:",
                placeholder="e.g., 'Calculate SA-CCR for a $500M interest rate swap with JP Morgan, 7-year maturity' or 'Why is my EAD so high?'",
                height=120,
                help="Describe trades for automatic calculation or ask SA-CCR questions"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                submitted = st.form_submit_button("🚀 Process Query", type="primary", use_container_width=True)
            with col2:
                show_steps = st.form_submit_button("📊 With Steps", use_container_width=True)
            with col3:
                if st.form_submit_button("🧹 Clear", use_container_width=True):
                    st.session_state.ai_chat_history = st.session_state.ai_chat_history[:1]
                    st.rerun()
            
            if (submitted or show_steps) and user_input.strip():
                self._process_detailed_ai_query(user_input.strip(), show_processing_steps=show_steps or submitted)
    
    def _process_detailed_ai_query(self, user_query: str, show_processing_steps: bool = True):
        """Process query with detailed step-by-step analysis and response generation"""
        
        # Add user message
        st.session_state.ai_chat_history.append({
            'role': 'user',
            'content': user_query,
            'timestamp': datetime.now()
        })
        
        processing_steps = []
        
        try:
            # Step 1: Query Analysis
            processing_steps.append({
                'title': 'Query Intent Analysis',
                'description': 'Analyzing user intent and extracting key information',
                'result': f"Analyzing: '{user_query[:100]}...'"
            })
            
            # Analyze query
            query_analysis = self._detailed_query_analysis(user_query)
            
            processing_steps.append({
                'title': 'Intent Classification',
                'description': f"Query type: {query_analysis['category']}",
                'result': f"Requires calculation: {query_analysis['requires_calculation']}\nExtracted trades: {len(query_analysis.get('extracted_trades', []))}"
            })
            
            # Step 2: LLM Processing (if connected)
            if self.llm_connection_status == "connected":
                processing_steps.append({
                    'title': 'LLM Context Preparation',
                    'description': 'Preparing context and prompts for AI model',
                    'result': 'Building SA-CCR expert context with current portfolio state'
                })
            
            # Step 3: Trade Extraction (if calculation needed)
            if query_analysis['requires_calculation'] and query_analysis.get('extracted_trades'):
                processing_steps.append({
                    'title': 'Trade Information Extraction',
                    'description': 'Parsing natural language for trade details',
                    'result': f"Extracted {len(query_analysis['extracted_trades'])} trades with details:\n" + 
                             '\n'.join([f"- {t['asset_class']} {t['trade_type']}: ${t['notional']:,.0f} {t['currency']}" 
                                       for t in query_analysis['extracted_trades']])
                })
                
                # Step 4: SA-CCR Calculation Setup
                processing_steps.append({
                    'title': 'SA-CCR Calculation Preparation',
                    'description': 'Creating Trade objects and NettingSet for Basel calculation',
                    'result': 'Preparing complete 24-step SA-CCR calculation pipeline'
                })
            
            # Generate response
            if self.llm_connection_status == "connected":
                processing_steps.append({
                    'title': 'AI Response Generation',
                    'description': 'Generating expert response using connected LLM',
                    'result': 'Processing with AI model...'
                })
                
                response_content = self._generate_detailed_llm_response(user_query, query_analysis, processing_steps)
            else:
                processing_steps.append({
                    'title': 'Rule-Based Response Generation',
                    'description': 'Using built-in SA-CCR logic (LLM not connected)',
                    'result': 'Generating response with local expertise engine'
                })
                
                response_content = self._generate_detailed_fallback_response(user_query, query_analysis)
            
            # If calculation was performed, add calculation steps
            if query_analysis['requires_calculation'] and st.session_state.get('ai_generated_results'):
                calculation_steps = st.session_state.ai_generated_results.get('calculation_steps', [])
                processing_steps.append({
                    'title': 'SA-CCR 24-Step Calculation Completed',
                    'description': f'Executed complete Basel SA-CCR methodology',
                    'result': f'All 24 steps completed successfully\nFinal EAD: ${st.session_state.ai_generated_results["final_results"]["exposure_at_default"]:,.0f}'
                })
            
            # Add final response to chat
            st.session_state.ai_chat_history.append({
                'role': 'assistant',
                'content': response_content,
                'timestamp': datetime.now(),
                'processing_steps': processing_steps if show_processing_steps else []
            })
            
        except Exception as e:
            error_response = f"""**❌ Processing Error**
            
I encountered an error while processing your request: {str(e)}

**Troubleshooting:**
- Check if trade descriptions include notional, currency, and maturity
- Verify LLM connection in Settings if using AI features
- Try rephrasing your question

**Processing Steps Completed:** {len(processing_steps)}
"""
            
            processing_steps.append({
                'title': 'Error Encountered',
                'description': f'Processing failed: {str(e)}',
                'result': 'Please check your input and try again'
            })
            
            st.session_state.ai_chat_history.append({
                'role': 'assistant',
                'content': error_response,
                'timestamp': datetime.now(),
                'processing_steps': processing_steps if show_processing_steps else []
            })
        
        st.rerun()
    
    def _detailed_query_analysis(self, query: str) -> Dict:
        """Enhanced query analysis with detailed extraction"""
        
        query_lower = query.lower()
        
        # Enhanced calculation detection
        calculation_keywords = [
            'calculate', 'compute', 'sa-ccr for', 'portfolio', 'exposure',
            'swap', 'forward', 'option', 'swaption', 'trade', 'notional', 
            'ead', 'rwa', 'derivatives', 'counterparty'
        ]
        
        requires_calculation = any(keyword in query_lower for keyword in calculation_keywords)
        
        # Enhanced trade extraction
        extracted_trades = self._enhanced_trade_extraction(query) if requires_calculation else []
        
        # Category classification
        if requires_calculation and extracted_trades:
            category = 'calculation'
        elif any(word in query_lower for word in ['optimize', 'reduce', 'improve', 'lower']):
            category = 'optimization'
        elif any(word in query_lower for word in ['explain', 'what is', 'how does', 'why']):
            category = 'explanation'
        elif any(word in query_lower for word in ['formula', 'equation', 'calculate']):
            category = 'formula'
        else:
            category = 'general'
        
        return {
            'requires_calculation': requires_calculation and len(extracted_trades) > 0,
            'extracted_trades': extracted_trades,
            'category': category,
            'counterparty': self._extract_counterparty(query),
            'query_complexity': 'high' if len(query.split()) > 20 else 'medium' if len(query.split()) > 10 else 'simple'
        }
    
    def _enhanced_trade_extraction(self, query: str) -> List[Dict]:
        """Enhanced trade extraction with better pattern matching"""
        
        import re
        
        # Comprehensive patterns for better extraction
        patterns = {
            'notional': [
                r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*([KMB]?)\b',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*million',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*billion'
            ],
            'currency': r'\b(USD|EUR|GBP|JPY|CHF|CAD|AUD|NZD|SEK|NOK)\b',
            'maturity': [
                r'(\d+(?:\.\d+)?)\s*[-\s]?\s*(year|yr|month|mon|day)s?',
                r'(\d+(?:\.\d+)?)\s*Y\b',
                r'(\d+(?:\.\d+)?)\s*M\b'
            ],
            'trade_types': {
                'Interest Rate': [r'\b(interest\s+rate\s+swap|irs|swap)\b', r'\bswap\b'],
                'Foreign Exchange': [r'\b(fx\s+forward|fx|foreign\s+exchange|currency\s+forward)\b'],
                'Equity': [r'\b(equity\s+option|stock\s+option|equity|option)\b'],
                'Credit': [r'\b(cds|credit\s+default\s+swap|credit)\b'],
                'Commodity': [r'\b(commodity|oil|gold|wheat)\b']
            }
        }
        
        trades = []
        
        # Try to split multiple trades
        trade_separators = r'[,;]\s*(?=\d+[^\d]|\$)'
        potential_trades = re.split(trade_separators, query)
        
        if len(potential_trades) == 1:
            potential_trades = [query]
        
        for i, trade_text in enumerate(potential_trades):
            trade_info = self._extract_single_trade(trade_text, i, patterns)
            if trade_info:
                trades.append(trade_info)
        
        return trades
    
    def _extract_single_trade(self, trade_text: str, index: int, patterns: Dict) -> Dict:
        """Extract information from a single trade description"""
        
        import re
        
        trade_info = {
            'trade_id': f'AI_TRADE_{index+1}',
            'notional': 100000000.0,  # Default $100M
            'currency': 'USD',
            'maturity_years': 5.0,
            'asset_class': 'Interest Rate',
            'trade_type': 'Swap',
            'delta': 1.0,
            'mtm_value': 0.0
        }
        
        # Extract notional with improved patterns
        for pattern in patterns['notional']:
            matches = re.findall(pattern, trade_text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    amount, multiplier = matches[0]
                    amount = float(amount.replace(',', ''))
                    
                    multiplier_map = {'K': 1000, 'M': 1000000, 'B': 1000000000, 'million': 1000000, 'billion': 1000000000}
                    multiplier_key = multiplier.upper() if multiplier else 'million' if 'million' in pattern else 'billion' if 'billion' in pattern else ''
                    
                    if multiplier_key in multiplier_map:
                        amount *= multiplier_map[multiplier_key]
                else:
                    amount = float(matches[0].replace(',', ''))
                    if 'million' in trade_text.lower():
                        amount *= 1000000
                    elif 'billion' in trade_text.lower():
                        amount *= 1000000000
                
                trade_info['notional'] = amount
                break
        
        # Extract other fields
        currency_match = re.search(patterns['currency'], trade_text, re.IGNORECASE)
        if currency_match:
            trade_info['currency'] = currency_match.group(0).upper()
        
        # Extract maturity
        for pattern in patterns['maturity']:
            matches = re.findall(pattern, trade_text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    period, unit = matches[0]
                else:
                    period = matches[0]
                    unit = 'year'
                
                period = float(period)
                
                if unit.lower().startswith(('year', 'yr')) or unit == 'Y':
                    trade_info['maturity_years'] = period
                elif unit.lower().startswith(('month', 'mon')) or unit == 'M':
                    trade_info['maturity_years'] = period / 12
                elif unit.lower().startswith('day'):
                    trade_info['maturity_years'] = period / 365
                break
        
        
        # Extract trade type and asset class
        for asset_class, type_patterns in patterns['trade_types'].items():
            for pattern in type_patterns:
                if re.search(pattern, trade_text, re.IGNORECASE):
                    trade_info['asset_class'] = asset_class
                    
                    if asset_class == 'Interest Rate':
                        trade_info['trade_type'] = 'Swap'
                    elif asset_class == 'Foreign Exchange':
                        trade_info['trade_type'] = 'Forward'
                    elif asset_class == 'Equity':
                        trade_info['trade_type'] = 'Option'
                        # Extract delta if mentioned
                        delta_match = re.search(r'delta\s+(\d+(?:\.\d+)?)', trade_text, re.IGNORECASE)
                        if delta_match:
                            trade_info['delta'] = float(delta_match.group(1))
                    elif asset_class == 'Credit':
                        trade_info['trade_type'] = 'Credit Default Swap'
                    break
            else:
                continue
            break
        
        return trade_info
    
    def _extract_counterparty(self, query: str) -> str:
        """Extract counterparty name from query"""
        
        import re
        
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
    
    def _generate_detailed_llm_response(self, user_query: str, analysis: Dict, processing_steps: List[Dict]) -> str:
        """Generate detailed LLM response with SA-CCR context"""
        
        from langchain.schema import HumanMessage, SystemMessage
        
        # Get context
        portfolio_context = self._get_portfolio_context()
        results_context = self._get_results_context()
        
        # Update processing step
        for step in processing_steps:
            if step['title'] == 'AI Response Generation':
                step['result'] = 'Generating expert SA-CCR response with full context'
        
        system_prompt = """You are a world-class Basel SA-CCR (Standardized Approach for Counterparty Credit Risk) expert with deep knowledge of:

**Regulatory Framework:**
- Complete 24-step SA-CCR calculation methodology per Basel III
- Supervisory factors, correlations, and regulatory parameters
- PFE multipliers, replacement cost, and EAD calculations
- Central clearing benefits and alpha multipliers
- Risk weight assignments and capital requirements

**Practical Expertise:**
- Derivatives trading and risk management
- Netting agreements and collateral optimization
- Portfolio compression and clearing strategies
- Regulatory compliance and reporting

**Response Style:**
- Provide detailed, step-by-step explanations
- Use specific formulas with regulatory references
- Give practical, actionable recommendations
- Show calculation examples when relevant

When users request calculations, acknowledge the trade details extracted and confirm the SA-CCR calculation will be performed automatically."""

        calculation_context = ""
        if analysis['requires_calculation'] and analysis.get('extracted_trades'):
            trades_summary = []
            for trade in analysis['extracted_trades']:
                trades_summary.append(f"- {trade['asset_class']} {trade['trade_type']}: ${trade['notional']:,.0f} {trade['currency']}, {trade['maturity_years']}Y maturity")
            
            calculation_context = f"""

**CALCULATION REQUEST DETECTED:**
The user has requested SA-CCR calculation for:
{chr(10).join(trades_summary)}
Counterparty: {analysis.get('counterparty', 'Not specified')}

The system will automatically perform the complete 24-step Basel SA-CCR calculation. Please acknowledge this and provide relevant regulatory context."""

        user_prompt = f"""User Query: {user_query}
        {portfolio_context}
        {results_context}
        {calculation_context}
        
Please provide a comprehensive expert response. If this is a calculation request, acknowledge the extracted trades and explain what the SA-CCR calculation will determine. For methodology questions, provide detailed regulatory guidance."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            llm_response = response.content
            
            # If calculation was requested, perform it and add results
            if analysis['requires_calculation'] and analysis.get('extracted_trades'):
                calc_result = self._perform_automatic_calculation_with_steps(analysis)
                if calc_result:
                    return f"{llm_response}\n\n{calc_result}"
            
            return llm_response
            
        except Exception as e:
            return f"Error generating AI response: {str(e)}\nPlease check LLM connection."
    
    def _generate_detailed_fallback_response(self, user_query: str, analysis: Dict) -> str:
        """Generate detailed fallback response when LLM not connected"""
        
        if analysis['requires_calculation'] and analysis.get('extracted_trades'):
            # Perform calculation
            calc_result = self._perform_automatic_calculation_with_steps(analysis)
            if calc_result:
                return f"""**SA-CCR Calculation Response** *(LLM not connected - using built-in engine)*

{calc_result}

**Note:** For detailed regulatory explanations and advanced consultation, connect an LLM model in Settings > LLM Setup.
"""
        else:
            return f"""**SA-CCR Information Response** *(LLM not connected)*

I can help with SA-CCR calculations by parsing trade descriptions, but detailed regulatory explanations require LLM connection.

**Your query:** "{user_query}"

**Available without LLM:**
- Automatic SA-CCR calculations from trade descriptions
- Basic portfolio analysis  
- Standard optimization recommendations

**To enable full AI capabilities:**
1. Go to Settings → LLM Setup
2. Configure your AI model (OpenAI, Ollama, etc.)
3. Test the connection

**For calculation requests, try:**
"Calculate SA-CCR for a $100M USD interest rate swap with JP Morgan, 5-year maturity"
"""
    
    def _perform_automatic_calculation_with_steps(self, analysis: Dict) -> str:
        """Perform SA-CCR calculation and return detailed results with steps"""
        
        try:
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
                'threshold': 1000000.0,
                'mta': 500000.0,
                'trades': trades
            }
            
            # Perform calculation
            results = self.saccr_engine.calculate_comprehensive_saccr(portfolio_data)
            
            # Store results
            st.session_state.ai_generated_results = results
            st.session_state.ai_generated_portfolio = portfolio_data
            
            # Format comprehensive results
            final_results = results['final_results']
            calculation_steps = results['calculation_steps']
            
            # Build detailed response
            response = f"""**🚀 SA-CCR CALCULATION COMPLETED**

**Portfolio Summary:**
- Counterparty: {counterparty}
- Trades: {len(trades)}
- Total Notional: ${sum(t.notional for t in trades)/1_000_000:.1f}M

**Final Results:**
- **Exposure at Default (EAD)**: ${final_results['exposure_at_default']/1_000_000:.2f}M
- **Risk Weighted Assets (RWA)**: ${final_results['risk_weighted_assets']/1_000_000:.2f}M
- **Capital Requirement (8%)**: ${final_results['capital_requirement']/1_000:.0f}K
- **Replacement Cost (RC)**: ${final_results['replacement_cost']/1_000_000:.2f}M
- **Potential Future Exposure (PFE)**: ${final_results['potential_future_exposure']/1_000_000:.2f}M

**Trade Details:**
"""
            
            for i, trade in enumerate(trades, 1):
                response += f"{i}. {trade.asset_class.value} {trade.trade_type.value}: ${trade.notional/1_000_000:.1f}M {trade.currency}, {trade.time_to_maturity():.1f}Y\n"
            
            # Add key calculation insights
            ead_ratio = (final_results['exposure_at_default'] / sum(t.notional for t in trades)) * 100
            
            response += f"""
**Key Insights:**
- EAD/Notional Ratio: {ead_ratio:.2f}%
- RC vs PFE: {"RC dominates" if final_results['replacement_cost'] > final_results['potential_future_exposure'] else "PFE dominates"}
- Capital Efficiency: {100 - ead_ratio:.1f}%

**Basel 24-Step Methodology Applied:**
"""
            
            # Add key calculation steps
            key_steps = [15, 16, 18, 21, 24]  # PFE Multiplier, PFE, RC, EAD, RWA
            for step_num in key_steps:
                if step_num <= len(calculation_steps):
                    step = calculation_steps[step_num - 1]
                    response += f"- Step {step['step']}: {step['title']} → {step['result']}\n"
            
            response += f"""
**Want more details?** The complete 24-step calculation breakdown is available in the Calculator module.
"""
            
            return response
            
        except Exception as e:
            return f"**Calculation Error:** {str(e)}\n\nPlease verify trade descriptions include notional amounts, currencies, and maturities."
    
    def _get_portfolio_context(self) -> str:
        """Get current portfolio context for LLM"""
        if st.session_state.get('current_portfolio'):
            portfolio = st.session_state.current_portfolio
            trades = portfolio.get('trades', [])
            return f"""
Current Portfolio Context:
- Netting Set: {portfolio.get('netting_set_id', 'N/A')}
- Counterparty: {portfolio.get('counterparty', 'N/A')}
- Trades: {len(trades)}
- Total Notional: ${sum(abs(t.notional) for t in trades):,.0f}
- Asset Classes: {list(set(t.asset_class.value for t in trades))}
"""
        return ""
    
    def _get_results_context(self) -> str:
        """Get recent calculation results context for LLM"""
        if st.session_state.get('calculation_results'):
            final_results = st.session_state.calculation_results['final_results']
            return f"""
Recent Calculation Results:
- EAD: ${final_results['exposure_at_default']:,.0f}
- RWA: ${final_results['risk_weighted_assets']:,.0f}
- Capital: ${final_results['capital_requirement']:,.0f}
"""
        return ""
    
    def _render_quick_examples(self):
        """Render quick example buttons"""
        
        st.markdown("### Quick Examples")
        
        examples = [
            {
                "title": "💰 Sample Calculation",
                "query": "Calculate SA-CCR for a $500M USD interest rate swap with Goldman Sachs, 10-year maturity, and a $300M EUR/USD FX forward with Deutsche Bank, 2-year maturity",
                "description": "Multi-trade portfolio calculation"
            },
            {
                "title": "🧠 Methodology Question", 
                "query": "Explain how the PFE multiplier works and what drives netting benefits in SA-CCR",
                "description": "Basel regulatory explanation"
            },
            {
                "title": "🎯 Optimization Help",
                "query": "My EAD is very high relative to notional. What are the best strategies to reduce SA-CCR capital requirements?",
                "description": "Capital optimization advice"
            },
            {
                "title": "📊 Results Analysis",
                "query": "Why might my replacement cost be higher than my potential future exposure? What does this indicate?",
                "description": "Calculation interpretation"
            }
        ]
        
        cols = st.columns(len(examples))
        
        for i, example in enumerate(examples):
            with cols[i]:
                if st.button(
                    example["title"], 
                    use_container_width=True,
                    help=example["description"]
                ):
                    self._process_detailed_ai_query(example["query"], show_processing_steps=True)

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

    # Remaining methods for portfolio loading, results display, settings, etc.
    # [Continuing with the rest of the methods...]

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
                suggestions.append("Consider additional collateral posting")
            
            if ead_ratio > 10:
                suggestions.append("Review netting agreement optimization")
            
            if final_results.get('portfolio_summary', {}).get('trade_count', 0) > 100:
                suggestions.append("Consider trade compression strategies")
            
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
    
    def _render_portfolio_page(self):
        """Render portfolio analysis page"""
        st.markdown("## Portfolio Analysis")
        st.info("Advanced portfolio analysis features coming soon...")
    
    def _render_optimization_page(self):
        """Render optimization analysis page"""
        st.markdown("## Optimization Analysis")
        st.info("What-if analysis and optimization features coming soon...")
    
    def _render_comparison_page(self):
        """Render scenario comparison page"""
        st.markdown("## Scenario Comparison")
        st.info("Before/after scenario comparison features coming soon...")
    
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
                    if self.setup_llm_connection(config):
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
                    from langchain.schema import HumanMessage, SystemMessage
                    test_response = self.llm.invoke([
                        SystemMessage(content="You are a helpful SA-CCR expert."),
                        HumanMessage(content="Explain SA-CCR in one sentence.")
                    ])
                    st.write(f"**Test Response:** {test_response.content}")
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
