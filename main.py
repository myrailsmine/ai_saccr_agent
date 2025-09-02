import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

# Configure logging to see any errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI SA-CCR Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def test_imports():
    """Test all imports and report which ones fail"""
    import_results = {}
    
    # Test core imports
    try:
        from src.engine.saccr_engine import SACCREngine
        import_results['SACCREngine'] = "✅ Success"
    except Exception as e:
        import_results['SACCREngine'] = f"❌ Error: {str(e)}"
    
    try:
        from src.data.database_manager import DatabaseManager
        import_results['DatabaseManager'] = "✅ Success"
    except Exception as e:
        import_results['DatabaseManager'] = f"❌ Error: {str(e)}"
    
    try:
        from src.config.config_manager import ConfigManager
        import_results['ConfigManager'] = "✅ Success"
    except Exception as e:
        import_results['ConfigManager'] = f"❌ Error: {str(e)}"
    
    try:
        from src.ui.components import UIComponents
        import_results['UIComponents'] = "✅ Success"
    except Exception as e:
        import_results['UIComponents'] = f"❌ Error: {str(e)}"
    
    try:
        from src.models.trade_models import Trade, NettingSet, Collateral
        import_results['Trade Models'] = "✅ Success"
    except Exception as e:
        import_results['Trade Models'] = f"❌ Error: {str(e)}"
    
    try:
        from src.utils.validators import TradeValidator
        import_results['TradeValidator'] = "✅ Success"
    except Exception as e:
        import_results['TradeValidator'] = f"❌ Error: {str(e)}"
    
    try:
        from src.utils.progress_tracker import ProgressTracker
        import_results['ProgressTracker'] = "✅ Success"
    except Exception as e:
        import_results['ProgressTracker'] = f"❌ Error: {str(e)}"
    
    return import_results

def main():
    """Simplified main function for debugging"""
    
    # Header
    st.title("🚀 SA-CCR Platform - Debug Mode")
    st.markdown("Debugging import and initialization issues...")
    
    # Test imports
    st.header("Import Test Results")
    
    with st.spinner("Testing imports..."):
        import_results = test_imports()
    
    # Display results
    for module, result in import_results.items():
        if "Success" in result:
            st.success(f"{module}: {result}")
        else:
            st.error(f"{module}: {result}")
    
    # Check if all imports successful
    all_successful = all("Success" in result for result in import_results.values())
    
    if all_successful:
        st.success("🎉 All imports successful! Loading full application...")
        
        # Try to load the full application
        try:
            load_full_application()
        except Exception as e:
            st.error(f"Error loading full application: {str(e)}")
            st.exception(e)
    else:
        st.error("❌ Some imports failed. Please fix the import errors first.")
        
        # Show directory structure
        st.header("Directory Structure Check")
        check_directory_structure()

def check_directory_structure():
    """Check if required directories and files exist"""
    
    required_paths = [
        "src/__init__.py",
        "src/engine/__init__.py",
        "src/engine/saccr_engine.py",
        "src/models/__init__.py", 
        "src/models/trade_models.py",
        "src/data/__init__.py",
        "src/data/database_manager.py",
        "src/config/__init__.py",
        "src/config/config_manager.py",
        "src/utils/__init__.py",
        "src/utils/validators.py",
        "src/utils/progress_tracker.py",
        "src/ui/__init__.py",
        "src/ui/components.py"
    ]
    
    st.subheader("Required Files Check:")
    
    for path in required_paths:
        if Path(path).exists():
            st.success(f"✅ {path}")
        else:
            st.error(f"❌ Missing: {path}")

def load_full_application():
    """Load the full SA-CCR application"""
    
    # Import all modules
    from src.engine.saccr_engine import SACCREngine
    from src.data.database_manager import DatabaseManager
    from src.config.config_manager import ConfigManager
    from src.ui.components import UIComponents
    from src.models.trade_models import Trade, NettingSet, Collateral
    from src.utils.validators import TradeValidator
    from src.utils.progress_tracker import ProgressTracker
    
    st.success("✅ All modules imported successfully!")
    
    # Try to initialize components
    try:
        config_manager = ConfigManager()
        st.success("✅ ConfigManager initialized")
    except Exception as e:
        st.error(f"❌ ConfigManager error: {e}")
        return
    
    try:
        db_manager = DatabaseManager()
        st.success("✅ DatabaseManager initialized")
    except Exception as e:
        st.error(f"❌ DatabaseManager error: {e}")
        return
    
    try:
        saccr_engine = SACCREngine(config_manager)
        st.success("✅ SACCREngine initialized")
    except Exception as e:
        st.error(f"❌ SACCREngine error: {e}")
        return
    
    st.success("🎉 All components initialized successfully!")
    
    # Show basic interface
    st.header("Basic SA-CCR Interface")
    
    # Simple trade input form
    with st.form("simple_trade_form"):
        st.subheader("Add Test Trade")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trade_id = st.text_input("Trade ID", value="TEST_001")
            counterparty = st.text_input("Counterparty", value="Test Bank")
        
        with col2:
            notional = st.number_input("Notional", value=100000000.0)
            currency = st.selectbox("Currency", ["USD", "EUR", "GBP"])
        
        with col3:
            maturity_years = st.number_input("Maturity (Years)", value=5.0)
        
        submitted = st.form_submit_button("Create Test Trade")
        
        if submitted:
            try:
                # Create a test trade
                test_trade = Trade(
                    trade_id=trade_id,
                    counterparty=counterparty,
                    asset_class="Interest Rate",
                    trade_type="Swap", 
                    notional=notional,
                    currency=currency,
                    underlying="Interest Rate Swap",
                    maturity_date=datetime.now() + timedelta(days=int(maturity_years * 365))
                )
                
                st.success(f"✅ Created test trade: {test_trade.trade_id}")
                st.json(test_trade.to_dict())
                
            except Exception as e:
                st.error(f"❌ Error creating trade: {e}")
                st.exception(e)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Fatal error: {str(e)}")
        st.exception(e)
        
        # Show basic information for debugging
        st.header("Debug Information")
        st.write(f"Current working directory: {Path.cwd()}")
        st.write(f"Python path: {list(Path.cwd().glob('*'))}")
