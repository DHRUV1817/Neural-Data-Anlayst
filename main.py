import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import requests
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from io import StringIO
import uuid
from dotenv import load_dotenv
import sqlite3
import tempfile

# Load environment variables
load_dotenv()

# ===========================
# Configuration and Constants
# ===========================

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

AVAILABLE_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B (Most Advanced)",
    "llama3-70b-8192": "Llama 3 70B (Reliable)",
    "mixtral-8x7b-32768": "Mixtral 8x7B (Fast & Efficient)",
    "gemma2-9b-it": "Gemma 2 9B (Lightweight)",
    "qwen-qwq-32b": "Qwen QwQ 32B (Reasoning)",
    "deepseek-r1-distill-llama-70b": "DeepSeek R1 70B (Specialized)"
}

# Page configuration
st.set_page_config(
    page_title="⚡ Neural Data Analyst Premium",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Custom CSS
# ===========================

def inject_custom_css():
    """Inject custom CSS for styling"""
    st.markdown("""
    <style>
        .main > div {
            padding-top: 2rem;
        }
        
        .stApp {
            background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
        }
        
        .main-header {
            background: linear-gradient(45deg, #ffd700, #ffff00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            font-size: 3rem;
            font-weight: 900;
            margin-bottom: 1rem;
            text-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
        }
        
        .subtitle {
            text-align: center;
            color: #cccccc;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: rgba(15, 15, 15, 0.95);
            border: 1px solid rgba(255, 215, 0, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            backdrop-filter: blur(20px);
        }
        
        .success-msg {
            background: rgba(0, 255, 0, 0.1);
            color: #00ff00;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #00ff00;
        }
        
        .error-msg {
            background: rgba(255, 68, 68, 0.1);
            color: #ff4444;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ff4444;
        }
        
        .neural-button {
            background: linear-gradient(45deg, #ffd700, #ffff00);
            color: #000000;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .neural-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(255, 215, 0, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# Session State Management
# ===========================

def initialize_session_state():
    """Initialize all session state variables"""
    # API configuration
    api_key = get_api_key()
    
    defaults = {
        'api_key': api_key or "",
        'api_connected': bool(api_key),
        'selected_model': DEFAULT_MODEL,
        'uploaded_data': None,
        'data_schema': "",
        'analysis_history': [],
        'session_id': str(uuid.uuid4()),
        'example_query': "",
        'recent_queries': [],
        'show_eda_results': False,
        'show_ai_insights': False,
        'show_advanced_analytics': False,
        'eda_results': None,
        'ai_insights_text': None,
        'show_model_selection': False,
        'current_query': ""
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Test API connection if key exists but not connected
    if api_key and not st.session_state.api_connected:
        test_api_connection_silent(api_key, st.session_state.selected_model)

def get_api_key() -> Optional[str]:
    """Get API key from various sources"""
    # Try Streamlit secrets first (with proper error handling)
    try:
        if hasattr(st, 'secrets'):
            return st.secrets.get('GROQ_API_KEY')
    except Exception:
        # No secrets file exists, which is fine
        pass
    
    # Try environment variable
    if 'GROQ_API_KEY' in os.environ:
        return os.environ['GROQ_API_KEY']
    
    # Try loading from .env file
    load_dotenv(override=True)
    return os.environ.get('GROQ_API_KEY')

# ===========================
# API Functions
# ===========================

def test_api_connection_silent(api_key: str, model: str) -> bool:
    """Test API connection silently"""
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say 'OK' in one word."}],
                "temperature": 0.1,
                "max_tokens": 10
            },
            timeout=10
        )
        success = response.status_code == 200
        if success:
            st.session_state.api_connected = True
        return success
    except Exception:
        return False

def make_api_call(model: str, prompt: str, timeout: int = 30) -> str:
    """Make API call to Groq"""
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {st.session_state.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1000
            },
            timeout=timeout
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"API error: {response.status_code}")
            
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}")

# ===========================
# Data Processing Functions
# ===========================

@st.cache_data
def load_csv_file(uploaded_file) -> pd.DataFrame:
    """Load CSV file with caching"""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        raise Exception(f"Error reading CSV: {str(e)}")

@st.cache_data
def load_json_file(uploaded_file) -> pd.DataFrame:
    """Load JSON file with caching"""
    try:
        return pd.read_json(uploaded_file)
    except Exception as e:
        raise Exception(f"Error reading JSON: {str(e)}")

def create_sample_data() -> pd.DataFrame:
    """Create sample sales data for demonstration"""
    np.random.seed(42)
    n_rows = 1000
    
    data = {
        'customer_id': range(1, n_rows + 1),
        'customer_name': [f"Customer_{i}" for i in range(1, n_rows + 1)],
        'product': np.random.choice(['Widget A', 'Widget B', 'Widget C', 'Gadget X', 'Gadget Y'], n_rows),
        'sales_amount': np.random.normal(2000, 500, n_rows).round(2),
        'order_date': pd.date_range('2023-01-01', periods=n_rows, freq='D'),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
        'sales_rep': np.random.choice(['John Smith', 'Jane Doe', 'Bob Johnson', 'Alice Brown'], n_rows),
        'customer_age': np.random.randint(25, 70, n_rows),
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_rows),
        'discount_percent': np.random.uniform(0, 20, n_rows).round(1)
    }
    
    return pd.DataFrame(data)

def generate_database_schema(df: pd.DataFrame) -> Dict[str, str]:
    """Generate database schema from DataFrame"""
    table_name = "uploaded_data"
    column_definitions = []
    
    for col in df.columns:
        # Clean column name
        clean_col = col.replace(' ', '_').replace('-', '_').replace('.', '_')
        clean_col = ''.join(c for c in clean_col if c.isalnum() or c == '_')
        
        # Determine SQL data type
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            sql_type = "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            sql_type = "DECIMAL(10,2)"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            sql_type = "DATETIME"
        elif pd.api.types.is_bool_dtype(dtype):
            sql_type = "BOOLEAN"
        else:
            max_length = df[col].astype(str).str.len().max() if not df[col].empty else 50
            sql_type = f"VARCHAR({max(50, int(max_length))})" if max_length <= 50 else "TEXT"
        
        column_definitions.append(f"    {clean_col} {sql_type}")
    
    sql_schema = f"CREATE TABLE {table_name} (\n" + ",\n".join(column_definitions) + "\n);"
    simple_schema = f"{table_name}(" + ", ".join([
        col.replace(' ', '_').replace('-', '_').replace('.', '_') 
        for col in df.columns
    ]) + ")"
    
    return {
        "sql_schema": sql_schema,
        "simple_schema": simple_schema
    }

# ===========================
# UI Components
# ===========================

def render_header():
    """Render the main header"""
    st.markdown('<h1 class="main-header">⚡ NEURAL DATA ANALYST</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Premium AI-Powered Business Intelligence Suite</p>', unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with API configuration and controls"""
    with st.sidebar:
        st.markdown("## 🔐 Neural Configuration")
        
        # Debug info (collapsed by default)
        with st.expander("🔧 Debug Info", expanded=False):
            render_debug_info()
        
        # API configuration
        if st.session_state.api_key:
            st.success("✅ API Key loaded from environment")
            
            # Model selection
            model = st.selectbox(
                "AI Model",
                list(AVAILABLE_MODELS.keys()),
                format_func=lambda x: AVAILABLE_MODELS[x],
                index=0,
                key="model_selector"
            )
            st.session_state.selected_model = model
            
            # Connection status
            if st.session_state.api_connected:
                st.markdown('<div class="success-msg">⚡ Neural Link: Active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-msg">⚡ Neural Link: Connecting...</div>', unsafe_allow_html=True)
        else:
            render_api_setup_instructions()
        
        # History section
        st.markdown("---")
        st.markdown("## 📊 Analysis History")
        
        if st.button("🗂️ View History", key="view_history"):
            show_history()
            
        if st.button("🗑️ Clear History", key="clear_history"):
            st.session_state.analysis_history = []
            st.success("History cleared!")

def render_debug_info():
    """Render debug information panel"""
    st.write(f"API Key in session: {'Yes' if st.session_state.api_key else 'No'}")
    if st.session_state.api_key:
        st.write(f"API Key (masked): {st.session_state.api_key[:10]}...{st.session_state.api_key[-5:]}")
    st.write(f"API Connected: {st.session_state.api_connected}")
    st.write(f"Environment GROQ_API_KEY: {'Set' if os.environ.get('GROQ_API_KEY') else 'Not set'}")
    
    if st.button("🔄 Reload API Key", key="reload_api"):
        reload_api_key()
    
    if st.button("🧪 Test API Connection", key="test_api"):
        test_api_connection()

def render_api_setup_instructions():
    """Render API setup instructions"""
    st.error("❌ No API key configured")
    st.markdown("""
    **Setup Required:**
    
    **For local development:**
    Create `.env` file:
    ```
    GROQ_API_KEY=your_api_key_here
    ```
    
    **For Streamlit Cloud:**
    Add to app secrets:
    ```toml
    GROQ_API_KEY = "your_api_key_here"
    ```
    
    **Get API key:** [Groq Console](https://console.groq.com/keys)
    """)

def reload_api_key():
    """Reload API key from environment"""
    api_key = get_api_key()
    if api_key:
        st.session_state.api_key = api_key
        if test_api_connection_silent(api_key, st.session_state.selected_model):
            st.session_state.api_connected = True
            st.success("✅ API key reloaded and tested successfully!")
        else:
            st.error("❌ API key loaded but connection test failed")
        st.rerun()
    else:
        st.error("No API key found in .env file")

def test_api_connection():
    """Test API connection with user feedback"""
    if st.session_state.api_key:
        with st.spinner("Testing API connection..."):
            success = test_api_connection_silent(st.session_state.api_key, st.session_state.selected_model)
            if success:
                st.session_state.api_connected = True
                st.success("✅ API connection successful!")
            else:
                st.error("❌ API connection failed")
    else:
        st.error("No API key to test")

# ===========================
# Data Upload and Display
# ===========================

def render_data_upload():
    """Render data upload section"""
    st.markdown("## 📊 Data Upload & Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or JSON file",
        type=['csv', 'json'],
        help="Upload your data file for comprehensive analysis"
    )
    
    if uploaded_file is not None:
        process_uploaded_file(uploaded_file)
    else:
        render_sample_data_option()

def process_uploaded_file(uploaded_file):
    """Process uploaded file and display results"""
    try:
        # Load data based on file type
        if uploaded_file.name.endswith('.csv'):
            df = load_csv_file(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = load_json_file(uploaded_file)
        else:
            st.error("Unsupported file type")
            return
        
        # Store in session state
        st.session_state.uploaded_data = df
        
        # Generate schema
        schema_info = generate_database_schema(df)
        st.session_state.data_schema = schema_info["simple_schema"]
        
        # Display success message and metrics
        st.success(f"✅ {uploaded_file.name} loaded successfully!")
        display_data_metrics(df, uploaded_file.size)
        
        # Display schema
        display_database_schema(schema_info, df)
        
        # Create visualizations
        create_data_visualizations(df)
        
        # Display action buttons
        display_analysis_actions(df)
        
        # Data preview
        with st.expander("👀 Data Preview", expanded=False):
            st.dataframe(df.head(100), use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

def render_sample_data_option():
    """Render option to load sample data"""
    st.info("👆 Upload a CSV or JSON file to get started")
    
    if st.button("📋 Load Sample Data", help="Load sample sales data for testing"):
        sample_data = create_sample_data()
        st.session_state.uploaded_data = sample_data
        schema_info = generate_database_schema(sample_data)
        st.session_state.data_schema = schema_info["simple_schema"]
        st.success("✅ Sample data loaded!")
        st.rerun()

def display_data_metrics(df: pd.DataFrame, file_size: int):
    """Display key metrics about the loaded data"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Rows", f"{len(df):,}")
    with col2:
        st.metric("📋 Columns", len(df.columns))
    with col3:
        st.metric("💾 Size", f"{file_size / 1024:.1f} KB")
    with col4:
        st.metric("❓ Missing", f"{df.isnull().sum().sum():,}")

def display_database_schema(schema_info: Dict[str, str], df: pd.DataFrame):
    """Display database schema information"""
    with st.expander("🗄️ Database Schema", expanded=True):
        st.markdown("**Generated Schema for AI Queries:**")
        st.code(st.session_state.data_schema, language="sql")
        
        st.markdown("**Full SQL Schema:**")
        st.code(schema_info["sql_schema"], language="sql")
        
        # Column details
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📊 Numeric Columns:**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                for col in numeric_cols:
                    st.write(f"• {col} ({df[col].dtype})")
            else:
                st.write("None found")
                
        with col2:
            st.markdown("**📝 Text Columns:**")
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            if text_cols:
                for col in text_cols:
                    st.write(f"• {col} (text)")
            else:
                st.write("None found")

def display_analysis_actions(df: pd.DataFrame):
    """Display analysis action buttons"""
    st.markdown("### 🚀 Analysis Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔬 Complete EDA", key="eda_button", help="Comprehensive Exploratory Data Analysis"):
            with st.spinner("Performing comprehensive EDA analysis..."):
                perform_eda(df)
                st.session_state.show_eda_results = True
                st.session_state.show_ai_insights = False
                st.session_state.show_advanced_analytics = False
    
    with col2:
        if st.button("🤖 AI Insights", key="ai_insights", help="Generate AI-powered insights"):
            if check_api_availability():
                with st.spinner("🤖 Generating AI insights..."):
                    generate_ai_insights(df)
                    st.session_state.show_ai_insights = True
                    st.session_state.show_eda_results = False
                    st.session_state.show_advanced_analytics = False
    
    with col3:
        if st.button("📊 Advanced Analytics", key="advanced_analytics", help="Advanced statistical analysis"):
            st.session_state.show_advanced_analytics = True
            st.session_state.show_eda_results = False
            st.session_state.show_ai_insights = False
    
    # Display results based on selection
    display_analysis_results(df)

def display_analysis_results(df: pd.DataFrame):
    """Display analysis results based on user selection"""
    if st.session_state.show_eda_results and st.session_state.eda_results:
        display_eda_results(st.session_state.eda_results)
    elif st.session_state.show_ai_insights and st.session_state.ai_insights_text:
        display_ai_insights(st.session_state.ai_insights_text)
    elif st.session_state.show_advanced_analytics:
        display_advanced_analytics(df)

# ===========================
# Visualization Functions
# ===========================

def create_data_visualizations(df: pd.DataFrame):
    """Create multiple visualizations for the uploaded data"""
    st.markdown("### 📊 Data Visualizations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    tabs = st.tabs(["📈 Overview", "📊 Distributions", "🔗 Relationships", "📋 Summary"])
    
    with tabs[0]:
        create_overview_visualizations(df, numeric_cols, categorical_cols)
    
    with tabs[1]:
        create_distribution_visualizations(df, numeric_cols)
    
    with tabs[2]:
        create_relationship_visualizations(df, numeric_cols, categorical_cols)
    
    with tabs[3]:
        create_summary_statistics(df, numeric_cols, categorical_cols)

def create_overview_visualizations(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]):
    """Create overview visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        if len(numeric_cols) >= 2:
            # Correlation heatmap
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          title="Correlation Heatmap",
                          color_continuous_scale="RdBu_r")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        if len(categorical_cols) > 0:
            # Categorical distribution
            col = categorical_cols[0]
            value_counts = df[col].value_counts().head(10)
            fig = px.pie(values=value_counts.values, 
                       names=value_counts.index,
                       title=f"Distribution of {col}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def create_distribution_visualizations(df: pd.DataFrame, numeric_cols: List[str]):
    """Create distribution visualizations"""
    if len(numeric_cols) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_col = st.selectbox("Select numeric column for histogram", numeric_cols)
            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)

def create_relationship_visualizations(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]):
    """Create relationship visualizations"""
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, key="x_scatter")
        with col2:
            y_col = st.selectbox("Y-axis", numeric_cols, key="y_scatter", 
                               index=1 if len(numeric_cols) > 1 else 0)
        
        color_col = None
        if categorical_cols:
            color_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
            color_col = color_col if color_col != "None" else None
        
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                       title=f"{x_col} vs {y_col}")
        st.plotly_chart(fig, use_container_width=True)

def create_summary_statistics(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]):
    """Create summary statistics"""
    st.markdown("#### 📋 Data Summary")
    
    summary_data = {
        "Metric": ["Total Rows", "Total Columns", "Numeric Columns", "Categorical Columns", "Missing Values", "Memory Usage"],
        "Value": [
            f"{len(df):,}",
            f"{len(df.columns)}",
            f"{len(numeric_cols)}",
            f"{len(categorical_cols)}",
            f"{df.isnull().sum().sum():,}",
            f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ===========================
# Query Interface
# ===========================

def render_query_interface():
    """Render natural language query interface"""
    st.markdown("## 🚀 AI Query Interface")
    
    # Show current schema
    if st.session_state.data_schema:
        with st.expander("🗄️ Current Data Schema", expanded=False):
            st.code(st.session_state.data_schema, language="sql")
    
    # Query input
    query_input = st.text_area(
        "Natural Language Query",
        value=st.session_state.example_query,
        placeholder="Example: Show me the top 10 customers by total sales amount",
        height=100,
        help="Describe what you want to analyze in plain English"
    )
    
    # Clear example query after use
    if st.session_state.example_query and query_input == st.session_state.example_query:
        st.session_state.example_query = ""
    
    # Analysis buttons
    col1, col2 = st.columns(2)
    
    with col1:
        api_available = check_api_availability()
        if st.button("🧠 Analyze Query", 
                    disabled=not api_available or not query_input.strip(),
                    help="Generate SQL and insights for your query"):
            if api_available and query_input.strip():
                analyze_single_query(query_input.strip())
    
    with col2:
        if st.button("⚔️ Model Battle", 
                    disabled=not api_available or not query_input.strip(),
                    help="Compare multiple AI models on your query"):
            if api_available and query_input.strip():
                st.session_state.current_query = query_input.strip()
                st.session_state.show_model_selection = True
                st.rerun()
    
    # Status messages
    display_api_status()
    
    # Recent queries
    display_recent_queries()
    
    # Model selection interface
    if st.session_state.show_model_selection:
        render_model_selection_interface()

def check_api_availability() -> bool:
    """Check if API is available"""
    return bool(st.session_state.api_key and len(st.session_state.api_key) > 10)

def display_api_status():
    """Display API connection status"""
    if not check_api_availability():
        st.warning("⚠️ **AI Features Disabled**: API key not detected. Use the '🔄 Reload API Key' button in the sidebar.")
    else:
        st.success("✅ **AI Features Active**: Ready for natural language queries and model battles!")

def display_recent_queries():
    """Display recent queries"""
    if st.session_state.recent_queries:
        with st.expander("📝 Recent Queries", expanded=False):
            for i, recent_query in enumerate(st.session_state.recent_queries[-5:]):
                if st.button(f"🔄 {recent_query[:60]}...", key=f"recent_{i}"):
                    st.session_state.example_query = recent_query
                    st.rerun()

# ===========================
# Analysis Functions
# ===========================

def analyze_single_query(query: str):
    """Analyze query with single model"""
    # Add to recent queries
    if query not in st.session_state.recent_queries:
        st.session_state.recent_queries.append(query)
        st.session_state.recent_queries = st.session_state.recent_queries[-10:]
    
    with st.spinner(f"🧠 Analyzing with {st.session_state.selected_model}..."):
        try:
            # Generate SQL and insights
            sql_result = generate_sql(query)
            insights_result = generate_insights(query)
            
            # Save to history
            save_analysis_to_history({
                "type": "Single Query Analysis",
                "query": query,
                "schema": st.session_state.data_schema,
                "sql_result": sql_result,
                "insights": insights_result,
                "model": st.session_state.selected_model
            })
            
            # Display results
            display_query_results(sql_result, insights_result)
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

def generate_sql(query: str) -> str:
    """Generate SQL from natural language query"""
    prompt = f"""Convert this natural language query to SQL:

Database Schema: {st.session_state.data_schema}

Natural Language Query: {query}

Instructions:
- Use the exact column names from the schema
- Generate clean, optimized SQL
- Include appropriate WHERE, GROUP BY, ORDER BY clauses
- Use proper SQL syntax
- Return only the SQL query without explanations

SQL Query:"""

    return make_api_call(st.session_state.selected_model, prompt)

def generate_insights(query: str) -> str:
    """Generate business insights from query"""
    prompt = f"""Provide detailed business insights for this data analysis query:

Database Schema: {st.session_state.data_schema}

Query: {query}

Generate 4-5 key business insights in this format:
**Insight Title 1**: Detailed explanation of what this analysis reveals about the business
**Insight Title 2**: Another important finding or recommendation
(continue for 4-5 insights)

Focus on:
- Business implications
- Actionable recommendations  
- Data patterns and trends
- Strategic insights
- Potential opportunities or risks

Business Insights:"""

    return make_api_call(st.session_state.selected_model, prompt)

def display_query_results(sql_result: str, insights_result: str):
    """Display query analysis results"""
    st.markdown("## 🎯 Analysis Results")
    
    tabs = st.tabs(["🔍 SQL Query", "💡 AI Insights", "🔄 Execute Query"])
    
    with tabs[0]:
        st.markdown("### 🔍 Generated SQL Query")
        st.code(sql_result, language='sql')
        
        if st.button("📋 Copy SQL", key="copy_sql"):
            st.success("SQL copied to clipboard! (Use Ctrl+C to copy from the code block above)")
    
    with tabs[1]:
        st.markdown("### 💡 AI-Powered Business Insights")
        insights = parse_insights(insights_result)
        
        for i, insight in enumerate(insights):
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #ffd700;">💡 {insight['title']}</h4>
                <p>{insight['text']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("### 🔄 Execute Query on Your Data")
        
        if st.session_state.uploaded_data is not None:
            if st.button("▶️ Run SQL on Uploaded Data", key="execute_sql"):
                execute_sql_on_data(sql_result, st.session_state.uploaded_data)
        else:
            st.info("Upload data first to execute SQL queries")

def execute_sql_on_data(sql_query: str, df: pd.DataFrame):
    """Execute SQL query on the uploaded DataFrame"""
    try:
        # Create temporary SQLite database
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
            conn = sqlite3.connect(tmp_file.name)
            
            # Write DataFrame to SQLite
            df.to_sql('uploaded_data', conn, if_exists='replace', index=False)
            
            # Clean and execute SQL
            clean_sql = sql_query.strip()
            if clean_sql.lower().startswith('sql:'):
                clean_sql = clean_sql[4:].strip()
            
            # Execute query
            result_df = pd.read_sql_query(clean_sql, conn)
            conn.close()
            
            # Display results
            st.success("✅ Query executed successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows Returned", len(result_df))
            with col2:
                st.metric("Columns", len(result_df.columns))
            
            st.markdown("#### 📊 Query Results")
            st.dataframe(result_df, use_container_width=True)
            
            # Auto visualization
            if len(result_df) > 0:
                auto_visualize_results(result_df)
                
    except Exception as e:
        st.error(f"Error executing SQL: {str(e)}")
        st.info("💡 Tip: The AI-generated SQL might need adjustment for your specific data structure")

def auto_visualize_results(result_df: pd.DataFrame):
    """Automatically create visualizations for query results"""
    st.markdown("#### 📈 Auto-Generated Visualization")
    
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 1 and len(result_df) <= 50:
        text_cols = result_df.select_dtypes(include=['object']).columns.tolist()
        if text_cols:
            fig = px.bar(result_df, 
                       x=text_cols[0], 
                       y=numeric_cols[0],
                       title=f"{numeric_cols[0]} by {text_cols[0]}")
            st.plotly_chart(fig, use_container_width=True)
    elif len(numeric_cols) >= 1 and len(result_df) > 10:
        fig = px.line(result_df, 
                     y=numeric_cols[0],
                     title=f"Trend: {numeric_cols[0]}")
        st.plotly_chart(fig, use_container_width=True)

# ===========================
# Model Comparison Functions
# ===========================

def render_model_selection_interface():
    """Render model selection interface for battle"""
    st.markdown("---")
    st.markdown("## ⚔️ Model Battle Setup")
    st.markdown(f"**Query:** {st.session_state.current_query}")
    
    st.markdown("### 🎯 Select Models for Battle")
    
    col1, col2 = st.columns(2)
    
    selected_models = []
    
    with col1:
        st.markdown("**🚀 High-Performance Models:**")
        if st.checkbox("Llama 3.3 70B (Most Advanced)", key="battle_llama33", value=True):
            selected_models.append("llama-3.3-70b-versatile")
        if st.checkbox("Llama 3 70B (Reliable)", key="battle_llama3", value=True):
            selected_models.append("llama3-70b-8192")
        if st.checkbox("DeepSeek R1 70B (Specialized)", key="battle_deepseek", value=False):
            selected_models.append("deepseek-r1-distill-llama-70b")
    
    with col2:
        st.markdown("**⚡ Fast & Efficient Models:**")
        if st.checkbox("Mixtral 8x7B (Fast & Efficient)", key="battle_mixtral", value=True):
            selected_models.append("mixtral-8x7b-32768")
        if st.checkbox("Gemma 2 9B (Lightweight)", key="battle_gemma", value=False):
            selected_models.append("gemma2-9b-it")
        if st.checkbox("Qwen QwQ 32B (Reasoning)", key="battle_qwen", value=False):
            selected_models.append("qwen-qwq-32b")
    
    if selected_models:
        st.success(f"✅ **Selected Models:** {len(selected_models)} models ready for battle")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_rounds = st.selectbox("Test Rounds", [1, 2, 3], index=0)
        with col2:
            timeout_seconds = st.selectbox("Timeout (seconds)", [10, 20, 30], index=1)
        with col3:
            if st.button("❌ Cancel", key="cancel_battle"):
                st.session_state.show_model_selection = False
                st.rerun()
        
        if st.button("🚀 Start Model Battle", key="start_battle", type="primary"):
            st.session_state.show_model_selection = False
            run_model_comparison(selected_models, test_rounds, timeout_seconds)
    else:
        st.warning("⚠️ Please select at least one model for the battle")
        if st.button("❌ Cancel", key="cancel_no_models"):
            st.session_state.show_model_selection = False
            st.rerun()

def run_model_comparison(selected_models: List[str], rounds: int, timeout: int):
    """Run model comparison with selected models"""
    st.markdown("## ⚔️ Model Battle Arena")
    st.markdown(f"*Testing {len(selected_models)} models with {rounds} round(s) each...*")
    
    total_tests = len(selected_models) * rounds
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    test_count = 0
    
    for model in selected_models:
        model_results = []
        
        for round_num in range(rounds):
            test_count += 1
            status_text.text(f"Testing {model} (Round {round_num + 1}/{rounds})...")
            progress_bar.progress(test_count / total_tests)
            
            try:
                start_time = time.time()
                response = generate_comparison_response(model, timeout)
                response_time = time.time() - start_time
                
                score = score_model_response(response, response_time)
                
                model_results.append({
                    'response': response,
                    'response_time': response_time * 1000,
                    'score': score,
                    'success': True,
                    'round': round_num + 1
                })
            except Exception as e:
                model_results.append({
                    'response': f"Error: {str(e)}",
                    'response_time': 0,
                    'score': 0,
                    'success': False,
                    'round': round_num + 1
                })
            
            time.sleep(0.5)  # Rate limiting
        
        # Calculate averages
        successful_results = [r for r in model_results if r['success']]
        if successful_results:
            avg_score = sum(r['score'] for r in successful_results) / len(successful_results)
            avg_time = sum(r['response_time'] for r in successful_results) / len(successful_results)
            best_response = max(successful_results, key=lambda x: x['score'])['response']
        else:
            avg_score = 0
            avg_time = 0
            best_response = "All attempts failed"
        
        results.append({
            'model': model,
            'avg_score': avg_score,
            'avg_response_time': avg_time,
            'success_rate': len(successful_results) / len(model_results) * 100,
            'best_response': best_response,
            'all_results': model_results,
            'success': len(successful_results) > 0
        })
    
    progress_bar.empty()
    status_text.empty()
    
    # Save to history
    save_analysis_to_history({
        "type": "Model Comparison",
        "query": st.session_state.current_query,
        "results": results
    })
    
    display_comparison_results(results)

def generate_comparison_response(model: str, timeout: int) -> str:
    """Generate response for model comparison"""
    prompt = f"""Analyze this query and provide SQL + business insight:

Schema: {st.session_state.data_schema}
Query: {st.session_state.current_query}

Respond in this exact format:
SQL: [your SQL query here]
INSIGHT: [your business insight here]

Keep response concise and focused."""

    return make_api_call(model, prompt, timeout)

def score_model_response(response: str, response_time: float) -> int:
    """Score model response based on quality and speed"""
    response_lower = response.lower()
    
    # Quality scoring
    has_sql = any(keyword in response_lower for keyword in ['select', 'from', 'where', 'group by', 'order by'])
    sql_score = 40 if has_sql else 0
    
    has_insight = any(keyword in response_lower for keyword in ['insight', 'analysis', 'recommendation', 'business'])
    insight_score = 30 if has_insight else 0
    
    length_score = min(len(response) / 20, 20)
    speed_score = max(0, 10 - (response_time * 2)) if response_time > 0 else 0
    
    total_score = sql_score + insight_score + length_score + speed_score
    return max(0, min(100, round(total_score)))

def display_comparison_results(results: List[Dict]):
    """Display model comparison results"""
    sorted_results = sorted([r for r in results if r['success']], key=lambda x: x['avg_score'], reverse=True)
    
    if not sorted_results:
        st.error("No successful results to display")
        return
    
    # Winner announcement
    winner = sorted_results[0]
    fastest = min(sorted_results, key=lambda x: x['avg_response_time'])
    most_reliable = max(sorted_results, key=lambda x: x['success_rate'])
    
    st.markdown("### 🏆 Battle Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(45deg, #FFD700, #FFA500); padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #000; margin: 0;">🏆 HIGHEST SCORE</h3>
            <h4 style="color: #000; margin: 5px 0;">{winner['model'].replace('-', ' ').title()}</h4>
            <p style="color: #000; margin: 0;">Avg Score: {winner['avg_score']:.1f}/100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(45deg, #40E0D0, #48D1CC); padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #000; margin: 0;">⚡ FASTEST</h3>
            <h4 style="color: #000; margin: 5px 0;">{fastest['model'].replace('-', ' ').title()}</h4>
            <p style="color: #000; margin: 0;">Avg: {fastest['avg_response_time']:.0f}ms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(45deg, #98FB98, #90EE90); padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #000; margin: 0;">🎯 MOST RELIABLE</h3>
            <h4 style="color: #000; margin: 5px 0;">{most_reliable['model'].replace('-', ' ').title()}</h4>
            <p style="color: #000; margin: 0;">{most_reliable['success_rate']:.0f}% Success</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance chart
    display_performance_chart(sorted_results)
    
    # Detailed results
    display_detailed_results(sorted_results)

def display_performance_chart(results: List[Dict]):
    """Display performance comparison chart"""
    st.markdown("### 📊 Performance Comparison")
    
    models = [r['model'].replace('-', ' ').replace('versatile', '').replace('8192', '').title() for r in results]
    scores = [r['avg_score'] for r in results]
    times = [r['avg_response_time'] for r in results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Average Score',
        x=models,
        y=scores,
        yaxis='y',
        marker_color='#FFD700',
        text=[f"{s:.1f}" for s in scores],
        textposition='auto'
    ))
    
    fig.add_trace(go.Scatter(
        name='Response Time (ms)',
        x=models,
        y=times,
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title=f'Model Performance: "{st.session_state.current_query[:50]}..."',
        xaxis=dict(title='Models'),
        yaxis=dict(title='Score (0-100)', side='left'),
        yaxis2=dict(title='Response Time (ms)', side='right', overlaying='y'),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_detailed_results(results: List[Dict]):
    """Display detailed results for each model"""
    st.markdown("### 📋 Detailed Results")
    
    for i, result in enumerate(results):
        with st.expander(f"{'🥇' if i == 0 else '🥈' if i == 1 else '🥉' if i == 2 else '📊'} {result['model']} - Avg Score: {result['avg_score']:.1f}/100"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Score", f"{result['avg_score']:.1f}/100")
            with col2:
                st.metric("Avg Speed", f"{result['avg_response_time']:.0f}ms")
            with col3:
                st.metric("Success Rate", f"{result['success_rate']:.0f}%")
            with col4:
                st.metric("Total Tests", len(result['all_results']))
            
            st.markdown("**Best Response:**")
            display_model_response(result['best_response'], i)

def display_model_response(response: str, index: int):
    """Display formatted model response"""
    if "SQL:" in response and "INSIGHT:" in response:
        parts = response.split("INSIGHT:")
        sql_part = parts[0].replace("SQL:", "").strip()
        insight_part = parts[1].strip()
        
        st.markdown("**SQL:**")
        st.code(sql_part, language='sql')
        st.markdown("**Insight:**")
        st.markdown(insight_part)
    else:
        st.text_area("", response, height=150, key=f"response_{index}")

# ===========================
# EDA Functions
# ===========================

def perform_eda(df: pd.DataFrame):
    """Perform comprehensive EDA analysis"""
    eda_results = {
        'overview': generate_overview_stats(df),
        'distributions': generate_distribution_charts(df),
        'correlations': generate_correlation_analysis(df),
        'insights': generate_eda_insights(df),
        'data_quality': analyze_data_quality(df)
    }
    
    st.session_state.eda_results = eda_results
    
    # Save to history
    save_analysis_to_history({
        "type": "EDA",
        "data_shape": df.shape,
        "results": "EDA analysis completed"
    })

def generate_overview_stats(df: pd.DataFrame) -> Dict:
    """Generate overview statistics"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    
    overview = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(numeric_cols),
        'categorical_columns': len(categorical_cols),
        'datetime_columns': len(datetime_cols),
        'missing_values_total': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    
    if len(numeric_cols) > 0:
        overview['summary_stats'] = df[numeric_cols].describe()
    
    return overview

def generate_distribution_charts(df: pd.DataFrame) -> Dict:
    """Generate distribution charts"""
    charts = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]  # Limit to 6
    
    if len(numeric_cols) > 0:
        # Create subplots for distributions
        fig = make_subplots(
            rows=(len(numeric_cols) + 1) // 2,
            cols=2,
            subplot_titles=[col for col in numeric_cols]
        )
        
        for i, col in enumerate(numeric_cols):
            row = (i // 2) + 1
            col_pos = (i % 2) + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, showlegend=False),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title="Numeric Distributions",
            height=300 * ((len(numeric_cols) + 1) // 2)
        )
        
        charts['distributions'] = fig
    
    return charts

def generate_correlation_analysis(df: pd.DataFrame) -> Dict:
    """Generate correlation analysis"""
    correlations = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        
        # Heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
            color_continuous_scale="RdBu_r"
        )
        correlations['heatmap'] = fig
        
        # Top correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs)
            corr_df['Abs_Correlation'] = abs(corr_df['Correlation'])
            corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
            correlations['top_correlations'] = corr_df.head(10)
    
    return correlations

def generate_eda_insights(df: pd.DataFrame) -> List[Dict]:
    """Generate EDA insights"""
    insights = []
    
    # Basic insights
    if df.isnull().sum().sum() > 0:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        insights.append({
            'title': 'Missing Data Alert',
            'description': f'Dataset contains {missing_pct:.1f}% missing values across all cells.'
        })
    
    if df.duplicated().sum() > 0:
        dup_pct = (df.duplicated().sum() / len(df)) * 100
        insights.append({
            'title': 'Duplicate Rows Found',
            'description': f'{df.duplicated().sum()} duplicate rows detected ({dup_pct:.1f}% of data).'
        })
    
    # Numeric insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols[:3]:  # Top 3 numeric columns
            skewness = df[col].skew()
            if abs(skewness) > 1:
                insights.append({
                    'title': f'Skewed Distribution: {col}',
                    'description': f'{col} shows {"positive" if skewness > 0 else "negative"} skew ({skewness:.2f}).'
                })
    
    return insights[:5]  # Limit to 5 insights

def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """Analyze data quality"""
    quality = {}
    
    # Missing values chart
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        fig = px.bar(
            x=missing_data.index,
            y=missing_data.values,
            title="Missing Values by Column",
            labels={'x': 'Column', 'y': 'Missing Count'}
        )
        quality['missing_values'] = fig
    
    # Data types chart
    dtype_counts = df.dtypes.value_counts()
    fig = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index.astype(str),
        title="Data Types Distribution"
    )
    quality['data_types'] = fig
    
    # Duplicates info
    quality['duplicates'] = {
        'count': df.duplicated().sum(),
        'percentage': (df.duplicated().sum() / len(df)) * 100
    }
    
    return quality

def display_eda_results(results: Dict):
    """Display EDA results"""
    st.markdown("---")
    st.markdown("## 🔬 Comprehensive EDA Results")
    
    tabs = st.tabs(["📊 Overview", "📈 Distributions", "🔗 Correlations", "🎯 Insights", "📋 Data Quality"])
    
    with tabs[0]:
        display_overview_tab(results.get('overview', {}))
    
    with tabs[1]:
        display_distributions_tab(results.get('distributions', {}))
    
    with tabs[2]:
        display_correlations_tab(results.get('correlations', {}))
    
    with tabs[3]:
        display_insights_tab(results.get('insights', []))
    
    with tabs[4]:
        display_data_quality_tab(results.get('data_quality', {}))

def display_overview_tab(overview: Dict):
    """Display overview statistics"""
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{overview.get('total_rows', 0):,}")
    with col2:
        st.metric("Total Columns", overview.get('total_columns', 0))
    with col3:
        st.metric("Numeric Columns", overview.get('numeric_columns', 0))
    with col4:
        st.metric("Categorical Columns", overview.get('categorical_columns', 0))
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Missing Values", f"{overview.get('missing_values_total', 0):,}")
    with col6:
        st.metric("Duplicate Rows", f"{overview.get('duplicate_rows', 0):,}")
    with col7:
        st.metric("Memory Usage", overview.get('memory_usage', '0 MB'))
    with col8:
        st.metric("DateTime Columns", overview.get('datetime_columns', 0))
    
    if 'summary_stats' in overview:
        st.markdown("### 📈 Summary Statistics")
        st.dataframe(overview['summary_stats'], use_container_width=True)

def display_distributions_tab(distributions: Dict):
    """Display distribution charts"""
    st.markdown("### 📈 Data Distributions")
    
    if distributions:
        for chart_name, chart in distributions.items():
            st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("No distribution charts available")

def display_correlations_tab(correlations: Dict):
    """Display correlation analysis"""
    st.markdown("### 🔗 Correlation Analysis")
    
    if 'heatmap' in correlations:
        st.plotly_chart(correlations['heatmap'], use_container_width=True)
    
    if 'top_correlations' in correlations:
        st.markdown("#### 🔝 Top Correlations")
        st.dataframe(correlations['top_correlations'], use_container_width=True)

def display_insights_tab(insights: List[Dict]):
    """Display generated insights"""
    st.markdown("### 🎯 Generated Insights")
    
    if insights:
        for insight in insights:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #ffd700;">💡 {insight['title']}</h4>
                <p>{insight['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No insights generated")

def display_data_quality_tab(quality: Dict):
    """Display data quality assessment"""
    st.markdown("### 📋 Data Quality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Missing Values")
        if 'missing_values' in quality:
            st.plotly_chart(quality['missing_values'], use_container_width=True)
        else:
            st.info("No missing values chart available")
    
    with col2:
        st.markdown("#### Data Types")
        if 'data_types' in quality:
            st.plotly_chart(quality['data_types'], use_container_width=True)
        else:
            st.info("No data types chart available")
    
    if 'duplicates' in quality:
        st.markdown("#### Duplicate Analysis")
        dup_info = quality['duplicates']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Duplicate Count", dup_info.get('count', 0))
        with col2:
            st.metric("Duplicate %", f"{dup_info.get('percentage', 0):.1f}%")

# ===========================
# AI Insights Functions
# ===========================

def generate_ai_insights(df: pd.DataFrame):
    """Generate AI-powered insights"""
    try:
        # Prepare data summary
        summary = f"""
        Dataset Analysis:
        - Rows: {len(df):,}
        - Columns: {len(df.columns)}
        - Schema: {st.session_state.data_schema}
        - Missing values: {df.isnull().sum().sum():,}
        
        Column types:
        {df.dtypes.to_string()}
        
        Sample data:
        {df.head(3).to_string()}
        """
        
        prompt = f"""Analyze this dataset and provide 5 key business insights:
        
        {summary}
        
        Format as:
        1. **Insight Title**: Description
        2. **Insight Title**: Description
        (etc.)
        
        Focus on business value, patterns, and actionable recommendations."""
        
        insights_text = make_api_call(st.session_state.selected_model, prompt)
        st.session_state.ai_insights_text = insights_text
        
        # Save to history
        save_analysis_to_history({
            "type": "AI Insights",
            "data_shape": df.shape,
            "insights": insights_text
        })
        
    except Exception as e:
        st.error(f"Failed to generate AI insights: {str(e)}")
        st.session_state.ai_insights_text = f"Error generating insights: {str(e)}"

def display_ai_insights(insights_text: str):
    """Display AI-generated insights"""
    st.markdown("---")
    st.markdown("## 🤖 AI-Generated Insights")
    
    insights = parse_insights(insights_text)
    
    for insight in insights:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #ffd700;">💡 {insight['title']}</h4>
            <p>{insight['text']}</p>
        </div>
        """, unsafe_allow_html=True)

def parse_insights(raw_insights: str) -> List[Dict]:
    """Parse insights from API response"""
    insights = []
    lines = raw_insights.strip().split('\n')
    
    current_insight = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for patterns like "**Title**:" or "1. **Title**:"
        if line.startswith('**') and '**:' in line:
            if current_insight:
                insights.append(current_insight)
            
            parts = line.split('**:', 1)
            title = parts[0].replace('**', '').strip().lstrip('1234567890.- ')
            text = parts[1].strip() if len(parts) > 1 else ''
            current_insight = {'title': title, 'text': text}
        
        elif line.startswith(('1.', '2.', '3.', '4.', '5.')) and '**' in line:
            if current_insight:
                insights.append(current_insight)
            
            # Extract from numbered format
            clean_line = line.lstrip('1234567890. ').strip()
            if '**:' in clean_line:
                parts = clean_line.split('**:', 1)
                title = parts[0].replace('**', '').strip()
                text = parts[1].strip() if len(parts) > 1 else ''
            else:
                title = f"Insight {len(insights) + 1}"
                text = clean_line
            
            current_insight = {'title': title, 'text': text}
        
        elif current_insight and line:
            current_insight['text'] += ' ' + line
    
    if current_insight:
        insights.append(current_insight)
    
    # Fallback if no insights parsed
    if not insights and raw_insights.strip():
        insights.append({
            'title': 'AI Analysis',
            'text': raw_insights.strip()
        })
    
    return insights[:5]

# ===========================
# Advanced Analytics Functions
# ===========================

def display_advanced_analytics(df: pd.DataFrame):
    """Display advanced analytics"""
    st.markdown("---")
    st.markdown("## 📊 Advanced Analytics")
    
    tabs = st.tabs(["📈 Statistical Summary", "🔍 Outlier Detection", "📊 Correlation Analysis", "🎯 Distribution Analysis"])
    
    with tabs[0]:
        show_statistical_summary(df)
    
    with tabs[1]:
        show_outlier_analysis(df)
    
    with tabs[2]:
        show_correlation_analysis(df)
    
    with tabs[3]:
        show_distribution_analysis(df)

def show_statistical_summary(df: pd.DataFrame):
    """Show advanced statistical summary"""
    st.markdown("### 📈 Advanced Statistical Summary")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        # Basic statistics
        st.markdown("#### 📊 Descriptive Statistics")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Advanced statistics
        st.markdown("#### 📏 Advanced Distribution Metrics")
        advanced_stats = pd.DataFrame({
            'Column': numeric_cols,
            'Skewness': [df[col].skew() for col in numeric_cols],
            'Kurtosis': [df[col].kurtosis() for col in numeric_cols],
            'Missing %': [df[col].isnull().sum() / len(df) * 100 for col in numeric_cols],
            'Zeros %': [(df[col] == 0).sum() / len(df) * 100 for col in numeric_cols],
            'Unique Values': [df[col].nunique() for col in numeric_cols]
        })
        st.dataframe(advanced_stats, use_container_width=True, hide_index=True)
        
        # Interpretation
        st.markdown("#### 🎯 Statistical Interpretation")
        for col in numeric_cols[:3]:
            skewness = df[col].skew()
            kurtosis = df[col].kurtosis()
            
            skew_interpretation = (
                "Normal" if abs(skewness) < 0.5 else
                "Slightly Skewed" if abs(skewness) < 1 else
                "Moderately Skewed" if abs(skewness) < 2 else
                "Highly Skewed"
            )
            
            kurt_interpretation = (
                "Normal" if abs(kurtosis) < 1 else
                "Heavy-tailed" if kurtosis > 1 else
                "Light-tailed"
            )
            
            st.markdown(f"**{col}:** {skew_interpretation} distribution, {kurt_interpretation} shape")
    else:
        st.info("No numeric columns found for statistical analysis")

def show_outlier_analysis(df: pd.DataFrame):
    """Show outlier detection analysis"""
    st.markdown("### 🔍 Advanced Outlier Detection")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        # Summary for all columns
        st.markdown("#### 📊 Outlier Summary (All Columns)")
        
        outlier_summary = []
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_summary.append({
                    'Column': col,
                    'Outlier Count': len(outliers),
                    'Outlier %': len(outliers) / len(df) * 100,
                    'Lower Bound': lower_bound,
                    'Upper Bound': upper_bound
                })
        
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary)
            st.dataframe(outlier_df, use_container_width=True, hide_index=True)
            
            # Detailed analysis
            st.markdown("#### 🎯 Detailed Outlier Analysis")
            selected_col = st.selectbox("Select column for detailed analysis", numeric_cols, key="outlier_detail")
            
            if selected_col:
                analyze_column_outliers(df, selected_col)
    else:
        st.info("No numeric columns found for outlier analysis")

def analyze_column_outliers(df: pd.DataFrame, column: str):
    """Analyze outliers for a specific column"""
    data = df[column].dropna()
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Outliers", len(outliers))
    with col2:
        st.metric("Outlier %", f"{len(outliers)/len(df)*100:.1f}%")
    with col3:
        st.metric("IQR", f"{IQR:.2f}")
    with col4:
        st.metric("Range", f"{upper_bound - lower_bound:.2f}")
    
    # Box plot
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=data,
        name=column,
        boxpoints='outliers'
    ))
    fig.update_layout(
        title=f'Box Plot with Outliers: {column}',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show outlier values
    if len(outliers) > 0:
        st.markdown("#### 📋 Outlier Values")
        st.dataframe(outliers[[column]].head(20), use_container_width=True)

def show_correlation_analysis(df: pd.DataFrame):
    """Show correlation analysis"""
    st.markdown("### 📊 Advanced Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        
        # Enhanced heatmap
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Enhanced Correlation Matrix",
                       color_continuous_scale="RdBu_r")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation strength analysis
        st.markdown("#### 🎯 Correlation Strength Analysis")
        
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                strength = (
                    "Very Strong" if abs(corr_value) >= 0.8 else
                    "Strong" if abs(corr_value) >= 0.6 else
                    "Moderate" if abs(corr_value) >= 0.4 else
                    "Weak" if abs(corr_value) >= 0.2 else
                    "Very Weak"
                )
                
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_value,
                    'Abs Correlation': abs(corr_value),
                    'Strength': strength
                })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.sort_values('Abs Correlation', ascending=False)
            
            st.dataframe(corr_df[['Variable 1', 'Variable 2', 'Correlation', 'Strength']].head(15), 
                       use_container_width=True, hide_index=True)
            
            # Strong correlations warning
            strong_corr = corr_df[corr_df['Abs Correlation'] >= 0.6]
            if len(strong_corr) > 0:
                st.markdown("#### ⚠️ Strong Correlations (>0.6)")
                st.dataframe(strong_corr[['Variable 1', 'Variable 2', 'Correlation']], 
                           use_container_width=True, hide_index=True)
                st.warning("Strong correlations may indicate multicollinearity issues in modeling.")
    else:
        st.info("Need at least 2 numeric columns for correlation analysis")

def show_distribution_analysis(df: pd.DataFrame):
    """Show distribution analysis"""
    st.markdown("### 🎯 Advanced Distribution Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(numeric_cols) > 0:
        st.markdown("#### 📊 Numeric Distribution Analysis")
        
        selected_col = st.selectbox("Select numeric column", numeric_cols, key="dist_numeric")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            data = df[selected_col].dropna()
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data,
                name='Distribution',
                opacity=0.7,
                nbinsx=30
            ))
            fig.update_layout(
                title=f'Distribution of {selected_col}',
                xaxis_title=selected_col,
                yaxis_title='Frequency',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=data,
                name=selected_col,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
            fig.update_layout(
                title=f'Box Plot of {selected_col}',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution statistics
        st.markdown("#### 📈 Distribution Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{data.mean():.2f}")
        with col2:
            st.metric("Median", f"{data.median():.2f}")
        with col3:
            st.metric("Std Dev", f"{data.std():.2f}")
        with col4:
            st.metric("Range", f"{data.max() - data.min():.2f}")
    
    if len(categorical_cols) > 0:
        st.markdown("#### 🏷️ Categorical Distribution Analysis")
        
        selected_cat = st.selectbox("Select categorical column", categorical_cols, key="dist_categorical")
        
        value_counts = df[selected_cat].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = px.bar(
                x=value_counts.index[:15],
                y=value_counts.values[:15],
                title=f'Top Categories in {selected_cat}'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart
            fig = px.pie(
                values=value_counts.values[:10],
                names=value_counts.index[:10],
                title=f'Distribution of {selected_cat}'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Category statistics
        st.markdown("#### 📊 Category Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Unique Categories", df[selected_cat].nunique())
        with col2:
            st.metric("Most Frequent", value_counts.index[0])
        with col3:
            st.metric("Frequency", value_counts.iloc[0])
        with col4:
            st.metric("Missing Values", df[selected_cat].isnull().sum())

# ===========================
# History Functions
# ===========================

def save_analysis_to_history(analysis_record: Dict):
    """Save analysis to history"""
    record = {
        "timestamp": datetime.now().isoformat(),
        "session_id": st.session_state.session_id,
        **analysis_record
    }
    st.session_state.analysis_history.append(record)

def show_history():
    """Display analysis history"""
    st.markdown("## 📊 Analysis History")
    
    history = st.session_state.analysis_history
    
    if not history:
        st.info("No analysis history found.")
        return
    
    for i, record in enumerate(reversed(history[-10:])):
        with st.expander(f"{record['type']} - {record['timestamp'][:19]}"):
            if record['type'] == 'Single Query Analysis':
                st.markdown(f"**Query:** {record['query']}")
                st.markdown("**SQL Result:**")
                st.code(record.get('sql_result', 'N/A'), language='sql')
                
            elif record['type'] == 'Model Comparison':
                st.markdown(f"**Query:** {record['query']}")
                st.markdown("**Results:**")
                for result in record.get('results', []):
                    if isinstance(result, dict) and 'model' in result and 'avg_score' in result:
                        st.markdown(f"- {result['model']}: {result['avg_score']:.1f}/100")
                    
            elif record['type'] == 'EDA':
                st.markdown(f"**Data Shape:** {record.get('data_shape', 'N/A')}")
                st.markdown("**Analysis completed successfully**")
            
            elif record['type'] == 'AI Insights':
                st.markdown(f"**Data Shape:** {record.get('data_shape', 'N/A')}")
                st.markdown("**Insights generated successfully**")

# ===========================
# Main Application
# ===========================

def main():
    """Main application entry point"""
    # Initialize
    inject_custom_css()
    initialize_session_state()
    
    # Render UI
    render_header()
    render_sidebar()
    
    # Main content area
    if st.session_state.uploaded_data is not None:
        render_query_interface()
    else:
        render_data_upload()

if __name__ == "__main__":
    main()