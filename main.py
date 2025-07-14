import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import requests
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from io import StringIO
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom modules
from eda_analyzer import EDAAnalyzer
from database_manager import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="⚡ Neural Data Analyst Premium",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

class NeuralDataAnalyst:
    def __init__(self):
        # Initialize with error handling
        try:
            self.db_manager = DatabaseManager()
        except:
            # Create a basic fallback if DatabaseManager fails
            self.db_manager = None
            
        self.eda_analyzer = EDAAnalyzer()
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        # Get API key from multiple sources
        api_key = None
        
        # Try different sources in order of preference
        try:
            # 1. Try Streamlit secrets first (for deployment)
            if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                api_key = st.secrets['GROQ_API_KEY']
                print("API key loaded from Streamlit secrets")
            # 2. Try environment variable (most common)
            elif 'GROQ_API_KEY' in os.environ:
                api_key = os.environ['GROQ_API_KEY']
                print("API key loaded from environment variable")
            # 3. Try loading from .env file directly
            else:
                from dotenv import load_dotenv
                load_dotenv(override=True)  # Force reload
                if 'GROQ_API_KEY' in os.environ:
                    api_key = os.environ['GROQ_API_KEY']
                    print("API key loaded from .env file")
        except Exception as e:
            print(f"Error loading API key: {e}")
            
        # Debug: Check if we found the API key
        if api_key:
            print(f"✅ API key found: {api_key[:10]}...{api_key[-5:] if len(api_key) > 15 else api_key}")
        else:
            print("❌ No API key found")
            
        if 'api_key' not in st.session_state:
            st.session_state.api_key = api_key or ""
        if 'api_connected' not in st.session_state:
            st.session_state.api_connected = bool(api_key)
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None
        if 'data_schema' not in st.session_state:
            st.session_state.data_schema = ""
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = "llama-3.3-70b-versatile"
        if 'example_query' not in st.session_state:
            st.session_state.example_query = ""
        if 'recent_queries' not in st.session_state:
            st.session_state.recent_queries = []
        if 'show_eda_results' not in st.session_state:
            st.session_state.show_eda_results = False
        if 'show_ai_insights' not in st.session_state:
            st.session_state.show_ai_insights = False
        if 'show_advanced_analytics' not in st.session_state:
            st.session_state.show_advanced_analytics = False
        if 'eda_results' not in st.session_state:
            st.session_state.eda_results = None
        if 'ai_insights_text' not in st.session_state:
            st.session_state.ai_insights_text = None
        if 'show_model_selection' not in st.session_state:
            st.session_state.show_model_selection = False
        if 'current_query' not in st.session_state:
            st.session_state.current_query = ""
            
        # Force test connection if we have an API key but haven't connected
        if api_key and not st.session_state.api_connected:
            print("Testing API connection...")
            self.test_api_connection_silent(api_key, st.session_state.selected_model)
            
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">⚡ NEURAL DATA ANALYST</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Premium AI-Powered Business Intelligence Suite</p>', unsafe_allow_html=True)
        
    def render_api_config(self):
        """Render API configuration section"""
        with st.sidebar:
            st.markdown("## 🔐 Neural Configuration")
            
            # Debug section
            with st.expander("🔧 Debug Info", expanded=False):
                st.write(f"API Key in session: {'Yes' if st.session_state.api_key else 'No'}")
                if st.session_state.api_key:
                    st.write(f"API Key (masked): {st.session_state.api_key[:10]}...{st.session_state.api_key[-5:]}")
                st.write(f"API Connected: {st.session_state.api_connected}")
                st.write(f"Environment GROQ_API_KEY: {'Set' if os.environ.get('GROQ_API_KEY') else 'Not set'}")
                
                # Test button
                if st.button("🔄 Reload API Key", key="reload_api"):
                    from dotenv import load_dotenv
                    load_dotenv(override=True)
                    api_key = os.environ.get('GROQ_API_KEY')
                    if api_key:
                        st.session_state.api_key = api_key
                        # Test the API key immediately
                        test_success = self.test_api_connection_silent(api_key, st.session_state.selected_model)
                        if test_success:
                            st.session_state.api_connected = True
                            st.success("✅ API key reloaded and tested successfully!")
                        else:
                            st.error("❌ API key loaded but connection test failed")
                        st.rerun()
                    else:
                        st.error("No API key found in .env file")
                
                if st.button("🧪 Test API Connection", key="test_api"):
                    if st.session_state.api_key:
                        with st.spinner("Testing API connection..."):
                            success = self.test_api_connection_silent(st.session_state.api_key, st.session_state.selected_model)
                            if success:
                                st.session_state.api_connected = True
                                st.success("✅ API connection successful!")
                            else:
                                st.error("❌ API connection failed")
                    else:
                        st.error("No API key to test")
            
            # Check if API key is configured
            has_api_key = bool(st.session_state.api_key)
            
            if has_api_key:
                st.success("✅ API Key loaded from environment")
                
                # Model selection
                model = st.selectbox(
                    "AI Model",
                    [
                        "llama-3.3-70b-versatile",
                        "llama3-70b-8192",
                        "mixtral-8x7b-32768",
                        "gemma2-9b-it",
                        "qwen-qwq-32b",
                        "deepseek-r1-distill-llama-70b"
                    ],
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
                
            # History section
            st.markdown("---")
            st.markdown("## 📊 Analysis History")
            
            if st.button("🗂️ View History", key="view_history"):
                self.show_history()
                
            if st.button("🗑️ Clear History", key="clear_history"):
                if self.db_manager:
                    self.db_manager.clear_history(st.session_state.session_id)
                    st.success("History cleared!")
                else:
                    st.session_state.analysis_history = []
                    st.success("History cleared!")
                
    def generate_database_schema(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate database schema from DataFrame"""
        schema_parts = []
        table_name = "uploaded_data"
        
        # Start table definition
        schema_parts.append(f"CREATE TABLE {table_name} (")
        
        column_definitions = []
        for col in df.columns:
            # Clean column name (remove spaces, special chars)
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
                # Check if it's a short text field
                max_length = df[col].astype(str).str.len().max() if not df[col].empty else 50
                if max_length <= 50:
                    sql_type = f"VARCHAR({max(50, int(max_length))})"
                else:
                    sql_type = "TEXT"
            
            column_definitions.append(f"    {clean_col} {sql_type}")
        
        schema_parts.append(",\n".join(column_definitions))
        schema_parts.append(");")
        
        schema = "\n".join(schema_parts)
        
        # Add simple format for AI queries
        simple_schema = f"{table_name}(" + ", ".join([
            f"{col.replace(' ', '_').replace('-', '_').replace('.', '_')}" 
            for col in df.columns
        ]) + ")"
        
        return {
            "sql_schema": schema,
            "simple_schema": simple_schema
        }
    
    def test_api_connection_silent(self, api_key: str, model: str) -> bool:
        """Test API connection silently and return success status"""
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
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
                print("✅ API connection test successful")
            else:
                print(f"❌ API connection test failed: {response.status_code}")
            return success
        except Exception as e:
            print(f"❌ API connection test error: {e}")
            return False

    def render_data_upload(self):
        """Render data upload section"""
        st.markdown("## 📊 Data Upload & Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or JSON file",
            type=['csv', 'json'],
            help="Upload your data file for comprehensive analysis"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                st.session_state.uploaded_data = df
                
                # Generate and store database schema
                schema_info = self.generate_database_schema(df)
                st.session_state.data_schema = schema_info["simple_schema"]
                
                # Success message with file info
                st.success(f"✅ {uploaded_file.name} loaded successfully!")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Rows", f"{len(df):,}")
                with col2:
                    st.metric("📋 Columns", len(df.columns))
                with col3:
                    st.metric("💾 Size", f"{uploaded_file.size / 1024:.1f} KB")
                with col4:
                    st.metric("❓ Missing", f"{df.isnull().sum().sum():,}")
                
                # Database Schema Section
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
                
                # Multiple Visualizations
                self.create_multiple_visualizations(df)
                
                # Action buttons
                st.markdown("### 🚀 Analysis Actions")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔬 Complete EDA", key="eda_button", help="Comprehensive Exploratory Data Analysis"):
                        with st.spinner("Performing comprehensive EDA analysis..."):
                            st.session_state.eda_results = self.eda_analyzer.perform_complete_eda(df)
                            st.session_state.show_eda_results = True
                            st.session_state.show_ai_insights = False
                            st.session_state.show_advanced_analytics = False
                        
                with col2:
                    if st.button("🤖 AI Insights", key="ai_insights", help="Generate AI-powered insights"):
                        if st.session_state.api_key and len(st.session_state.api_key) > 10:  # Better API key check
                            with st.spinner("🤖 Generating AI insights..."):
                                try:
                                    summary = f"Dataset: {len(df)} rows, {len(df.columns)} columns, Missing: {df.isnull().sum().sum()}"
                                    prompt = f"Analyze this dataset and provide 3 key business insights: {summary}"
                                    st.session_state.ai_insights_text = self.make_api_call(st.session_state.selected_model, prompt)
                                    st.session_state.show_ai_insights = True
                                    st.session_state.show_eda_results = False
                                    st.session_state.show_advanced_analytics = False
                                except Exception as e:
                                    st.error(f"AI insights failed: {str(e)}")
                                    st.session_state.ai_insights_text = f"Error generating insights: {str(e)}"
                                    st.session_state.show_ai_insights = True
                        else:
                            st.error("❌ API key not found or invalid. Check the sidebar debug info.")
                            st.session_state.show_ai_insights = False
                
                with col3:
                    if st.button("📊 Advanced Analytics", key="advanced_analytics", help="Advanced statistical analysis"):
                        st.session_state.show_advanced_analytics = True
                        st.session_state.show_eda_results = False
                        st.session_state.show_ai_insights = False
                
                # Display results based on session state
                if st.session_state.show_eda_results and st.session_state.eda_results:
                    st.markdown("---")
                    st.markdown("## 🔬 EDA Results")
                    
                    # Simple EDA display
                    overview = st.session_state.eda_results.get('overview', {})
                    if overview:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Rows", f"{overview.get('total_rows', 0):,}")
                        with col2:
                            st.metric("Columns", overview.get('total_columns', 0))
                        with col3:
                            st.metric("Missing", f"{overview.get('missing_values_total', 0):,}")
                        with col4:
                            st.metric("Duplicates", f"{overview.get('duplicate_rows', 0):,}")
                    
                    # Show insights
                    insights = st.session_state.eda_results.get('insights', [])
                    if insights:
                        st.markdown("### 💡 Key Insights")
                        for insight in insights[:3]:
                            st.markdown(f"**{insight.get('title', 'Insight')}:** {insight.get('description', 'No description')}")
                
                elif st.session_state.show_ai_insights and st.session_state.ai_insights_text:
                    st.markdown("---")
                    st.markdown("## 🤖 AI Insights")
                    st.markdown(st.session_state.ai_insights_text)
                
                elif st.session_state.show_advanced_analytics:
                    st.markdown("---")
                    st.markdown("## 📊 Advanced Analytics")
                    
                    # Simple analytics
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.markdown("### 📈 Statistics")
                        st.dataframe(df[numeric_cols].describe())
                        
                        if len(numeric_cols) >= 2:
                            st.markdown("### 🔗 Correlations")
                            corr_matrix = df[numeric_cols].corr()
                            fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix")
                            st.plotly_chart(fig, use_container_width=True)
                
                # Data preview
                with st.expander("👀 Data Preview", expanded=False):
                    st.dataframe(df.head(100), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            # Show sample data option when no file is uploaded
            st.info("👆 Upload a CSV or JSON file to get started")
            
            if st.button("📋 Load Sample Data", help="Load sample sales data for testing"):
                sample_data = self.create_sample_data()
                st.session_state.uploaded_data = sample_data
                schema_info = self.generate_database_schema(sample_data)
                st.session_state.data_schema = schema_info["simple_schema"]
                st.success("✅ Sample data loaded!")
                st.rerun()
                
    def create_sample_data(self) -> pd.DataFrame:
        """Create sample sales data for demonstration"""
        np.random.seed(42)
        n_rows = 1000
        
        # Generate sample sales data
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
        
    def generate_ai_insights(self, df: pd.DataFrame):
        """Generate AI-powered insights about the data"""
        with st.spinner("🤖 Generating AI insights..."):
            try:
                # Prepare data summary for AI
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
                
                insights = self.make_api_call(st.session_state.selected_model, prompt)
                
                st.markdown("### 🤖 AI-Generated Insights")
                st.markdown(insights)
                
                # Save to history
                analysis_record = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "AI Insights",
                    "data_shape": df.shape,
                    "insights": insights,
                    "session_id": st.session_state.session_id
                }
                
                if self.db_manager:
                    self.db_manager.save_analysis(analysis_record)
                else:
                    st.session_state.analysis_history.append(analysis_record)
                
            except Exception as e:
                st.error(f"Failed to generate AI insights: {str(e)}")
                
    def show_advanced_analytics(self, df: pd.DataFrame):
        """Show advanced analytics options"""
        st.markdown("### 📊 Advanced Analytics")
        
        analytics_tabs = st.tabs(["📈 Statistical Summary", "🔍 Outlier Detection", "📊 Correlation Analysis"])
        
        with analytics_tabs[0]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.markdown("#### 📈 Statistical Summary")
                summary_stats = df[numeric_cols].describe()
                st.dataframe(summary_stats, use_container_width=True)
                
                # Skewness and kurtosis
                st.markdown("#### 📏 Distribution Metrics")
                dist_metrics = pd.DataFrame({
                    'Column': numeric_cols,
                    'Skewness': [df[col].skew() for col in numeric_cols],
                    'Kurtosis': [df[col].kurtosis() for col in numeric_cols]
                })
                st.dataframe(dist_metrics, use_container_width=True, hide_index=True)
            else:
                st.info("No numeric columns found for statistical analysis")
        
        with analytics_tabs[1]:
            self.perform_outlier_analysis(df)
            
        with analytics_tabs[2]:
            self.perform_correlation_analysis(df)
            
    def perform_outlier_analysis(self, df: pd.DataFrame):
        """Perform outlier analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.info("No numeric columns found for outlier analysis")
            return
            
        st.markdown("#### 🔍 Outlier Analysis")
        
        selected_col = st.selectbox("Select column for outlier analysis", numeric_cols)
        
        if selected_col:
            data = df[selected_col].dropna()
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Outliers", len(outliers))
            with col2:
                st.metric("Outlier Percentage", f"{len(outliers)/len(df)*100:.1f}%")
            
            # Visualization
            fig = go.Figure()
            
            # Normal points
            normal_data = df[~df.index.isin(outliers.index)]
            fig.add_trace(go.Scatter(
                x=normal_data.index,
                y=normal_data[selected_col],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=4)
            ))
            
            # Outliers
            if len(outliers) > 0:
                fig.add_trace(go.Scatter(
                    x=outliers.index,
                    y=outliers[selected_col],
                    mode='markers',
                    name='Outliers',
                    marker=dict(color='red', size=8, symbol='x')
                ))
            
            fig.update_layout(
                title=f'Outlier Detection: {selected_col}',
                xaxis_title='Index',
                yaxis_title=selected_col,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    def perform_correlation_analysis(self, df: pd.DataFrame):
        """Perform correlation analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for correlation analysis")
            return
            
        st.markdown("#### 🔗 Correlation Analysis")
        
        # Correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Heatmap
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Correlation Matrix",
                       color_continuous_scale="RdBu_r")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations
        st.markdown("#### 🔝 Strongest Correlations")
        
        # Get correlation pairs
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
            
            # Display top 10 correlations
            top_corr = corr_df.head(10)[['Variable 1', 'Variable 2', 'Correlation']]
            st.dataframe(top_corr, use_container_width=True, hide_index=True)

    def create_multiple_visualizations(self, df: pd.DataFrame):
        """Create multiple visualizations for the uploaded data"""
        st.markdown("### 📊 Data Visualizations")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create tabs for different visualization types
        viz_tabs = st.tabs(["📈 Overview", "📊 Distributions", "🔗 Relationships", "📋 Summary"])
        
        with viz_tabs[0]:  # Overview
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
        
        with viz_tabs[1]:  # Distributions
            if len(numeric_cols) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    selected_num_col = st.selectbox("Select numeric column for histogram", numeric_cols)
                    fig = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col2:
                    # Box plot
                    fig = px.box(df, y=selected_num_col, title=f"Box Plot of {selected_num_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Multiple histograms
                if len(numeric_cols) > 1:
                    st.markdown("#### All Numeric Distributions")
                    fig = make_subplots(
                        rows=(len(numeric_cols) + 1) // 2, 
                        cols=2,
                        subplot_titles=numeric_cols[:6]  # Limit to 6 for performance
                    )
                    
                    for i, col in enumerate(numeric_cols[:6]):
                        row = (i // 2) + 1
                        col_pos = (i % 2) + 1
                        fig.add_trace(
                            go.Histogram(x=df[col], name=col, showlegend=False),
                            row=row, col=col_pos
                        )
                    
                    fig.update_layout(height=300 * ((len(numeric_cols[:6]) + 1) // 2))
                    st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[2]:  # Relationships
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols, key="x_scatter")
                with col2:
                    y_col = st.selectbox("Y-axis", numeric_cols, key="y_scatter", 
                                       index=1 if len(numeric_cols) > 1 else 0)
                
                # Color by categorical if available
                color_col = None
                if categorical_cols:
                    color_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                    color_col = color_col if color_col != "None" else None
                
                # Scatter plot
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                               title=f"{x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Scatter matrix for first 4 numeric columns
                if len(numeric_cols) >= 3:
                    st.markdown("#### Scatter Matrix")
                    cols_for_matrix = numeric_cols[:4]
                    fig = px.scatter_matrix(df[cols_for_matrix], 
                                          title="Scatter Matrix (First 4 Numeric Columns)")
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[3]:  # Summary
            # Data summary table
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
            
            # Column information
            st.markdown("#### 📝 Column Information")
            col_info = []
            for col in df.columns:
                col_info.append({
                    "Column": col,
                    "Type": str(df[col].dtype),
                    "Non-Null": f"{df[col].count():,}",
                    "Null": f"{df[col].isnull().sum():,}",
                    "Unique": f"{df[col].nunique():,}",
                    "Sample Values": str(df[col].dropna().head(3).tolist())[:50] + "..."
                })
            
            col_info_df = pd.DataFrame(col_info)
            st.dataframe(col_info_df, use_container_width=True, hide_index=True)
            
    def test_api_connection(self, api_key: str, model: str) -> bool:
        """Test API connection"""
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Say 'Connection successful!' in exactly 3 words."}],
                    "temperature": 0.1,
                    "max_tokens": 50
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
            return False
            
    def perform_eda(self, df: pd.DataFrame):
        """Perform comprehensive EDA"""
        st.markdown("## 🔬 Comprehensive EDA Results")
        
        with st.spinner("Performing comprehensive analysis..."):
            eda_results = self.eda_analyzer.perform_complete_eda(df)
            
            # Save EDA to history
            eda_record = {
                "timestamp": datetime.now().isoformat(),
                "type": "EDA",
                "data_shape": df.shape,
                "results": "EDA analysis completed",
                "session_id": st.session_state.session_id
            }
            
            if self.db_manager:
                self.db_manager.save_analysis(eda_record)
            else:
                st.session_state.analysis_history.append(eda_record)
            
            # Display EDA results
            self.display_eda_results(eda_results)
            
    def display_eda_results(self, results: Dict):
        """Display EDA results"""
        tabs = st.tabs([
            "📊 Overview", 
            "📈 Distributions", 
            "🔗 Correlations", 
            "🎯 Insights",
            "📋 Data Quality"
        ])
        
        with tabs[0]:  # Overview
            self.display_overview(results['overview'])
            
        with tabs[1]:  # Distributions
            self.display_distributions(results['distributions'])
            
        with tabs[2]:  # Correlations
            self.display_correlations(results['correlations'])
            
        with tabs[3]:  # Insights
            self.display_insights(results['insights'])
            
        with tabs[4]:  # Data Quality
            self.display_data_quality(results['data_quality'])
            
    def display_overview(self, overview: Dict):
        """Display overview section"""
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{overview['total_rows']:,}")
        with col2:
            st.metric("Total Columns", overview['total_columns'])
        with col3:
            st.metric("Numeric Columns", overview['numeric_columns'])
        with col4:
            st.metric("Categorical Columns", overview['categorical_columns'])
            
        if 'summary_stats' in overview:
            st.markdown("### 📈 Summary Statistics")
            st.dataframe(overview['summary_stats'], use_container_width=True)
            
    def display_distributions(self, distributions: Dict):
        """Display distribution plots"""
        st.markdown("### 📈 Data Distributions")
        
        for chart_name, chart_data in distributions.items():
            st.plotly_chart(chart_data, use_container_width=True)
            
    def display_correlations(self, correlations: Dict):
        """Display correlation analysis"""
        st.markdown("### 🔗 Correlation Analysis")
        
        if 'heatmap' in correlations:
            st.plotly_chart(correlations['heatmap'], use_container_width=True)
            
        if 'top_correlations' in correlations:
            st.markdown("#### 🔝 Top Correlations")
            st.dataframe(correlations['top_correlations'], use_container_width=True)
            
    def display_insights(self, insights: List[Dict]):
        """Display AI-generated insights"""
        st.markdown("### 🎯 AI-Generated Insights")
        
        for i, insight in enumerate(insights):
            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #ffd700;">💡 {insight['title']}</h4>
                    <p>{insight['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
    def display_data_quality(self, quality: Dict):
        """Display data quality metrics"""
        st.markdown("### 📋 Data Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Missing Values")
            if 'missing_values' in quality:
                st.plotly_chart(quality['missing_values'], use_container_width=True)
                
        with col2:
            st.markdown("#### Data Types")
            if 'data_types' in quality:
                st.plotly_chart(quality['data_types'], use_container_width=True)
                
    def render_query_interface(self):
        """Render natural language query interface"""
        st.markdown("## 🚀 AI Query Interface")
        
        # Show current schema
        if st.session_state.data_schema:
            with st.expander("🗄️ Current Data Schema", expanded=False):
                st.code(st.session_state.data_schema, language="sql")
        
        # Natural Language Query input
        query_input = st.text_area(
            "Natural Language Query",
            value=st.session_state.example_query,
            placeholder="Example: Show me the top 10 customers by total sales amount",
            height=100,
            help="Describe what you want to analyze in plain English"
        )
        
        # Clear the example query after it's been used
        if st.session_state.example_query and query_input == st.session_state.example_query:
            st.session_state.example_query = ""
        
        # Analysis buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Check API key more thoroughly
            api_available = bool(st.session_state.api_key and len(st.session_state.api_key) > 10)
            analyze_disabled = not api_available or not query_input.strip()
            
            if st.button("🧠 Analyze Query", 
                        key="analyze_single", 
                        disabled=analyze_disabled,
                        help="Generate SQL and insights for your query"):
                if not api_available:
                    st.error("❌ API key required. Check sidebar debug info.")
                elif query_input.strip():
                    self.analyze_single_query(query_input.strip(), st.session_state.data_schema)
                else:
                    st.error("Please enter a query")
                    
        with col2:
            compare_disabled = not api_available or not query_input.strip()
            if st.button("⚔️ Model Battle", 
                        key="model_comparison", 
                        disabled=compare_disabled,
                        help="Compare multiple AI models on your query"):
                if not api_available:
                    st.error("❌ API key required. Check sidebar debug info.")
                elif query_input.strip():
                    # Store query and show model selection
                    st.session_state.current_query = query_input.strip()
                    st.session_state.show_model_selection = True
                    st.rerun()
                else:
                    st.error("Please enter a query")
        
        # Show connection status
        if not api_available:
            st.warning("⚠️ **AI Features Disabled**: API key not detected. Use the '🔄 Reload API Key' button in the sidebar.")
        else:
            st.success("✅ **AI Features Active**: Ready for natural language queries and model battles!")
        
        # Recent queries section (simplified)
        if hasattr(st.session_state, 'recent_queries') and st.session_state.recent_queries:
            with st.expander("📝 Recent Queries", expanded=False):
                for i, recent_query in enumerate(st.session_state.recent_queries[-5:]):  # Show last 5
                    if st.button(f"🔄 {recent_query[:60]}...", key=f"recent_{i}"):
                        st.session_state.example_query = recent_query
                        st.rerun()
        
        # Show model selection interface if activated
        if st.session_state.show_model_selection:
            self.show_model_selection_interface()
                    
    def analyze_single_query(self, query: str, schema: str = ""):
        """Analyze query with single model"""
        # Add to recent queries
        if query not in st.session_state.recent_queries:
            st.session_state.recent_queries.append(query)
            # Keep only last 10 queries
            st.session_state.recent_queries = st.session_state.recent_queries[-10:]
        
        with st.spinner(f"🧠 Analyzing with {st.session_state.get('selected_model', 'AI model')}..."):
            try:
                # Generate SQL
                sql_result = self.generate_sql(query, schema)
                
                # Generate insights
                insights_result = self.generate_insights(query, schema)
                
                # Save to history
                analysis_record = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "Single Query Analysis",
                    "query": query,
                    "schema": schema,
                    "sql_result": sql_result,
                    "insights": insights_result,
                    "model": st.session_state.get('selected_model'),
                    "session_id": st.session_state.session_id
                }
                
                if self.db_manager:
                    self.db_manager.save_analysis(analysis_record)
                else:
                    st.session_state.analysis_history.append(analysis_record)
                
                # Display results
                self.display_query_results(sql_result, insights_result)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                
    def generate_sql(self, query: str, schema: str = "") -> str:
        """Generate SQL from natural language"""
        schema_context = schema if schema else st.session_state.data_schema
        
        prompt = f"""Convert this natural language query to SQL:

Database Schema: {schema_context}

Natural Language Query: {query}

Instructions:
- Use the exact column names from the schema
- Generate clean, optimized SQL
- Include appropriate WHERE, GROUP BY, ORDER BY clauses
- Use proper SQL syntax
- Return only the SQL query without explanations

SQL Query:"""

        return self.make_api_call(st.session_state.selected_model, prompt)
        
    def generate_insights(self, query: str, schema: str = "") -> str:
        """Generate business insights"""
        schema_context = schema if schema else st.session_state.data_schema
        
        prompt = f"""Provide detailed business insights for this data analysis query:

Database Schema: {schema_context}

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

        return self.make_api_call(st.session_state.selected_model, prompt)
        
    def make_api_call(self, model: str, prompt: str) -> str:
        """Make API call to Groq"""
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
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
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"API error: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
            
    def display_query_results(self, sql_result: str, insights_result: str):
        """Display query analysis results"""
        st.markdown("## 🎯 Analysis Results")
        
        tabs = st.tabs(["🔍 SQL Query", "💡 AI Insights", "🔄 Execute Query"])
        
        with tabs[0]:
            st.markdown("### 🔍 Generated SQL Query")
            st.code(sql_result, language='sql')
            
            # Copy button simulation
            if st.button("📋 Copy SQL", key="copy_sql"):
                st.success("SQL copied to clipboard! (Use Ctrl+C to copy from the code block above)")
                
        with tabs[1]:
            st.markdown("### 💡 AI-Powered Business Insights")
            # Parse and display insights
            insights = self.parse_insights_improved(insights_result)
            
            for i, insight in enumerate(insights):
                with st.container():
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
                    self.execute_sql_on_data(sql_result, st.session_state.uploaded_data)
            else:
                st.info("Upload data first to execute SQL queries")
                
    def execute_sql_on_data(self, sql_query: str, df: pd.DataFrame):
        """Execute SQL query on the uploaded DataFrame"""
        try:
            import sqlite3
            import tempfile
            
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
                
                # Show results
                st.markdown("#### 📊 Query Results")
                st.dataframe(result_df, use_container_width=True)
                
                # Visualization if possible
                if len(result_df) > 0:
                    self.auto_visualize_results(result_df)
                
        except Exception as e:
            st.error(f"Error executing SQL: {str(e)}")
            st.info("💡 Tip: The AI-generated SQL might need adjustment for your specific data structure")
            
    def auto_visualize_results(self, result_df: pd.DataFrame):
        """Automatically create visualizations for query results"""
        if len(result_df) == 0:
            return
            
        st.markdown("#### 📈 Auto-Generated Visualization")
        
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Simple bar chart if we have one numeric column
        if len(numeric_cols) == 1 and len(result_df) <= 50:
            if len(result_df.columns) >= 2:
                # Use first non-numeric column as x-axis
                text_cols = result_df.select_dtypes(include=['object']).columns.tolist()
                if text_cols:
                    fig = px.bar(result_df, 
                               x=text_cols[0], 
                               y=numeric_cols[0],
                               title=f"{numeric_cols[0]} by {text_cols[0]}")
                    st.plotly_chart(fig, use_container_width=True)
                    
        # Line chart for time series-like data
        elif len(numeric_cols) >= 1 and len(result_df) > 10:
            fig = px.line(result_df, 
                         y=numeric_cols[0],
                         title=f"Trend: {numeric_cols[0]}")
            st.plotly_chart(fig, use_container_width=True)
            
    def parse_insights_improved(self, raw_insights: str) -> List[Dict]:
        """Parse insights from API response with improved formatting"""
        insights = []
        
        # Split by lines and look for patterns
        lines = raw_insights.strip().split('\n')
        
        current_insight = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for bold titles or numbered items
            if line.startswith('**') and line.endswith('**:'):
                if current_insight:
                    insights.append(current_insight)
                
                title = line.replace('**', '').replace(':', '').strip()
                current_insight = {'title': title, 'text': ''}
                
            elif line.startswith('**') and '**:' in line:
                if current_insight:
                    insights.append(current_insight)
                
                parts = line.split('**:', 1)
                title = parts[0].replace('**', '').strip()
                text = parts[1].strip() if len(parts) > 1 else ''
                current_insight = {'title': title, 'text': text}
                
            elif current_insight and line:
                # Continue building the current insight
                if current_insight['text']:
                    current_insight['text'] += ' ' + line
                else:
                    current_insight['text'] = line
                    
            elif line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                # Handle numbered or bulleted insights
                if current_insight:
                    insights.append(current_insight)
                
                # Extract title and text
                clean_line = line.lstrip('1234567890.-• ').strip()
                if ':' in clean_line:
                    parts = clean_line.split(':', 1)
                    title = parts[0].strip()
                    text = parts[1].strip()
                else:
                    title = f"Insight {len(insights) + 1}"
                    text = clean_line
                    
                current_insight = {'title': title, 'text': text}
        
        # Add the last insight
        if current_insight:
            insights.append(current_insight)
            
        # If no insights were parsed, create one from the raw text
        if not insights and raw_insights.strip():
            insights.append({
                'title': 'AI Analysis',
                'text': raw_insights.strip()
            })
            
        return insights[:5]  # Limit to 5 insights
                
    def run_model_comparison(self, query: str, schema: str = ""):
        """Run model comparison analysis"""
        # Double-check API key
        if not st.session_state.api_key or len(st.session_state.api_key) < 10:
            st.error("🔑 Valid API key required for model comparison feature")
            return
            
        models = [
            "llama-3.3-70b-versatile",
            "llama3-70b-8192", 
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        
        st.markdown("## ⚔️ Model Comparison Arena")
        st.markdown("*Comparing multiple AI models on your query...*")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, model in enumerate(models):
            status_text.text(f"Testing {model}...")
            progress_bar.progress((i + 1) / len(models))
            
            try:
                start_time = time.time()
                
                # Create a more focused prompt for model comparison
                comparison_prompt = f"""Analyze this query and provide SQL + business insight:

Schema: {schema if schema else st.session_state.data_schema}
Query: {query}

Respond in this exact format:
SQL: [your SQL query here]
INSIGHT: [your business insight here]

Keep response concise and focused."""

                response = self.make_api_call(model, comparison_prompt)
                response_time = time.time() - start_time
                
                # Score the response
                score = self.score_response(response, response_time)
                
                results.append({
                    'model': model,
                    'response': response,
                    'response_time': response_time * 1000,  # Convert to ms
                    'score': score,
                    'success': True
                })
                
            except Exception as e:
                results.append({
                    'model': model,
                    'response': f"Error: {str(e)}",
                    'response_time': 0,
                    'score': 0,
                    'success': False
                })
                
            # Add small delay between requests to avoid rate limiting
            time.sleep(1)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Save comparison to history
        comparison_record = {
            "timestamp": datetime.now().isoformat(),
            "type": "Model Comparison",
            "query": query,
            "schema": schema,
            "results": [
                {
                    'model': r['model'],
                    'score': r['score'],
                    'success': r['success'],
                    'response_time': r['response_time']
                } for r in results  # Store simplified version for history
            ],
            "session_id": st.session_state.session_id
        }
        
        if self.db_manager:
            self.db_manager.save_analysis(comparison_record)
        else:
            st.session_state.analysis_history.append(comparison_record)
        
        # Display comparison results
        self.display_comparison_results(results)
            
    def score_response(self, response: str, response_time: float) -> int:
        """Score the model response based on quality and speed"""
        try:
            # Quality scoring based on content
            response_lower = response.lower()
            
            # Check for SQL presence (40 points max)
            has_sql = any(keyword in response_lower for keyword in ['select', 'from', 'where', 'group by', 'order by'])
            sql_score = 40 if has_sql else 0
            
            # Check for insight presence (30 points max)
            has_insight = any(keyword in response_lower for keyword in ['insight', 'analysis', 'recommendation', 'business', 'trend'])
            insight_score = 30 if has_insight else 0
            
            # Response length and completeness (20 points max)
            length_score = min(len(response) / 20, 20)  # 1 point per 20 characters, max 20
            
            # Speed scoring (10 points max) - faster is better
            if response_time > 0:
                speed_score = max(0, 10 - (response_time * 2))  # Penalty for slow responses
            else:
                speed_score = 0
            
            # Calculate total score
            total_score = sql_score + insight_score + length_score + speed_score
            
            # Ensure score is between 0 and 100
            return max(0, min(100, round(total_score)))
            
        except Exception:
            return 0
        
    def display_comparison_results(self, results: List[Dict]):
        """Display model comparison results with enhanced formatting"""
        if not results:
            st.error("No results to display")
            return
            
        # Sort by score for ranking
        sorted_results = sorted([r for r in results if r['success']], key=lambda x: x['score'], reverse=True)
        failed_results = [r for r in results if not r['success']]
        
        # Winner summary
        if sorted_results:
            winner = sorted_results[0]
            fastest = min([r for r in results if r['success']], key=lambda x: x['response_time'], default=winner)
            
            # Create winner announcement
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #FFD700, #FFA500); padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #000; margin: 0;">🏆 WINNER</h3>
                    <h4 style="color: #000; margin: 5px 0;">{winner['model']}</h4>
                    <p style="color: #000; margin: 0;">Score: {winner['score']}/100</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #40E0D0, #48D1CC); padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #000; margin: 0;">⚡ FASTEST</h3>
                    <h4 style="color: #000; margin: 5px 0;">{fastest['model']}</h4>
                    <p style="color: #000; margin: 0;">{fastest['response_time']:.0f}ms</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                success_rate = len(sorted_results) / len(results) * 100
                st.markdown(f"""
                <div style="background: linear-gradient(45deg, #98FB98, #90EE90); padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: #000; margin: 0;">📊 SUCCESS</h3>
                    <h4 style="color: #000; margin: 5px 0;">{len(sorted_results)}/{len(results)}</h4>
                    <p style="color: #000; margin: 0;">{success_rate:.0f}% Success Rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
        
        # Detailed results in tabs
        if len(sorted_results) > 0:
            # Create tabs for each successful model
            tab_names = [f"{'🥇' if i == 0 else '🥈' if i == 1 else '🥉' if i == 2 else '📊'} {result['model']}" 
                        for i, result in enumerate(sorted_results)]
            
            if len(tab_names) > 0:
                tabs = st.tabs(tab_names)
                
                for i, (tab, result) in enumerate(zip(tabs, sorted_results)):
                    with tab:
                        # Performance metrics
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.metric("🏆 Rank", f"#{i+1}")
                        with metric_col2:
                            st.metric("📊 Score", f"{result['score']}/100")
                        with metric_col3:
                            st.metric("⚡ Speed", f"{result['response_time']:.0f}ms")
                        with metric_col4:
                            quality = "Excellent" if result['score'] >= 80 else "Good" if result['score'] >= 60 else "Fair" if result['score'] >= 40 else "Poor"
                            st.metric("🎯 Quality", quality)
                        
                        st.markdown("---")
                        
                        # Model response
                        st.markdown("### 📝 Response")
                        
                        # Try to parse SQL and Insight separately
                        response = result['response']
                        
                        if "SQL:" in response and "INSIGHT:" in response:
                            parts = response.split("INSIGHT:")
                            sql_part = parts[0].replace("SQL:", "").strip()
                            insight_part = parts[1].strip()
                            
                            st.markdown("**🔍 Generated SQL:**")
                            st.code(sql_part, language='sql')
                            
                            st.markdown("**💡 Business Insight:**")
                            st.markdown(insight_part)
                        else:
                            # Show full response if parsing fails
                            st.markdown("**📄 Full Response:**")
                            st.text_area("", response, height=200, key=f"response_full_{i}")
        
        # Show failed models if any
        if failed_results:
            st.markdown("---")
            st.markdown("### ❌ Failed Models")
            
            for result in failed_results:
                with st.expander(f"❌ {result['model']} - Failed"):
                    st.error(f"Error: {result['response']}")
        
        # Add comparison summary
        if len(sorted_results) > 1:
            st.markdown("---")
            st.markdown("### 📈 Performance Comparison")
            
            # Create comparison chart
            import plotly.graph_objects as go
            
            models = [r['model'].replace('-', '<br>') for r in sorted_results]
            scores = [r['score'] for r in sorted_results]
            times = [r['response_time'] for r in sorted_results]
            
            fig = go.Figure()
            
            # Add score bars
            fig.add_trace(go.Bar(
                name='Score',
                x=models,
                y=scores,
                yaxis='y',
                marker_color='#FFD700'
            ))
            
            # Add response time line
            fig.add_trace(go.Scatter(
                name='Response Time (ms)',
                x=models,
                y=times,
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=8)
            ))
            
            # Update layout for dual y-axis
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis=dict(title='Models'),
                yaxis=dict(title='Score (0-100)', side='left'),
                yaxis2=dict(title='Response Time (ms)', side='right', overlaying='y'),
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
                
    def show_history(self):
        """Show analysis history"""
        st.markdown("## 📊 Analysis History")
        
        if self.db_manager:
            history = self.db_manager.get_history(st.session_state.session_id)
        else:
            history = st.session_state.analysis_history
        
        if not history:
            st.info("No analysis history found.")
            return
            
        for i, record in enumerate(reversed(history[-10:])):  # Show last 10 records
            with st.expander(f"{record['type']} - {record['timestamp'][:19]}"):
                if record['type'] == 'Single Query Analysis':
                    st.markdown(f"**Query:** {record['query']}")
                    if record.get('schema'):
                        st.markdown(f"**Schema:** {record['schema']}")
                    st.markdown("**SQL Result:**")
                    st.code(record['sql_result'], language='sql')
                    
                elif record['type'] == 'Model Comparison':
                    st.markdown(f"**Query:** {record['query']}")
                    st.markdown("**Results:**")
                    for result in record['results']:
                        st.markdown(f"- {result['model']}: {result['score']}/100")
                        
                elif record['type'] == 'EDA':
                    st.markdown(f"**Data Shape:** {record['data_shape']}")
                    st.markdown("**Analysis completed successfully**")
    
    def show_statistical_summary(self, df: pd.DataFrame):
        """Show statistical summary without API"""
        st.markdown("### 📊 Statistical Summary")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            # Additional stats
            st.markdown("#### 📈 Additional Statistics")
            additional_stats = pd.DataFrame({
                'Column': numeric_cols,
                'Skewness': [df[col].skew() for col in numeric_cols],
                'Kurtosis': [df[col].kurtosis() for col in numeric_cols],
                'Missing %': [df[col].isnull().sum() / len(df) * 100 for col in numeric_cols]
            })
            st.dataframe(additional_stats, use_container_width=True, hide_index=True)
        else:
            st.info("No numeric columns found for statistical analysis")
    
    def show_outlier_detection(self, df: pd.DataFrame):
        """Show outlier detection without API"""
        st.markdown("### 🔍 Outlier Detection")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for outlier analysis", numeric_cols, key="outlier_col")
            
            if selected_col:
                data = df[selected_col].dropna()
                
                # IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Outliers", len(outliers))
                with col2:
                    st.metric("Outlier %", f"{len(outliers)/len(df)*100:.1f}%")
                with col3:
                    st.metric("IQR", f"{IQR:.2f}")
                
                # Show outlier values
                if len(outliers) > 0:
                    st.markdown("#### 🎯 Outlier Values")
                    st.dataframe(outliers[[selected_col]].head(20), use_container_width=True)
        else:
            st.info("No numeric columns found for outlier analysis")
    
    def show_correlation_matrix(self, df: pd.DataFrame):
        """Show correlation matrix without API"""
        st.markdown("### 🔗 Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            
            # Heatmap
            fig = px.imshow(corr_matrix, 
                           text_auto=True, 
                           aspect="auto",
                           title="Correlation Matrix",
                           color_continuous_scale="RdBu_r")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Strong correlations
            st.markdown("#### 🔝 Strongest Correlations")
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
                st.dataframe(corr_df.head(10)[['Variable 1', 'Variable 2', 'Correlation']], 
                           use_container_width=True, hide_index=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")
    
    def show_model_selection_interface(self):
        """Show persistent model selection interface"""
        st.markdown("---")
        st.markdown("## ⚔️ Model Battle Setup")
        st.markdown(f"**Query:** {st.session_state.current_query}")
        
        # Available models
        available_models = {
            "llama-3.3-70b-versatile": "Llama 3.3 70B (Most Advanced)",
            "llama3-70b-8192": "Llama 3 70B (Reliable)",
            "mixtral-8x7b-32768": "Mixtral 8x7B (Fast & Efficient)",
            "gemma2-9b-it": "Gemma 2 9B (Lightweight)",
            "qwen-qwq-32b": "Qwen QwQ 32B (Reasoning)",
            "deepseek-r1-distill-llama-70b": "DeepSeek R1 70B (Specialized)"
        }
        
        st.markdown("### 🎯 Select Models for Battle")
        
        # Model selection with checkboxes - using session state keys
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🚀 High-Performance Models:**")
            llama33_selected = st.checkbox("Llama 3.3 70B (Most Advanced)", key="battle_llama33", value=True)
            llama3_selected = st.checkbox("Llama 3 70B (Reliable)", key="battle_llama3", value=True)
            deepseek_selected = st.checkbox("DeepSeek R1 70B (Specialized)", key="battle_deepseek", value=False)
                
        with col2:
            st.markdown("**⚡ Fast & Efficient Models:**")
            mixtral_selected = st.checkbox("Mixtral 8x7B (Fast & Efficient)", key="battle_mixtral", value=True)
            gemma_selected = st.checkbox("Gemma 2 9B (Lightweight)", key="battle_gemma", value=False)
            qwen_selected = st.checkbox("Qwen QwQ 32B (Reasoning)", key="battle_qwen", value=False)
        
        # Build selected models list
        selected_models = []
        if llama33_selected:
            selected_models.append("llama-3.3-70b-versatile")
        if llama3_selected:
            selected_models.append("llama3-70b-8192")
        if deepseek_selected:
            selected_models.append("deepseek-r1-distill-llama-70b")
        if mixtral_selected:
            selected_models.append("mixtral-8x7b-32768")
        if gemma_selected:
            selected_models.append("gemma2-9b-it")
        if qwen_selected:
            selected_models.append("qwen-qwq-32b")
        
        # Show selection summary
        if selected_models:
            st.success(f"✅ **Selected Models:** {len(selected_models)} models ready for battle")
            
            # Battle configuration
            col1, col2, col3 = st.columns(3)
            with col1:
                test_rounds = st.selectbox("Test Rounds", [1, 2, 3], index=0, help="Number of times to test each model")
            with col2:
                timeout_seconds = st.selectbox("Timeout (seconds)", [10, 20, 30], index=1, help="Max time to wait for each response")
            with col3:
                # Cancel button
                if st.button("❌ Cancel", key="cancel_battle"):
                    st.session_state.show_model_selection = False
                    st.rerun()
            
            # Start battle button
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🚀 Start Model Battle", key="start_battle_persist", type="primary"):
                    st.session_state.show_model_selection = False  # Hide selection interface
                    self.run_model_comparison_with_selection(
                        st.session_state.current_query, 
                        st.session_state.data_schema, 
                        selected_models, 
                        test_rounds, 
                        timeout_seconds
                    )
        else:
            st.warning("⚠️ Please select at least one model for the battle")
            
            # Cancel button for when no models selected
            if st.button("❌ Cancel", key="cancel_battle_no_models"):
                st.session_state.show_model_selection = False
                st.rerun()
    
    def show_model_selection_and_run(self, query: str, schema: str = ""):
        """Legacy method - now redirects to persistent interface"""
        st.session_state.current_query = query
        st.session_state.show_model_selection = True
    
    def run_model_comparison_with_selection(self, query: str, schema: str, selected_models: list, rounds: int, timeout: int):
        """Run model comparison with selected models"""
        if not selected_models:
            st.error("No models selected")
            return
            
        # Double-check API key
        if not st.session_state.api_key or len(st.session_state.api_key) < 10:
            st.error("🔑 Valid API key required for model comparison feature")
            return
        
        st.markdown("## ⚔️ Model Battle Arena")
        st.markdown(f"*Testing {len(selected_models)} models with {rounds} round(s) each...*")
        
        # Create progress bar
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
                    
                    # Create a focused prompt
                    comparison_prompt = f"""Analyze this query and provide SQL + business insight:

Schema: {schema if schema else st.session_state.data_schema}
Query: {query}

Respond in this exact format:
SQL: [your SQL query here]
INSIGHT: [your business insight here]

Keep response concise and focused."""

                    response = self.make_api_call_with_timeout(model, comparison_prompt, timeout)
                    response_time = time.time() - start_time
                    
                    # Score the response
                    score = self.score_response(response, response_time)
                    
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
                
                # Small delay between requests
                time.sleep(0.5)
            
            # Calculate average performance for this model
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
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display enhanced results
        self.display_enhanced_comparison_results(results, query)
    
    def make_api_call_with_timeout(self, model: str, prompt: str, timeout: int) -> str:
        """Make API call with custom timeout"""
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
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
    
    def display_enhanced_comparison_results(self, results: List[Dict], query: str):
        """Display enhanced comparison results with multiple rounds"""
        if not results:
            st.error("No results to display")
            return
        
        # Sort by average score
        sorted_results = sorted([r for r in results if r['success']], key=lambda x: x['avg_score'], reverse=True)
        failed_results = [r for r in results if not r['success']]
        
        # Enhanced winner announcement
        if sorted_results:
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
            
            st.markdown("---")
            
            # Performance comparison chart
            st.markdown("### 📊 Performance Comparison")
            
            models = [r['model'].replace('-', ' ').replace('versatile', '').replace('8192', '').title() for r in sorted_results]
            scores = [r['avg_score'] for r in sorted_results]
            times = [r['avg_response_time'] for r in sorted_results]
            
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
                title=f'Model Performance: "{query[:50]}..."',
                xaxis=dict(title='Models'),
                yaxis=dict(title='Score (0-100)', side='left'),
                yaxis2=dict(title='Response Time (ms)', side='right', overlaying='y'),
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            st.markdown("### 📋 Detailed Results")
            for i, result in enumerate(sorted_results):
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
                    best_response = result['best_response']
                    
                    if "SQL:" in best_response and "INSIGHT:" in best_response:
                        parts = best_response.split("INSIGHT:")
                        sql_part = parts[0].replace("SQL:", "").strip()
                        insight_part = parts[1].strip()
                        
                        st.markdown("**SQL:**")
                        st.code(sql_part, language='sql')
                        st.markdown("**Insight:**")
                        st.markdown(insight_part)
                    else:
                        st.text_area("", best_response, height=150, key=f"best_response_{i}")
        
        # Show failed models
        if failed_results:
            st.markdown("### ❌ Failed Models")
            for result in failed_results:
                st.error(f"**{result['model']}**: All attempts failed")
    
    def perform_eda_inline(self, df: pd.DataFrame):
        """Perform comprehensive EDA and display results inline"""
        st.markdown("---")
        st.markdown("## 🔬 Comprehensive EDA Results")
        
        try:
            eda_results = self.eda_analyzer.perform_complete_eda(df)
            
            # Save EDA to history
            eda_record = {
                "timestamp": datetime.now().isoformat(),
                "type": "EDA",
                "data_shape": df.shape,
                "results": "EDA analysis completed",
                "session_id": st.session_state.session_id
            }
            
            if self.db_manager:
                self.db_manager.save_analysis(eda_record)
            else:
                st.session_state.analysis_history.append(eda_record)
            
            # Display EDA results inline
            self.display_eda_results_inline(eda_results)
            
        except Exception as e:
            st.error(f"EDA analysis failed: {str(e)}")
    
    def display_eda_results_inline(self, results: Dict):
        """Display EDA results inline without navigation"""
        # Create tabs for different sections
        eda_tabs = st.tabs([
            "📊 Overview", 
            "📈 Distributions", 
            "🔗 Correlations", 
            "🎯 Insights",
            "📋 Data Quality"
        ])
        
        with eda_tabs[0]:  # Overview
            st.markdown("### 📊 Dataset Overview")
            overview = results.get('overview', {})
            
            if overview:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", f"{overview.get('total_rows', 0):,}")
                with col2:
                    st.metric("Total Columns", overview.get('total_columns', 0))
                with col3:
                    st.metric("Numeric Columns", overview.get('numeric_columns', 0))
                with col4:
                    st.metric("Categorical Columns", overview.get('categorical_columns', 0))
                
                # Additional metrics
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
            
        with eda_tabs[1]:  # Distributions
            st.markdown("### 📈 Data Distributions")
            distributions = results.get('distributions', {})
            
            if distributions:
                for chart_name, chart_data in distributions.items():
                    if hasattr(chart_data, 'update_layout'):  # Check if it's a plotly figure
                        st.plotly_chart(chart_data, use_container_width=True)
                    else:
                        st.write(f"Chart: {chart_name}")
            else:
                st.info("No distribution charts available")
            
        with eda_tabs[2]:  # Correlations
            st.markdown("### 🔗 Correlation Analysis")
            correlations = results.get('correlations', {})
            
            if 'heatmap' in correlations:
                st.plotly_chart(correlations['heatmap'], use_container_width=True)
                
            if 'top_correlations' in correlations:
                st.markdown("#### 🔝 Top Correlations")
                st.dataframe(correlations['top_correlations'], use_container_width=True)
                
            if 'scatter_matrix' in correlations:
                st.plotly_chart(correlations['scatter_matrix'], use_container_width=True)
            
        with eda_tabs[3]:  # Insights
            st.markdown("### 🎯 Generated Insights")
            insights = results.get('insights', [])
            
            if insights:
                for i, insight in enumerate(insights):
                    with st.container():
                        st.markdown(f"""
                        <div style="background: rgba(15, 15, 15, 0.95); border: 1px solid rgba(255, 215, 0, 0.3); border-radius: 15px; padding: 20px; margin: 10px 0;">
                            <h4 style="color: #ffd700;">💡 {insight.get('title', f'Insight {i+1}')}</h4>
                            <p>{insight.get('description', 'No description available')}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No insights generated")
                
        with eda_tabs[4]:  # Data Quality
            st.markdown("### 📋 Data Quality Assessment")
            quality = results.get('data_quality', {})
            
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
            
            # Show duplicates info if available
            if 'duplicates' in quality:
                st.markdown("#### Duplicate Analysis")
                dup_info = quality['duplicates']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Duplicate Count", dup_info.get('count', 0))
                with col2:
                    st.metric("Duplicate %", f"{dup_info.get('percentage', 0):.1f}%")
    
    def generate_ai_insights_inline(self, df: pd.DataFrame):
        """Generate AI-powered insights and display inline"""
        st.markdown("---")
        st.markdown("## 🤖 AI-Generated Insights")
        
        try:
            # Prepare data summary for AI
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
            
            insights = self.make_api_call(st.session_state.selected_model, prompt)
            
            # Display insights in a nice format
            st.markdown("### 💡 Business Intelligence Report")
            
            # Parse insights and display
            parsed_insights = self.parse_insights_improved(insights)
            
            for i, insight in enumerate(parsed_insights):
                st.markdown(f"""
                <div style="background: rgba(15, 15, 15, 0.95); border: 1px solid rgba(255, 215, 0, 0.3); border-radius: 15px; padding: 20px; margin: 10px 0;">
                    <h4 style="color: #ffd700;">💡 {insight['title']}</h4>
                    <p>{insight['text']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Save to history
            analysis_record = {
                "timestamp": datetime.now().isoformat(),
                "type": "AI Insights",
                "data_shape": df.shape,
                "insights": insights,
                "session_id": st.session_state.session_id
            }
            
            if self.db_manager:
                self.db_manager.save_analysis(analysis_record)
            else:
                st.session_state.analysis_history.append(analysis_record)
            
        except Exception as e:
            st.error(f"Failed to generate AI insights: {str(e)}")
    
    def show_advanced_analytics_inline(self, df: pd.DataFrame):
        """Show advanced analytics inline"""
        st.markdown("---")
        st.markdown("## 📊 Advanced Analytics")
        
        analytics_tabs = st.tabs(["📈 Statistical Summary", "🔍 Outlier Detection", "📊 Correlation Analysis", "🎯 Distribution Analysis"])
        
        with analytics_tabs[0]:
            self.show_statistical_summary_advanced(df)
            
        with analytics_tabs[1]:
            self.show_outlier_analysis_advanced(df)
            
        with analytics_tabs[2]:
            self.show_correlation_analysis_advanced(df)
            
        with analytics_tabs[3]:
            self.show_distribution_analysis_advanced(df)
    
    def show_statistical_summary_advanced(self, df: pd.DataFrame):
        """Advanced statistical summary"""
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
            
            # Statistical interpretation
            st.markdown("#### 🎯 Statistical Interpretation")
            for col in numeric_cols[:3]:  # Limit to first 3 columns
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
    
    def show_outlier_analysis_advanced(self, df: pd.DataFrame):
        """Advanced outlier analysis"""
        st.markdown("### 🔍 Advanced Outlier Detection")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Outlier summary for all columns
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
                
                # Detailed analysis for selected column
                st.markdown("#### 🎯 Detailed Outlier Analysis")
                selected_col = st.selectbox("Select column for detailed analysis", numeric_cols, key="detailed_outlier")
                
                if selected_col:
                    data = df[selected_col].dropna()
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Outliers", len(outliers))
                    with col2:
                        st.metric("Outlier %", f"{len(outliers)/len(df)*100:.1f}%")
                    with col3:
                        st.metric("IQR", f"{IQR:.2f}")
                    with col4:
                        st.metric("Range", f"{upper_bound - lower_bound:.2f}")
                    
                    # Visualization
                    fig = go.Figure()
                    
                    # Box plot
                    fig.add_trace(go.Box(
                        y=data,
                        name=selected_col,
                        boxpoints='outliers'
                    ))
                    
                    fig.update_layout(
                        title=f'Box Plot with Outliers: {selected_col}',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show actual outlier values
                    if len(outliers) > 0:
                        st.markdown("#### 📋 Outlier Values")
                        st.dataframe(outliers[[selected_col]].head(20), use_container_width=True)
        else:
            st.info("No numeric columns found for outlier analysis")
    
    def show_correlation_analysis_advanced(self, df: pd.DataFrame):
        """Advanced correlation analysis"""
        st.markdown("### 📊 Advanced Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            # Correlation matrix
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
                
                # Strong correlations highlight
                strong_corr = corr_df[corr_df['Abs Correlation'] >= 0.6]
                if len(strong_corr) > 0:
                    st.markdown("#### ⚠️ Strong Correlations (>0.6)")
                    st.dataframe(strong_corr[['Variable 1', 'Variable 2', 'Correlation']], 
                               use_container_width=True, hide_index=True)
                    st.warning("Strong correlations may indicate multicollinearity issues in modeling.")
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")
    
    def show_distribution_analysis_advanced(self, df: pd.DataFrame):
        """Advanced distribution analysis"""
        st.markdown("### 🎯 Advanced Distribution Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            st.markdown("#### 📊 Numeric Distribution Analysis")
            
            selected_num_col = st.selectbox("Select numeric column", numeric_cols, key="dist_numeric")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram with normal curve overlay
                data = df[selected_num_col].dropna()
                
                fig = go.Figure()
                
                # Histogram
                fig.add_trace(go.Histogram(
                    x=data,
                    name='Distribution',
                    opacity=0.7,
                    nbinsx=30
                ))
                
                fig.update_layout(
                    title=f'Distribution of {selected_num_col}',
                    xaxis_title=selected_num_col,
                    yaxis_title='Frequency',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # QQ plot approximation using percentiles
                fig = go.Figure()
                
                # Box plot
                fig.add_trace(go.Box(
                    y=data,
                    name=selected_num_col,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ))
                
                fig.update_layout(
                    title=f'Box Plot of {selected_num_col}',
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
            
            selected_cat_col = st.selectbox("Select categorical column", categorical_cols, key="dist_categorical")
            
            value_counts = df[selected_cat_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig = px.bar(
                    x=value_counts.index[:15],  # Top 15
                    y=value_counts.values[:15],
                    title=f'Top Categories in {selected_cat_col}'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Pie chart
                fig = px.pie(
                    values=value_counts.values[:10],  # Top 10
                    names=value_counts.index[:10],
                    title=f'Distribution of {selected_cat_col}'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Category statistics
            st.markdown("#### 📊 Category Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Unique Categories", df[selected_cat_col].nunique())
            with col2:
                st.metric("Most Frequent", value_counts.index[0])
            with col3:
                st.metric("Frequency", value_counts.iloc[0])
            with col4:
                st.metric("Missing Values", df[selected_cat_col].isnull().sum())
                    
    def run(self):
        """Main application runner"""
        self.render_header()
        
        # Render sidebar configuration
        self.render_api_config()
        
        # Show API key warning if not configured, but don't block the app
        if not st.session_state.api_key:
            with st.expander("🔑 API Key Configuration (Optional for Basic Features)", expanded=False):
                st.warning("**AI-powered features require a Groq API key**")
                st.markdown("""
                **For local development:**
                1. Create a `.env` file in your project directory
                2. Add: `GROQ_API_KEY=your_api_key_here`
                3. Restart the application
                
                **For Streamlit Cloud:**
                1. Go to your app settings
                2. Add to secrets: `GROQ_API_KEY = "your_api_key_here"`
                3. Redeploy the app
                
                **Get your API key:** [Groq Console](https://console.groq.com/keys)
                
                **Available without API key:**
                - Data upload and preview
                - Statistical analysis
                - Data visualizations
                - Correlation analysis
                - Outlier detection
                - Complete EDA reports
                """)
        
        # Main content area - always show regardless of API key status
        if st.session_state.uploaded_data is not None:
            # Show query interface when data is loaded
            st.markdown("---")
            self.render_query_interface()
        else:
            # Show data upload when no data is loaded
            self.render_data_upload()

def main():
    """Main function"""
    app = NeuralDataAnalyst()
    app.run()

if __name__ == "__main__":
    main()