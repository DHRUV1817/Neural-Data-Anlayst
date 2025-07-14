import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import base64
from io import BytesIO

# Additional advanced features for Neural Data Analyst

class AdvancedFeatures:
    """Advanced features and utilities for the Neural Data Analyst"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def render_advanced_analytics_dashboard(self, df: pd.DataFrame):
        """Render advanced analytics dashboard"""
        st.markdown("## 🔬 Advanced Analytics Dashboard")
        
        tabs = st.tabs([
            "📊 Interactive Plots", 
            "🎯 Smart Recommendations", 
            "📈 Trend Analysis",
            "🔍 Anomaly Detection",
            "📋 Report Generator"
        ])
        
        with tabs[0]:
            self.render_interactive_plots(df)
            
        with tabs[1]:
            self.render_smart_recommendations(df)
            
        with tabs[2]:
            self.render_trend_analysis(df)
            
        with tabs[3]:
            self.render_anomaly_detection(df)
            
        with tabs[4]:
            self.render_report_generator(df)
            
    def render_interactive_plots(self, df: pd.DataFrame):
        """Render interactive plotting interface"""
        st.markdown("### 📊 Interactive Plot Builder")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            plot_type = st.selectbox(
                "Plot Type",
                ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin", "Heatmap", "3D Scatter"]
            )
            
        with col2:
            x_column = st.selectbox("X-axis", df.columns)
            
        with col3:
            y_column = st.selectbox("Y-axis", df.columns)
            
        # Color and size options
        col1, col2 = st.columns(2)
        with col1:
            color_column = st.selectbox("Color by", ["None"] + list(df.columns))
        with col2:
            size_column = st.selectbox("Size by", ["None"] + list(df.select_dtypes(include=[np.number]).columns))
            
        # Generate plot based on selections
        if st.button("🎨 Generate Plot"):
            fig = self.create_dynamic_plot(df, plot_type, x_column, y_column, color_column, size_column)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
        # Plot gallery
        with st.expander("🖼️ Quick Plot Gallery"):
            self.render_plot_gallery(df)
            
    def create_dynamic_plot(self, df: pd.DataFrame, plot_type: str, x_col: str, y_col: str, 
                           color_col: str = None, size_col: str = None):
        """Create dynamic plot based on user selections"""
        try:
            kwargs = {
                'data_frame': df,
                'x': x_col,
                'title': f'{plot_type} Plot: {x_col} vs {y_col}'
            }
            
            if y_col and y_col != x_col:
                kwargs['y'] = y_col
                
            if color_col and color_col != "None":
                kwargs['color'] = color_col
                
            if size_col and size_col != "None" and plot_type in ["Scatter", "3D Scatter"]:
                kwargs['size'] = size_col
                
            if plot_type == "Scatter":
                fig = px.scatter(**kwargs)
            elif plot_type == "Line":
                fig = px.line(**kwargs)
            elif plot_type == "Bar":
                fig = px.bar(**kwargs)
            elif plot_type == "Histogram":
                fig = px.histogram(df, x=x_col, title=f'Histogram: {x_col}')
            elif plot_type == "Box":
                fig = px.box(**kwargs)
            elif plot_type == "Violin":
                fig = px.violin(**kwargs)
            elif plot_type == "3D Scatter":
                z_col = st.selectbox("Z-axis", df.select_dtypes(include=[np.number]).columns)
                kwargs['z'] = z_col
                fig = px.scatter_3d(**kwargs)
            elif plot_type == "Heatmap":
                numeric_df = df.select_dtypes(include=[np.number])
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")
            else:
                return None
                
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating plot: {str(e)}")
            return None
            
    def render_plot_gallery(self, df: pd.DataFrame):
        """Render quick plot gallery"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Quick correlation plot
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                               title="Quick Correlation View")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Quick distribution plot
                fig = px.histogram(df, x=numeric_cols[0], title="Quick Distribution")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
    def render_smart_recommendations(self, df: pd.DataFrame):
        """Render smart analysis recommendations"""
        st.markdown("### 🎯 Smart Analysis Recommendations")
        
        recommendations = self.generate_analysis_recommendations(df)
        
        for i, rec in enumerate(recommendations):
            with st.expander(f"💡 {rec['title']}", expanded=i == 0):
                st.markdown(f"**Recommendation:** {rec['description']}")
                st.markdown(f"**Rationale:** {rec['rationale']}")
                
                if st.button(f"Apply Recommendation", key=f"apply_rec_{i}"):
                    self.apply_recommendation(df, rec)
                    
    def generate_analysis_recommendations(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Generate smart analysis recommendations"""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Missing data recommendation
        missing_data = df.isnull().sum()
        high_missing = missing_data[missing_data > len(df) * 0.1]
        
        if len(high_missing) > 0:
            recommendations.append({
                'title': 'Missing Data Analysis',
                'description': f'Analyze missing data patterns in {len(high_missing)} columns with >10% missing values',
                'rationale': 'Understanding missing data patterns can reveal data collection issues or systematic biases',
                'action': 'missing_analysis'
            })
            
        # Correlation analysis recommendation
        if len(numeric_cols) > 2:
            recommendations.append({
                'title': 'Correlation Deep Dive',
                'description': 'Perform comprehensive correlation analysis with feature selection recommendations',
                'rationale': 'Identifying highly correlated features can improve model performance and interpretability',
                'action': 'correlation_analysis'
            })
            
        # Outlier detection recommendation
        if len(numeric_cols) > 0:
            recommendations.append({
                'title': 'Outlier Detection & Treatment',
                'description': 'Identify and analyze outliers using multiple statistical methods',
                'rationale': 'Outliers can significantly impact analysis results and model performance',
                'action': 'outlier_analysis'
            })
            
        # Segmentation recommendation
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            recommendations.append({
                'title': 'Customer/Data Segmentation',
                'description': 'Perform clustering analysis to identify natural data segments',
                'rationale': 'Segmentation can reveal hidden patterns and improve targeted strategies',
                'action': 'segmentation_analysis'
            })
            
        # Time series recommendation
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            recommendations.append({
                'title': 'Time Series Analysis',
                'description': 'Analyze temporal patterns and trends in your data',
                'rationale': 'Time-based analysis can reveal seasonality, trends, and forecasting opportunities',
                'action': 'time_series_analysis'
            })
            
        return recommendations
        
    def apply_recommendation(self, df: pd.DataFrame, recommendation: Dict[str, str]):
        """Apply a smart recommendation"""
        action = recommendation.get('action')
        
        if action == 'missing_analysis':
            self.perform_missing_analysis(df)
        elif action == 'correlation_analysis':
            self.perform_correlation_analysis(df)
        elif action == 'outlier_analysis':
            self.perform_outlier_analysis(df)
        elif action == 'segmentation_analysis':
            self.perform_segmentation_analysis(df)
        elif action == 'time_series_analysis':
            self.perform_time_series_analysis(df)
            
    def perform_missing_analysis(self, df: pd.DataFrame):
        """Perform detailed missing data analysis"""
        st.markdown("#### 🔍 Missing Data Analysis Results")
        
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percent.values
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        
        if len(missing_df) > 0:
            fig = px.bar(missing_df, x='Column', y='Missing_Percentage',
                        title='Missing Data by Column (%)')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing data found in the dataset!")
            
    def perform_correlation_analysis(self, df: pd.DataFrame):
        """Perform detailed correlation analysis"""
        st.markdown("#### 🔗 Advanced Correlation Analysis")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            # Hierarchical clustering of correlations
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
            
            distance_matrix = 1 - np.abs(corr_matrix)
            condensed_distances = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_distances, method='average')
            
            fig = go.Figure()
            dendro = dendrogram(linkage_matrix, labels=corr_matrix.columns, no_plot=True)
            
            # Create dendrogram plot
            for i in range(len(dendro['icoord'])):
                x = dendro['icoord'][i]
                y = dendro['dcoord'][i]
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', 
                                       line=dict(color='gold', width=2),
                                       showlegend=False))
                                       
            fig.update_layout(
                title="Feature Clustering Dendrogram",
                xaxis_title="Features",
                yaxis_title="Distance",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    def render_trend_analysis(self, df: pd.DataFrame):
        """Render trend analysis interface"""
        st.markdown("### 📈 Trend Analysis")
        
        date_cols = df.select_dtypes(include=['datetime64']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(date_cols) == 0:
            st.warning("No datetime columns found. Try converting date columns to datetime format.")
            
            # Offer to convert columns
            potential_date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if potential_date_cols:
                date_col = st.selectbox("Select date column to convert:", potential_date_cols)
                if st.button("Convert to DateTime"):
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        st.success(f"Converted {date_col} to datetime!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Conversion failed: {str(e)}")
            return
            
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Date Column", date_cols)
        with col2:
            value_col = st.selectbox("Value Column", numeric_cols)
            
        if st.button("🔍 Analyze Trends"):
            self.perform_trend_analysis(df, date_col, value_col)
            
    def perform_trend_analysis(self, df: pd.DataFrame, date_col: str, value_col: str):
        """Perform trend analysis"""
        st.markdown("#### 📊 Trend Analysis Results")
        
        # Time series plot
        fig = px.line(df.sort_values(date_col), x=date_col, y=value_col,
                     title=f'{value_col} Over Time')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Rolling statistics
        df_sorted = df.sort_values(date_col).copy()
        df_sorted['7_day_avg'] = df_sorted[value_col].rolling(window=7, min_periods=1).mean()
        df_sorted['30_day_avg'] = df_sorted[value_col].rolling(window=30, min_periods=1).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sorted[date_col], y=df_sorted[value_col], 
                               name='Original', mode='lines'))
        fig.add_trace(go.Scatter(x=df_sorted[date_col], y=df_sorted['7_day_avg'], 
                               name='7-Day Average', mode='lines'))
        fig.add_trace(go.Scatter(x=df_sorted[date_col], y=df_sorted['30_day_avg'], 
                               name='30-Day Average', mode='lines'))
        
        fig.update_layout(title="Trend with Moving Averages", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    def render_anomaly_detection(self, df: pd.DataFrame):
        """Render anomaly detection interface"""
        st.markdown("### 🔍 Anomaly Detection")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for anomaly detection.")
            return
            
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Target Column", numeric_cols)
        with col2:
            method = st.selectbox("Detection Method", 
                                ["IQR", "Z-Score", "Isolation Forest", "Local Outlier Factor"])
            
        if st.button("🎯 Detect Anomalies"):
            self.perform_anomaly_detection(df, target_col, method)
            
    def perform_anomaly_detection(self, df: pd.DataFrame, target_col: str, method: str):
        """Perform anomaly detection"""
        st.markdown("#### 🎯 Anomaly Detection Results")
        
        data = df[target_col].dropna()
        anomalies = []
        
        if method == "IQR":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomalies = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
            
        elif method == "Z-Score":
            z_scores = np.abs((data - data.mean()) / data.std())
            anomalies = df[z_scores > 3]
            
        elif method == "Isolation Forest":
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
            anomalies = df[outlier_labels == -1]
            
        elif method == "Local Outlier Factor":
            from sklearn.neighbors import LocalOutlierFactor
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            outlier_labels = lof.fit_predict(data.values.reshape(-1, 1))
            anomalies = df[outlier_labels == -1]
            
        # Visualization
        fig = go.Figure()
        
        # Normal data points
        normal_data = df[~df.index.isin(anomalies.index)]
        fig.add_trace(go.Scatter(
            x=normal_data.index,
            y=normal_data[target_col],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=6)
        ))
        
        # Anomalies
        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomalies[target_col],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10, symbol='x')
        ))
        
        fig.update_layout(
            title=f'Anomaly Detection: {target_col} ({method})',
            xaxis_title='Index',
            yaxis_title=target_col,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data Points", len(df))
        with col2:
            st.metric("Anomalies Found", len(anomalies))
        with col3:
            st.metric("Anomaly Rate", f"{len(anomalies)/len(df)*100:.2f}%")
            
        if len(anomalies) > 0:
            with st.expander("🔍 Anomaly Details"):
                st.dataframe(anomalies[[target_col]], use_container_width=True)
                
    def render_report_generator(self, df: pd.DataFrame):
        """Render automated report generator"""
        st.markdown("### 📋 Automated Report Generator")
        
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Technical Analysis", "Data Quality Report", "Custom Report"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            include_charts = st.checkbox("Include Charts", value=True)
        with col2:
            include_recommendations = st.checkbox("Include Recommendations", value=True)
            
        if st.button("📄 Generate Report"):
            report_content = self.generate_report(df, report_type, include_charts, include_recommendations)
            
            # Display report
            st.markdown("#### 📊 Generated Report")
            st.markdown(report_content)
            
            # Download option
            self.create_download_link(report_content, f"neural_analyst_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            
    def generate_report(self, df: pd.DataFrame, report_type: str, include_charts: bool, include_recommendations: bool) -> str:
        """Generate automated report"""
        report = f"""
# Neural Data Analyst Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Report Type:** {report_type}

## Dataset Overview
- **Total Rows:** {len(df):,}
- **Total Columns:** {len(df.columns)}
- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- **Missing Values:** {df.isnull().sum().sum():,} ({df.isnull().sum().sum() / df.size * 100:.1f}%)

## Column Information
"""
        
        # Column details
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            
            report += f"- **{col}** ({dtype}): {null_count} missing, {unique_count} unique values\n"
            
        # Numeric summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report += "\n## Numeric Summary\n"
            summary_stats = df[numeric_cols].describe()
            report += summary_stats.to_markdown()
            
        # Key insights
        if include_recommendations:
            report += "\n## Key Insights & Recommendations\n"
            recommendations = self.generate_analysis_recommendations(df)
            for i, rec in enumerate(recommendations[:5], 1):
                report += f"{i}. **{rec['title']}:** {rec['description']}\n"
                
        return report
        
    def create_download_link(self, content: str, filename: str):
        """Create download link for report"""
        b64 = base64.b64encode(content.encode()).decode()
        href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}">📥 Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)
        
    def render_data_comparison_tool(self):
        """Render data comparison tool for multiple datasets"""
        st.markdown("## ⚖️ Data Comparison Tool")
        
        st.markdown("Upload multiple datasets to compare their characteristics:")
        
        uploaded_files = st.file_uploader(
            "Choose CSV files for comparison",
            type=['csv'],
            accept_multiple_files=True
        )
        
        if len(uploaded_files) >= 2:
            datasets = {}
            
            for file in uploaded_files:
                try:
                    df = pd.read_csv(file)
                    datasets[file.name] = df
                except Exception as e:
                    st.error(f"Error loading {file.name}: {str(e)}")
                    
            if len(datasets) >= 2:
                self.perform_dataset_comparison(datasets)
                
    def perform_dataset_comparison(self, datasets: Dict[str, pd.DataFrame]):
        """Perform comparison between multiple datasets"""
        st.markdown("### 📊 Dataset Comparison Results")
        
        # Basic comparison table
        comparison_data = []
        
        for name, df in datasets.items():
            comparison_data.append({
                'Dataset': name,
                'Rows': len(df),
                'Columns': len(df.columns),
                'Numeric Columns': len(df.select_dtypes(include=[np.number]).columns),
                'Text Columns': len(df.select_dtypes(include=['object']).columns),
                'Missing Values': df.isnull().sum().sum(),
                'Memory (MB)': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"
            })
            
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visual comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Rows Comparison', 'Columns Comparison', 
                          'Missing Values', 'Memory Usage'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        names = list(datasets.keys())
        
        # Rows comparison
        fig.add_trace(
            go.Bar(x=names, y=[len(datasets[name]) for name in names], name="Rows"),
            row=1, col=1
        )
        
        # Columns comparison
        fig.add_trace(
            go.Bar(x=names, y=[len(datasets[name].columns) for name in names], name="Columns"),
            row=1, col=2
        )
        
        # Missing values comparison
        fig.add_trace(
            go.Bar(x=names, y=[datasets[name].isnull().sum().sum() for name in names], name="Missing"),
            row=2, col=1
        )
        
        # Memory usage comparison
        fig.add_trace(
            go.Bar(x=names, y=[datasets[name].memory_usage(deep=True).sum() / 1024**2 for name in names], name="Memory"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Dataset Comparison Dashboard")
        st.plotly_chart(fig, use_container_width=True)
        
    def render_data_profiling_tool(self, df: pd.DataFrame):
        """Render comprehensive data profiling tool"""
        st.markdown("## 🔬 Data Profiling Tool")
        
        if st.button("🚀 Generate Complete Data Profile"):
            with st.spinner("Generating comprehensive data profile..."):
                profile = self.generate_data_profile(df)
                self.display_data_profile(profile)
                
    def generate_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        profile = {
            'overview': {},
            'column_profiles': {},
            'data_quality': {},
            'relationships': {},
            'recommendations': []
        }
        
        # Overview
        profile['overview'] = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_cells': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Column profiles
        for col in df.columns:
            col_profile = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_percentage': df[col].isnull().sum() / len(df) * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': df[col].nunique() / len(df) * 100
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_profile.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis()
                })
            else:
                col_profile.update({
                    'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    'most_frequent_count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
                })
                
            profile['column_profiles'][col] = col_profile
            
        return profile
        
    def display_data_profile(self, profile: Dict[str, Any]):
        """Display data profile results"""
        st.markdown("### 📊 Complete Data Profile")
        
        # Overview metrics
        overview = profile['overview']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{overview['shape'][0]:,}")
        with col2:
            st.metric("Columns", overview['shape'][1])
        with col3:
            st.metric("Missing Cells", f"{overview['missing_cells']:,}")
        with col4:
            st.metric("Duplicates", f"{overview['duplicate_rows']:,}")
            
        # Column details table
        st.markdown("#### 📋 Column Details")
        
        col_data = []
        for col, details in profile['column_profiles'].items():
            col_data.append({
                'Column': col,
                'Type': details['dtype'],
                'Missing %': f"{details['null_percentage']:.1f}%",
                'Unique %': f"{details['unique_percentage']:.1f}%",
                'Details': f"Min: {details.get('min', 'N/A')}, Max: {details.get('max', 'N/A')}" if 'min' in details else f"Most Frequent: {details.get('most_frequent', 'N/A')}"
            })
            
        col_df = pd.DataFrame(col_data)
        st.dataframe(col_df, use_container_width=True)

# Usage in main app
def integrate_advanced_features():
    """Integration function for advanced features"""
    return """
    # Add this to your main.py file:
    
    from advanced_features import AdvancedFeatures
    
    # In your NeuralDataAnalyst class:
    def __init__(self):
        # ... existing code ...
        self.advanced_features = AdvancedFeatures(self.db_manager)
    
    # Add this after your existing data upload section:
    if st.session_state.uploaded_data is not None:
        if st.button("🔬 Advanced Analytics", key="advanced_analytics"):
            self.advanced_features.render_advanced_analytics_dashboard(st.session_state.uploaded_data)
            
        if st.button("🔍 Data Profiling", key="data_profiling"):
            self.advanced_features.render_data_profiling_tool(st.session_state.uploaded_data)
    
    # Add dataset comparison in sidebar:
    with st.sidebar:
        st.markdown("---")
        if st.button("⚖️ Compare Datasets"):
            self.advanced_features.render_data_comparison_tool()
    """