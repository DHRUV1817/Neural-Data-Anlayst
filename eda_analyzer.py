import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import scipy with error handling
try:
    from scipy import stats
    from scipy.stats import chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class EDAAnalyzer:
    """Comprehensive Exploratory Data Analysis with advanced visualizations"""
    
    def __init__(self):
        self.color_palette = [
            '#FFD700', '#FF6B6B', '#4ECDC4', '#45B7D1', 
            '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'
        ]
        
    def perform_complete_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive EDA analysis"""
        try:
            results = {
                'overview': self.generate_overview(df),
                'distributions': self.analyze_distributions(df),
                'correlations': self.analyze_correlations(df),
                'insights': self.generate_insights(df),
                'data_quality': self.assess_data_quality(df),
                'advanced_analysis': self.perform_advanced_analysis(df)
            }
            
            return results
        except Exception as e:
            # Return basic results if advanced analysis fails
            return {
                'overview': self.generate_overview(df),
                'distributions': {},
                'correlations': {},
                'insights': [{'title': 'Analysis Error', 'description': f'Error during analysis: {str(e)}'}],
                'data_quality': {},
                'advanced_analysis': {}
            }
        
    def generate_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate dataset overview"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            
            overview = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(categorical_cols),
                'datetime_columns': len(datetime_cols),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                'duplicate_rows': df.duplicated().sum(),
                'missing_values_total': df.isnull().sum().sum()
            }
            
            if len(numeric_cols) > 0:
                overview['summary_stats'] = df[numeric_cols].describe()
                
            return overview
        except Exception as e:
            return {
                'total_rows': len(df) if df is not None else 0,
                'total_columns': len(df.columns) if df is not None else 0,
                'numeric_columns': 0,
                'categorical_columns': 0,
                'datetime_columns': 0,
                'memory_usage': '0 MB',
                'duplicate_rows': 0,
                'missing_values_total': 0,
                'error': str(e)
            }
        
    def analyze_distributions(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Analyze data distributions with multiple chart types"""
        distributions = {}
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # Numeric distributions
            if len(numeric_cols) > 0:
                distributions.update(self.create_numeric_distributions(df, numeric_cols))
                
            # Categorical distributions
            if len(categorical_cols) > 0:
                distributions.update(self.create_categorical_distributions(df, categorical_cols))
                
        except Exception as e:
            distributions['error'] = self.create_error_plot(f"Distribution analysis failed: {str(e)}")
            
        return distributions
        
    def create_error_plot(self, error_message: str) -> go.Figure:
        """Create an error plot when analysis fails"""
        fig = go.Figure()
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Analysis Error",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig
        
    def create_numeric_distributions(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, go.Figure]:
        """Create numeric distribution plots"""
        plots = {}
        
        try:
            # Multi-histogram plot
            if len(numeric_cols) <= 6:
                rows = (len(numeric_cols) + 2) // 3
                fig = make_subplots(
                    rows=rows, cols=3,
                    subplot_titles=list(numeric_cols),
                    vertical_spacing=0.08
                )
                
                for i, col in enumerate(numeric_cols):
                    row = (i // 3) + 1
                    col_pos = (i % 3) + 1
                    
                    # Filter out non-finite values
                    data = df[col].dropna()
                    if len(data) > 0:
                        fig.add_trace(
                            go.Histogram(
                                x=data,
                                name=col,
                                marker_color=self.color_palette[i % len(self.color_palette)],
                                opacity=0.7,
                                showlegend=False
                            ),
                            row=row, col=col_pos
                        )
                    
                fig.update_layout(
                    title="📊 Numeric Distributions Overview",
                    height=300 * rows,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                plots['numeric_histograms'] = fig
                
            # Box plots for outlier detection
            if len(numeric_cols) > 0:
                fig = go.Figure()
                for i, col in enumerate(numeric_cols[:8]):  # Limit to 8 columns
                    data = df[col].dropna()
                    if len(data) > 0:
                        fig.add_trace(go.Box(
                            y=data,
                            name=col,
                            marker_color=self.color_palette[i % len(self.color_palette)]
                        ))
                    
                fig.update_layout(
                    title="📦 Box Plots - Outlier Detection",
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                plots['box_plots'] = fig
                
            # Violin plots for distribution shapes
            if len(numeric_cols) > 0:
                fig = go.Figure()
                for i, col in enumerate(numeric_cols[:6]):
                    data = df[col].dropna()
                    if len(data) > 1:  # Need at least 2 points for violin plot
                        fig.add_trace(go.Violin(
                            y=data,
                            name=col,
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor=self.color_palette[i % len(self.color_palette)],
                            opacity=0.6
                        ))
                    
                fig.update_layout(
                    title="🎻 Violin Plots - Distribution Shapes",
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                plots['violin_plots'] = fig
                
        except Exception as e:
            plots['numeric_error'] = self.create_error_plot(f"Numeric distribution error: {str(e)}")
            
        return plots
        
    def create_categorical_distributions(self, df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, go.Figure]:
        """Create categorical distribution plots"""
        plots = {}
        
        try:
            # Bar charts for categorical variables
            for i, col in enumerate(categorical_cols[:4]):  # Limit to 4 columns
                value_counts = df[col].value_counts().head(15)  # Top 15 categories
                
                if len(value_counts) > 0:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=value_counts.index.astype(str),
                            y=value_counts.values,
                            marker_color=self.color_palette[i % len(self.color_palette)],
                            text=value_counts.values,
                            textposition='auto'
                        )
                    ])
                    
                    fig.update_layout(
                        title=f"📊 {col} - Value Distribution",
                        xaxis_title=col,
                        yaxis_title="Count",
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    plots[f'categorical_{col}'] = fig
                
            # Pie chart for first categorical variable
            if len(categorical_cols) > 0:
                col = categorical_cols[0]
                value_counts = df[col].value_counts().head(10)
                
                if len(value_counts) > 0:
                    fig = go.Figure(data=[go.Pie(
                        labels=value_counts.index.astype(str),
                        values=value_counts.values,
                        hole=0.3,
                        marker_colors=self.color_palette
                    )])
                    
                    fig.update_layout(
                        title=f"🥧 {col} - Proportion Analysis",
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    plots['pie_chart'] = fig
                    
        except Exception as e:
            plots['categorical_error'] = self.create_error_plot(f"Categorical distribution error: {str(e)}")
            
        return plots
        
    def analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between variables"""
        correlations = {}
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 1:
                # Correlation matrix
                corr_matrix = df[numeric_cols].corr()
                
                # Heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdYlBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation")
                ))
                
                fig.update_layout(
                    title="🔥 Correlation Heatmap",
                    height=max(400, len(numeric_cols) * 30),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                correlations['heatmap'] = fig
                
                # Top correlations
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                corr_matrix_masked = corr_matrix.mask(mask)
                
                # Get top positive and negative correlations
                corr_pairs = []
                for i in range(len(corr_matrix_masked.columns)):
                    for j in range(len(corr_matrix_masked.columns)):
                        if pd.notna(corr_matrix_masked.iloc[i, j]):
                            corr_pairs.append({
                                'Variable 1': corr_matrix_masked.columns[i],
                                'Variable 2': corr_matrix_masked.columns[j],
                                'Correlation': corr_matrix_masked.iloc[i, j]
                            })
                            
                if corr_pairs:
                    corr_df = pd.DataFrame(corr_pairs)
                    corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
                    correlations['top_correlations'] = corr_df.head(10)
                    
                # Scatter plot matrix for top correlated variables
                if len(numeric_cols) >= 2:
                    top_corr_cols = corr_df.head(3)[['Variable 1', 'Variable 2']].values.flatten()
                    unique_cols = list(set(top_corr_cols))[:4]  # Max 4 variables
                    
                    if len(unique_cols) >= 2:
                        try:
                            fig = px.scatter_matrix(
                                df[unique_cols].dropna(),
                                dimensions=unique_cols,
                                color_discrete_sequence=self.color_palette
                            )
                            
                            fig.update_layout(
                                title="🎯 Scatter Plot Matrix - Top Correlated Variables",
                                height=600,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            correlations['scatter_matrix'] = fig
                        except Exception:
                            pass  # Skip if scatter matrix fails
                        
        except Exception as e:
            correlations['error'] = f"Correlation analysis failed: {str(e)}"
                    
        return correlations
        
    def generate_insights(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Generate AI-powered insights about the data"""
        insights = []
        
        try:
            # Basic statistics insights
            insights.append({
                'title': '📊 Dataset Overview',
                'description': f"Dataset contains {len(df):,} rows and {len(df.columns)} columns. "
                              f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB. "
                              f"Missing values: {df.isnull().sum().sum():,} ({df.isnull().sum().sum() / df.size * 100:.1f}%)."
            })
            
            # Numeric columns insights
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                try:
                    # Find columns with high variance
                    variances = df[numeric_cols].var().sort_values(ascending=False)
                    high_var_col = variances.index[0]
                    
                    insights.append({
                        'title': '📈 Variance Analysis',
                        'description': f"'{high_var_col}' shows the highest variance ({variances.iloc[0]:.2f}), "
                                      f"indicating significant spread in values. This column might contain outliers "
                                      f"or represent a key differentiating factor in your dataset."
                    })
                    
                    # Skewness analysis
                    skewed_cols = []
                    for col in numeric_cols:
                        try:
                            skewness = df[col].skew()
                            if abs(skewness) > 1:
                                skewed_cols.append((col, skewness))
                        except:
                            continue
                            
                    if skewed_cols:
                        insights.append({
                            'title': '📏 Distribution Skewness',
                            'description': f"Found {len(skewed_cols)} heavily skewed columns. "
                                          f"Most skewed: '{skewed_cols[0][0]}' (skewness: {skewed_cols[0][1]:.2f}). "
                                          f"Consider log transformation or outlier treatment for better modeling."
                        })
                except Exception:
                    pass
                    
            # Categorical insights
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                try:
                    cardinalities = []
                    for col in categorical_cols:
                        unique_count = df[col].nunique()
                        cardinalities.append((col, unique_count))
                        
                    cardinalities.sort(key=lambda x: x[1], reverse=True)
                    
                    insights.append({
                        'title': '🏷️ Categorical Analysis',
                        'description': f"'{cardinalities[0][0]}' has the highest cardinality ({cardinalities[0][1]} unique values). "
                                      f"High cardinality columns might need encoding strategies for machine learning. "
                                      f"Consider grouping rare categories or using embedding techniques."
                    })
                except Exception:
                    pass
                
            # Missing data patterns
            try:
                missing_data = df.isnull().sum()
                missing_cols = missing_data[missing_data > 0].sort_values(ascending=False)
                
                if len(missing_cols) > 0:
                    insights.append({
                        'title': '❓ Missing Data Patterns',
                        'description': f"'{missing_cols.index[0]}' has the most missing values ({missing_cols.iloc[0]:,} - "
                                      f"{missing_cols.iloc[0] / len(df) * 100:.1f}%). "
                                      f"Analyze if missing data is random or systematic. "
                                      f"Consider imputation strategies or feature engineering."
                    })
            except Exception:
                pass
                
            # Correlation insights
            if len(numeric_cols) > 1:
                try:
                    corr_matrix = df[numeric_cols].corr()
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    corr_matrix_masked = corr_matrix.mask(mask)
                    
                    max_corr = 0
                    max_pair = None
                    for i in range(len(corr_matrix_masked.columns)):
                        for j in range(len(corr_matrix_masked.columns)):
                            if pd.notna(corr_matrix_masked.iloc[i, j]):
                                if abs(corr_matrix_masked.iloc[i, j]) > abs(max_corr):
                                    max_corr = corr_matrix_masked.iloc[i, j]
                                    max_pair = (corr_matrix_masked.columns[i], corr_matrix_masked.columns[j])
                                    
                    if max_pair and abs(max_corr) > 0.5:
                        insights.append({
                            'title': '🔗 Strong Correlations',
                            'description': f"Strong correlation found between '{max_pair[0]}' and '{max_pair[1]}' "
                                          f"(r = {max_corr:.3f}). This suggests potential multicollinearity. "
                                          f"Consider feature selection or dimensionality reduction techniques."
                        })
                except Exception:
                    pass
                
        except Exception as e:
            insights.append({
                'title': 'Analysis Error',
                'description': f"Error generating insights: {str(e)}"
            })
            
        return insights
        
    def assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality with visualizations"""
        quality = {}
        
        try:
            # Missing values heatmap
            if df.isnull().sum().sum() > 0:
                missing_data = df.isnull().sum().sort_values(ascending=False)
                missing_data = missing_data[missing_data > 0]
                
                if len(missing_data) > 0:
                    fig = go.Figure([go.Bar(
                        x=missing_data.index,
                        y=missing_data.values,
                        marker_color='#FF6B6B',
                        text=missing_data.values,
                        textposition='auto'
                    )])
                    
                    fig.update_layout(
                        title="❓ Missing Values by Column",
                        xaxis_title="Columns",
                        yaxis_title="Missing Count",
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    quality['missing_values'] = fig
                
            # Data types distribution
            dtype_counts = df.dtypes.value_counts()
            
            if len(dtype_counts) > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=[str(dtype) for dtype in dtype_counts.index],
                    values=dtype_counts.values,
                    hole=0.3,
                    marker_colors=self.color_palette
                )])
                
                fig.update_layout(
                    title="🔧 Data Types Distribution",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                quality['data_types'] = fig
            
            # Duplicate analysis
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                quality['duplicates'] = {
                    'count': duplicates,
                    'percentage': duplicates / len(df) * 100
                }
                
        except Exception as e:
            quality['error'] = f"Data quality assessment failed: {str(e)}"
            
        return quality
        
    def perform_advanced_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced statistical analysis"""
        advanced = {}
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Outlier detection using IQR method
            if len(numeric_cols) > 0:
                outlier_counts = {}
                for col in numeric_cols:
                    try:
                        data = df[col].dropna()
                        if len(data) > 0:
                            Q1 = data.quantile(0.25)
                            Q3 = data.quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                            outlier_counts[col] = len(outliers)
                    except Exception:
                        outlier_counts[col] = 0
                        
                if outlier_counts:
                    outlier_df = pd.DataFrame(list(outlier_counts.items()), 
                                            columns=['Column', 'Outlier_Count'])
                    outlier_df = outlier_df.sort_values('Outlier_Count', ascending=False)
                    advanced['outliers'] = outlier_df
                    
            # Statistical tests
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) >= 2 and SCIPY_AVAILABLE:
                try:
                    col1, col2 = categorical_cols[0], categorical_cols[1]
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    
                    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        advanced['chi_square_test'] = {
                            'variables': [col1, col2],
                            'chi2_statistic': chi2,
                            'p_value': p_value,
                            'interpretation': 'Dependent' if p_value < 0.05 else 'Independent'
                        }
                except Exception:
                    pass  # Skip if test fails
                    
        except Exception as e:
            advanced['error'] = f"Advanced analysis failed: {str(e)}"
                
        return advanced