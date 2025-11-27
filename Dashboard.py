## Og Code:
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.database import init_database, get_all_experiments
from utils.visualization import create_metrics_comparison_chart

# Initialize database
init_database()

st.set_page_config(
    page_title="ML Benchmark Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🪩
st.title("⚙️ ML Ensemble Based Benchmark Tool")
st.markdown("##### Comprehensive Machine Learning Model Comparison and Benchmarking Platform")

# Sidebar navigation
# st.sidebar.title("Navigation")
# st.sidebar.markdown("**Current Page:** Dashboard")
# st.sidebar.markdown("---")
# st.sidebar.markdown("**Available Pages:**")
# st.sidebar.markdown("- 📊 Data Management")
# st.sidebar.markdown("- 🤖 Model Training")
# st.sidebar.markdown("- 📈 Model Comparison")
# st.sidebar.markdown("- 🔍 Model Explainability")

# Main dashboard content
col1, col2, col3 = st.columns(3)

# Get experiment statistics
experiments = get_all_experiments()

with col1:
    st.metric("Total Experiments", len(experiments))

with col2:
    if experiments:
        classification_count = sum(1 for exp in experiments if exp[4] == 'classification')
        st.metric("Classification Experiments", classification_count)
    else:
        st.metric("Classification Experiments", 0)

with col3:
    if experiments:
        regression_count = sum(1 for exp in experiments if exp[4] == 'regression')
        st.metric("Regression Experiments", regression_count)
    else:
        st.metric("Regression Experiments", 0)

st.markdown("---")

# Recent experiments section
st.subheader("📋 Recent Experiments")

if experiments:
    # Convert to DataFrame for better display
    df_experiments = pd.DataFrame(experiments, columns=[
        'ID', 'Name', 'Dataset', 'Models', 'Task Type', 'Best Model', 
        'Best Score', 'Timestamp'
    ])
    
    # Display recent experiments table
    st.dataframe(
        df_experiments.sort_values('Timestamp', ascending=False).head(10),
        use_container_width=True,
        hide_index=True
    )
    
    # Performance overview chart
    st.subheader("📊 Performance Overview")
    
    if len(df_experiments) > 1:
        # Create performance comparison chart
        fig = create_metrics_comparison_chart(df_experiments)
        st.plotly_chart(fig, use_container_width=True)
    
    # Task type distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Task Type Distribution")
        task_counts = df_experiments['Task Type'].value_counts()
        fig_pie = px.pie(
            values=task_counts.values,
            names=task_counts.index,
            title="Distribution of Task Types"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Best Models Distribution")
        model_counts = df_experiments['Best Model'].value_counts().head(5)
        fig_bar = px.bar(
            x=model_counts.values,
            y=model_counts.index,
            orientation='h',
            title="Top 5 Best Performing Models",
            labels={'x': 'Count', 'y': 'Model'}
        )
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed experiment analysis
    st.subheader("🔍 Experiment Details")
    
    selected_experiment = st.selectbox(
        "Select an experiment to view details:",
        options=df_experiments['Name'].tolist(),
        index=0
    )
    
    if selected_experiment:
        exp_details = df_experiments[df_experiments['Name'] == selected_experiment].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Dataset:** {exp_details['Dataset']}")
            st.info(f"**Task Type:** {exp_details['Task Type']}")
        
        with col2:
            st.success(f"**Best Model:** {exp_details['Best Model']}")
            st.success(f"**Best Score:** {exp_details['Best Score']:.4f}")
        
        with col3:
            st.warning(f"**Models Tested:** {exp_details['Models']}")
            st.warning(f"**Date:** {exp_details['Timestamp']}")

else:
    st.info("🚀 No experiments found. Start by uploading data and training models!")
    st.markdown("### Getting Started")
    st.markdown("""
    1. **📊 Data Management**: Upload your dataset or choose from built-in datasets
    2. **🤖 Model Training**: Select models and configure training parameters
    3. **📈 Model Comparison**: Compare ensemble vs individual model performance
    4. **🔍 Model Explainability**: Analyze feature importance and model decisions
    """)

# Footer
st.markdown("---")
st.markdown("*ML Benchmark Tool - Built with Streamlit*")


