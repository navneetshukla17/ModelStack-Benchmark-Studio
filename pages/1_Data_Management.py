import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    fetch_california_housing, load_diabetes, make_classification, make_regression
)
from sklearn.model_selection import train_test_split
from utils.data_utils import detect_task_type, preprocess_data
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Data Management - ML Benchmark Tool",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Management")
st.markdown("### Load, explore, and preprocess your datasets")

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# Data source selection
st.subheader("🔗 Data Source")
data_source = st.radio(
    "Choose your data source:",
    ["Built-in Datasets", "Upload CSV File"],
    horizontal=True
)

if data_source == "Built-in Datasets":
    # Built-in dataset selection
    st.subheader("📚 Built-in Datasets")
    
    dataset_options = {
        "Classification": {
            "Iris": load_iris,
            "Wine": load_wine,
            "Breast Cancer": load_breast_cancer,
            "Digits": load_digits
        },
        "Regression": {
            "California Housing": fetch_california_housing,
            "Diabetes": load_diabetes
        }
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        task_category = st.selectbox("Task Category:", ["Classification", "Regression"])
    
    with col2:
        dataset_name = st.selectbox("Dataset:", list(dataset_options[task_category].keys()))
    
    if st.button("Load Dataset"):
        try:
            # Load the selected dataset
            data_loader = dataset_options[task_category][dataset_name]
            
            # Load the dataset
            data = data_loader()
            
            # Create DataFrame with safe feature name handling
            if hasattr(data, 'feature_names') and data.feature_names is not None:
                feature_names = data.feature_names
            else:
                # Fallback for datasets without feature_names (e.g., some versions of Digits)
                feature_names = [f"feature_{i}" for i in range(data.data.shape[1])]
            
            df = pd.DataFrame(data.data, columns=feature_names)
            df['target'] = data.target
            
            st.session_state.dataset = df
            st.session_state.task_type = task_category.lower()
            
            st.success(f"✅ {dataset_name} dataset loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")

else:
    # CSV file upload
    st.subheader("📁 Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with your dataset"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.dataset = df
            
            # Auto-detect task type
            detected_task = detect_task_type(df)
            st.session_state.task_type = detected_task
            
            st.success("✅ CSV file uploaded successfully!")
            st.info(f"🔍 Detected task type: {detected_task}")
            
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

# Dataset exploration and preprocessing
if st.session_state.dataset is not None:
    df = st.session_state.dataset
    
    st.markdown("---")
    st.subheader("🔍 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Numeric Columns", df.select_dtypes(include=[np.number]).shape[1])
    
    # Dataset preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Task type configuration
    st.subheader("⚙️ Task Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Task type selection (with auto-detection)
        task_type = st.selectbox(
            "Task Type:",
            ["classification", "regression"],
            index=0 if st.session_state.task_type == "classification" else 1
        )
        st.session_state.task_type = task_type
    
    with col2:
        # Target column selection
        target_column = st.selectbox(
            "Target Column:",
            df.columns.tolist(),
            index=len(df.columns) - 1  # Default to last column
        )
    
    # Data preprocessing
    st.subheader("🔧 Data Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        handle_missing = st.selectbox(
            "Handle Missing Values:",
            ["Drop rows with missing values", "Fill with mean/mode", "Fill with median"]
        )
    
    with col2:
        test_size = st.slider(
            "Test Set Size:",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05
        )
    
    if st.button("Preprocess Data"):
        try:
            # Prepare features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Preprocess data
            X_processed, y_processed = preprocess_data(
                X, y, 
                task_type=task_type,
                handle_missing=handle_missing
            )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed,
                test_size=test_size,
                random_state=42,
                stratify=y_processed if task_type == 'classification' else None
            )
            
            # Store in session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            st.success("✅ Data preprocessing completed!")
            
            # Display preprocessing results
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"Training set: {st.session_state.X_train.shape[0]} samples")
                st.info(f"Features: {st.session_state.X_train.shape[1]}")
            
            with col2:
                st.info(f"Test set: {st.session_state.X_test.shape[0]} samples")
                st.info(f"Task type: {task_type}")
            
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")
    
    # Data visualization
    if st.session_state.X_train is not None:
        st.markdown("---")
        st.subheader("📊 Data Visualization")
        
        # Target distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Target Distribution**")
            if task_type == 'classification':
                fig = px.histogram(y=st.session_state.y_train, title="Target Class Distribution")
            else:
                fig = px.histogram(x=st.session_state.y_train, title="Target Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Feature Correlation Heatmap**")
            # Select numeric columns for correlation
            numeric_cols = st.session_state.X_train.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = st.session_state.X_train[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numeric features for correlation analysis")
        
        # Feature importance preview (basic variance analysis)
        st.write("**Feature Variance Analysis**")
        if len(numeric_cols) > 0:
            variances = st.session_state.X_train[numeric_cols].var().sort_values(ascending=False)
            fig = px.bar(
                x=variances.values,
                y=variances.index,
                orientation='h',
                title="Feature Variance (Higher variance may indicate more information)"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Please select and load a dataset to get started!")

# Navigation hint
if st.session_state.X_train is not None:
    st.success("🎉 Data is ready! You can now proceed to **Model Training** page.")
