import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    BaggingClassifier, BaggingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
import joblib
import os
from datetime import datetime
from utils.model_utils import get_model_configs, train_ensemble_models
from utils.database import save_experiment
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Model Training - ML Benchmark Tool",
    page_icon="🤖",  # 🏹
    layout="wide"
)

st.title("🏹 Model Training")
st.markdown("### Train and compare individual models with ensemble methods")

# Check if data is available
if 'X_train' not in st.session_state or st.session_state.X_train is None:
    st.warning("⚠️ No training data available. Please go to Data Management page first.")
    st.stop()

# Get data from session state
X_train = st.session_state.X_train
X_test = st.session_state.X_test
y_train = st.session_state.y_train
y_test = st.session_state.y_test
task_type = st.session_state.task_type

st.success(f"✅ Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
st.info(f"📊 Task type: {task_type.title()}")

# Model selection
st.subheader("🔧 Model Configuration")

# Get available models
model_configs = get_model_configs(task_type)

col1, col2 = st.columns(2)

with col1:
    st.write("**Individual Models**")
    selected_models = st.multiselect(
        "Select models to train:",
        list(model_configs['individual'].keys()),
        default=list(model_configs['individual'].keys())[:3]
    )

with col2:
    st.write("**Ensemble Methods**")
    selected_ensembles = st.multiselect(
        "Select ensemble methods:",
        list(model_configs['ensemble'].keys()),
        default=list(model_configs['ensemble'].keys())[:2]
    )

# Training configuration
st.subheader("⚙️ Training Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)

with col2:
    random_state = st.number_input("Random state:", value=42, min_value=0)

with col3:
    if "experiment_name" not in st.session_state:
        st.session_state['experiment_name'] = f"Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    experiment_name = st.text_input(
        "Experiment name:",
        value=st.session_state['experiment_name'],
        key="eperiment_name"
    )

# Training button
if st.button("🚀 Start Training", type="primary"):
    if not selected_models and not selected_ensembles:
        st.error("Please select at least one model to train!")
        st.stop()
    
    # Create models directory
    models_dir = "trained_models"
    os.makedirs(models_dir, exist_ok=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    total_models = len(selected_models) + len(selected_ensembles)
    current_model = 0
    
    # Train individual models
    st.subheader("🔄 Training Individual Models")
    individual_results = {}
    
    for model_name in selected_models:
        current_model += 1
        progress_bar.progress(current_model / total_models)
        status_text.text(f"Training {model_name}...")
        
        try:
            # Get model
            model = model_configs['individual'][model_name]
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='accuracy' if task_type == 'classification' else 'r2'
            )
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if task_type == 'classification':
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted')
                }
            else:
                metrics = {
                    'r2': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred)
                }
            
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            
            individual_results[model_name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred
            }
            
            # Save model
            model_path = os.path.join(models_dir, f"{experiment_name}_{model_name}.joblib")
            joblib.dump(model, model_path)
            
            st.success(f"✅ {model_name} trained successfully!")
            
        except Exception as e:
            st.error(f"❌ Error training {model_name}: {str(e)}")
    
    # Train ensemble models
    if selected_ensembles:
        st.subheader("🔄 Training Ensemble Models")
        ensemble_results = {}
        
        # Get base models for ensembles
        base_models = [
            individual_results[name]['model'] 
            for name in selected_models 
            if name in individual_results
        ]
        
        if len(base_models) < 2:
            st.warning("⚠️ Need at least 2 individual models for ensemble methods. Using default base models.")
            # Use some default models
            if task_type == 'classification':
                base_models = [
                    RandomForestClassifier(random_state=random_state),
                    LogisticRegression(random_state=random_state, max_iter=1000)
                ]
            else:
                base_models = [
                    RandomForestRegressor(random_state=random_state),
                    LinearRegression()
                ]
            
            # Train base models
            for model in base_models:
                model.fit(X_train, y_train)
        
        for ensemble_name in selected_ensembles:
            current_model += 1
            progress_bar.progress(current_model / total_models)
            status_text.text(f"Training {ensemble_name}...")
            
            try:
                # Train ensemble
                ensemble_model = train_ensemble_models(
                    ensemble_name, base_models, X_train, y_train, task_type, random_state
                )
                
                # Cross-validation
                cv_scores = cross_val_score(
                    ensemble_model, X_train, y_train,
                    cv=cv_folds,
                    scoring='accuracy' if task_type == 'classification' else 'r2'
                )
                
                # Predictions
                y_pred = ensemble_model.predict(X_test)
                
                # Calculate metrics
                if task_type == 'classification':
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='weighted'),
                        'recall': recall_score(y_test, y_pred, average='weighted'),
                        'f1': f1_score(y_test, y_pred, average='weighted')
                    }
                else:
                    metrics = {
                        'r2': r2_score(y_test, y_pred),
                        'mse': mean_squared_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'mae': mean_absolute_error(y_test, y_pred)
                    }
                
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                
                ensemble_results[ensemble_name] = {
                    'model': ensemble_model,
                    'metrics': metrics,
                    'predictions': y_pred
                }
                
                # Save model
                model_path = os.path.join(models_dir, f"{experiment_name}_{ensemble_name}.joblib")
                joblib.dump(ensemble_model, model_path)
                
                st.success(f"✅ {ensemble_name} trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Error training {ensemble_name}: {str(e)}")
    
    # Combine results
    all_results = {**individual_results, **ensemble_results}
    
    # Store results in session state
    st.session_state.training_results = all_results
    st.session_state.experiment_name = experiment_name
    
    progress_bar.progress(1.0)
    status_text.text("Training completed!")
    
    # Display results summary
    st.subheader("📊 Training Results Summary")
    
    if all_results:
        # Create results DataFrame
        results_data = []
        for model_name, result in all_results.items():
            row = {'Model': model_name}
            row.update(result['metrics'])
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        
        # Find best model
        if task_type == 'classification':
            best_model_idx = results_df['accuracy'].idxmax()
            best_score = results_df.loc[best_model_idx, 'accuracy']
            scoring_metric = 'accuracy'
        else:
            best_model_idx = results_df['r2'].idxmax()
            best_score = results_df.loc[best_model_idx, 'r2']
            scoring_metric = 'r2'
        
        best_model_name = results_df.loc[best_model_idx, 'Model']
        
        st.success(f"🏆 Best model: **{best_model_name}** with {scoring_metric}: {best_score:.4f}")
        
        # Save experiment to database
        try:
            save_experiment(
                name=experiment_name,
                dataset_name=getattr(st.session_state, 'dataset_name', 'Custom Dataset'),
                models_tested=list(all_results.keys()),
                task_type=task_type,
                best_model=best_model_name,
                best_score=best_score,
                results=results_data
            )
            st.info("💾 Experiment saved to database!")
        except Exception as e:
            st.error(f"Error saving experiment: {str(e)}")
        
        # Visualization
        st.subheader("📈 Performance Comparison")
        
        # Performance comparison chart
        if task_type == 'classification':
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        else:
            metrics_to_plot = ['r2', 'mse', 'rmse', 'mae']
        
        for metric in metrics_to_plot:
            if metric in results_df.columns:
                fig = px.bar(
                    results_df,
                    x='Model',
                    y=metric,
                    title=f"{metric.upper()} Comparison",
                    color=metric,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("No models were successfully trained!")

# Display existing results if available
elif 'training_results' in st.session_state:
    st.subheader("📊 Previous Training Results")
    st.info(f"Experiment: {st.session_state.experiment_name}")
    
    results_data = []
    for model_name, result in st.session_state.training_results.items():
        row = {'Model': model_name}
        row.update(result['metrics'])
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)

else:
    st.info("👆 Configure your models and click 'Start Training' to begin!")

# Navigation hint
if 'training_results' in st.session_state:
    st.success("🎉 Models trained! You can now proceed to **Model Comparison** page.")
