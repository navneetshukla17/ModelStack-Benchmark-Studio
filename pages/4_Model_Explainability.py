import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import SHAP, with fallback if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Model Explainability - ML Benchmark Tool",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Model Explainability")
st.markdown("### Understand how your models make predictions")

# Check if training results are available
if 'training_results' not in st.session_state or not st.session_state.training_results:
    st.warning("⚠️ No training results available. Please go to Model Training page first.")
    st.stop()

# Get data from session state
results = st.session_state.training_results
task_type = st.session_state.task_type
X_train = st.session_state.X_train
X_test = st.session_state.X_test
y_test = st.session_state.y_test
experiment_name = st.session_state.experiment_name

# SHAP availability check
if not SHAP_AVAILABLE:
    st.warning("⚠️ SHAP library not available. Using alternative explainability methods.")

# Model selection for explainability
st.subheader("🎯 Select Model for Analysis")

col1, col2 = st.columns(2)

with col1:
    selected_model_name = st.selectbox(
        "Choose a model to analyze:",
        list(results.keys())
    )

with col2:
    analysis_type = st.selectbox(
        "Analysis Type:",
        ["Feature Importance", "SHAP Analysis", "Permutation Importance", "Prediction Analysis"]
    )

if selected_model_name:
    model = results[selected_model_name]['model']
    model_metrics = results[selected_model_name]['metrics']
    
    # Display model information
    st.subheader(f"📊 Model: {selected_model_name}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if task_type == 'classification':
            st.metric("Accuracy", f"{model_metrics['accuracy']:.4f}")
        else:
            st.metric("R²", f"{model_metrics['r2']:.4f}")
    
    with col2:
        st.metric("CV Mean", f"{model_metrics['cv_mean']:.4f}")
    
    with col3:
        st.metric("CV Std", f"{model_metrics['cv_std']:.4f}")
    
    # Feature Importance Analysis
    if analysis_type == "Feature Importance":
        st.subheader("🎯 Feature Importance Analysis")
        
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]
            importances = model.feature_importances_
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig_bar = px.bar(
                    importance_df.head(20),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Top 20 Feature Importances - {selected_model_name}",
                    color='Importance',
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Pie chart for top features
                top_features = importance_df.head(10)
                fig_pie = px.pie(
                    top_features,
                    values='Importance',
                    names='Feature',
                    title="Top 10 Features Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Feature importance table
            st.write("**Feature Importance Rankings**")
            st.dataframe(importance_df.head(20), hide_index=True)
            
            # Cumulative importance
            importance_df['Cumulative'] = importance_df['Importance'].cumsum() / importance_df['Importance'].sum()
            
            fig_cumulative = px.line(
                importance_df.head(30),
                x=range(1, min(31, len(importance_df) + 1)),
                y='Cumulative',
                title="Cumulative Feature Importance",
                labels={'x': 'Number of Features', 'y': 'Cumulative Importance'}
            )
            fig_cumulative.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="80% threshold")
            fig_cumulative.add_hline(y=0.9, line_dash="dash", line_color="orange", annotation_text="90% threshold")
            st.plotly_chart(fig_cumulative, use_container_width=True)
            
        else:
            st.info("This model doesn't support native feature importance. Try Permutation Importance instead.")
    
    # SHAP Analysis
    elif analysis_type == "SHAP Analysis":
        st.subheader("🔍 SHAP (SHapley Additive exPlanations) Analysis")
        
        if SHAP_AVAILABLE:
            try:
                # Create SHAP explainer
                explainer = shap.Explainer(model)
                
                # Calculate SHAP values for a sample of test data
                sample_size = min(100, len(X_test))
                X_sample = X_test.iloc[:sample_size] if hasattr(X_test, 'iloc') else X_test[:sample_size]
                
                with st.spinner("Calculating SHAP values..."):
                    shap_values = explainer(X_sample)
                
                # SHAP Summary Plot
                st.write("**SHAP Summary Plot**")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, show=False)
                st.pyplot(fig)
                plt.close()
                
                # SHAP Feature Importance
                st.write("**SHAP Feature Importance**")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                st.pyplot(fig)
                plt.close()
                
                # Individual prediction explanation
                st.write("**Individual Prediction Explanation**")
                sample_idx = st.slider("Select sample to explain:", 0, len(X_sample)-1, 0)
                
                if hasattr(X_sample, 'iloc'):
                    sample_data = X_sample.iloc[sample_idx:sample_idx+1]
                else:
                    sample_data = X_sample[sample_idx:sample_idx+1]
                
                # SHAP Waterfall plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(shap_values[sample_idx], show=False)
                st.pyplot(fig)
                plt.close()
                
                # Sample details
                st.write("**Sample Details**")
                if hasattr(X_sample, 'columns'):
                    sample_df = pd.DataFrame([sample_data.iloc[0]], columns=X_sample.columns)
                else:
                    sample_df = pd.DataFrame([sample_data[0]], columns=[f'Feature_{i}' for i in range(len(sample_data[0]))])
                
                st.dataframe(sample_df, hide_index=True)
                
                # Prediction for this sample
                prediction = model.predict(sample_data)[0]
                st.info(f"Model prediction for this sample: {prediction:.4f}")
                
            except Exception as e:
                st.error(f"Error calculating SHAP values: {str(e)}")
                st.info("SHAP analysis failed. This might be due to model compatibility issues.")
        
        else:
            st.error("SHAP library is not available. Please install it using: pip install shap")
    
    # Permutation Importance
    elif analysis_type == "Permutation Importance":
        st.subheader("🔄 Permutation Importance Analysis")
        
        with st.spinner("Calculating permutation importance..."):
            try:
                # Calculate permutation importance
                perm_importance = permutation_importance(
                    model, X_test, y_test,
                    n_repeats=10,
                    random_state=42,
                    scoring='accuracy' if task_type == 'classification' else 'r2'
                )
                
                # Create results DataFrame
                feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]
                perm_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance_Mean': perm_importance.importances_mean,
                    'Importance_Std': perm_importance.importances_std
                }).sort_values('Importance_Mean', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart with error bars
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=perm_df['Importance_Mean'].head(20),
                        y=perm_df['Feature'].head(20),
                        orientation='h',
                        error_x=dict(
                            type='data',
                            array=perm_df['Importance_Std'].head(20)
                        ),
                        name='Permutation Importance'
                    ))
                    
                    fig.update_layout(
                        title=f"Permutation Importance - {selected_model_name}",
                        xaxis_title="Importance",
                        yaxis_title="Features",
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Importance vs Standard Deviation
                    fig_scatter = px.scatter(
                        perm_df.head(20),
                        x='Importance_Std',
                        y='Importance_Mean',
                        hover_name='Feature',
                        title="Importance vs Variability",
                        labels={'Importance_Std': 'Standard Deviation', 'Importance_Mean': 'Mean Importance'}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Permutation importance table
                st.write("**Permutation Importance Rankings**")
                st.dataframe(perm_df.head(20), hide_index=True)
                
            except Exception as e:
                st.error(f"Error calculating permutation importance: {str(e)}")
    
    # Prediction Analysis
    elif analysis_type == "Prediction Analysis":
        st.subheader("🎯 Prediction Analysis")
        
        # Get predictions
        predictions = results[selected_model_name]['predictions']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if task_type == 'classification':
                # Prediction confidence analysis
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)
                    max_proba = np.max(proba, axis=1)
                    
                    fig_conf = px.histogram(
                        x=max_proba,
                        title="Prediction Confidence Distribution",
                        labels={'x': 'Max Probability', 'y': 'Count'},
                        nbins=20
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
                    
                    # Low confidence predictions
                    low_conf_threshold = st.slider("Low confidence threshold:", 0.5, 0.9, 0.7, 0.05)
                    low_conf_mask = max_proba < low_conf_threshold
                    st.metric("Low Confidence Predictions", f"{low_conf_mask.sum()} ({(low_conf_mask.sum()/len(predictions)*100):.1f}%)")
                
                else:
                    st.info("This model doesn't support probability predictions.")
            
            else:
                # Prediction vs Actual
                fig_pred = px.scatter(
                    x=y_test,
                    y=predictions,
                    title=f"Predictions vs Actual - {selected_model_name}",
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'}
                )
                
                # Add perfect prediction line
                min_val = min(y_test.min(), predictions.min())
                max_val = max(y_test.max(), predictions.max())
                fig_pred.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", dash="dash")
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
        
        with col2:
            # Residual analysis (for regression)
            if task_type == 'regression':
                residuals = y_test - predictions
                
                # Residual distribution
                fig_resid = px.histogram(
                    x=residuals,
                    title="Residual Distribution",
                    labels={'x': 'Residuals', 'y': 'Count'},
                    nbins=20
                )
                st.plotly_chart(fig_resid, use_container_width=True)
                
                # Residual statistics
                st.write("**Residual Statistics**")
                resid_stats = {
                    'Mean': residuals.mean(),
                    'Std': residuals.std(),
                    'Min': residuals.min(),
                    'Max': residuals.max(),
                    'Skewness': pd.Series(residuals).skew()
                }
                
                for stat, value in resid_stats.items():
                    st.metric(stat, f"{value:.4f}")
            
            else:
                # Confusion matrix details
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, predictions)
                
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title=f"Confusion Matrix - {selected_model_name}",
                    color_continuous_scale='Blues'
                )
                fig_cm.update_layout(
                    xaxis_title="Predicted Label",
                    yaxis_title="True Label"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
        
        # Detailed prediction analysis
        st.write("**Detailed Prediction Analysis**")
        
        # Create analysis DataFrame
        if hasattr(X_test, 'index'):
            analysis_df = X_test.copy()
            analysis_df['Actual'] = y_test
            analysis_df['Predicted'] = predictions
        else:
            feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
            analysis_df = pd.DataFrame(X_test, columns=feature_names)
            analysis_df['Actual'] = y_test
            analysis_df['Predicted'] = predictions
        
        if task_type == 'regression':
            analysis_df['Residual'] = analysis_df['Actual'] - analysis_df['Predicted']
            analysis_df['Abs_Error'] = np.abs(analysis_df['Residual'])
        else:
            analysis_df['Correct'] = analysis_df['Actual'] == analysis_df['Predicted']
        
        # Show worst predictions
        if task_type == 'regression':
            worst_predictions = analysis_df.nlargest(10, 'Abs_Error')
            st.write("**Worst Predictions (Highest Absolute Error)**")
        else:
            incorrect_predictions = analysis_df[analysis_df['Correct'] == False]
            if len(incorrect_predictions) > 0:
                worst_predictions = incorrect_predictions.head(10)
                st.write("**Incorrect Predictions**")
            else:
                worst_predictions = analysis_df.head(10)
                st.write("**All predictions are correct! Showing sample predictions:**")
        
        st.dataframe(worst_predictions, hide_index=True)

# Model comparison explainability
st.subheader("⚖️ Model Comparison - Explainability")

if len(results) > 1:
    st.write("**Feature Importance Comparison Across Models**")
    
    # Collect feature importances from models that support it
    importance_data = {}
    
    for model_name, result in results.items():
        model = result['model']
        if hasattr(model, 'feature_importances_'):
            feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]
            importance_data[model_name] = dict(zip(feature_names, model.feature_importances_))
    
    if importance_data:
        # Create comparison DataFrame
        importance_comparison = pd.DataFrame(importance_data).fillna(0)
        
        # Plot heatmap
        fig_heatmap = px.imshow(
            importance_comparison.T,
            aspect="auto",
            title="Feature Importance Heatmap Across Models",
            labels={'x': 'Features', 'y': 'Models', 'color': 'Importance'},
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Top consistent features
        feature_consistency = importance_comparison.mean(axis=1).sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Most Consistently Important Features**")
            top_consistent = feature_consistency.head(10)
            fig_consistent = px.bar(
                x=top_consistent.values,
                y=top_consistent.index,
                orientation='h',
                title="Average Importance Across Models"
            )
            fig_consistent.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_consistent, use_container_width=True)
        
        with col2:
            st.write("**Feature Importance Variance**")
            feature_variance = importance_comparison.var(axis=1).sort_values(ascending=False)
            top_variable = feature_variance.head(10)
            fig_variable = px.bar(
                x=top_variable.values,
                y=top_variable.index,
                orientation='h',
                title="Importance Variance Across Models",
                color=top_variable.values,
                color_continuous_scale='Reds'
            )
            fig_variable.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_variable, use_container_width=True)
    
    else:
        st.info("No models with feature importance available for comparison.")

# Export explainability results
st.subheader("💾 Export Explainability Results")

if st.button("📊 Download Feature Importance Data"):
    if hasattr(model, 'feature_importances_'):
        feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        csv = importance_df.to_csv(index=False)
        st.download_button(
            label="Download Feature Importance CSV",
            data=csv,
            file_name=f"{experiment_name}_{selected_model_name}_feature_importance.csv",
            mime="text/csv"
        )
    else:
        st.warning("Selected model doesn't support feature importance export.")
