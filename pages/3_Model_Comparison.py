import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Model Comparison - ML Benchmark Tool",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Model Comparison")
st.markdown("### Compare ensemble models with individual models across all metrics")

# Check if training results are available
if 'training_results' not in st.session_state or not st.session_state.training_results:
    st.warning("⚠️ No training results available. Please go to Model Training page first.")
    st.stop()

# Get data from session state
results = st.session_state.training_results
task_type = st.session_state.task_type
y_test = st.session_state.y_test
experiment_name = st.session_state.experiment_name

st.success(f"✅ Comparing {len(results)} models from experiment: **{experiment_name}**")

# Create results DataFrame
results_data = []
for model_name, result in results.items():
    row = {'Model': model_name, 'Type': 'Ensemble' if any(ens in model_name for ens in ['Voting', 'Bagging', 'Stacking']) else 'Individual'}
    row.update(result['metrics'])
    results_data.append(row)

results_df = pd.DataFrame(results_data)

# Overview metrics
st.subheader("📊 Performance Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    individual_models = results_df[results_df['Type'] == 'Individual']
    st.metric("Individual Models", len(individual_models))

with col2:
    ensemble_models = results_df[results_df['Type'] == 'Ensemble']
    st.metric("Ensemble Models", len(ensemble_models))

with col3:
    if task_type == 'classification':
        best_score = results_df['accuracy'].max()
        st.metric("Best Accuracy", f"{best_score:.4f}")
    else:
        best_score = results_df['r2'].max()
        st.metric("Best R²", f"{best_score:.4f}")

with col4:
    if task_type == 'classification':
        best_model = results_df.loc[results_df['accuracy'].idxmax(), 'Model']
    else:
        best_model = results_df.loc[results_df['r2'].idxmax(), 'Model']
    st.metric("Best Model", best_model[:15] + "..." if len(best_model) > 15 else best_model)

# Detailed comparison table
st.subheader("📋 Detailed Metrics Comparison")
st.dataframe(results_df, use_container_width=True, hide_index=True)

# Performance comparison visualizations
st.subheader("📈 Performance Visualizations")

# Metric selection for detailed analysis
if task_type == 'classification':
    available_metrics = ['accuracy', 'precision', 'recall', 'f1', 'cv_mean']
    default_metric = 'accuracy'
else:
    available_metrics = ['r2', 'mse', 'rmse', 'mae', 'cv_mean']
    default_metric = 'r2'

selected_metric = st.selectbox(
    "Select metric for detailed analysis:",
    available_metrics,
    index=available_metrics.index(default_metric)
)

# Create comparison charts
col1, col2 = st.columns(2)

with col1:
    # Bar chart comparison
    fig_bar = px.bar(
        results_df,
        x='Model',
        y=selected_metric,
        color='Type',
        title=f"{selected_metric.upper()} Comparison by Model Type",
        color_discrete_map={'Individual': '#3498db', 'Ensemble': '#e74c3c'}
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    # Box plot by type
    fig_box = px.box(
        results_df,
        x='Type',
        y=selected_metric,
        title=f"{selected_metric.upper()} Distribution by Model Type",
        color='Type',
        color_discrete_map={'Individual': '#3498db', 'Ensemble': '#e74c3c'}
    )
    st.plotly_chart(fig_box, use_container_width=True)

# Multi-metric radar chart
st.subheader("🎯 Multi-Metric Radar Chart")

selected_models = st.multiselect(
    "Select models to compare (max 5):",
    results_df['Model'].tolist(),
    default=results_df['Model'].head(3).tolist(),
    max_selections=5
)

if selected_models:
    # Prepare data for radar chart
    if task_type == 'classification':
        radar_metrics = ['accuracy', 'precision', 'recall', 'f1']
    else:
        # For regression, normalize MSE and RMSE (invert so higher is better)
        radar_metrics = ['r2', 'mae', 'mse', 'rmse']
    
    fig_radar = go.Figure()
    
    for model_name in selected_models:
        model_data = results_df[results_df['Model'] == model_name].iloc[0]
        
        if task_type == 'classification':
            values = [model_data[metric] for metric in radar_metrics]
        else:
            # For regression, normalize values (invert MAE, MSE, RMSE)
            values = []
            for metric in radar_metrics:
                if metric in ['mae', 'mse', 'rmse']:
                    # Invert and normalize these metrics (lower is better)
                    max_val = results_df[metric].max()
                    values.append(1 - (model_data[metric] / max_val))
                else:
                    values.append(model_data[metric])
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_metrics,
            fill='toself',
            name=model_name,
            opacity=0.6
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Multi-Metric Performance Comparison"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

# Cross-validation stability analysis
st.subheader("📊 Cross-Validation Stability Analysis")

col1, col2 = st.columns(2)

with col1:
    # CV mean vs std scatter plot
    fig_cv = px.scatter(
        results_df,
        x='cv_std',
        y='cv_mean',
        color='Type',
        size=[1] * len(results_df),  # Equal size for all points
        hover_name='Model',
        title="Cross-Validation: Mean vs Standard Deviation",
        labels={'cv_std': 'CV Standard Deviation', 'cv_mean': 'CV Mean Score'},
        color_discrete_map={'Individual': '#3498db', 'Ensemble': '#e74c3c'}
    )
    fig_cv.add_annotation(
        x=results_df['cv_std'].max() * 0.7,
        y=results_df['cv_mean'].min() * 1.1,
        text="Lower std = More stable",
        showarrow=True,
        arrowhead=2
    )
    st.plotly_chart(fig_cv, use_container_width=True)

with col2:
    # Model stability ranking
    results_df['stability_score'] = results_df['cv_mean'] / (results_df['cv_std'] + 1e-6)
    stability_ranking = results_df.nlargest(10, 'stability_score')[['Model', 'cv_mean', 'cv_std', 'stability_score']]
    
    fig_stability = px.bar(
        stability_ranking,
        x='Model',
        y='stability_score',
        title="Model Stability Ranking (Mean/Std)",
        color='stability_score',
        color_continuous_scale='viridis'
    )
    fig_stability.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_stability, use_container_width=True)

# Ensemble vs Individual Analysis
st.subheader("🤖 Ensemble vs Individual Model Analysis")

individual_df = results_df[results_df['Type'] == 'Individual']
ensemble_df = results_df[results_df['Type'] == 'Ensemble']

if len(individual_df) > 0 and len(ensemble_df) > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        # Average performance comparison
        comparison_data = []
        for metric in available_metrics:
            if metric in results_df.columns:
                comparison_data.append({
                    'Metric': metric,
                    'Individual (Avg)': individual_df[metric].mean(),
                    'Ensemble (Avg)': ensemble_df[metric].mean(),
                    'Improvement': ensemble_df[metric].mean() - individual_df[metric].mean()
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.write("**Average Performance Comparison**")
        st.dataframe(comparison_df, hide_index=True)
    
    with col2:
        # Performance improvement visualization
        fig_improvement = px.bar(
            comparison_df,
            x='Metric',
            y='Improvement',
            title="Ensemble Improvement over Individual Models",
            color='Improvement',
            color_continuous_scale='RdYlGn'
        )
        fig_improvement.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_improvement, use_container_width=True)

# Confusion Matrix (for classification tasks)
if task_type == 'classification':
    st.subheader("🎯 Confusion Matrix Analysis")
    
    selected_model_cm = st.selectbox(
        "Select model for confusion matrix:",
        list(results.keys())
    )
    
    if selected_model_cm:
        y_pred = results[selected_model_cm]['predictions']
        cm = confusion_matrix(y_test, y_pred)
        
        # Create confusion matrix heatmap
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title=f"Confusion Matrix - {selected_model_cm}",
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("**Classification Report**")
        st.dataframe(report_df)

# Prediction Analysis
st.subheader("🔍 Prediction Analysis")

if task_type == 'regression':
    # Prediction vs Actual scatter plots
    selected_models_pred = st.multiselect(
        "Select models for prediction analysis:",
        list(results.keys()),
        default=list(results.keys())[:2]
    )
    
    if selected_models_pred:
        n_models = len(selected_models_pred)
        cols = st.columns(min(n_models, 2))
        
        for i, model_name in enumerate(selected_models_pred):
            col_idx = i % 2
            with cols[col_idx]:
                y_pred = results[model_name]['predictions']
                
                fig_pred = px.scatter(
                    x=y_test,
                    y=y_pred,
                    title=f"Predictions vs Actual - {model_name}",
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'}
                )
                
                # Add perfect prediction line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                fig_pred.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", dash="dash")
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)

# Model performance summary
st.subheader("📈 Performance Summary")

# Best performing models summary
st.write("**Top 5 Performing Models**")
if task_type == 'classification':
    top_models = results_df.nlargest(5, 'accuracy')[['Model', 'Type', 'accuracy', 'precision', 'recall', 'f1']]
else:
    top_models = results_df.nlargest(5, 'r2')[['Model', 'Type', 'r2', 'mse', 'mae']]

st.dataframe(top_models, hide_index=True)

# Key insights
st.subheader("💡 Key Insights")

insights = []

# Best overall model
if task_type == 'classification':
    best_model = results_df.loc[results_df['accuracy'].idxmax()]
    insights.append(f"🏆 **Best Overall Model**: {best_model['Model']} with {best_model['accuracy']:.4f} accuracy")
else:
    best_model = results_df.loc[results_df['r2'].idxmax()]
    insights.append(f"🏆 **Best Overall Model**: {best_model['Model']} with {best_model['r2']:.4f} R²")

# Ensemble vs Individual performance
if len(ensemble_df) > 0 and len(individual_df) > 0:
    if task_type == 'classification':
        ensemble_avg = ensemble_df['accuracy'].mean()
        individual_avg = individual_df['accuracy'].mean()
        metric_name = 'accuracy'
    else:
        ensemble_avg = ensemble_df['r2'].mean()
        individual_avg = individual_df['r2'].mean()
        metric_name = 'R²'
    
    if ensemble_avg > individual_avg:
        improvement = ((ensemble_avg - individual_avg) / individual_avg) * 100
        insights.append(f"📈 **Ensemble Advantage**: Ensemble models perform {improvement:.1f}% better on average ({metric_name})")
    else:
        decline = ((individual_avg - ensemble_avg) / individual_avg) * 100
        insights.append(f"📉 **Individual Advantage**: Individual models perform {decline:.1f}% better on average ({metric_name})")

# Most stable model
most_stable = results_df.loc[results_df['stability_score'].idxmax()]
insights.append(f"⚖️ **Most Stable Model**: {most_stable['Model']} (lowest CV variance)")

for insight in insights:
    st.info(insight)

# Export results
st.subheader("💾 Export Results")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Download Results CSV"):
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{experiment_name}_results.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📈 Download Performance Summary"):
        summary_data = {
            'experiment_name': experiment_name,
            'task_type': task_type,
            'total_models': len(results_df),
            'best_model': best_model['Model'],
            'best_score': best_model[default_metric],
            'insights': insights
        }
        
        summary_text = f"""
# {experiment_name} - Performance Summary

## Experiment Details
- Task Type: {task_type.title()}
- Total Models: {len(results_df)}
- Individual Models: {len(individual_df)}
- Ensemble Models: {len(ensemble_df)}

## Best Performance
- Best Model: {best_model['Model']}
- Best Score: {best_model[default_metric]:.4f}

## Key Insights
{chr(10).join(insights)}

## Detailed Results
{results_df.to_string(index=False)}
        """
        
        st.download_button(
            label="Download Summary",
            data=summary_text,
            file_name=f"{experiment_name}_summary.txt",
            mime="text/plain"
        )
