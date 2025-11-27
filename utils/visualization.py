import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_metrics_comparison_chart(df_experiments):
    """
    Create a comprehensive metrics comparison chart for experiments.
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Over Time', 'Task Type Distribution', 
                       'Best Score Distribution', 'Model Performance'),
        specs=[[{"secondary_y": False}, {"type": "pie"}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Convert timestamp to datetime
    df_experiments['Timestamp'] = pd.to_datetime(df_experiments['Timestamp'])
    
    # 1. Performance over time
    fig.add_trace(
        go.Scatter(
            x=df_experiments['Timestamp'],
            y=df_experiments['Best Score'],
            mode='lines+markers',
            name='Best Score',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # 2. Task type distribution (pie chart)
    task_counts = df_experiments['Task Type'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=task_counts.index,
            values=task_counts.values,
            name="Task Types",
            marker_colors=['#FF6B6B', '#4ECDC4']
        ),
        row=1, col=2
    )
    
    # 3. Best score distribution (histogram)
    fig.add_trace(
        go.Histogram(
            x=df_experiments['Best Score'],
            name='Score Distribution',
            marker_color='green',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # 4. Model performance (bar chart)
    model_counts = df_experiments['Best Model'].value_counts().head(5)
    fig.add_trace(
        go.Bar(
            x=model_counts.index,
            y=model_counts.values,
            name='Model Frequency',
            marker_color='purple'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Experiment Overview Dashboard",
        showlegend=False,
        height=600
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Best Score", row=2, col=1)
    fig.update_xaxes(title_text="Models", row=2, col=2, tickangle=45)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    return fig

def create_model_performance_radar(results_df, metrics, models_to_compare=None):
    """
    Create a radar chart comparing model performance across multiple metrics.
    """
    if models_to_compare is None:
        models_to_compare = results_df['Model'].head(5).tolist()
    
    fig = go.Figure()
    
    for model in models_to_compare:
        model_data = results_df[results_df['Model'] == model].iloc[0]
        values = [model_data[metric] for metric in metrics]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=model,
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Comparison (Radar Chart)"
    )
    
    return fig

def create_ensemble_vs_individual_comparison(results_df, metric='accuracy'):
    """
    Create a comparison chart between ensemble and individual models.
    """
    # Separate ensemble and individual models
    ensemble_models = results_df[results_df['Type'] == 'Ensemble']
    individual_models = results_df[results_df['Type'] == 'Individual']
    
    fig = go.Figure()
    
    # Box plot for individual models
    if len(individual_models) > 0:
        fig.add_trace(go.Box(
            y=individual_models[metric],
            name='Individual Models',
            marker_color='lightblue',
            boxmean=True
        ))
    
    # Box plot for ensemble models
    if len(ensemble_models) > 0:
        fig.add_trace(go.Box(
            y=ensemble_models[metric],
            name='Ensemble Models',
            marker_color='lightcoral',
            boxmean=True
        ))
    
    fig.update_layout(
        title=f'{metric.title()} Comparison: Ensemble vs Individual Models',
        yaxis_title=metric.title(),
        xaxis_title='Model Type'
    )
    
    return fig

def create_cross_validation_stability_plot(results_df):
    """
    Create a scatter plot showing cross-validation stability (mean vs std).
    """
    fig = px.scatter(
        results_df,
        x='cv_std',
        y='cv_mean',
        color='Type',
        size=[10] * len(results_df),
        hover_name='Model',
        title='Cross-Validation Stability Analysis',
        labels={
            'cv_std': 'CV Standard Deviation (Lower = More Stable)',
            'cv_mean': 'CV Mean Score (Higher = Better)'
        },
        color_discrete_map={'Individual': '#3498db', 'Ensemble': '#e74c3c'}
    )
    
    # Add annotations for interpretation
    fig.add_annotation(
        x=results_df['cv_std'].max() * 0.8,
        y=results_df['cv_mean'].min() * 1.1,
        text="Ideal: High Mean, Low Std",
        showarrow=True,
        arrowhead=2,
        bgcolor="yellow",
        opacity=0.7
    )
    
    return fig

def create_feature_importance_comparison(importance_data, top_n=15):
    """
    Create a heatmap comparing feature importance across models.
    """
    # Convert to DataFrame if it's a dictionary
    if isinstance(importance_data, dict):
        importance_df = pd.DataFrame(importance_data)
    else:
        importance_df = importance_data
    
    # Get top N features by average importance
    avg_importance = importance_df.mean(axis=1).sort_values(ascending=False)
    top_features = avg_importance.head(top_n).index
    
    # Create heatmap
    fig = px.imshow(
        importance_df.loc[top_features].T,
        aspect="auto",
        title=f"Top {top_n} Features - Importance Comparison Across Models",
        labels={'x': 'Features', 'y': 'Models', 'color': 'Importance'},
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=max(400, len(importance_df.columns) * 30)
    )
    
    return fig

def create_prediction_analysis_plot(y_true, y_pred, model_name, task_type='regression'):
    """
    Create prediction analysis plots.
    """
    if task_type == 'regression':
        # Scatter plot for regression
        fig = go.Figure()
        
        # Actual vs Predicted scatter
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', opacity=0.6),
            text=[f'Actual: {a:.2f}<br>Predicted: {p:.2f}' for a, p in zip(y_true, y_pred)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Prediction Analysis - {model_name}',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values'
        )
        
    else:
        # Confusion matrix for classification
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title=f'Confusion Matrix - {model_name}',
            labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'},
            color_continuous_scale='Blues'
        )
    
    return fig

def create_residual_analysis_plot(y_true, y_pred, model_name):
    """
    Create residual analysis plots for regression models.
    """
    residuals = y_true - y_pred
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residuals vs Predicted', 'Residual Distribution')
    )
    
    # Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue', opacity=0.6)
        ),
        row=1, col=1
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Residual distribution (histogram)
    fig.add_trace(
        go.Histogram(
            x=residuals,
            name='Distribution',
            marker_color='green',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text=f"Residual Analysis - {model_name}",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_xaxes(title_text="Residuals", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    return fig

def create_learning_curve_plot(train_scores, val_scores, train_sizes):
    """
    Create learning curves to analyze model performance vs training size.
    """
    fig = go.Figure()
    
    # Training scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=np.mean(train_scores, axis=1),
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue'),
        error_y=dict(
            type='data',
            array=np.std(train_scores, axis=1),
            visible=True
        )
    ))
    
    # Validation scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=np.mean(val_scores, axis=1),
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red'),
        error_y=dict(
            type='data',
            array=np.std(val_scores, axis=1),
            visible=True
        )
    ))
    
    fig.update_layout(
        title='Learning Curves',
        xaxis_title='Training Set Size',
        yaxis_title='Score',
        hovermode='x'
    )
    
    return fig
