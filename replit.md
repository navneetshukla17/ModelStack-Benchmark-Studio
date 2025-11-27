# ML Benchmark Tool

## Overview

The ML Benchmark Tool is a comprehensive Streamlit-based web application for machine learning model comparison and benchmarking. It provides a complete workflow for data management, model training, performance comparison, and model explainability analysis. The tool supports both classification and regression tasks with built-in datasets and custom CSV uploads, focusing on ensemble methods alongside individual model comparisons.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit with multi-page application structure
- **Layout**: Wide layout with expandable sidebar navigation
- **Visualization**: Plotly for interactive charts and graphs, with matplotlib/seaborn fallbacks
- **State Management**: Streamlit session state for data persistence across pages
- **Page Structure**: Main dashboard with dedicated pages for data management, model training, comparison, and explainability

### Backend Architecture
- **Core Framework**: Python with scikit-learn as the primary ML library
- **Model Support**: Comprehensive set of individual models (Random Forest, SVM, KNN, etc.) and ensemble methods (Voting, Bagging, Stacking)
- **Data Processing**: Automated task type detection and preprocessing pipelines
- **Experiment Tracking**: SQLite database for persistent experiment storage
- **Model Persistence**: Joblib for model serialization and storage

### Data Storage Solutions
- **Database**: SQLite for experiment tracking and results storage
- **Schema**: Single experiments table with JSON serialization for complex data
- **File Storage**: Local file system for model artifacts and temporary data
- **Session Storage**: In-memory storage via Streamlit session state for workflow continuity

### Key Design Patterns
- **Modular Architecture**: Separate utility modules for database, models, data processing, and visualization
- **Auto-detection**: Intelligent task type detection based on target variable characteristics
- **Ensemble Focus**: Built-in ensemble method implementations with automatic base model selection
- **Error Handling**: Graceful degradation when optional dependencies (SHAP) are unavailable

### Model Training Pipeline
- **Individual Models**: Support for 8-9 different algorithm types per task
- **Ensemble Methods**: Voting (soft/hard), Bagging, and Stacking classifiers/regressors
- **Cross-validation**: Built-in cross-validation for robust performance estimation
- **Metrics**: Task-appropriate metrics (accuracy, precision, recall, F1 for classification; MSE, MAE, R² for regression)

### Explainability Framework
- **Primary**: SHAP integration for advanced model interpretability
- **Fallback**: Permutation importance when SHAP is unavailable
- **Feature Analysis**: Interactive feature importance visualizations
- **Model-agnostic**: Works across all supported model types

## External Dependencies

### Core ML Libraries
- **scikit-learn**: Primary machine learning library for models, metrics, and preprocessing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing foundation

### Visualization and UI
- **streamlit**: Web application framework and UI components
- **plotly**: Interactive plotting and visualization library
- **matplotlib**: Static plotting fallback
- **seaborn**: Statistical visualization enhancements

### Optional Dependencies
- **shap**: Advanced model explainability (graceful degradation if unavailable)
- **joblib**: Model serialization and persistence

### Data Sources
- **Built-in Datasets**: Scikit-learn dataset collection (Iris, Wine, Breast Cancer, Housing, etc.)
- **Custom Data**: CSV file upload support with automatic preprocessing

### Database
- **sqlite3**: Lightweight database for experiment tracking and results persistence
- **json**: Serialization for complex data structures in database storage