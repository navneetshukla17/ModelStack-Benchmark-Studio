from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    BaggingClassifier, BaggingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
import numpy as np

def get_model_configs(task_type='classification'):
    """
    Get model configurations for individual and ensemble models.
    """
    if task_type == 'classification':
        individual_models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        }
        
        ensemble_models = {
            'Voting Classifier (Soft)': 'voting_soft',
            'Voting Classifier (Hard)': 'voting_hard',
            'Bagging Classifier': 'bagging',
            'Stacking Classifier': 'stacking'
        }
        
    else:  # regression
        individual_models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'SVR': SVR(),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'AdaBoost': AdaBoostRegressor(random_state=42)
        }
        
        ensemble_models = {
            'Voting Regressor': 'voting',
            'Bagging Regressor': 'bagging',
            'Stacking Regressor': 'stacking'
        }
    
    return {
        'individual': individual_models,
        'ensemble': ensemble_models
    }

def train_ensemble_models(ensemble_type, base_models, X_train, y_train, task_type='classification', random_state=42):
    """
    Train ensemble models using base models.
    """
    if task_type == 'classification':
        if ensemble_type == 'voting_soft':
            # Ensure all models support probability
            prob_models = []
            for model in base_models:
                if hasattr(model, 'predict_proba'):
                    prob_models.append(model)
                elif hasattr(model, 'decision_function'):
                    # For SVM, enable probability
                    if hasattr(model, 'probability'):
                        model.probability = True
                    prob_models.append(model)
            
            if len(prob_models) < 2:
                # Fallback to hard voting
                ensemble = VotingClassifier(
                    estimators=[(f'model_{i}', model) for i, model in enumerate(base_models)],
                    voting='hard'
                )
            else:
                ensemble = VotingClassifier(
                    estimators=[(f'model_{i}', model) for i, model in enumerate(prob_models)],
                    voting='soft'
                )
                
        elif ensemble_type == 'voting_hard':
            ensemble = VotingClassifier(
                estimators=[(f'model_{i}', model) for i, model in enumerate(base_models)],
                voting='hard'
            )
            
        elif ensemble_type == 'bagging':
            # Use Random Forest as base estimator for bagging
            ensemble = BaggingClassifier(
                base_estimator=DecisionTreeClassifier(random_state=random_state),
                n_estimators=10,
                random_state=random_state
            )
            
        elif ensemble_type == 'stacking':
            # Use logistic regression as meta-learner
            ensemble = StackingClassifier(
                estimators=[(f'model_{i}', model) for i, model in enumerate(base_models)],
                final_estimator=LogisticRegression(random_state=random_state),
                cv=5
            )
        else:
            # Default fallback for unknown ensemble types
            ensemble = VotingClassifier(
                estimators=[(f'model_{i}', model) for i, model in enumerate(base_models)],
                voting='hard'
            )
    
    else:  # regression
        if ensemble_type == 'voting':
            ensemble = VotingRegressor(
                estimators=[(f'model_{i}', model) for i, model in enumerate(base_models)]
            )
            
        elif ensemble_type == 'bagging':
            ensemble = BaggingRegressor(
                base_estimator=DecisionTreeRegressor(random_state=random_state),
                n_estimators=10,
                random_state=random_state
            )
            
        elif ensemble_type == 'stacking':
            ensemble = StackingRegressor(
                estimators=[(f'model_{i}', model) for i, model in enumerate(base_models)],
                final_estimator=LinearRegression(),
                cv=5
            )
        else:
            # Default fallback for unknown ensemble types
            ensemble = VotingRegressor(
                estimators=[(f'model_{i}', model) for i, model in enumerate(base_models)]
            )
    
    # Train the ensemble
    ensemble.fit(X_train, y_train)
    return ensemble

def evaluate_model_complexity(model):
    """
    Evaluate the complexity of a model.
    """
    complexity_info = {
        'model_type': type(model).__name__,
        'parameters': model.get_params(),
        'complexity_score': 1.0  # Default complexity score
    }
    
    # Adjust complexity based on model type
    if hasattr(model, 'n_estimators'):
        complexity_info['complexity_score'] = model.n_estimators / 100.0
    elif hasattr(model, 'max_depth') and model.max_depth:
        complexity_info['complexity_score'] = model.max_depth / 10.0
    elif hasattr(model, 'C'):  # SVM
        complexity_info['complexity_score'] = np.log10(model.C + 1)
    
    return complexity_info

def get_model_interpretability_score(model):
    """
    Assign interpretability scores to different model types.
    """
    interpretable_models = {
        'LinearRegression': 1.0,
        'LogisticRegression': 1.0,
        'DecisionTreeClassifier': 0.9,
        'DecisionTreeRegressor': 0.9,
        'GaussianNB': 0.8,
        'KNeighborsClassifier': 0.7,
        'KNeighborsRegressor': 0.7,
        'RandomForestClassifier': 0.6,
        'RandomForestRegressor': 0.6,
        'GradientBoostingClassifier': 0.5,
        'GradientBoostingRegressor': 0.5,
        'SVC': 0.3,
        'SVR': 0.3,
        'VotingClassifier': 0.4,
        'VotingRegressor': 0.4,
        'BaggingClassifier': 0.5,
        'BaggingRegressor': 0.5,
        'StackingClassifier': 0.3,
        'StackingRegressor': 0.3
    }
    
    model_name = type(model).__name__
    return interpretable_models.get(model_name, 0.5)

def create_model_summary(model, metrics, task_type):
    """
    Create a comprehensive summary of a trained model.
    """
    summary = {
        'model_name': type(model).__name__,
        'task_type': task_type,
        'metrics': metrics,
        'complexity': evaluate_model_complexity(model),
        'interpretability_score': get_model_interpretability_score(model),
        'parameters': model.get_params()
    }
    
    # Add model-specific information
    if hasattr(model, 'feature_importances_'):
        summary['has_feature_importance'] = True
    else:
        summary['has_feature_importance'] = False
    
    if hasattr(model, 'predict_proba'):
        summary['supports_probability'] = True
    else:
        summary['supports_probability'] = False
    
    return summary
