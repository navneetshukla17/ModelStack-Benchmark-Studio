import sqlite3
import json
from datetime import datetime
import os

def init_database():
    """
    Initialize the SQLite database for experiment tracking.
    """
    conn = sqlite3.connect('ml_experiments.db')
    cursor = conn.cursor()
    
    # Create experiments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            models_tested TEXT NOT NULL,
            task_type TEXT NOT NULL,
            best_model TEXT NOT NULL,
            best_score REAL NOT NULL,
            timestamp TEXT NOT NULL,
            results TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def save_experiment(name, dataset_name, models_tested, task_type, best_model, best_score, results):
    """
    Save an experiment to the database.
    """
    conn = sqlite3.connect('ml_experiments.db')
    cursor = conn.cursor()
    
    # Convert models list to string
    models_str = ', '.join(models_tested)
    
    # Convert results to JSON string
    results_json = json.dumps(results)
    
    # Get current timestamp
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT INTO experiments 
        (name, dataset_name, models_tested, task_type, best_model, best_score, timestamp, results)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, dataset_name, models_str, task_type, best_model, best_score, timestamp, results_json))
    
    conn.commit()
    conn.close()
    
    return cursor.lastrowid

def get_all_experiments():
    """
    Retrieve all experiments from the database.
    """
    if not os.path.exists('ml_experiments.db'):
        return []
    
    conn = sqlite3.connect('ml_experiments.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, name, dataset_name, models_tested, task_type, best_model, best_score, timestamp
        FROM experiments
        ORDER BY timestamp DESC
    ''')
    
    experiments = cursor.fetchall()
    conn.close()
    
    return experiments

def get_experiment_by_id(experiment_id):
    """
    Retrieve a specific experiment by ID.
    """
    conn = sqlite3.connect('ml_experiments.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM experiments WHERE id = ?
    ''', (experiment_id,))
    
    experiment = cursor.fetchone()
    conn.close()
    
    if experiment:
        # Parse results JSON
        results = json.loads(experiment[8])
        return {
            'id': experiment[0],
            'name': experiment[1],
            'dataset_name': experiment[2],
            'models_tested': experiment[3].split(', '),
            'task_type': experiment[4],
            'best_model': experiment[5],
            'best_score': experiment[6],
            'timestamp': experiment[7],
            'results': results
        }
    
    return None

def delete_experiment(experiment_id):
    """
    Delete an experiment from the database.
    """
    conn = sqlite3.connect('ml_experiments.db')
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM experiments WHERE id = ?', (experiment_id,))
    
    conn.commit()
    conn.close()
    
    return cursor.rowcount > 0

def get_experiments_summary():
    """
    Get summary statistics of all experiments.
    """
    if not os.path.exists('ml_experiments.db'):
        return {
            'total_experiments': 0,
            'classification_experiments': 0,
            'regression_experiments': 0,
            'best_classification_score': 0,
            'best_regression_score': 0,
            'most_used_models': [],
            'recent_experiments': []
        }
    
    conn = sqlite3.connect('ml_experiments.db')
    cursor = conn.cursor()
    
    # Total experiments
    cursor.execute('SELECT COUNT(*) FROM experiments')
    total_experiments = cursor.fetchone()[0]
    
    # Classification experiments
    cursor.execute('SELECT COUNT(*) FROM experiments WHERE task_type = "classification"')
    classification_experiments = cursor.fetchone()[0]
    
    # Regression experiments
    cursor.execute('SELECT COUNT(*) FROM experiments WHERE task_type = "regression"')
    regression_experiments = cursor.fetchone()[0]
    
    # Best scores
    cursor.execute('SELECT MAX(best_score) FROM experiments WHERE task_type = "classification"')
    best_classification_score = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT MAX(best_score) FROM experiments WHERE task_type = "regression"')
    best_regression_score = cursor.fetchone()[0] or 0
    
    # Most used models
    cursor.execute('SELECT best_model, COUNT(*) as count FROM experiments GROUP BY best_model ORDER BY count DESC LIMIT 5')
    most_used_models = cursor.fetchall()
    
    # Recent experiments
    cursor.execute('SELECT name, best_model, best_score, timestamp FROM experiments ORDER BY timestamp DESC LIMIT 5')
    recent_experiments = cursor.fetchall()
    
    conn.close()
    
    return {
        'total_experiments': total_experiments,
        'classification_experiments': classification_experiments,
        'regression_experiments': regression_experiments,
        'best_classification_score': best_classification_score,
        'best_regression_score': best_regression_score,
        'most_used_models': most_used_models,
        'recent_experiments': recent_experiments
    }

def update_experiment(experiment_id, **kwargs):
    """
    Update an existing experiment.
    """
    conn = sqlite3.connect('ml_experiments.db')
    cursor = conn.cursor()
    
    # Build update query dynamically
    update_fields = []
    values = []
    
    for field, value in kwargs.items():
        if field in ['name', 'dataset_name', 'models_tested', 'task_type', 'best_model', 'best_score', 'results']:
            update_fields.append(f"{field} = ?")
            if field == 'models_tested' and isinstance(value, list):
                values.append(', '.join(value))
            elif field == 'results':
                values.append(json.dumps(value))
            else:
                values.append(value)
    
    if update_fields:
        query = f"UPDATE experiments SET {', '.join(update_fields)} WHERE id = ?"
        values.append(experiment_id)
        
        cursor.execute(query, values)
        conn.commit()
    
    conn.close()
    
    return cursor.rowcount > 0
