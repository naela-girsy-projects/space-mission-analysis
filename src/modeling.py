"""
Predictive modeling functions for space mission analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def prepare_features_and_target(df, target_column, feature_columns):
    """
    Prepare features and target variable for modeling.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        target_column (str): Name of the target column
        feature_columns (list): List of feature column names
        
    Returns:
        tuple: X (features) and y (target)
    """
    # Select features and target
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    return X, y


def build_preprocessing_pipeline(categorical_features, numerical_features):
    """
    Build scikit-learn preprocessing pipeline for mixed data types.
    
    Args:
        categorical_features (list): List of categorical feature names
        numerical_features (list): List of numerical feature names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    # Transformer for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Transformer for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combine transformers in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor


def train_mission_success_model(df, feature_columns, target_column='Mission_Status', 
                              test_size=0.2, random_state=42):
    """
    Train a model to predict mission success.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        feature_columns (dict): Dictionary with 'categorical' and 'numerical' features
        target_column (str): Name of the target column (success/failure)
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary with model, performance metrics, and other information
    """
    results = {}
    
    # Prepare data
    # For binary classification, convert target to binary (Success = 1, Otherwise = 0)
    df_model = df.copy()
    df_model['Success_Binary'] = (df_model[target_column] == 'Success').astype(int)
    
    # Select features and target
    X, y = prepare_features_and_target(
        df_model, 
        'Success_Binary', 
        feature_columns['categorical'] + feature_columns['numerical']
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        feature_columns['categorical'], 
        feature_columns['numerical']
    )
    
    # Create model pipelines
    models = {
        'logistic_regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
        ]),
        'random_forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=random_state, n_estimators=100))
        ]),
        'svm': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(random_state=random_state, probability=True))
        ]),
        'neural_network': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MLPClassifier(random_state=random_state, max_iter=1000, hidden_layer_sizes=(100, 50)))
        ])
    }
    
    # Train and evaluate models
    model_results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store results
        model_results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
    
    # Find best model
    best_model_name = max(model_results, key=lambda x: model_results[x]['f1_score'])
    best_model = model_results[best_model_name]
    
    # Store information in results
    results['all_models'] = model_results
    results['best_model'] = {
        'name': best_model_name,
        'model': best_model['model'],
        'accuracy': best_model['accuracy'],
        'precision': best_model['precision'],
        'recall': best_model['recall'],
        'f1_score': best_model['f1_score'],
        'confusion_matrix': best_model['confusion_matrix']
    }
    
    # Calculate feature importance if best model is Random Forest
    if best_model_name == 'random_forest':
        # Get feature names after preprocessing
        categorical_feat_count = len(preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(
            feature_columns['categorical']))
        numerical_feat_count = len(feature_columns['numerical'])
        
        # Get feature importances
        importances = best_model['model'].named_steps['classifier'].feature_importances_
        
        # Since we can't easily get the transformed feature names, create indices
        feature_indices = range(len(importances))
        
        # Create a dataframe of feature importances
        feature_importance = pd.DataFrame({
            'Feature_Index': feature_indices,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        results['feature_importance'] = feature_importance
    
    return results


def build_cost_estimation_model(df, feature_columns, cost_column='Cost_USD', 
                              test_size=0.2, random_state=42):
    """
    Build a model to estimate mission costs.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        feature_columns (dict): Dictionary with 'categorical' and 'numerical' features
        cost_column (str): Name of the cost column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary with model, performance metrics, and other information
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {}
    
    # Prepare data - filter out rows with missing cost
    df_model = df.dropna(subset=[cost_column]).copy()
    
    # Select features and target
    X, y = prepare_features_and_target(
        df_model, 
        cost_column, 
        feature_columns['categorical'] + feature_columns['numerical']
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        feature_columns['categorical'], 
        feature_columns['numerical']
    )
    
    # Create model pipelines
    regression_models = {
        'linear_regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'random_forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=random_state, n_estimators=100))
        ])
    }
    
    # Train and evaluate models
    model_results = {}
    for name, model in regression_models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        model_results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }
    
    # Find best model (based on R² score)
    best_model_name = max(model_results, key=lambda x: model_results[x]['r2_score'])
    best_model = model_results[best_model_name]
    
    # Store information in results
    results['all_models'] = model_results
    results['best_model'] = {
        'name': best_model_name,
        'model': best_model['model'],
        'mse': best_model['mse'],
        'rmse': best_model['rmse'],
        'mae': best_model['mae'],
        'r2_score': best_model['r2_score']
    }
    
    return results


def classify_mission_type(df, feature_columns, type_column, 
                        test_size=0.2, random_state=42):
    """
    Build a classifier to categorize mission types.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        feature_columns (dict): Dictionary with 'categorical' and 'numerical' features
        type_column (str): Name of the mission type column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary with model, performance metrics, and other information
    """
    from sklearn.metrics import classification_report
    
    results = {}
    
    # Prepare data
    df_model = df.copy()
    
    # Select features and target
    X, y = prepare_features_and_target(
        df_model, 
        type_column, 
        feature_columns['categorical'] + feature_columns['numerical']
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        feature_columns['categorical'], 
        feature_columns['numerical']
    )
    
    # Create model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=random_state, n_estimators=100))
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Store results
    results['model'] = model
    results['accuracy'] = accuracy
    results['classification_report'] = class_report
    
    return results


def predict_new_mission(model, new_data, feature_columns, preprocessor=None):
    """
    Predict outcomes for new mission data.
    
    Args:
        model: Trained model
        new_data (pandas.DataFrame): New mission data
        feature_columns (list): List of feature column names
        preprocessor (optional): Preprocessing pipeline if not included in model
        
    Returns:
        dict: Prediction results
    """
    # Ensure new_data has all required columns
    missing_columns = [col for col in feature_columns if col not in new_data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in new data: {missing_columns}")
    
    # Select features
    X_new = new_data[feature_columns].copy()
    
    # Preprocess if necessary
    if preprocessor:
        X_new_processed = preprocessor.transform(X_new)
    else:
        # Assume preprocessing is part of the model pipeline
        X_new_processed = X_new
    
    # Make predictions
    if hasattr(model, 'predict_proba'):
        # For classification with probability estimates
        y_prob = model.predict_proba(X_new)
        y_pred = model.predict(X_new)
        
        return {
            'prediction': y_pred,
            'probability': y_prob
        }
    else:
        # For regression or classification without probability
        y_pred = model.predict(X_new)
        
        return {
            'prediction': y_pred
        }
# In modeling.py

# Add these new functions to your existing modeling.py file

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance on both training and test sets.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Dictionary containing various performance metrics
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Get probabilities for AUC calculation
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'auc': roc_auc_score(y_test, y_test_proba)
    }

def plot_feature_importance(feature_importances, feature_names, top_n=20):
    """
    Plot feature importance graph.
    
    Args:
        feature_importances: Array of feature importance scores
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Create DataFrame of features and their importance scores
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(top_n), x='Importance', y='Feature')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.show()

def plot_model_comparison(model_results):
    """
    Plot comparison of different models' performance.
    
    Args:
        model_results: Dictionary containing results for each model
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    # Prepare data for plotting
    comparison_data = pd.DataFrame([
        {
            'Model': model_name,
            'Test Accuracy': results['test_accuracy'],
            'AUC': results['auc']
        }
        for model_name, results in model_results.items()
    ])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    metrics_data = pd.melt(comparison_data, 
                          id_vars=['Model'], 
                          value_vars=['Test Accuracy', 'AUC'],
                          var_name='Metric')
    
    sns.barplot(x='Model', y='value', hue='Metric', data=metrics_data)
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()