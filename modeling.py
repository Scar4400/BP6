import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_model(X_train, y_train, model_type='rf'):
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'xgb':
        model = XGBClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    elif model_type == 'lgbm':
        model = LGBMClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'num_leaves': [31, 63, 127]
        }
    elif model_type == 'lr':
        model = LogisticRegression(random_state=42)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    else:
        raise ValueError("Invalid model type. Choose 'rf', 'xgb', 'lgbm', or 'lr'.")

    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model doesn't have feature importances or coefficients.")

    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    return feature_importance

def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {
        'Random Forest': train_model(X_train, y_train, 'rf'),
        'XGBoost': train_model(X_train, y_train, 'xgb'),
        'LightGBM': train_model(X_train, y_train, 'lgbm'),
        'Logistic Regression': train_model(X_train, y_train, 'lr')
    }

    results = {}
    feature_importances = {}

    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test)
        feature_importances[name] = get_feature_importance(model, X.columns)

    return models, results, feature_importances

def predict_matches(model, X):
    return model.predict_proba(X)

if __name__ == "__main__":
    from data_fetcher import get_all_data
    from data_preprocessing import preprocess_data
    from feature_engineering import engineer_features

    pinnacle_data, livescore_data, api_football_data = get_all_data()
    preprocessed_data = preprocess_data(pinnacle_data, livescore_data, api_football_data)
    engineered_data = engineer_features(preprocessed_data)

    X = engineered_data.drop('result', axis=1)
    y = engineered_data['result']

    models, results, feature_importances = train_and_evaluate_models(X, y)

    print("Model Results:")
    for name, metrics in results.items():
        print(f"{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()

    print("Top 10 Feature Importances:")
    for name, importance in feature_importances.items():
        print(f"{name}:")
        print(importance.head(10))
        print()

    # Example of predicting new matches
    new_matches = X.head(5)  # Replace with actual new match data
    predictions = predict_matches(models['Random Forest'], new_matches)
    print("Predictions for new matches:")
    print(predictions)
