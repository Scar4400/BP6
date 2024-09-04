import pandas as pd
from data_fetcher import get_all_data
from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from modeling import train_and_evaluate_models, predict_matches

def integrate_data():
    # Fetch data from all sources
    pinnacle_data, livescore_data, api_football_data = get_all_data()

    # Preprocess the data
    preprocessed_data = preprocess_data(pinnacle_data, livescore_data, api_football_data)

    # Engineer features
    engineered_data = engineer_features(preprocessed_data)

    return engineered_data

def train_models(data):
    X = data.drop('result', axis=1)
    y = data['result']

    models, results, feature_importances = train_and_evaluate_models(X, y)

    return models, results, feature_importances

def predict_new_matches(models, new_data):
    predictions = {}
    for name, model in models.items():
        predictions[name] = predict_matches(model, new_data)

    return predictions

def run_pipeline():
    # Integrate data
    integrated_data = integrate_data()

    # Train models
    models, results, feature_importances = train_models(integrated_data)

    # Print results
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
    new_matches = integrated_data.drop('result', axis=1).head(5)  # Replace with actual new match data
    predictions = predict_new_matches(models, new_matches)

    print("Predictions for new matches:")
    for name, preds in predictions.items():
        print(f"{name}:")
        print(preds)
        print()

if __name__ == "__main__":
    run_pipeline()

