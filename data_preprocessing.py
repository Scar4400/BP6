import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_pinnacle_data(data):
    df = pd.DataFrame(data['sports'])
    # Extract relevant features
    df = df[['id', 'name', 'events']]
    df = df.explode('events').reset_index(drop=True)
    df = pd.concat([df.drop('events', axis=1), pd.json_normalize(df['events'])], axis=1)
    return df

def preprocess_livescore_data(data):
    df = pd.DataFrame(data['results'])
    # Extract relevant features
    df = df[['id', 'name', 'country', 'sport', 'type']]
    return df

def preprocess_api_football_data(data):
    df = pd.DataFrame(data['api']['odds'])
    # Extract relevant features
    df = df[['fixture', 'bookmakers']]
    df = df.explode('bookmakers').reset_index(drop=True)
    df = pd.concat([df.drop('bookmakers', axis=1), pd.json_normalize(df['bookmakers'])], axis=1)
    return df

def combine_data(pinnacle_df, livescore_df, api_football_df):
    # Merge dataframes based on common columns or indices
    combined_df = pd.merge(pinnacle_df, livescore_df, on='id', how='outer')
    combined_df = pd.merge(combined_df, api_football_df, left_on='id', right_on='fixture', how='outer')
    return combined_df

def handle_missing_values(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    # Impute numeric columns with mean
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    # Impute categorical columns with most frequent value
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    return df

def encode_categorical_variables(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names(categorical_columns))

    # Combine encoded categorical variables with numeric variables
    final_df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
    return final_df

def scale_numeric_features(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def preprocess_data(pinnacle_data, livescore_data, api_football_data):
    pinnacle_df = preprocess_pinnacle_data(pinnacle_data)
    livescore_df = preprocess_livescore_data(livescore_data)
    api_football_df = preprocess_api_football_data(api_football_data)

    combined_df = combine_data(pinnacle_df, livescore_df, api_football_df)
    combined_df = handle_missing_values(combined_df)
    combined_df = encode_categorical_variables(combined_df)
    combined_df = scale_numeric_features(combined_df)

    return combined_df

if __name__ == "__main__":
    from data_fetcher import get_all_data

    pinnacle_data, livescore_data, api_football_data = get_all_data()
    preprocessed_data = preprocess_data(pinnacle_data, livescore_data, api_football_data)
    print(preprocessed_data.head())
    print(preprocessed_data.info())
