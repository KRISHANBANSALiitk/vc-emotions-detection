import numpy as np
import pandas as pd
import yaml
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Load parameters from YAML
def load_params(params_path):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        max_features = params['build_features']['max_features']
        return max_features
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found at: {params_path}")
    except KeyError:
        raise KeyError("Check the YAML structure. 'build_features' or 'max_features' key is missing.")
    except Exception as e:
        raise Exception(f"Error while loading parameters: {e}")

# Fetch the data from data/processed
def fetch_data(train_path='./data/processed/train_processed.csv', test_path='./data/processed/test_processed.csv'):
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        return train_data, test_data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")
    except pd.errors.EmptyDataError:
        raise Exception("One of the CSV files is empty or unreadable.")
    except Exception as e:
        raise Exception(f"Error while fetching data: {e}")

# Apply Bag of Words on X
def tfidf(train_data, test_data, max_features):
    try:
        if 'content' not in train_data.columns or 'content' not in test_data.columns:
            raise KeyError("The 'content' column is missing in one of the datasets.")
        if 'sentiment' not in train_data.columns or 'sentiment' not in test_data.columns:
            raise KeyError("The 'sentiment' column is missing in one of the datasets.")

        train_data['content'] = train_data['content'].fillna('')
        test_data['content'] = test_data['content'].fillna('')

        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        return train_df, test_df

    except Exception as e:
        raise Exception(f"Error during Bag of Words transformation: {e}")

# Store the data inside data/features
def store_data(train_df, test_df, save_dir='data/features'):
    try:
        os.makedirs(save_dir, exist_ok=True)
        train_df.to_csv(os.path.join(save_dir, 'train_tfidf.csv'), index=False)
        test_df.to_csv(os.path.join(save_dir, 'test_tfidf.csv'), index=False)
    except Exception as e:
        raise Exception(f"Error while storing data: {e}")

# Define main pipeline
def main():
    try:
        max_features = load_params('params.yaml')
        train_data, test_data = fetch_data()
        train_df, test_df = tfidf(train_data, test_data, max_features)

        save_dir = 'data/features'
        store_data(train_df, test_df, save_dir)

        print("Feature engineering completed successfully.")
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == '__main__':
    main()
