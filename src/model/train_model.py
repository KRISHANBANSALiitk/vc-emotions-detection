import numpy as np
import pandas as pd
import pickle
import yaml

from sklearn.ensemble import GradientBoostingClassifier

# Load parameters from YAML
def load_params(params_path):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        n_estimators = params['train_model']['n_estimators']
        learning_rate = params['train_model']['learning_rate']
        return n_estimators, learning_rate
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found at: {params_path}")
    except KeyError:
        raise KeyError("Check the YAML structure. 'train_model', 'n_estimators', or 'learning_rate' key is missing.")
    except Exception as e:
        raise Exception(f"Error while loading parameters: {e}")

# Fetch the data from data/features
def fetch_data(train_path='./data/features/train_bow.csv'):
    try:
        train_data = pd.read_csv(train_path)

        if train_data.shape[1] < 2:
            raise ValueError("Training data must have at least one feature and one label column.")

        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values

        return X_train, y_train

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")
    except pd.errors.EmptyDataError:
        raise Exception("CSV file is empty or unreadable.")
    except Exception as e:
        raise Exception(f"Error while fetching training data: {e}")

# Define and train Gradient Boosting Model
def GBM_model(X_train, y_train, n_estimators, learning_rate):
    try:
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        clf.fit(X_train, y_train)
        return clf
    except Exception as e:
        raise Exception(f"Error while training Gradient Boosting Model: {e}")

# Save the model
def save_model(model, model_path='model.pkl'):
    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error while saving the model: {e}")

# Define main pipeline
def main():
    try:
        n_estimators, learning_rate = load_params('params.yaml')
        X_train, y_train = fetch_data()
        clf = GBM_model(X_train, y_train, n_estimators, learning_rate)

        model_path = 'model.pkl'
        save_model(clf, model_path)

        print("Model building and saving completed successfully.")

    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == '__main__':
    main()
