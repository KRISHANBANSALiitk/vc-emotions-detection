import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Load the model
def load_model(model_path='model.pkl'):
    try:
        with open(model_path, 'rb') as file:
            clf = pickle.load(file)
        return clf
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    except Exception as e:
        raise Exception(f"Error while loading the model: {e}")

# Load test data
def load_test_data(test_data_path='./data/features/test_bow.csv'):
    try:
        test_data = pd.read_csv(test_data_path)
        if test_data.shape[1] < 2:
            raise ValueError("Test data must have at least one feature and one label column.")

        X_test = test_data.iloc[:, 0:-1]
        y_test = test_data.iloc[:, -1]

        return X_test, y_test

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")
    except pd.errors.EmptyDataError:
        raise Exception("CSV file is empty or unreadable.")
    except Exception as e:
        raise Exception(f"Error while loading test data: {e}")

# Prediction and evaluation
def pred_eval(X_test, y_test, clf):
    try:
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        return metrics_dict

    except Exception as e:
        raise Exception(f"Error during prediction or evaluation: {e}")

# Save metrics to json
def save_metrics(metrics_dict, file_path='metrics.json'):
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        raise Exception(f"Error while saving metrics: {e}")

# Define main pipeline
def main():
    try:
        clf = load_model('model.pkl')
        X_test, y_test = load_test_data()
        metrics_dict = pred_eval(X_test, y_test, clf)

        file_path = 'metrics.json'
        save_metrics(metrics_dict, file_path)

        print("Model evaluation completed successfully.")

    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == '__main__':
    main()
