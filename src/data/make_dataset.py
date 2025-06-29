import numpy as np
import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
import logging

# ===================== Logging Configuration =====================
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ========================= Functions =============================

def load_params(params_path: str) -> float:
    """Load test_size parameter from YAML file."""
    try:
        logger.info("Loading parameters from YAML file.")
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        # Extract test_size from YAML
        test_size = params['make_dataset']['test_size']
        logger.info(f"Parameters loaded successfully: test_size={test_size}")
        return test_size
    except FileNotFoundError:
        logger.error(f"YAML file not found at path: {params_path}")
        raise
    except KeyError:
        logger.error("Missing 'make_dataset' or 'test_size' key in YAML file.")
        raise
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def read_data(url: str) -> pd.DataFrame:
    """Read CSV data from the given URL."""
    try:
        logger.info(f"Reading data from URL: {url}")
        df = pd.read_csv(url)
        logger.info("Data read successfully from URL.")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV from URL: {e}")
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean the dataset."""
    try:
        logger.info("Starting data processing.")

        # Drop tweet_id column
        df = df.drop(columns=['tweet_id'])

        # Check if 'sentiment' column exists
        if 'sentiment' not in df.columns:
            logger.error("'sentiment' column not found in dataset.")
            raise KeyError("'sentiment' column not found in dataset.")

        # Filter dataset to include only 'happiness' and 'sadness'
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()

        # Map sentiments to binary labels
        final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})

        # Check if dataset is empty after filtering
        if final_df.empty:
            logger.error("Filtered dataset is empty after selecting 'happiness' and 'sadness'.")
            raise ValueError("Filtered dataset is empty.")

        logger.info("Data processing completed successfully.")
        return final_df

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Save train and test data to CSV files."""
    try:
        logger.info(f"Saving processed data to {data_path}")

        # Create directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)

        # Save train and test data
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

        logger.info("Data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

# ========================= Main Pipeline =========================

def main() -> None:
    """Main data ingestion pipeline."""
    try:
        logger.info("Starting data ingestion pipeline.")

        # Load parameters
        test_size = load_params('params.yaml')

        # Validate test_size
        if not 0 < test_size < 1:
            logger.error("Test size should be between 0 and 1.")
            raise ValueError("Test size should be between 0 and 1.")

        # Read and process data
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = process_data(df)

        # Split data into train and test sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        logger.info(f"Data split successfully: train_size={len(train_data)}, test_size={len(test_data)}")

        # Save data
        data_path = os.path.join("emotions_detector/data", "raw")
        save_data(data_path, train_data, test_data)

        logger.info("Data ingestion completed successfully.")
        print("Data ingestion completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Pipeline failed: {e}")

if __name__ == '__main__':
    main()
