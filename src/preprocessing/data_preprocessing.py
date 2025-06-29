import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Download required nltk packages
nltk.download('wordnet')
nltk.download('stopwords')

# Fetch data from data/raw
def fetch_data(train_path='./emotions_detector/data/raw/train.csv', test_path='./emotions_detector/data/raw/test.csv'):
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        # Handle missing values
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)

        return train_data, test_data

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")
    except pd.errors.EmptyDataError:
        raise Exception("CSV file is empty or unreadable.")
    except Exception as e:
        raise Exception(f"Error while fetching data: {e}")

# Text cleaning function
def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(y) for y in text]
        return " ".join(text)
    except Exception as e:
        raise Exception(f"Error during lemmatization: {e}")

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        Text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(Text)
    except Exception as e:
        raise Exception(f"Error during stop word removal: {e}")

def removing_numbers(text):
    try:
        text = ''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        raise Exception(f"Error during number removal: {e}")

def lower_case(text):
    try:
        text = text.split()
        text = [y.lower() for y in text]
        return " ".join(text)
    except Exception as e:
        raise Exception(f"Error during lower casing: {e}")

def removing_punctuations(text):
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        raise Exception(f"Error during punctuation removal: {e}")

def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        raise Exception(f"Error during URL removal: {e}")

def remove_small_sentences(df):
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
    except Exception as e:
        raise Exception(f"Error while removing small sentences: {e}")

# Data Transformation function
def transform_data(df):
    try:
        if 'content' not in df.columns:
            raise KeyError("The 'content' column is missing in the dataset.")

        df.content = df.content.apply(lambda content: lower_case(content))
        df.content = df.content.apply(lambda content: remove_stop_words(content))
        df.content = df.content.apply(lambda content: removing_numbers(content))
        df.content = df.content.apply(lambda content: removing_punctuations(content))
        df.content = df.content.apply(lambda content: removing_urls(content))
        df.content = df.content.apply(lambda content: lemmatization(content))
        return df

    except Exception as e:
        raise Exception(f"Error during data transformation: {e}")

# Data Storing function
def store_data(train_df, test_df, save_dir='emotions_detector/data/processed'):
    try:
        os.makedirs(save_dir, exist_ok=True)
        train_df.to_csv(os.path.join(save_dir, 'train_processed.csv'), index=False)
        test_df.to_csv(os.path.join(save_dir, 'test_processed.csv'), index=False)
    except Exception as e:
        raise Exception(f"Error while storing data: {e}")

# Main pipeline
def main():
    try:
        train_data, test_data = fetch_data()
        train_processed = transform_data(train_data)
        test_processed = transform_data(test_data)

        save_dir = 'emotions_detector/data/processed'
        store_data(train_processed, test_processed, save_dir)

        print("Data processing completed successfully.")

    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == '__main__':
    main()
