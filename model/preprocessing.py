import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def preprocess_data(df):
    # Tokenize the 'externalStatus' column
    tokenizer = Tokenizer(num_words=5000, oov_token="OOV")
    tokenizer.fit_on_texts(df['externalStatus'])
    X = tokenizer.texts_to_sequences(df['externalStatus'])
    X = pad_sequences(X, padding='post')

    # Encode the 'internalStatus' column
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['internalStatus'])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save tokenizer and encoder
    joblib.dump(tokenizer, 'tokenizer.joblib')
    joblib.dump(encoder, 'encoder.joblib')

    return X_train, X_test, y_train, y_test, tokenizer, encoder
