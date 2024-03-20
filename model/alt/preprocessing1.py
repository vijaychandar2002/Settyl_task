from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Convert the 'externalStatus' column to a TF-IDF vector
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['externalStatus'])

    # Encode the 'internalStatus' column
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['internalStatus'])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
