import pandas as pd
import requests
from preprocessing import preprocess_data
from models import create_transformer_model, create_lstm_model
from train import train_model
from evaluate1 import evaluate_model
import requests

# Load the data
url = "https://gist.githubusercontent.com/farhaan-settyl/ecf9c1e7ab7374f18e4400b7a3d2a161/raw/f94652f217eeca83e36dab9d08727caf79ebdecf/dataset.json"
response = requests.get(url)
data = response.json()
df = pd.json_normalize(data)

# Preprocess the data
X_train, X_test, y_train, y_test, tokenizer, encoder = preprocess_data(df)

# Create the LSTM model
model = create_lstm_model(df)

# Train the model
model = train_model(X_train, y_train, model)

# Evaluate the model
evaluate_model(model, X_test, y_test)

# Save the model
model.save('lstm_model.h5')
