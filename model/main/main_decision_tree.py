import pandas as pd
from preprocessing1 import preprocess_data
from models import create_decision_tree_model
from train1 import train_model
from evaluate1 import evaluate_model
import requests
import pickle

# Load the data
url = "https://gist.githubusercontent.com/farhaan-settyl/ecf9c1e7ab7374f18e4400b7a3d2a161/raw/f94652f217eeca83e36dab9d08727caf79ebdecf/dataset.json"
response = requests.get(url)
data = response.json()
df = pd.json_normalize(data)

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(df)

# Create the Decision Tree model
model = create_decision_tree_model()

# Train the model
model = train_model(X_train, y_train, model)

# Evaluate the model
evaluate_model(model, X_test, y_test)

# Save the model
with open('dt_model.pkl', 'wb') as file:
    pickle.dump(model, file)
