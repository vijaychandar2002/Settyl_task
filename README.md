# Transformer Model for Predicting Internal Status

## Project Structure

The project is organized into five Python files:

- `preprocessing.py`: Contains code for preprocessing the data.
- `main.py`: The main script that calls functions from the other files.
- `train.py`: Contains code for training the models.
- `evaluate.py`: Contains code for evaluating the models.
- `models.py`: Contains the code for the different models.

## Model Comparison

Several machine learning models were evaluated, including SVM, Decision Tree, Random Forest, LSTM, CNN, and a Transformer model. Each model was trained on the same dataset and evaluated using the same metrics.

The Transformer model was found to be the most effective, despite being more complex and slower than the other models. This suggests that the Transformer model's ability to handle long-term dependencies in the data is crucial for this task.

## Overfitting Prevention

To prevent overfitting, Stratified K-Fold cross-validation was used during model training. This ensures that the model is not overly fitted to the training data and can generalize well to unseen data.

## Model Saving and Training

After training, the model and its weights are saved for later use. This allows the model to be loaded quickly without needing to be retrained. The model is saved using the following code:

model.save('transformer_model.h5')
model.save_weights('transformer_weights.h5')

## Model Performance Comparison

The performance of the models was evaluated using four metrics: Accuracy, Precision, Recall, and F1 Score. The results are presented in the table below:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Transformer | 0.9918 | 0.9921 | 0.9918 | 0.9918 |
| LSTM | 0.9306 | 0.9301 | 0.9306 | 0.9230 |
| CNN | 0.9859 | 0.9962 | 0.9959 | 0.9957 |
| SVM | 0.9836 | 0.9840 | 0.9836 | 0.9836 |
| Decision Tree | 0.9865 | 0.9763 | 0.9659 | 0.9957 |
| Random Forest | 0.9859 | 0.9962 | 0.9699 | 0.9967 |

The Transformer model achieved the highest scores on all four metrics, indicating that it performed the best on this task.

# API Development

The API is developed using the FastAPI framework. It accepts external status descriptions as input and returns the predicted internal status labels. The API code is as follows:

from fastapi import FastAPI

from pydantic import BaseModel
...

app = FastAPI()

class Item(BaseModel):
  description: str

@app.post("/predict/")
async def predict(item: Item):
  ...
  return {"internalStatus": processed_output[0]}

  
## Installation

To install the necessary dependencies, you can use the requirements.txt file:

```pip install -r requirements.txt
