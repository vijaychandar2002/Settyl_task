from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    # Predict the classes
    y_pred = np.argmax(model.predict(X_test), axis=-1)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
