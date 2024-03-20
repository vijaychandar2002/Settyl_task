from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from fastapi.responses import HTMLResponse

# Define your custom function
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
  # Normalization and Attention
  x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
  x = tf.keras.layers.MultiHeadAttention(key_dim=head_size,
                                         num_heads=num_heads,
                                         dropout=dropout)(x, x)
  x = tf.keras.layers.Dropout(dropout)(x)
  res = x + inputs

  # Feed Forward Part
  x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
  x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1,
                             activation='relu')(x)
  x = tf.keras.layers.Dropout(dropout)(x)
  x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
  return x + res


# Define your model architecture
def create_transformer_model(num_classes):
  inputs = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32)
  x = tf.keras.layers.Embedding(input_dim=5000, output_dim=16)(inputs)
  x = transformer_encoder(x,
                          head_size=256,
                          num_heads=4,
                          ff_dim=4 * 256,
                          dropout=0.1)
  x = transformer_encoder(x,
                          head_size=256,
                          num_heads=4,
                          ff_dim=4 * 256,
                          dropout=0.1)
  x = transformer_encoder(x,
                          head_size=256,
                          num_heads=4,
                          ff_dim=4 * 256,
                          dropout=0.1)
  x = tf.keras.layers.GlobalAveragePooling1D()(x)
  outputs = tf.keras.layers.Dense(
      num_classes,
      activation='softmax',
      kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
  model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
  return model


# Define the number of unique classes in your data
num_classes = 15

# Create the model
model = create_transformer_model(num_classes)

# Load your saved weights into the model
model.load_weights('transformer_weights.h5')

# Load your tokenizer and encoder
tokenizer = joblib.load('tokenizer.joblib')
encoder = joblib.load('encoder.joblib')

app = FastAPI()


class Item(BaseModel):
  externalStatus: str


@app.post("/predict/")
async def predict(item: Item):
  # Preprocess the input
  processed_input = tokenizer.texts_to_sequences([item.externalStatus])
  processed_input = pad_sequences(processed_input, padding='post')

  # Make prediction
  prediction = model.predict(processed_input)

  # Postprocess the prediction
  predicted_label = np.argmax(prediction, axis=-1)
  processed_output = encoder.inverse_transform(predicted_label)

  return {"internalStatus": processed_output[0]}

@app.get("/", response_class=HTMLResponse)
@app.get("/predict/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>My FastAPI Application</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f0f0f0;
                }
                .container {
                    width: 80%;
                    margin: auto;
                    background-color: #fff;
                    padding: 20px;
                    box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                }
                p {
                    color: #666;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Welcome to my FastAPI application!</h1>
                <p>Use a POST request to /predict to get predictions.</p>
            </div>
        </body>
    </html>
    """

