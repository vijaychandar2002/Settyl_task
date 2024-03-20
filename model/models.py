import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = tf.keras.layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def create_transformer_model(df):
    inputs = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32)
    x = tf.keras.layers.Embedding(input_dim=5000, output_dim=16)(inputs)
    x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=4*256, dropout=0.1)
    x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=4*256, dropout=0.1)
    x = transformer_encoder(x, head_size=256, num_heads=4, ff_dim=4*256, dropout=0.1)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(len(df['internalStatus'].unique()), activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def create_lstm_model(df):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=5000, output_dim=16),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(df['internalStatus'].unique()), activation='softmax')
    ])
    return model

def create_cnn_model(df):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=5000, output_dim=16),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(df['internalStatus'].unique()), activation='softmax')
    ])
    return model

def create_svm_model():
    model = SVC()
    return model

def create_rf_model():
    model = RandomForestClassifier()
    return model

def create_decision_tree_model():
    model = DecisionTreeClassifier()
    return model



