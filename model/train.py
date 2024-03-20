from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping

def train_model(X_train, y_train, model):
    # Initialize the StratifiedKFold class
    skf = StratifiedKFold(n_splits=5)

    # Loop over each split
    for train_index, val_index in skf.split(X_train, y_train):
        # Split the data
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        history = model.fit(X_train_fold, y_train_fold, epochs=10, validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping])

    return model
