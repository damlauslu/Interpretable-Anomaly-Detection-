import numpy as np
import tensorflow as tf
import random
import os

from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from BehaviourAnalysis import load_datasets
from keras.layers import LSTM, RepeatVector, TimeDistributed

import matplotlib.pyplot as plt

ModelPath = "ModularizedClasses/ForTraining/"
DataPath = "ModularizedClasses/ForTraining/behaviours/"

# Build Autoencoder
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def build_encoder(input_dim):
    set_seed(42)
    input_dim = input_dim

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation='relu',  activity_regularizer=l2(1e-4))(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(8, activation='relu')(encoded)

    decoded = Dense(16, activation='relu')  (encoded)
    output_layer = Dense(input_dim,     activation='linear')(decoded)

    autoencod = Model(input_layer, output_layer)
    
    return autoencod
    

def build_lstm_autoencoder(input_dim, timesteps=1):
    set_seed(42)
    input_layer = Input(shape=(timesteps, input_dim))
    encoded = LSTM(16, activation='relu', return_sequences=False)(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(16, activation='relu', return_sequences=True)(encoded)
    output_layer = TimeDistributed(Dense(input_dim, activation='linear'))(decoded)
    autoencoder = Model(input_layer, output_layer)
    return autoencoder

def TrainModel(model_path, data_path, timesteps=1):
    os.makedirs(model_path, exist_ok=True)

    # Get datasets as tuple (train, cv, test)
    datasets = load_datasets(data_path)
    if datasets is None:
        raise ValueError("❌ Datasets could not be loaded. Check the path or file format.")
    
    # Unpack the tuple
    train, cv, test = datasets
    
    X_train = train.drop(columns=["source"]).values
    X_cv = cv.drop(columns=["source"]).values
    X_test = test.drop(columns=["source"]).values

    # Reshape for LSTM: (samples, timesteps, features)
    def reshape_for_lstm(X, timesteps):
        n_samples = X.shape[0] // timesteps
        X = X[:n_samples * timesteps]
        return X.reshape((n_samples, timesteps, X.shape[1]))

    X_train = reshape_for_lstm(X_train, timesteps)
    X_cv = reshape_for_lstm(X_cv, timesteps)
    X_test = reshape_for_lstm(X_test, timesteps)

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Build the LSTM autoencoder model
    model = build_lstm_autoencoder(X_train.shape[2], timesteps)
    model.compile(optimizer='adam', loss='mse')
    
    history = model.fit(
        X_train, X_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_cv, X_cv),
        callbacks=[early_stop]
    )
    
    # Save the model
    model.save(model_path + "lstm_autoencoder_model.keras")
    
    return model, history

def plot_training_history(history):    
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()