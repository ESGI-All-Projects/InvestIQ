import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def train_LSTM_model(df):
    data = df['c']
    # Taille de la fenetre
    window_size = 30

    # Créer des séquences d'entraînement
    X_train, y_train = [], []
    for i in range(window_size, len(data)):
        X_train.append(data.iloc[i - window_size:i])
        y_train.append(data.iloc[i])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = get_model(1)
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    return model


def get_model(input_size):
    model = Sequential()

    # Première couche LSTM
    model.add(LSTM(units=64, return_sequences=True, input_shape=(input_size, 1)))
    model.add(Dropout(0.2))

    # Deuxième couche LSTM
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))

    # Couche de sortie
    model.add(Dense(units=1))

    # Compiler le modèle
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def evaluate_model(df, model):
    data = df['c']
    window_size = 30

    X_test, y_test = [], []
    for i in range(window_size, len(data)):
        X_test.append(data.iloc[i - window_size:i])
        y_test.append(data.iloc[i])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test)

    # Visualiser les prédictions
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, color='blue', label='Prix réel')
    plt.plot(predictions, color='red', label='Prédictions')
    plt.title('Prédiction du Prix des Actions')
    plt.xlabel('Temps')
    plt.ylabel('Prix des Actions')
    plt.legend()
    plt.show()
