import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from tqdm import tqdm
# from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from keras import backend as K
from keras.losses import binary_crossentropy
from sklearn.preprocessing import StandardScaler
import joblib

def train_LSTM_model(df_train, df_val, features_column, model_name, epochs=100):
    data_train = df_train[features_column]
    data_val = df_val[features_column]
    # Taille de la fenetre
    window_size = 30

    # Normalisation standard
    scaler = StandardScaler()
    data_train_scaled = scaler.fit_transform(data_train)
    data_val_scaled = scaler.transform(data_val)

    # Créer des séquences d'entraînement
    X_train, y_train = [], []
    X_val, y_val = [], []
    for i in range(window_size, len(data_train_scaled)):
        X_train.append(data_train_scaled[i - window_size:i])
        y_train.append(data_train_scaled[i][0])
    for i in range(window_size, len(data_val_scaled)):
        X_val.append(data_val_scaled[i - window_size:i])
        y_val.append(data_val_scaled[i][0])


    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(features_column)))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], len(features_column)))

    model = get_model(window_size, len(features_column))

    # Configurer TensorBoard
    log_dir = "logs/LSTM/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Entraîner le modèle avec TensorBoard callback
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard_callback]
    )

    model.save(f"models/LSTM/{model_name}.keras")
    joblib.dump(scaler, f"models/LSTM/scaler_{model_name}.pkl")


def get_model(window_size, input_size):
    model = Sequential()

    # Première couche LSTM
    model.add(LSTM(units=128, return_sequences=True, input_shape=(window_size, input_size)))
    # model.add(Dropout(0.2))

    # Deuxième couche LSTM
    model.add(LSTM(units=128, return_sequences=False))
    # model.add(Dropout(0.2))

    # Couche de sortie
    # model.add(Dense(units=1, activation='sigmoid'))
    model.add(Dense(units=1))

    # Compiler le modèle
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    # model.compile(optimizer='adam', loss=custom_loss)

    return model


def custom_loss(y_true, y_pred):
    # Prédictions et valeurs réelles comme directions (1 pour augmentation, 0 pour diminution)
    y_true_diff = K.sign(y_true[1:] - y_true[:-1])
    # y_pred_diff = K.sign(y_pred[1:] - y_true[:-1])

    # Conversion des directions en classes binaires (1 pour augmentation, 0 pour diminution)
    y_true_binary = (y_true_diff + 1) / 2
    # y_pred_binary = (y_pred_diff + 1) / 2

    # Utilisation de l'entropie croisée binaire comme perte
    # loss = binary_crossentropy(y_true_binary, y_pred_binary)
    loss = binary_crossentropy(y_true_binary, y_pred[1:])

    return K.mean(loss)


def display_prediction(df, model, scaler, features_column):
    data = df[features_column]
    window_size = 30
    data_scaled = scaler.transform(data)

    X_test, y_test = [], []
    for i in range(window_size, len(data_scaled)):
        X_test.append(data_scaled[i - window_size:i])
        y_test.append(data_scaled[i][0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features_column)))

    y_test = np.array(y_test).reshape(-1, 1)
    predictions = model.predict(X_test)

    # Calcule la proportion de bonne prédiction pour l'augmentation ou la diminution des prix
    y_test_diff = np.sign(y_test[1:] - y_test[:-1])
    y_pred_diff = np.sign(predictions[1:] - predictions[:-1])

    propotion = np.sum(y_test_diff == y_pred_diff)/len(y_test_diff)
    print(propotion)
    # x0 = X_test[0]
    # predictions_long_term = []
    # for _ in tqdm(X_test):
    #     p = model.predict(np.array([x0]))
    #     predictions_long_term.append(p[0])
    #
    #     # Update x0 by removing the oldest value and adding the latest prediction
    #     x0 = np.append(x0[1:], p[0]).reshape(window_size, 1)

    scaler_first_feature = StandardScaler()
    scaler_first_feature.mean_ = np.array([scaler.mean_[0]])
    scaler_first_feature.scale_ = np.array([scaler.scale_[0]])
    scaler_first_feature.var_ = np.array([scaler.scale_[0] ** 2])
    scaler_first_feature.n_samples_seen_ = scaler.n_samples_seen_
    # Visualiser les prédictions
    plt.figure(figsize=(10, 6))
    plt.plot(scaler_first_feature.inverse_transform(y_test), color='blue', label='Prix réel')
    plt.plot(scaler_first_feature.inverse_transform(predictions), color='red', label='Prédictions')
    # plt.plot(scaler.inverse_transform(predictions_long_term), color='black', label='Prédictions à long terme', linestyle='dashed')
    plt.title('Prédiction du Prix des Actions')
    plt.xlabel('Temps')
    plt.ylabel('Prix des Actions')
    plt.legend()
    plt.show()

def evaluate_model(df, model, dates, scaler, features_column):
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z') for date in dates]

    data = df[features_column]
    window_size = 30
    data_scaled = scaler.transform(data)

    X_test, y_test = [], []
    for i in range(window_size, len(data_scaled)):
        X_test.append(data_scaled[i - window_size:i])
        y_test.append(data_scaled[i][0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features_column)))

    scaler_first_feature = StandardScaler()
    scaler_first_feature.mean_ = np.array([scaler.mean_[0]])
    scaler_first_feature.scale_ = np.array([scaler.scale_[0]])
    scaler_first_feature.var_ = np.array([scaler.scale_[0] ** 2])
    scaler_first_feature.n_samples_seen_ = scaler.n_samples_seen_

    predictions = model.predict(X_test)
    predictions = scaler_first_feature.inverse_transform(predictions)

    portfolio_value_IA = 10000
    portfolio_value_stock_evolution = 10000 - data['c'].iloc[window_size]
    portfolio_value_random = 10000

    position_IA = 0
    position_random = 0

    previous_price = data['c'].iloc[window_size]
    previous_price_IA = data['c'].iloc[window_size]

    portfolio_value_IA_list = []
    portfolio_value_stock_evolution_list = []
    portfolio_value_random_list = []
    for price, pred in zip(data['c'].iloc[window_size:], predictions):
        if pred < previous_price_IA:
        # if pred < 0.5:
            if position_IA == 1:
                portfolio_value_IA += previous_price
                position_IA = 0
                portfolio_value_IA_list.append(portfolio_value_IA)
            else:
                portfolio_value_IA_list.append(portfolio_value_IA)
        else:
            if position_IA == 0:
                portfolio_value_IA_list.append(portfolio_value_IA)
                portfolio_value_IA -= previous_price
                position_IA = 1
            else:
                portfolio_value_IA_list.append(portfolio_value_IA + previous_price)

        if np.random.random() < 0.5:
            if position_random == 1:
                portfolio_value_random += previous_price
                position_random = 0
                portfolio_value_random_list.append(portfolio_value_random)
            else:
                portfolio_value_random_list.append(portfolio_value_random)
        else:
            if position_random == 0:
                portfolio_value_random_list.append(portfolio_value_random)
                portfolio_value_random -= previous_price
                position_random = 1
            else:
                portfolio_value_random_list.append(portfolio_value_random + previous_price)

        portfolio_value_stock_evolution_list.append(portfolio_value_stock_evolution + previous_price)
        previous_price = price
        previous_price_IA = pred

    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.plot(dates[window_size:], portfolio_value_IA_list, label='IA optimize', color='red')
    plt.plot(dates[window_size:], portfolio_value_stock_evolution_list, label='Baseline', color='blue')
    plt.plot(dates[window_size:], portfolio_value_random_list, label='Random action', color='green')
    plt.xlabel('Dates')
    plt.ylabel('Portfolio Value')
    plt.title('AAPL Portfolio Over Time: IA vs. Random action vs. Baseline')
    plt.legend()

    # Formater l'axe des x pour afficher les dates correctement
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate(rotation=45)
    plt.show()


if __name__ == '__main__':
    # Paramètres de la génération des données
    time_steps = 200
    t = np.arange(time_steps) * 0.1

    # Création de la combinaison de fonctions sinus
    signal = np.sin(t) + np.sin(2 * t) + np.sin(3 * t)

    # Visualisation des données
    # plt.plot(t, signal)
    # plt.title("Combinaison de signaux sinus")
    # plt.xlabel("Time")
    # plt.ylabel("Amplitude")
    # plt.show()

    df_train = pd.DataFrame(signal[:160], columns=['c'])
    df_test = pd.DataFrame(signal[160:], columns=['c'])

    train_LSTM_model(df_train, df_test, 'test_mse', epochs=150)

    model = tf.keras.models.load_model('models/LSTM/test_mse.keras')
    display_prediction(df_test, model)


