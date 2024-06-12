import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
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

    model.save("models/LSTM/lstm.keras")


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

def display_prediction(df, model):
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

def evaluate_model(df, model, dates):
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z') for date in dates]

    data = df['c']
    window_size = 30

    data_momentum = []
    beta = 0.995
    m = data.iloc[window_size]
    for d in data[window_size:]:
        m = beta * m + (1 - beta) * d
        data_momentum.append(m)


    X_test, y_test = [], []
    for i in range(window_size, len(data)):
        X_test.append(data.iloc[i - window_size:i])
        y_test.append(data.iloc[i])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test)

    portfolio_value_IA = 10000
    portfolio_value_stock_evolution = 10000 - data.iloc[window_size]
    portfolio_value_my_algo = 10000
    position_IA = 0
    position_my_algo = 0

    previous_price = data.iloc[window_size]
    previous_price_IA = data.iloc[window_size]
    previous_price_momentum = data_momentum[window_size]

    portfolio_value_IA_list = []
    portfolio_value_stock_evolution_list = []
    portfolio_value_my_algo_list = []
    for price, price_momentum, pred in zip(data.iloc[window_size:], data_momentum, predictions):
        if pred < previous_price_IA:
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

        # if price < previous_price:
        #     if position_my_algo == 1:
        #         portfolio_value_my_algo += price
        #         position_my_algo = 0
        #         portfolio_value_my_algo_list.append(portfolio_value_my_algo)
        #     else:
        #         portfolio_value_my_algo_list.append(portfolio_value_my_algo)
        # else:
        #     if position_my_algo == 0:
        #         portfolio_value_my_algo_list.append(portfolio_value_my_algo)
        #         portfolio_value_my_algo -= price
        #         position_my_algo = 1
        #     else:
        #         portfolio_value_my_algo_list.append(portfolio_value_my_algo + price)

        if price_momentum < previous_price_momentum:
            if position_my_algo == 1:
                portfolio_value_my_algo += price
                position_my_algo = 0
                portfolio_value_my_algo_list.append(portfolio_value_my_algo)
            else:
                portfolio_value_my_algo_list.append(portfolio_value_my_algo)
        else:
            if position_my_algo == 0:
                portfolio_value_my_algo_list.append(portfolio_value_my_algo)
                portfolio_value_my_algo -= price
                position_my_algo = 1
            else:
                portfolio_value_my_algo_list.append(portfolio_value_my_algo + price)

        portfolio_value_stock_evolution_list.append(portfolio_value_stock_evolution + previous_price)
        previous_price = price
        previous_price_IA = pred
        previous_price_momentum = price_momentum

    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.plot(dates[window_size:], portfolio_value_IA_list, label='IA optimize Portfolio Value', color='red')
    plt.plot(dates[window_size:], portfolio_value_stock_evolution_list, label='Baseline Portfolio Value', color='blue')
    plt.plot(dates[window_size:], portfolio_value_my_algo_list, label='My algo Portfolio Value', color='green')
    plt.xlabel('Dates')
    plt.ylabel('Portfolio Value')
    plt.title('AAPL : IA Portfolio Value vs. Baseline Over Time')
    plt.legend()

    # Formater l'axe des x pour afficher les dates correctement
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.gcf().autofmt_xdate(rotation=45)

    plt.figure(figsize=(10, 6))
    plt.plot(dates[window_size:], data_momentum, label='Price momentum', color='black')
    plt.plot(dates[window_size:], data[window_size:], label='Price', color='blue')
    plt.xlabel('Dates')
    plt.ylabel('Portfolio Value')
    plt.title('AAPL : IA Portfolio Value vs. Baseline Over Time')
    plt.legend()

    # Formater l'axe des x pour afficher les dates correctement
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.gcf().autofmt_xdate(rotation=45)

    plt.show()



