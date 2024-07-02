import pandas as pd

# from stable_baselines3 import PPO
# from sb3_contrib import RecurrentPPO
import tensorflow as tf

# from environnement.stock_market_env import StockTradingEnv
# from models.train_model import train_model, evaluate_model
from models.train_LSTM_model import train_LSTM_model, evaluate_model, display_prediction

data = pd.read_csv("data/processed/historical_data_bars_1D_AAPL_with_indicators.csv", sep=',')
data = data.drop(['h','l','n','o','v','vw'], axis=1)
# for column in data.columns:
#     if 'days' in column:
#         data = data.drop([column], axis=1)


data_train = data[data['t'] < '2022-01-01']
data_val = data[(data['t'] < '2023-01-01') & (data['t'] >= '2022-01-01')]
data_test = data[data['t'] >= '2023-01-01']

date_train = data_train['t']
date_val = data_val['t']
date_test = data_test['t']

data_train = data_train.drop('t', axis=1)
data_val = data_val.drop('t', axis=1)
data_test = data_test.drop('t', axis=1)



# env_train = StockTradingEnv(data_train)
# env_test = StockTradingEnv(data_test)

# train_model(env_train)
#
# # model = RecurrentPPO.load("models/first-model")
# model = PPO.load("models/first-model")
#
# evaluate_model(env_train, model, date_train)
# evaluate_model(env_test, model, date_test)

train_LSTM_model(data_train, data_val)
# model = tf.keras.models.load_model('models/LSTM/lstm_1D.keras')

# display_prediction(data_test[:500], model)
# evaluate_model(data_test, model, date_test)

