import pandas as pd

from stable_baselines3 import PPO

from environnement.stock_market_env import StockTradingEnv
from models.train_model import train_model, evaluate_model

data = pd.read_csv("data/processed/historical_data_bars_1H_AAPL_with_indicators.csv", sep=',')[500:]

data_train = data[data['t'] < '2023-01-01']
data_test = data[data['t'] >= '2023-01-01']

date = data_test['t']

data_train = data_train.drop('t', axis=1)
data_test = data_test.drop('t', axis=1)

env_train = StockTradingEnv(data_train)
env_test = StockTradingEnv(data_test)

# train_model(env_train)


model = PPO.load("models/first-model")
evaluate_model(env_test, model, date)

