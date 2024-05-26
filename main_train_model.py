import pandas as pd

from environnement.stock_market_env import StockTradingEnv
from models.train_model import train_model

data = pd.read_csv("data/processed/historical_data_bars_1H_AAPL_with_indicators.csv", sep=',')[500:]
data = data.drop('t', axis=1)
env = StockTradingEnv(data)

train_model(env)

