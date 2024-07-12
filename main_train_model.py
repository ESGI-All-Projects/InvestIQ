import pandas as pd
import joblib

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
# from sb3_contrib import RecurrentPPO
import tensorflow as tf

from environnement.stock_market_env import StockTradingWindowEnv, StockTradingIndicatorsEnv, MultiStockTradingEnv, MultiStockRepartitionTradingEnv
from models.train_model_PPO import train_model
from models.train_model_PPO import evaluate_model_window, evaluate_model_indicators, evaluate_model_multi_actions
from models.train_LSTM_model import train_LSTM_model, display_prediction, custom_loss
from models.train_LSTM_model import evaluate_model as evaluate_model_LSTM


def get_data_AAPL():
    data = pd.read_csv("data/processed/historical_data_bars_1H_AAPL_with_indicators.csv", sep=',')
    # data = data.drop(['h', 'l', 'n', 'o', 'v', 'vw'], axis=1)[300:]
    data = data.drop(['h', 'l', 'o', 'vw'], axis=1)[300:]

    data_train = data[data['t'] < '2022-01-01']
    data_val = data[(data['t'] < '2023-01-01') & (data['t'] >= '2022-01-01')]
    data_test = data[data['t'] >= '2023-01-01']

    date_train = data_train['t']
    date_val = data_val['t']
    date_test = data_test['t']

    data_train = data_train.drop('t', axis=1)
    data_val = data_val.drop('t', axis=1)
    data_test = data_test.drop('t', axis=1)

    return data_train, data_val, data_test, date_train, date_val, date_test

def get_data_DJIA():
    data = pd.read_csv("data/processed/DJIA/historical_data_bars_1H_DJIA_with_indicators.csv", sep=',')
    # Suppression des colonnes pour 30 stocks
    delete_columns = [col for col in data.columns if col.startswith(("h__", "l__", "n__", "o__", "v__", "vw__"))]
    data = data.drop(delete_columns, axis=1)[500:]

    data_train = data[data['t'] < '2022-01-01']
    data_val = data[(data['t'] < '2023-01-01') & (data['t'] >= '2022-01-01')]
    data_test = data[data['t'] >= '2023-01-01']

    date_train = data_train['t']
    date_val = data_val['t']
    date_test = data_test['t']

    data_train = data_train.drop('t', axis=1)
    data_val = data_val.drop('t', axis=1)
    data_test = data_test.drop('t', axis=1)

    return data_train, data_val, data_test, date_train, date_val, date_test

def train_LSTM(data_train, data_val, data_test, date_test):
    train_LSTM_model(data_train, data_val, ['c', 'n', 'v'], 'AAPL_1H_cnv', epochs=100)

    model = tf.keras.models.load_model('models/LSTM/AAPL_1H_cnv.keras')
    scaler = joblib.load('models/LSTM/scaler_AAPL_1H_cnv.pkl')

    display_prediction(data_test, model, scaler, ['c', 'n', 'v'])
    evaluate_model_LSTM(data_test, model, date_test, scaler, ['c', 'n', 'v'])

def train_PPO_window(data_train, data_test, date_test):
    def make_env(data):
        def _init():
            return StockTradingWindowEnv(data)
        return _init

    env_train = DummyVecEnv([make_env(data_train['c'])])
    env_test = DummyVecEnv([make_env(data_test['c'])])
    env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True)
    env_test = VecNormalize(env_test, norm_obs=True, norm_reward=True)

    model_name = 'window30'
    train_model(env_train, model_name, total_timesteps=1_000_000)

    # model = RecurrentPPO.load("models/first-model")
    model = PPO.load(f"models/PPO/{model_name}")

    # evaluate_model_window(env_train, model, date_train)
    evaluate_model_window(env_test, model, date_test)

def train_PPO_indicators(data_train, data_test, date_test):
    def make_env(data):
        def _init():
            return StockTradingIndicatorsEnv(data)
        return _init

    env_train = DummyVecEnv([make_env(data_train)])
    env_test = DummyVecEnv([make_env(data_test)])
    env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True)
    env_test = VecNormalize(env_test, norm_obs=True, norm_reward=True)

    model_name = 'indicators_1H_1D'
    train_model(env_train, model_name, total_timesteps=1_000_000)

    # model = RecurrentPPO.load("models/first-model")
    model = PPO.load(f"models/PPO/{model_name}")

    # evaluate_model_indicators(env_train, model, date_train)
    evaluate_model_indicators(env_test, model, date_test)

def train_PPO_MultiActions(data_train, data_test, date_test):
    def make_env(data):
        def _init():
            return MultiStockTradingEnv(data)
        return _init

    env_train = DummyVecEnv([make_env(data_train)])
    env_test = DummyVecEnv([make_env(data_test)])
    env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True)
    env_test = VecNormalize(env_test, norm_obs=True, norm_reward=True)

    model_name = 'multi_actions'
    train_model(env_train, model_name, total_timesteps=1_000_000)

    model = PPO.load(f"models/PPO/{model_name}")

    # evaluate_model_multi_actions(env_train, model, date_train)
    evaluate_model_multi_actions(env_test, model, date_test)

def train_PPO_MultiActions_repartition():
    def make_env(data):
        def _init():
            return MultiStockRepartitionTradingEnv(data)
        return _init

    env_train = DummyVecEnv([make_env(data_train)])
    env_test = DummyVecEnv([make_env(data_test)])
    env_train = VecNormalize(env_train, norm_obs=True, norm_reward=True)
    env_test = VecNormalize(env_test, norm_obs=True, norm_reward=True)

    model_name = 'multi_actions_repartition'
    train_model(env_train, model_name, total_timesteps=5000, policy="SOFTMAX")

    model = PPO.load(f"models/PPO/{model_name}")

    # evaluate_model_multi_actions(env_train, model, date_train)
    evaluate_model_multi_actions(env_test, model, date_test)

data_train, data_val, data_test, date_train, date_val, date_test = get_data_AAPL()
# data_train, data_val, data_test, date_train, date_val, date_test = get_data_DJIA()

# train_LSTM(data_train, data_val, data_test, date_test)
# train_PPO_window(data_train, data_test, date_test)
train_PPO_indicators(data_train, data_test, date_test)
# train_PPO_MultiActions(data_train, data_test, date_test)
