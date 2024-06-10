import numpy as np

import matplotlib.dates as mdates
import datetime

import matplotlib.pyplot as plt
import gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack


def train_model(env):

    model = PPO("MlpPolicy",
                env,
                n_steps=2048,
                n_epochs=10,
                # ent_coef=0.005,
                learning_rate=0.0001,
                batch_size=128,
                tensorboard_log="logs/tensorboard_logs", # `tensorboard --logdir logs` in terminal to see graphs
                verbose=1)
    # MultiInputLstmPolicy MlpLstmPolicy
    # model = RecurrentPPO("MultiInputLstmPolicy",
    #                      env,
    #                      n_steps=2048,
    #                      n_epochs=10,
    #                      # ent_coef=0.005,
    #                      learning_rate=0.0001,
    #                      batch_size=128,
    #                      tensorboard_log="logs/tensorboard_logs", # `tensorboard --logdir logs` in terminal to see graphs
    #                      verbose=1)
    # policy = model.policy
    reward_logging_callback = RewardLoggingCallback(env)
    model.learn(total_timesteps=5_000, callback=reward_logging_callback)
    # model.learn(total_timesteps=5000)
    model.save("models/first-model")

def evaluate_model(env, model, dates):
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z') for date in dates]

    obs = env.reset()

    # cell and hidden state of the LSTM
    # lstm_states = None
    # num_envs = 1
    # Episode start signals are used to reset the lstm states
    # episode_starts = np.ones((num_envs,), dtype=bool)
    done = False
    actions = []
    portfolio_value = [env.calculate_buying_power()]
    portfolio_value_baseline = [env.calculate_buying_power_baseline()]
    while not done:
        # action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # episode_starts = done
        # action, _ = model.predict(obs)
        # obs, reward, done, _ = env.step(action)
        portfolio_value.append(env.calculate_buying_power())
        portfolio_value_baseline.append(env.calculate_buying_power_baseline())
        actions.append(action)

    total = len(actions)
    proportion_0 = actions.count(0) / total
    proportion_1 = actions.count(1) / total
    proportion_2 = actions.count(2) / total

    print("Proportion de 0:", proportion_0)
    print("Proportion de 1:", proportion_1)
    print("Proportion de 2:", proportion_2)

    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.plot(dates[30:], portfolio_value, label='IA optimize Portfolio Value', color='red')
    plt.plot(dates[30:], portfolio_value_baseline, label='Baseline Portfolio Value', color='blue')
    plt.xlabel('Dates')
    plt.ylabel('Portfolio Value')
    plt.title('AAPL : IA Portfolio Value vs. Baseline Over Time')
    plt.legend()

    # Formater l'axe des x pour afficher les dates correctement
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.gcf().autofmt_xdate(rotation=45)

    plt.show()



class RewardLoggingCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.env = env
    def _on_step(self) -> bool:
        # Calculate and log the reward
        reward = np.mean(self.locals["rewards"])
        self.logger.record("reward", reward)

        portfolio_value = self.env.calculate_buying_power()
        self.logger.record("portfolio_value", portfolio_value)
        return True