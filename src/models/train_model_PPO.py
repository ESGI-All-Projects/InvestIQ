import numpy as np

import matplotlib.dates as mdates
import datetime

import matplotlib.pyplot as plt
import gym
from stable_baselines3 import PPO
# from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn


def train_model(env, model_name, total_timesteps, policy="MLP"):
    if policy == "MLP":
        policy_config = "MlpPolicy"
    elif policy == "SOFTMAX":
        policy_config = SoftmaxPolicy
    else:
        raise Exception(f"policy {policy} unknow")
    model = PPO(policy_config,
                env,
                device='cuda',
                n_steps=2048,
                n_epochs=10,
                # ent_coef=0.005,
                learning_rate=0.0001,
                batch_size=128,
                tensorboard_log="logs/PPO", # `tensorboard --logdir logs` in terminal to see graphs
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
    reward_logging_callback = RewardLoggingCallback(env, n_steps=2048)
    model.learn(total_timesteps=total_timesteps, callback=reward_logging_callback)
    # model.learn(total_timesteps=total_timesteps)
    model.save(f"models/PPO/{model_name}")

class SoftmaxPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(SoftmaxPolicy, self).__init__(*args, **kwargs)
        # Override the action_net with a softmax layer
        self.action_net = nn.Sequential(
            nn.Linear(self.net_arch["pi"][-1], self.action_space.shape[0]),
            nn.Softmax(dim=1)
        )

    def _predict(self, observation, deterministic: bool = False):
        features = self.extract_features(observation)
        # Apply the action_net to get the softmax output
        action_probs = self.action_net(features)
        return action_probs, features

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        # return self.action_dist.proba_distribution(mean_actions)
        return mean_actions
def evaluate_model_window(env, model, dates):
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z') for date in dates]

    obs = env.reset()

    # cell and hidden state of the LSTM
    # lstm_states = None
    # num_envs = 1
    # Episode start signals are used to reset the lstm states
    # episode_starts = np.ones((num_envs,), dtype=bool)
    done = False
    actions = []
    portfolio_value = [env.venv.envs[0].calculate_buying_power()]
    portfolio_value_baseline = [env.venv.envs[0].calculate_buying_power_baseline()]
    while not done:
        # action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # episode_starts = done
        # action, _ = model.predict(obs)
        # obs, reward, done, _ = env.step(action)
        if not done:
            portfolio_value.append(env.venv.envs[0].calculate_buying_power())
            portfolio_value_baseline.append(env.venv.envs[0].calculate_buying_power_baseline())
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
    plt.plot(dates[30:-1], portfolio_value, label='IA optimize Portfolio Value', color='red')
    plt.plot(dates[30:-1], portfolio_value_baseline, label='Baseline Portfolio Value', color='blue')
    plt.xlabel('Dates')
    plt.ylabel('Portfolio Value')
    plt.title('AAPL : IA Portfolio Value vs. Baseline Over Time')
    plt.legend()

    # Formater l'axe des x pour afficher les dates correctement
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.gcf().autofmt_xdate(rotation=45)

    plt.show()

def evaluate_model_indicators(env, model, dates):
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z') for date in dates]

    obs = env.reset()

    # cell and hidden state of the LSTM
    # lstm_states = None
    # num_envs = 1
    # Episode start signals are used to reset the lstm states
    # episode_starts = np.ones((num_envs,), dtype=bool)
    done = False
    actions = []
    portfolio_value = []
    portfolio_value_baseline = []
    while not done:
        # action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # episode_starts = done
        # action, _ = model.predict(obs)
        # obs, reward, done, _ = env.step(action)
        if not done:
            portfolio_value.append(env.venv.envs[0].calculate_buying_power())
            portfolio_value_baseline.append(env.venv.envs[0].calculate_buying_power_baseline())
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
    plt.plot(dates[1:-1], portfolio_value, label='IA optimize Portfolio Value', color='red')
    plt.plot(dates[1:-1], portfolio_value_baseline, label='Baseline Portfolio Value', color='blue')
    plt.xlabel('Dates')
    plt.ylabel('Portfolio Value')
    plt.title('AAPL : IA Portfolio Value vs. Baseline Over Time')
    plt.legend()

    # Formater l'axe des x pour afficher les dates correctement
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.gcf().autofmt_xdate(rotation=45)

    plt.show()

def evaluate_model_multi_actions(env, model, dates):
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z') for date in dates]

    obs = env.reset()

    done = False
    actions = []
    portfolio_value = []
    portfolio_value_baseline = []
    cash = []
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # episode_starts = done
        # action, _ = model.predict(obs)
        # obs, reward, done, _ = env.step(action)
        if not done:
            portfolio_value.append(env.venv.envs[0].calculate_buying_power())
            portfolio_value_baseline.append(env.venv.envs[0].calculate_buying_power_baseline())
            cash.append(env.venv.envs[0].cash)
            actions.extend(action.flatten())

    total = len(actions)
    proportions = [actions.count(i) / total for i in range(3)]
    # proportion_0 = actions.count(0) / total
    # proportion_1 = actions.count(1) / total
    # proportion_2 = actions.count(2) / total

    print("Proportion de vente        :", proportions[0])
    print("Proportion de ne rien faire:", proportions[1])
    print("Proportion d'achat         :", proportions[2])

    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.plot(dates[1:-1], portfolio_value, label='IA optimize Portfolio Value', color='red')
    plt.plot(dates[1:-1], portfolio_value_baseline, label='Baseline Portfolio Value', color='blue')
    plt.plot(dates[1:-1], cash, label='IA cash non investi', color='black')
    plt.xlabel('Dates')
    plt.ylabel('Portfolio Value')
    plt.title('DJIA : IA Portfolio Value vs. Baseline Over Time')
    plt.legend()

    # Formater l'axe des x pour afficher les dates correctement
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.gcf().autofmt_xdate(rotation=45)

    plt.show()



class RewardLoggingCallback(BaseCallback):
    def __init__(self, env, n_steps=2048, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.env = env
        self.n_steps = n_steps
        self.step_counter = 0
        self.rewards_accum = []
    def _on_step(self) -> bool:
        # Calculate and log the reward
        # self.logger.record("reward", self.locals["rewards"])
        self.rewards_accum.append(self.locals["rewards"])
        self.step_counter += 1
        if self.step_counter % self.n_steps == 0:
            mean_reward = np.mean(self.rewards_accum)

            self.logger.record("reward_mean", mean_reward)

            # Reset accumulators
            self.rewards_accum = []
        # portfolio_value = self.env.venv.envs[0].gym_env.calculate_buying_power()
        # self.logger.record("portfolio_value", portfolio_value)
        return True