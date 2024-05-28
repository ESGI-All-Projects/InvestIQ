import numpy as np

import matplotlib.dates as mdates
import datetime

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


def train_model(env):
    model = PPO("MlpPolicy",
                env,
                tensorboard_log="logs/tensorboard_logs", # `tensorboard --logdir logs` in terminal to see graphs
                verbose=1)
    reward_logging_callback = RewardLoggingCallback(env)
    model.learn(total_timesteps=200_000, callback=reward_logging_callback)
    model.save("models/first-model")

def evaluate_model(env, model, dates):
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z') for date in dates]
    # Évaluation du modèle
    obs = env.reset()
    done = False
    portfolio_value = [env.calculate_buying_power()]
    portfolio_value_baseline = [env.calculate_buying_power_baseline()]
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        portfolio_value.append(env.calculate_buying_power())
        portfolio_value_baseline.append(env.calculate_buying_power_baseline())

    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.plot(dates, portfolio_value, label='IA optimize Portfolio Value', color='red')
    plt.plot(dates, portfolio_value_baseline, label='Baseline Portfolio Value', color='blue')
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