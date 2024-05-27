import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


def train_model(env):
    model = PPO("MlpPolicy",
                env,
                tensorboard_log="logs/tensorboard_logs", # `tensorboard --logdir logs` in terminal to see graphs
                verbose=1)
    reward_logging_callback = RewardLoggingCallback()
    model.learn(total_timesteps=25000, callback=reward_logging_callback)
    model.save("../../models/first-model")


class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Calculate and log the reward
        reward = np.mean(self.locals["rewards"])
        self.logger.record("reward", reward)
        return True