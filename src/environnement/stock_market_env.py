import gym
from gym import spaces
# import gymnasium as gym
# from gymnasium import spaces
# import pandas as pd
import numpy as np
# from stable_baselines.common.env_checker import check_env

class StockTradingEnv(gym.Env):
    def __init__(self, data, window_size=30):
        super(StockTradingEnv, self).__init__()

        # Données de candlestick et indicateurs
        self.data = data

        # Espaces d'observation et d'action
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(data.columns)+1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, len(data.columns)),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # Actions discrètes: 0 = vendre, 1 = conserver, 2 = acheter

        # Variables d'état
        self.window_size = window_size
        self.current_step = self.window_size
        self.portfolio_value = 10000  # Valeur initiale du portefeuille
        self.position = 0  # Position actuelle: -1 = short, 0 = cash, 1 = long

        # To calculate baseline
        self.portfolio_value_base = 10000 - self.data.iloc[self.current_step]['c']

    def reset(self):
        # Réinitialiser l'environnement à l'état initial
        self.current_step = self.window_size
        self.portfolio_value = 10000
        self.position = 0
        self.portfolio_value_base = 10000 - self.data.iloc[self.current_step]['c']
        return self._next_observation()

    def step(self, action):
        # Exécuter une action dans l'environnement et retourner le nouvel état, la récompense, et si l'épisode est terminé
        assert self.action_space.contains(action)

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1  # Terminer l'épisode à la fin des données

        # Calculer la récompense
        reward = self._calculate_reward(action)

        # Mettre à jour le portefeuille en fonction de l'action
        self._take_action(action)

        # Obtenir la prochaine observation
        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        # Mettre à jour le portefeuille en fonction de l'action
        if action == 0:  # Vendre
            if self.position == 1:  # Si long, vendre
                self.portfolio_value += self.data.iloc[self.current_step - 1]['c']
                self.position = 0
        elif action == 1:  # Conserver
            pass
        elif action == 2:  # Acheter
            if self.position == 0:  # Si cash, acheter
            # if self.portfolio_value >=
                self.portfolio_value -= self.data.iloc[self.current_step - 1]['c']
                self.position = 1

    def _next_observation(self):
        # Retourner l'observation actuelle
        obs = self.data.iloc[self.current_step - self.window_size:self.current_step].values
        # obs = self.data.iloc[self.current_step].values
        # obs = np.append(obs, self.position)
        # return self.data.iloc[self.current_step].values
        return obs

    def _calculate_reward(self, action):
        # Calculer la récompense en fonction de l'action
        if action == 0:  # Vendre
            if self.position == 1:
                return self.data.iloc[self.current_step - 1]['c'] - self.data.iloc[self.current_step]['c']
            else:
                return 0
        elif action == 1:  # Conserver
            if self.position == 1:
                return self.data.iloc[self.current_step]['c'] - self.data.iloc[self.current_step - 1]['c']
                # return 0
            else:
                return 0
        elif action == 2:  # Acheter
            if self.position == 0:
                return self.data.iloc[self.current_step]['c'] - self.data.iloc[self.current_step - 1]['c']
            else:
                return 0

    def calculate_buying_power(self):
        if self.position == 1:
            return self.portfolio_value + self.data.iloc[self.current_step - 1]['c']
        else:
            return self.portfolio_value

    def calculate_buying_power_baseline(self):
        return self.portfolio_value_base + self.data.iloc[self.current_step]['c']
    def render(self, mode='human'):
        # Afficher une représentation visuelle de l'environnement (peut être vide dans cet exemple)
        pass

    def close(self):
        # Fermer toutes les ressources si nécessaire
        pass

# class StockTradingEnv(gym.Env):
#     def __init__(self, data, window_size=30):
#         super(StockTradingEnv, self).__init__()
#
#         # Données de candlestick et indicateurs
#         self.data = data
#
#         # Espaces d'observation et d'action
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(data.columns),), dtype=np.float32)
#         self.action_space = spaces.Discrete(3)  # Actions discrètes: 0 = vendre, 1 = conserver, 2 = acheter
#
#         # Variables d'état
#         self.current_step = 0
#         self.portfolio_value = 10000  # Valeur initiale du portefeuille
#         self.position = 0  # Position actuelle: -1 = short, 0 = cash, 1 = long
#
#         # To calculate baseline
#         self.portfolio_value_base = 10000 - self.data.iloc[self.current_step]['c']
#
#     def reset(self, seed=None, options=None):
#         # Réinitialiser l'environnement à l'état initial
#         self.current_step = 0
#         self.portfolio_value = 10000
#         self.position = 0
#         self.portfolio_value_base = 10000 - self.data.iloc[self.current_step]['c']
#         return self._next_observation(), {}
#
#     def step(self, action):
#         # Exécuter une action dans l'environnement et retourner le nouvel état, la récompense, et si l'épisode est terminé
#         assert self.action_space.contains(action)
#
#         self.current_step += 1
#         done = self.current_step >= len(self.data) - 1  # Terminer l'épisode à la fin des données
#
#         # Calculer la récompense
#         reward = self._calculate_reward(action)
#
#         # Mettre à jour le portefeuille en fonction de l'action
#         self._take_action(action)
#
#         # Obtenir la prochaine observation
#         obs = self._next_observation()
#
#         return obs, reward, done, False, {}
#
#     def _take_action(self, action):
#         # Mettre à jour le portefeuille en fonction de l'action
#         if action == 0:  # Vendre
#             if self.position == 1:  # Si long, vendre
#                 self.portfolio_value += self.data.iloc[self.current_step - 1]['c']
#                 self.position = 0
#         elif action == 1:  # Conserver
#             pass
#         elif action == 2:  # Acheter
#             if self.position == 0:  # Si cash, acheter
#             # if self.portfolio_value >=
#                 self.portfolio_value -= self.data.iloc[self.current_step - 1]['c']
#                 self.position = 1
#
#     def _next_observation(self):
#         # Retourner l'observation actuelle
#         obs = self.data.iloc[self.current_step].values
#         # obs = self.data.iloc[self.current_step].values
#         # obs = np.append(obs, self.position)
#         # return self.data.iloc[self.current_step].values
#         return obs
#
#     def _calculate_reward(self, action):
#         # Calculer la récompense en fonction de l'action
#         if action == 0:  # Vendre
#             if self.position == 1:
#                 return self.data.iloc[self.current_step - 1]['c'] - self.data.iloc[self.current_step]['c']
#             else:
#                 return 0
#         elif action == 1:  # Conserver
#             if self.position == 1:
#                 return self.data.iloc[self.current_step]['c'] - self.data.iloc[self.current_step - 1]['c']
#                 # return 0
#             else:
#                 return 0
#         elif action == 2:  # Acheter
#             if self.position == 0:
#                 return self.data.iloc[self.current_step]['c'] - self.data.iloc[self.current_step - 1]['c']
#             else:
#                 return 0
#
#     def calculate_buying_power(self):
#         if self.position == 1:
#             return self.portfolio_value + self.data.iloc[self.current_step - 1]['c']
#         else:
#             return self.portfolio_value
#
#     def calculate_buying_power_baseline(self):
#         return self.portfolio_value_base + self.data.iloc[self.current_step]['c']
#     def render(self, mode='human'):
#         # Afficher une représentation visuelle de l'environnement (peut être vide dans cet exemple)
#         pass
#
#     def close(self):
#         # Fermer toutes les ressources si nécessaire
#         pass
