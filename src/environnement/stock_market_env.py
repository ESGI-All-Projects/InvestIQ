import gym
from gym import spaces
# import gymnasium as gym
# from gymnasium import spaces
# import pandas as pd
import numpy as np
# from stable_baselines.common.env_checker import check_env



class StockTradingWindowEnv(gym.Env):
    def __init__(self, data, window_size=30):
        super(StockTradingWindowEnv, self).__init__()

        # Données de candlestick et indicateurs
        self.data = data

        # Espaces d'observation et d'action
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(data.columns)+1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size + 1,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # Actions discrètes: 0 = vendre, 1 = conserver, 2 = acheter

        # Variables d'état
        self.window_size = window_size
        self.current_step = self.window_size
        self.previous_portfolio_value = 10000  # Montant 1 step avant du total cash + action
        self.portfolio_value = 10000  # actuel total cash + action
        self.cash_amount = 10000  # Valeur initiale du portefeuille
        self.position = 0  # Position actuelle: 0 = cash, 1 = long

        # To calculate baseline
        self.cash_amount_base = 10000 - self.data.iloc[self.current_step]

    def reset(self):
        # Réinitialiser l'environnement à l'état initial
        self.current_step = self.window_size
        self.previous_portfolio_value = 10000
        self.portfolio_value = 10000
        self.cash_amount = 10000
        self.position = 0
        self.cash_amount_base = 10000 - self.data.iloc[self.current_step]
        return self._next_observation()

    def step(self, action):
        # Exécuter une action dans l'environnement et retourner le nouvel état, la récompense, et si l'épisode est terminé
        assert self.action_space.contains(action)

        # Mettre à jour le portefeuille en fonction de l'action
        self._take_action(action)

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1  # Terminer l'épisode à la fin des données

        # Calculer la récompense
        reward = self._calculate_reward()

        # Obtenir la prochaine observation
        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        self.previous_portfolio_value = self.portfolio_value
        # Mettre à jour le portefeuille en fonction de l'action
        if action == 0:  # Vendre
            if self.position == 1:  # Si action en possession
                self.cash_amount += self.data.iloc[self.current_step]
                self.portfolio_value = self.cash_amount
                self.position = 0
        elif action == 1:  # Conserver
            if self.position == 1:
                self.portfolio_value = self.cash_amount + self.data.iloc[self.current_step]
        elif action == 2:  # Acheter
            if self.position == 0:  # Si cash, acheter
                self.portfolio_value = self.cash_amount
                self.cash_amount -= self.data.iloc[self.current_step]
                self.position = 1

    def _next_observation(self):
        # Retourner l'observation actuelle
        obs = self.data.iloc[self.current_step - self.window_size:self.current_step].values
        # obs = self.data.iloc[self.current_step].values
        obs = np.append(obs, self.position)
        return obs

    def _calculate_reward(self):
        return self.portfolio_value - self.previous_portfolio_value

    def calculate_buying_power(self):
        return self.portfolio_value

    def calculate_buying_power_baseline(self):
        return self.cash_amount_base + self.data.iloc[self.current_step - 1]
    def render(self, mode='human'):
        # Afficher une représentation visuelle de l'environnement (peut être vide dans cet exemple)
        pass

    def close(self):
        # Fermer toutes les ressources si nécessaire
        pass


class StockTradingIndicatorsEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingIndicatorsEnv, self).__init__()

        # Données de candlestick et indicateurs
        self.data = data

        # Espaces d'observation et d'action
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(data.columns)+1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(data.columns) + 1,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # Actions discrètes: 0 = vendre, 1 = conserver, 2 = acheter

        # Variables d'état
        self.current_step = 0
        self.previous_portfolio_value = 10000  # Montant 1 step avant du total cash + action
        self.portfolio_value = 10000  # actuel total cash + action
        self.cash_amount = 10000  # Valeur initiale du portefeuille
        self.position = 0  # Position actuelle: 0 = cash, 1 = long

        # To calculate baseline
        self.cash_amount_base = 10000 - self.data.iloc[self.current_step]['c']

    def reset(self):
        # Réinitialiser l'environnement à l'état initial
        self.current_step = 0
        self.previous_portfolio_value = 10000
        self.portfolio_value = 10000
        self.cash_amount = 10000
        self.position = 0
        self.cash_amount_base = 10000 - self.data.iloc[self.current_step]['c']
        return self._next_observation()

    def step(self, action):
        # Exécuter une action dans l'environnement et retourner le nouvel état, la récompense, et si l'épisode est terminé
        assert self.action_space.contains(action)

        # Mettre à jour le portefeuille en fonction de l'action
        self._take_action(action)

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1  # Terminer l'épisode à la fin des données

        # Calculer la récompense
        reward = self._calculate_reward()

        # Obtenir la prochaine observation
        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        self.previous_portfolio_value = self.portfolio_value
        # Mettre à jour le portefeuille en fonction de l'action
        if action == 0:  # Vendre
            if self.position == 1:  # Si action en possession
                self.cash_amount += self.data.iloc[self.current_step]['c']
                self.portfolio_value = self.cash_amount
                self.position = 0
        elif action == 1:  # Conserver
            if self.position == 1:
                self.portfolio_value = self.cash_amount + self.data.iloc[self.current_step]['c']
        elif action == 2:  # Acheter
            if self.position == 0:  # Si cash, acheter
                self.portfolio_value = self.cash_amount
                self.cash_amount -= self.data.iloc[self.current_step]['c']
                self.position = 1

    def _next_observation(self):
        # Retourner l'observation actuelle
        obs = self.data.iloc[self.current_step].values
        obs = np.append(obs, self.position)
        return obs

    def _calculate_reward(self):
        return self.portfolio_value - self.previous_portfolio_value

    def calculate_buying_power(self):
        return self.portfolio_value

    def calculate_buying_power_baseline(self):
        return self.cash_amount_base + self.data.iloc[self.current_step - 1]['c']
    def render(self, mode='human'):
        # Afficher une représentation visuelle de l'environnement (peut être vide dans cet exemple)
        pass

    def close(self):
        # Fermer toutes les ressources si nécessaire
        pass



class MultiStockTradingEnv(gym.Env):
    def __init__(self, stock_data):
        super(MultiStockTradingEnv, self).__init__()

        # Stock data for 30 stocks (DataFrame with rows as time steps and columns as stocks' features)
        self.stock_data = stock_data
        self.n_stocks = 30
        self.n_features = self.stock_data.shape[1] // self.n_stocks

        # Action space: Buy, hold or sell for each stock (discrete for simplicity here)
        self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)

        # Observation space: Last closing price, indicators, position, invested amount, available cash
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_stocks * (self.n_features + 2) + 1),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = 100000  # Initial cash in hand
        self.positions = np.zeros(self.n_stocks)  # Number of shares held
        self.invested_amount = np.zeros(self.n_stocks)  # Amount invested in each stock
        self.previous_portfolio_value = self.cash  # Initial portfolio value (cash only)

        return self._next_observation()

    def _next_observation(self):
        # Get data for the current step
        obs = self.stock_data.iloc[self.current_step].values

        # Add position, invested amount, and available cash to the observation
        full_obs = np.concatenate([
            obs,
            self.positions,
            self.invested_amount,
            self.cash
        ], axis=1)

        return full_obs

    def step(self, action):
        self._take_action(action)

        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1

        reward = self._calculate_reward()
        self.previous_portfolio_value = self._calculate_portfolio_value()
        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        # First, execute sell actions
        for i, act in enumerate(action):
            if act == 0:  # Sell
                stock_price = self.stock_data.iloc[self.current_step, i]
                self.cash += self.positions[i] * stock_price
                self.positions[i] = 0
                self.invested_amount[i] = 0

        # Then, execute buy actions
        for i, act in enumerate(action):
            if act == 2:  # Buy
                stock_price = self.stock_data.iloc[self.current_step, i]
                if self.cash > 1000:  # Buy one share if enough cash
                    self.positions[i] += 1000/stock_price
                    self.invested_amount[i] += 1000
                    self.cash -= 1000

    def _calculate_portfolio_value(self):
        portfolio_value = self.cash + np.sum(self.positions * self.stock_data.iloc[self.current_step, :self.n_stocks])
        return portfolio_value

    def _calculate_reward(self):
        reward = self._calculate_portfolio_value() - self.previous_portfolio_value
        return reward

    def render(self, mode='human'):
        pass

    def close(self):
        pass