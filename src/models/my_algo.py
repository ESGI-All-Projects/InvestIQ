import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime


def calcul_momentum(data, beta=0.995):
    m = data.iloc[0]
    data_momentum = [m]
    for d in data.iloc[1:]:
        m = beta * m + (1 - beta) * d
        data_momentum.append(m)

    return data_momentum

def evaluate_momentum_algo(df, beta_list):
    df = df[df['t'] > '2021-01-01']
    dates = df['t']
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z') for date in dates]

    data = df['c']

    # Calcul max portfolio value
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    print("Maximum portfolio value gain :", 10000 + sum(gain))

    price_momentum = {}
    portfolio_momentum = {}
    position_momentum = {}
    previous_price_momentum = {}
    portfolio_values_momentum = {}
    for beta in beta_list:
        price_momentum[str(beta)] = calcul_momentum(data, beta=beta)
        portfolio_momentum[str(beta)] = 10000
        position_momentum[str(beta)] = 0
        previous_price_momentum[str(beta)] = price_momentum[str(beta)][0]
        portfolio_values_momentum[str(beta)] = []

    portfolio_value_stock_evolution = 10000 - data.iloc[0]

    portfolio_value_stock_evolution_list = []
    for beta in beta_list:
        for price, p_momentum in zip(data.iloc[1:], price_momentum[str(beta)][1:]):
            if p_momentum < previous_price_momentum[str(beta)]:
                if position_momentum[str(beta)] == 1:
                    portfolio_momentum[str(beta)] += price
                    position_momentum[str(beta)] = 0
                    portfolio_values_momentum[str(beta)].append(portfolio_momentum[str(beta)])
                else:
                    portfolio_values_momentum[str(beta)].append(portfolio_momentum[str(beta)])
            else:
                if position_momentum[str(beta)] == 0:
                    portfolio_values_momentum[str(beta)].append(portfolio_momentum[str(beta)])
                    portfolio_momentum[str(beta)] -= price
                    position_momentum[str(beta)] = 1
                else:
                    portfolio_values_momentum[str(beta)].append(portfolio_momentum[str(beta)] + price)

            previous_price_momentum[str(beta)] = p_momentum

    for price in data.iloc[1:]:
        portfolio_value_stock_evolution_list.append(portfolio_value_stock_evolution + price)

    # Tracer les résultats
    colors = plt.cm.viridis(np.linspace(0, 1, len(beta_list)))
    plt.figure(figsize=(10, 6))
    plt.plot(dates[1:], portfolio_value_stock_evolution_list, label='Baseline Portfolio Value', color='red')
    for beta, color in zip(beta_list, colors):
        plt.plot(dates[1:], portfolio_values_momentum[str(beta)], label=f'Portfolio value {beta}', color=color)
    plt.xlabel('Dates')
    plt.ylabel('Portfolio Value')
    plt.title('AAPL : Momentum Portfolio Value vs. Baseline Over Time')
    plt.legend()

    # Formater l'axe des x pour afficher les dates correctement
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.gcf().autofmt_xdate(rotation=45)
    plt.show()


def evalutate_diff_algo(df, diff_list):
    df = df[df['t'] > '2023-01-01']
    dates = df['t']
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z') for date in dates]

    data = df['c']
    max_diff = max(diff_list)


    # Calcul max portfolio value
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    # portfolio_values_perfect = 10000 + np.cumsum(gain)
    print("Maximum portfolio value gain :", 10000 + sum(gain))

    portfolio_diff = {}
    position_diff = {}
    previous_price_diff = {}
    portfolio_values_diff = {}
    for diff in diff_list:
        portfolio_diff[str(diff)] = 10000
        position_diff[str(diff)] = 0
        previous_price_diff[str(diff)] = data.iloc[0]
        portfolio_values_diff[str(diff)] = []

    portfolio_value_stock_evolution = 10000 - data.iloc[0]

    portfolio_value_stock_evolution_list = []
    for diff in diff_list:
        for i, price in enumerate(data.iloc[diff:]):
            if price < previous_price_diff[str(diff)]:
                if position_diff[str(diff)] == 1:
                    portfolio_diff[str(diff)] += price
                    position_diff[str(diff)] = 0
                    portfolio_values_diff[str(diff)].append(portfolio_diff[str(diff)])
                else:
                    portfolio_values_diff[str(diff)].append(portfolio_diff[str(diff)])
            else:
                if position_diff[str(diff)] == 0:
                    portfolio_values_diff[str(diff)].append(portfolio_diff[str(diff)])
                    portfolio_diff[str(diff)] -= price
                    position_diff[str(diff)] = 1
                else:
                    portfolio_values_diff[str(diff)].append(portfolio_diff[str(diff)] + price)

            previous_price_diff[str(diff)] = data.iloc[i + 1]

    for price in data:
        portfolio_value_stock_evolution_list.append(portfolio_value_stock_evolution + price)

    # Tracer les résultats
    colors = plt.cm.viridis(np.linspace(0, 1, len(diff_list)))
    plt.figure(figsize=(10, 6))
    plt.plot(dates[max_diff:], portfolio_value_stock_evolution_list[max_diff:], label='Baseline Portfolio Value', color='red')
    # plt.plot(dates[max_diff:], portfolio_values_perfect[max_diff:], label='Perfect Portfolio Value', color='black')
    for diff, color in zip(diff_list, colors):
        plt.plot(dates[max_diff:], portfolio_values_diff[str(diff)][max_diff - diff:], label=f'Portfolio value {diff}', color=color)
    plt.xlabel('Dates')
    plt.ylabel('Portfolio Value')
    plt.title('AAPL : Diff Portfolio Value vs. Baseline Over Time')
    plt.legend()

    # Formater l'axe des x pour afficher les dates correctement
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.gcf().autofmt_xdate(rotation=45)
    plt.show()

def plot_stock_evolution(df):
    df = df[df['t'] > '2023-01-01']
    dates = df['t']
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z') for date in dates]

    plt.figure(figsize=(10, 6))
    plt.plot(dates, df['c'])
    plt.xlabel('Date')
    plt.ylabel('Montant ($)')
    plt.title('AAPL : Evolution de la valeur du stock au cours du temps en dollar')
    plt.legend()

    # Formater l'axe des x pour afficher les dates correctement
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.gcf().autofmt_xdate(rotation=45)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("../../data/processed/historical_data_bars_1H_AAPL_with_indicators.csv")
    # evaluate_momentum_algo(df, [0.9, 0.95, 0.99, 0.995, 0.999])
    # evalutate_diff_algo(df, [1, 10, 20, 40, 80])
    plot_stock_evolution(df)