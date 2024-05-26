import pandas as pd


def calculate_rsi(prices, period=14):
    prices = pd.Series(prices)
    # Calculate daily price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate the average gain and average loss
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()

    # Calculate the RS (Relative Strength)
    rs = avg_gain / avg_loss

    # Calculate the RSI (Relative Strength Index)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    def calculate_ema(prices, period):
        return prices.ewm(span=period, adjust=True).mean()

    # Calculer les EMA à court terme et à long terme
    ema_short = calculate_ema(prices, short_period)
    ema_long = calculate_ema(prices, long_period)

    # Calculer la ligne MACD
    macd_line = ema_short - ema_long

    # Calculer la ligne de signal
    signal_line = calculate_ema(macd_line, signal_period)

    # Calculer l'histogramme
    macd_histogram = macd_line - signal_line

    return macd_line, signal_line, macd_histogram




def calculate_adx(high, low, close, period=14):
    def calculate_true_range(high, low, close):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range

    def calculate_dm(high, low):
        up_move = high.diff()
        down_move = low.diff() * -1

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        return plus_dm, minus_dm

    def calculate_ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    # Calculer TR, +DM, -DM
    tr = calculate_true_range(high, low, close)
    plus_dm, minus_dm = calculate_dm(high, low)

    # Calculer les EMA de TR, +DM, -DM
    tr_ema = calculate_ema(tr, period)
    plus_dm_ema = calculate_ema(plus_dm, period)
    minus_dm_ema = calculate_ema(minus_dm, period)

    # Calculer les +DI et -DI
    plus_di = 100 * (plus_dm_ema / tr_ema)
    minus_di = 100 * (minus_dm_ema / tr_ema)

    # Calculer le DX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))

    # Calculer l'ADX
    adx = calculate_ema(dx, period)

    return adx, plus_di, minus_di
