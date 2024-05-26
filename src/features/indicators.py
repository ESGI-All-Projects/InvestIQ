import pandas as pd


def calculate_rsi(df, interval="days", period=14):
    df['t'] = pd.to_datetime(df['t'])

    if interval == 'days':
        df_interval = df.groupby(df['t'].dt.date).tail(1).copy()
        df_interval['days'] = df_interval['t'].dt.date
        df['days'] = df['t'].dt.date
    elif interval == 'hours':
        df_interval = df.groupby(df['t'].dt.to_period('h')).tail(1).copy()
        df_interval['hours'] = df_interval['t'].dt.to_period('h')
        df['hours'] = df['t'].dt.to_period('h')
    else:
        raise ValueError("Interval must be either 'days' or 'hours'")

    # Calculate daily price changes
    delta = df_interval['c'].diff()

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

    df_interval[f'RSI_{interval}'] = rsi
    df = df.merge(df_interval[[f'RSI_{interval}', interval]], on=interval, how='left')
    df = df.drop(interval, axis=1)

    return df


def calculate_macd(df, interval="days", short_period=12, long_period=26, signal_period=9):
    def calculate_ema(prices, period):
        return prices.ewm(span=period, adjust=True).mean()

    df['t'] = pd.to_datetime(df['t'])

    if interval == 'days':
        df_interval = df.groupby(df['t'].dt.date).tail(1).copy()
        df_interval['days'] = df_interval['t'].dt.date
        df['days'] = df['t'].dt.date
    elif interval == 'hours':
        df_interval = df.groupby(df['t'].dt.to_period('h')).tail(1).copy()
        df_interval['hours'] = df_interval['t'].dt.to_period('h')
        df['hours'] = df['t'].dt.to_period('h')
    else:
        raise ValueError("Interval must be either 'days' or 'hours'")

    # Calculer les EMA à court terme et à long terme
    ema_short = calculate_ema(df_interval['c'], short_period)
    ema_long = calculate_ema(df_interval['c'], long_period)

    # Calculer la ligne MACD
    macd_line = ema_short - ema_long

    # Calculer la ligne de signal
    signal_line = calculate_ema(macd_line, signal_period)

    # Calculer l'histogramme
    macd_histogram = macd_line - signal_line

    df_interval[f'MACD_line_{interval}'] = macd_line
    df_interval[f'signal_line_{interval}'] = signal_line
    df_interval[f'MACD_histogram_{interval}'] = macd_histogram

    df = df.merge(df_interval[[f'MACD_line_{interval}', f'signal_line_{interval}', f'MACD_histogram_{interval}', interval]], on=interval, how='left')
    df = df.drop(interval, axis=1)

    return df




def calculate_adx(df, interval="days", period=14):
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

    df['t'] = pd.to_datetime(df['t'])

    if interval == 'days':
        df_interval = df.groupby(df['t'].dt.date).tail(1).copy()
        df_interval['days'] = df_interval['t'].dt.date
        df['days'] = df['t'].dt.date
    elif interval == 'hours':
        df_interval = df.groupby(df['t'].dt.to_period('h')).tail(1).copy()
        df_interval['hours'] = df_interval['t'].dt.to_period('h')
        df['hours'] = df['t'].dt.to_period('h')
    else:
        raise ValueError("Interval must be either 'days' or 'hours'")

    # Calculer TR, +DM, -DM
    tr = calculate_true_range(df_interval['h'], df_interval['l'], df_interval['c'])
    plus_dm, minus_dm = calculate_dm(df_interval['h'], df_interval['l'])

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

    df_interval[f'ADX_{interval}'] = adx
    df_interval[f'plus_di_{interval}'] = plus_di
    df_interval[f'minus_di_{interval}'] = minus_di

    df = df.merge(df_interval[[f'ADX_{interval}', f'plus_di_{interval}', f'minus_di_{interval}', interval]], on=interval, how='left')
    df = df.drop(interval, axis=1)

    return df
