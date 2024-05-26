from features.indicators import calculate_adx, calculate_rsi, calculate_macd

def add_indicators(df_stock):
    df_stock = calculate_rsi(df_stock, interval="days")
    df_stock = calculate_rsi(df_stock, interval="hours")

    df_stock = calculate_macd(df_stock, interval="days")
    df_stock = calculate_macd(df_stock, interval="hours")

    df_stock = calculate_adx(df_stock, interval="days")
    df_stock = calculate_adx(df_stock, interval="hours")

    return df_stock
