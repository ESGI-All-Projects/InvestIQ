import pandas as pd

from features.build_features import add_indicators

df_AAPL = pd.read_csv('data/raw/historical_data_bars_1H_AAPL.csv')
df_AAPL = add_indicators(df_AAPL)
df_AAPL.to_csv("data/processed/historical_data_bars_1H_AAPL_with_indicators.csv", index=False)