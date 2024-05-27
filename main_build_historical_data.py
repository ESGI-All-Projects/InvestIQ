
from src.API.historical_data import get_historical_data

symbols = "AAPL"
historical_data_df = get_historical_data(symbols, "2017-01-01", "2017-02-01", "1Day")
historical_data_df.to_csv(f"data/raw/historical_data_bars_{symbols}.csv", index=False)