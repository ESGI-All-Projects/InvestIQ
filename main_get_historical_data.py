from API.historical_data import get_historical_data

def get_DJIA_history():
    symbols = ["AMZN",
               "AXP",
               "AMGN",
               "AAPL",
               "BA",
               "CAT",
               "CSCO",
               "CVX",
               "GS",
               "HD",
               "HON",
               "IBM",
               "INTC",
               "JNJ",
               "KO",
               "JPM",
               "MCD",
               "MMM",
               "MRK",
               "MSFT",
               "NKE",
               "PG",
               "TRV",
               "UNH",
               "CRM",
               "VZ",
               "V",
               "WMT",
               "DIS",
               "DOW"]
    for symbol in symbols:
        historical_data_1H_df = get_historical_data(symbol, "2017-01-01", "2024-05-01", "1H")
        historical_data_1H_df.to_csv(f"data/raw/historical_data_bars_1H_{symbol}.csv", index=False)

def get_symbols_history(symbols):
    for symbol in symbols:
        historical_data_1D_df = get_historical_data(symbol, "2017-01-01", "2024-05-01", "1D")
        historical_data_1D_df.to_csv(f"data/raw/historical_data_bars_1D_{symbol}.csv", index=False)

        historical_data_1H_df = get_historical_data(symbol, "2017-01-01", "2024-05-01", "1H")
        historical_data_1H_df.to_csv(f"data/raw/historical_data_bars_1H_{symbol}.csv", index=False)

get_symbols_history(["AAPL"])
get_DJIA_history()