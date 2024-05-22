import json
import requests
import pandas as pd
from src.API.authentification import load_token

def get_historical_data(symbols, start_date, end_date, timeframe):
    API_KEY, SECRET_KEY = load_token()
    url = "https://data.alpaca.markets/v2/stocks/bars?limit=1000&feed=sip&sort=asc"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }

    params = {
        "symbols": symbols,
        "timeframe": timeframe,
        "start": start_date,
        "end": end_date
    }

    response = requests.get(url, headers=headers, params=params)
    historical_data_json = json.loads(response.text)
    historical_data_bars = historical_data_json["bars"][symbols]

    historical_data_df = pd.DataFrame(historical_data_bars)

    return historical_data_df