import json
import requests
import pandas as pd
import time

from API.authentification import load_token


def get_historical_data(symbol, start_date, end_date, timeframe):
    API_KEY, SECRET_KEY = load_token()
    url = "https://data.alpaca.markets/v2/stocks/bars?feed=sip&sort=asc"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }

    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "start": start_date,
        "end": end_date,
        "page_token": ''
    }

    historical_data_df = pd.DataFrame()
    while params["page_token"] is not None:
        time.sleep(0.5)
        response = requests.get(url, headers=headers, params=params)
        historical_data_json = json.loads(response.text)
        historical_data_bars = historical_data_json["bars"][symbol]

        historical_data_df = pd.concat([historical_data_df, pd.DataFrame(historical_data_bars)])
        params["page_token"] = historical_data_json['next_page_token']
        print(f"{symbol} Progress : {historical_data_df['t'].max()} / {end_date}")

    return historical_data_df
