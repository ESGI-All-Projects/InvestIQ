import requests
import json

from API.authentification import load_token

def get_all_active_stocks():
    API_KEY, SECRET_KEY = load_token()
    url = "https://paper-api.alpaca.markets/v2/assets?status=active"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }

    response = requests.get(url, headers=headers)
    stocks_data = json.loads(response.text)
    stocks_data = [stock for stock in stocks_data if stock['tradable']]

    return stocks_data

def get_latest_bars(symbols):
    """
    The latest multi bars endpoint returns the latest minute-aggregated historical bar data for the ticker symbols provided.
    """
    API_KEY, SECRET_KEY = load_token()
    url = "https://data.alpaca.markets/v2/stocks/bars/latest?feed=iex"


    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }

    params = {
        "symbols": symbols,
    }

    response = requests.get(url, headers=headers, params=params)
    current_data = json.loads(response.text)

    return current_data
