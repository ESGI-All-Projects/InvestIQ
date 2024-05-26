import requests
import json

from API.authentification import load_token


def get_account():
    API_KEY, SECRET_KEY = load_token()
    url = "https://paper-api.alpaca.markets/v2/account"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }

    response = requests.get(url, headers=headers)

    return response.text

def get_account_portfolio_history(period="1A", timeframe="1D"):
    API_KEY, SECRET_KEY = load_token()
    url = "https://paper-api.alpaca.markets/v2/account/portfolio/history?intraday_reporting=market_hours&pnl_reset=per_day"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }

    params = {
        "period": period,
        "timeframe": timeframe,
    }

    response = requests.get(url, headers=headers, params=params)
    historical_data_json = json.loads(response.text)

    return historical_data_json

