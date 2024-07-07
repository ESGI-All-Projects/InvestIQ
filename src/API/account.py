import requests
import json
from datetime import datetime, timezone

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
    respose_json = json.loads(response.text)

    return respose_json

def get_account_portfolio_gain(start_date):
    API_KEY, SECRET_KEY = load_token()
    url = "https://paper-api.alpaca.markets/v2/account/portfolio/history?intraday_reporting=market_hours&pnl_reset=per_day"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }

    current_date = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    params = {
        "timeframe": "1D",
        "start": start_date.astimezone(timezone.utc).isoformat(),
        "end": current_date,
        "intraday_reporting": "continuous"
    }

    response = requests.get(url, headers=headers, params=params)
    historical_data_json = json.loads(response.text)

    start_wallet = historical_data_json["equity"][0]
    current_wallet = historical_data_json["equity"][-1]

    return 1 - (current_wallet - start_wallet)/start_wallet, current_date

