import requests

from API.authentification import load_token

def get_all_positions():
    API_KEY, SECRET_KEY = load_token()
    url = "https://paper-api.alpaca.markets/v2/positions"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }

    response = requests.get(url, headers=headers)
    return response.text

def get_position(symbol):
    API_KEY, SECRET_KEY = load_token()
    url = f"https://paper-api.alpaca.markets/v2/positions/{symbol}"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }

    response = requests.get(url, headers=headers)
    return response.text

def close_all_positions():
    API_KEY, SECRET_KEY = load_token()
    url = "https://paper-api.alpaca.markets/v2/positions"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }

    response = requests.delete(url, headers=headers)
    return response.text
def close_position(symbol):
    API_KEY, SECRET_KEY = load_token()
    url = f"https://paper-api.alpaca.markets/v2/positions/{symbol}"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }

    response = requests.delete(url, headers=headers)

    if str(response.status_code) == '200':
        return True
    else:
        return False

def create_order(symbol, amount, side='buy', type='market'):
    API_KEY, SECRET_KEY = load_token()
    url = "https://paper-api.alpaca.markets/v2/orders"

    payload = {
        "symbol": symbol,
        "side": side,
        "type": type,
        "time_in_force": "day",
        "notional": amount
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }

    response = requests.post(url, json=payload, headers=headers)

    if str(response.status_code) == '200':
        return True
    else:
        return False





