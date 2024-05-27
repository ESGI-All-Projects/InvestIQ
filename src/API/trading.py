import requests

from src.API.authentification import load_token

def get_all_positions():
    API_KEY, SECRET_KEY = load_token()
    url = "https://paper-api.alpaca.markets/v2/positions"

    headers = {
        "accept": "application/json",
        "APCA-API.txt-KEY-ID": API_KEY,
        "APCA-API.txt-SECRET-KEY": SECRET_KEY
    }

    response = requests.get(url, headers=headers)
    return response.text

def get_position(symbol):
    API_KEY, SECRET_KEY = load_token()
    url = f"https://paper-api.alpaca.markets/v2/positions/{symbol}"

    headers = {
        "accept": "application/json",
        "APCA-API.txt-KEY-ID": API_KEY,
        "APCA-API.txt-SECRET-KEY": SECRET_KEY
    }

    response = requests.get(url, headers=headers)
    return response.text

def close_all_positions():
    API_KEY, SECRET_KEY = load_token()
    url = "https://paper-api.alpaca.markets/v2/positions"

    headers = {
        "accept": "application/json",
        "APCA-API.txt-KEY-ID": API_KEY,
        "APCA-API.txt-SECRET-KEY": SECRET_KEY
    }

    response = requests.delete(url, headers=headers)
    return response.text
def close_position(symbol):
    API_KEY, SECRET_KEY = load_token()
    url = f"https://paper-api.alpaca.markets/v2/positions/{symbol}"

    headers = {
        "accept": "application/json",
        "APCA-API.txt-KEY-ID": API_KEY,
        "APCA-API.txt-SECRET-KEY": SECRET_KEY
    }

    response = requests.delete(url, headers=headers)

    if response.status_code == '200':
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
        "APCA-API.txt-KEY-ID": API_KEY,
        "APCA-API.txt-SECRET-KEY": SECRET_KEY
    }

    response = requests.get(url, json=payload, headers=headers)

    if response.status_code == 200:
        return True
    else:
        return False





