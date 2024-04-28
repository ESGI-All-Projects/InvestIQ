import os
from dotenv import load_dotenv, find_dotenv

def load_token():
    """
    Find .env file and load API_KEY, SECRET_KEY
    :return: API_KEY, SECRET_KEY
    """
    # find .env automagically by walking up directories until it's found
    dotenv_path = find_dotenv()

    # load up the entries as environment variables
    load_dotenv(dotenv_path)

    API_KEY = os.environ.get("API_KEY")
    SECRET_KEY = os.environ.get("SECRET_KEY")

    return API_KEY, SECRET_KEY