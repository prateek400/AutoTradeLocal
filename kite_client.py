from kiteconnect import KiteConnect, KiteTicker
from config import API_KEY, API_SECRET, ACCESS_TOKEN, REQUEST_TOKEN

def get_kite_client():
    kite = KiteConnect(api_key=API_KEY)
    # data = kite.generate_session(REQUEST_TOKEN, api_secret=API_SECRET)
    kite.set_access_token(ACCESS_TOKEN)
    # kite.set_access_token(ACCESS_TOKEN)
    return kite


def generate_access_token():
    kite = KiteConnect(api_key=API_KEY)
    print("Login URL:", kite.login_url())
    data = kite.generate_session(REQUEST_TOKEN, api_secret=API_SECRET)
    return data['access_token']


def get_kite_ticker():
    kws = KiteTicker(api_key=API_KEY, access_token=ACCESS_TOKEN)
    return kws
