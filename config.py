API_KEY = "l86kj2dwb93c0s59"
API_SECRET = "cmwozypvc76o4dahsrc5p7mraw3ss1r1"
ACCESS_TOKEN = "oS1PT0bOIvYhn2wpWKxEk06RWzThngcs"  # Get after manual login
REQUEST_TOKEN = "B0Mhmlo2ZGpezQB1XBLbkMC6NW6hC0HE"
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"


INSTRUMENT_CONFIG = {
    "NIFTY": {
        "strike_step": 50,
        "spread_width": 100,
        "min_iv": 10,
        "max_iv": 20,
        "exchange": "NSE",
        "trading_symbol": "NIFTY 50"
    },
    "BANKNIFTY": {
        "strike_step": 100,
        "spread_width": 200,
        "min_iv": 12,
        "max_iv": 25,
        "exchange": "NSE",
        "trading_symbol": "NIFTY BANK"
    },
    "RELIANCE": {      # Example stock
        "strike_step": 5,
        "spread_width": 20,
        "min_iv": 20,
        "max_iv": 50,
    },
    # Add more instruments as needed
}

def get_instrument_params(instrument_name):
    """
    Fetch instrument-specific params with defaults
    """
    default_params = {
        "strike_step": 50,
        "spread_width": 100,
        "min_iv": 10,
        "max_iv": 20,
    }
    return INSTRUMENT_CONFIG.get(instrument_name.upper(), default_params)
