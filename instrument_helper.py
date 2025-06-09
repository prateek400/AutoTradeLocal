import pandas as pd
from kite_client import get_kite_client

_instruments_df = None

def get_instruments():
    global _instruments_df
    if _instruments_df is None:
        kite = get_kite_client()
        instruments = kite.instruments()
        if not instruments:
            raise ValueError("No instruments returned from API")
        _instruments_df = pd.DataFrame(instruments)
        # print("Sample instrument:", instruments[0])

        required_columns = {"name", "segment", "tradingsymbol"}
        if not required_columns.issubset(_instruments_df.columns):
            raise ValueError(f"Missing expected columns: {required_columns - set(_instruments_df.columns)}")
    return _instruments_df

def get_tradingsymbol(instrument_name, segment="NSE"):
    df = get_instruments()
    result = df[(df["name"] == instrument_name) & (df["segment"] == segment)]["tradingsymbol"].values
    return result if len(result) > 0 else None
