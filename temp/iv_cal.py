from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq

def calculate_iv(spot_price, strike_price, time_to_expiry_days, option_price, risk_free_rate=0.065, option_type='C'):
    """
    Calculate Implied Volatility using the Black-Scholes model.

    Parameters:
        spot_price (float): Current price of the underlying (e.g., NIFTY).
        strike_price (float): Strike price of the option.
        time_to_expiry_days (float): Days until the option expires.
        option_price (float): Current market price (LTP) of the option.
        risk_free_rate (float): Annualized risk-free rate (default: 6.5%).
        option_type (str): 'C' for Call, 'P' for Put.

    Returns:
        float: Implied Volatility as a decimal (e.g., 0.155 for 15.5%).
    """

    T = time_to_expiry_days / 365.0

    def black_scholes_price(vol):
        d1 = (log(spot_price / strike_price) + (risk_free_rate + 0.5 * vol ** 2) * T) / (vol * sqrt(T))
        d2 = d1 - vol * sqrt(T)
        if option_type.upper() == 'C':
            return spot_price * norm.cdf(d1) - strike_price * exp(-risk_free_rate * T) * norm.cdf(d2)
        elif option_type.upper() == 'P':
            return strike_price * exp(-risk_free_rate * T) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'C' for Call or 'P' for Put.")

    def objective(vol):
        return black_scholes_price(vol) - option_price

    try:
        implied_vol = brentq(objective, 1e-5, 5.0)  # limit volatility between 0.001% and 500%
        return implied_vol
    except ValueError:
        return None  # No valid solution

