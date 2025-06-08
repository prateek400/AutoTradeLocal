from kite_client import get_kite_client

kite = get_kite_client()

def place_market_order(symbol, quantity, transaction_type):
    try:
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange="NSE",
            tradingsymbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=kite.ORDER_TYPE_MARKET,
            product=kite.PRODUCT_MIS
        )
        return order_id
    except Exception as e:
        print(f"Order Failed: {e}")
        return None

def place_sl_order(symbol, quantity, trigger_price, price, transaction_type):
    try:
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange="NSE",
            tradingsymbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=kite.ORDER_TYPE_SL,
            trigger_price=trigger_price,
            price=price,
            product=kite.PRODUCT_MIS
        )
        return order_id
    except Exception as e:
        print(f"SL Order Failed: {e}")
        return None
