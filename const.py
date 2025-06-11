from enum import Enum
from dataclasses import dataclass
from typing import Optional


class CandleInterval(Enum):
    MIN_1 = "minute"
    MIN_3 = "3minute"
    MIN_5 = "5minute"
    MIN_10 = "10minute"
    MIN_15 = "15minute"
    MIN_30 = "30minute"
    MIN_60 = "60minute"
    DAY = "day"


@dataclass(frozen=True)
class OptionInstrumentDetails:
    strike_step: int
    spread_width: int
    iron_condor_min_iv: Optional[float] = 10
    iron_condor_max_iv: Optional[float] = 20

class OptionInstrumentConfig(Enum):
    NIFTY = OptionInstrumentDetails(
        strike_step=50,
        spread_width=100,
        iron_condor_min_iv=10,
        iron_condor_max_iv=20
    )
    BANKNIFTY = OptionInstrumentDetails(
        strike_step=100,
        spread_width=200,
        iron_condor_min_iv=12,
        iron_condor_max_iv=25
    )

    @property
    def config(self) -> OptionInstrumentDetails:
        return self.value

@dataclass(frozen=True)
class BasicInstrumentDetails:
    kite_trading_symbol: str
    nse_lib_symbol: str
    option_detail : OptionInstrumentDetails
    exchange: Optional[str] = "NSE"

class Instrument(Enum):
    NIFTY = BasicInstrumentDetails(
        kite_trading_symbol="NIFTY 50",
        nse_lib_symbol="NIFTY",
        option_detail= OptionInstrumentDetails(
            strike_step=50,
            spread_width=100,
            iron_condor_min_iv=10,
            iron_condor_max_iv=20
        )
    )
    BANKNIFTY = BasicInstrumentDetails(
        kite_trading_symbol="NIFTY BANK",
        nse_lib_symbol="BANKNIFTY",
        option_detail= OptionInstrumentDetails(
            strike_step=100,
            spread_width=200,
            iron_condor_min_iv=12,
            iron_condor_max_iv=25
        )
    )

class OptionType(Enum):
    CE = "CE"
    PE = "PE"
