from typing import Dict, Optional
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf
import streamlit as st
from pycoingecko import CoinGeckoAPI

from lib.utils import unix_timestamp


#
# Asset classes
#


@dataclass
class AssetInfo:
    name: str
    symbol: str
    id: Optional[str] = field(default=None)
    image_url: Optional[str] = field(default=None)


@dataclass
class AssetHistory:
    history: pd.DataFrame


@dataclass
class Asset(AssetInfo, AssetHistory):
    """This class holds asset information

    Parameters
    ----------
    name : str
        Canonical name of the asset
    symbol : str
        Symbol of the asset. e.g. AAPL, GOOG, MSFT
    image_url : str or None
        Link to an image of the assets logo
    history : pandas.DataFrame
        Historical pricing data for the asset
    """


#
# Stock price functions
#


def fetch_stock_asset(ticker, interval, start_date, end_date) -> Asset:
    ticker_cls = yf.Ticker(ticker)
    df_history = ticker_cls.history(start=start_date, end=end_date, interval=interval)

    df_history = df_history.reset_index()
    history_fields = ["Date", "Open", "Close", "High", "Low"]
    df_history = df_history[history_fields].rename(
        {col: col.lower() for col in history_fields}, axis=1
    )
    df_history["price"] = df_history["close"]
    df_history.set_index("date", inplace=True)

    asset = Asset(
        name=ticker_cls.info["shortName"],
        symbol=ticker,
        image_url=ticker_cls.info["logo_url"],
        history=df_history,
    )
    return asset


#
# Cryptocurreny functions
#

cg = CoinGeckoAPI()


@st.cache(ttl=86400)
def fetch_crypto_coins_list() -> Dict[str, AssetInfo]:
    crypto_list = cg.get_coins_list()
    crypto_map = {
        coin["symbol"]: AssetInfo(
            name=coin["name"], id=coin["id"], symbol=coin["symbol"]
        )
        for coin in crypto_list
    }

    return crypto_map


def fetch_crypto_asset(
    ticker, interval, start_date, end_date, vs_currency="usd"
) -> Asset:
    from_unixtimestamp = unix_timestamp(start_date)
    to_unixtimestamp = unix_timestamp(end_date)

    asset_info = fetch_crypto_coins_list()[ticker.lower()]
    ticker_data = cg.get_coin_market_chart_range_by_id(
        asset_info.id,
        vs_currency,
        from_timestamp=from_unixtimestamp,
        to_timestamp=to_unixtimestamp,
    )

    df_history = pd.DataFrame(ticker_data["prices"], columns=["date", "price"])
    # Dates are currently in unix timestamp
    df_history["date"] = pd.to_datetime(df_history["date"], unit="ms")
    df_history = df_history.set_index("date")

    asset = Asset(
        name=asset_info.name,
        symbol=asset_info.symbol,
        history=df_history,
    )

    return asset
