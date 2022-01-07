import datetime as dt
from dataclasses import fields

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from prophet.plot import (
    plot_plotly,
    plot_components_plotly,
    add_changepoints_to_plot,
)
from st_aggrid import AgGrid, GridOptionsBuilder

from lib.asset_history import Asset, fetch_crypto_asset, fetch_stock_asset
from lib.predictions import (
    ProphetParamGrid,
    cv_hptune_prophet_model,
    fit_prophet_model,
    predict_forecast,
)


#
# Config
#

st.set_page_config(
    page_title="Streamlit | Stock Prediction",
    initial_sidebar_state="auto",
    layout="centered",
)
# Remove footer
style_streamlit = """<style>
    footer {visibility: hidden;}
</style>"""
st.markdown(style_streamlit, unsafe_allow_html=True)


TICKERS = ("AAPL", "GOOG", "MSFT")
CRYPTO = ("BTC", "ETH", "ADA", "DOT", "XMR")
START = dt.datetime(2015, 1, 1)
END = dt.datetime.today()
# Supported intervals from yfinance
INTERVALS = (
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
)


@st.cache(
    allow_output_mutation=True, hash_funcs={dt.datetime: lambda _: None}, ttl=86400
)
def fetch_price_data(ticker, interval) -> Asset:
    print(ticker, interval, type(ticker), type(interval))
    if ticker in TICKERS:
        asset = fetch_stock_asset(
            ticker=ticker, interval=interval, start_date=START, end_date=END
        )
    elif ticker in CRYPTO:
        asset = fetch_crypto_asset(
            ticker=ticker, interval=interval, start_date=START, end_date=END
        )

    return asset


@st.cache(allow_output_mutation=True)
def fit_model(df_history: pd.DataFrame):
    model = fit_prophet_model(df_history)
    return model


@st.cache(allow_output_mutation=True)
def fit_model_tune(df_history: pd.DataFrame, param_grid: ProphetParamGrid):
    model_args, tuning_results = cv_hptune_prophet_model(
        df_history, horizon=30, param_grid=param_grid
    )
    model = fit_prophet_model(df_history, **model_args)

    return model, tuning_results


#
# Sidebar Config
#

selected_ticker = st.sidebar.selectbox("Select Stock", TICKERS + CRYPTO)
start_time = st.sidebar.date_input(
    "Select START date",
    dt.datetime(2020, 1, 1),
    START,
    END,
)
end_time = st.sidebar.date_input(
    "Select END date",
    END,
    start_time,
    END,
)
# Crypto API doesn't allow interval sizes
if selected_ticker in TICKERS:
    history_interval = st.sidebar.select_slider(
        "Select histroy interval", INTERVALS, INTERVALS[8]
    )
else:
    history_interval = INTERVALS[8]
# Input to prophet
prediction_period = st.sidebar.slider("Select forecasting period", 10, 3 * 365, 365)

hp_tune = st.sidebar.checkbox("Hyper Parameter Tune", False)

if hp_tune:
    hp_fields = ProphetParamGrid(
        **{
            field.name: st.sidebar.multiselect(
                field.name, field.default_factory(), field.default_factory()[-1]
            )
            for field in fields(ProphetParamGrid)
        }
    )

#
# Introduction
#

st.title("Stock prediction")
st.write(
    """
This project tests out the features of [Streamlit](https://streamlit.io/), using historical stock prices and crypto market prices. 
We make predictions of future prices using [Prophet](https://facebook.github.io/prophet/) forecasts for fun.
"""
)

# Stock Data
asset = fetch_price_data(selected_ticker, history_interval)
# Slice by selected time range
history = asset.history.copy().loc[start_time:end_time]
history = history.reset_index()

# gb = GridOptionsBuilder.from_dataframe(stock_data)
# gb.configure_pagination(True)

# grid_options = gb.build()
# AgGrid(
#     stock_data,
#     gridOptions=grid_options,
#     theme="streamlit",
# )

col1, col2, col3 = st.columns((2, 1, 1))
with col1:
    st.subheader(f"Stock data for `{asset.name}`")
with col2:
    if asset.image_url is not None:
        st.image(asset.image_url, width=64)
with col3:
    st.metric(
        label="Price",
        value=f"{history.price.iloc[-1]:.2f}",
        delta=f"{100 * ((history.price.iloc[-1] -history.price.iloc[-2]) / history.price.iloc[-2]):+.2f}%",
    )

st.table(history.tail())

# Stock plots
plot_data = (
    (
        go.Scatter(x=history["date"], y=history["open"], name="stock_open"),
        go.Scatter(x=history["date"], y=history["close"], name="stock_close"),
    )
    if selected_ticker in TICKERS
    else (go.Scatter(x=history["date"], y=history["price"], name="crypto_price"),)
)
fig1 = go.Figure(
    data=plot_data,
    layout={"title_text": "Time Series data", "xaxis_rangeslider_visible": True},
)
st.plotly_chart(fig1)

#
# Stock price prediction
#

if hp_tune:
    model, tuning_results = fit_model_tune(history, hp_fields)

    st.subheader("Hyper Parameter Tuning Results")
    st.write(tuning_results)
else:
    model = fit_model(history)

forecast = predict_forecast(
    model, periods=prediction_period, frequency=history_interval
)

st.subheader("Forecast Analysis")
fig2 = plot_plotly(model, forecast)
st.plotly_chart(fig2)

st.subheader("Forecast Components")
fig3 = plot_components_plotly(model, forecast)
st.plotly_chart(fig3)

st.subheader("Change Points")
fig4 = model.plot(forecast)
a = add_changepoints_to_plot(fig4.gca(), model, forecast)
st.pyplot(fig4)
