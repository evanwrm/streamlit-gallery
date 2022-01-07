import inspect
import datetime as dt
from functools import reduce

import numpy as np
import pandas as pd
import streamlit as st
from recommenders.datasets.python_splitters import python_random_split

from lib.constants import (
    DEFAULT_BIN_COUNT,
    DEFAULT_COLS_ITEM,
    DEFAULT_COLS_RATING,
    DEFAULT_COLS_TIMESTAMP,
    DEFAULT_COLS_USER,
    DEFAULT_DELIMITER,
    DEFAULT_TOP_K,
    MAX_BIN_COUNT,
)
from lib.plots import (
    plot_boxplot,
    plot_categories,
    plot_distribution,
)
from lib.utils import (
    bin_feature,
    column_index,
    get_dataset_files,
    get_model_parameters,
    unique_categories,
    unix_timestamp,
)
from lib.recommenders import (
    MODELS,
    MODELS_MAP,
    write_metrics,
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

DATE_CUTOFF = dt.datetime(2017, 1, 1)


@st.cache(allow_output_mutation=True)
def fetch_dataset(path):
    return pd.read_csv(path)


@st.cache(allow_output_mutation=True)
def fetch_dataframe_description(df: pd.DataFrame):
    return df.describe(include="all").astype(str)


#
# Introduction
#

st.title("Steam Recommender System")
st.write(
    """
This project tests out the features of [Streamlit](https://streamlit.io/), specifically we create a recommender system for Steam games using two datasets from [Kaggle](https://www.kaggle.com/). 
A [list of games](https://www.kaggle.com/trolukovich/steam-games-complete-dataset) and their categories, and a [dataset of reviews](https://www.kaggle.com/najzeko/steam-reviews-2021). 
From this we'll apply a number of recsys models using [Microsoft recommenders](https://github.com/microsoft/recommenders).
"""
)

#
# Sidebar
#

st.sidebar.subheader("Models")
model_id = st.sidebar.selectbox(
    "Select Model", [m.id for m in MODELS], format_func=lambda m: MODELS_MAP[m].name
)
# Model
model_cls = MODELS_MAP[model_id]
# Model Settings
st.sidebar.subheader("Model Parameters")
model_parameters = get_model_parameters(model_cls)
model = model_cls(**model_parameters) if inspect.isclass(model_cls) else model_cls
st.sidebar.write("---")

# Settings
st.sidebar.subheader("Settings")
eda_view = st.sidebar.checkbox("Exploratory Data Analysis (EDA)", True)
top_k = st.sidebar.number_input("Top K Recommendations", 1, 100, DEFAULT_TOP_K)
date_cutoff = unix_timestamp(
    dt.datetime.combine(
        st.sidebar.date_input(
            "Date Cutoff",
            DATE_CUTOFF,
            dt.datetime.fromtimestamp(0),
            dt.datetime.today(),
        ),
        dt.datetime.min.time(),
    )
)
st.sidebar.write("---")

# Data
st.sidebar.subheader("Data")
dataset_paths = get_dataset_files("data")
review_dataset = st.sidebar.selectbox(
    "Review Dataset",
    options=dataset_paths,
    index=dataset_paths.index("data/reviews/steam_reviews.csv"),
)
feature_datasets = st.sidebar.multiselect(
    "Tertiary Datasets", options=dataset_paths, default=["data/games/steam.csv"]
)

# Steam data
df_reviews = fetch_dataset(review_dataset)
df_features_datasets = [fetch_dataset(path) for path in feature_datasets]

# Column selections
st.sidebar.subheader("Column Selection")
user_col = st.sidebar.selectbox(
    "User Column",
    options=df_reviews.columns,
    index=column_index(df_reviews, DEFAULT_COLS_USER),
)
item_col = st.sidebar.selectbox(
    "Item Column",
    options=df_reviews.columns,
    index=column_index(df_reviews, DEFAULT_COLS_ITEM),
)
rating_col = st.sidebar.selectbox(
    "Rating Column",
    options=df_reviews.columns,
    index=column_index(df_reviews, DEFAULT_COLS_RATING),
)
timestamp_col = st.sidebar.selectbox(
    "Timestamp Column",
    options=df_reviews.columns,
    index=column_index(df_reviews, DEFAULT_COLS_TIMESTAMP),
)

# Clean data
df_reviews[rating_col] = 1 * pd.to_numeric(df_reviews[rating_col])
df_reviews = df_reviews[df_reviews[timestamp_col] < date_cutoff]
steam_dataset = reduce(
    lambda left, right: pd.merge(
        left,
        right,
        left_on=item_col,
        right_on=next(
            col for col in [item_col, *DEFAULT_COLS_ITEM] if col in right.columns
        ),
    ),
    [df_reviews, *df_features_datasets],
)

# Get additional features
st.sidebar.subheader("Feature Content Selection")
user_features_cols = st.sidebar.multiselect("User Features", steam_dataset.columns)
item_features_cols = st.sidebar.multiselect("Item Features", steam_dataset.columns)
feature_bins_container, delimiter_container = st.sidebar.columns(2)
with feature_bins_container:
    feature_bins = st.number_input(
        "Feature Bin Count", 1, MAX_BIN_COUNT, DEFAULT_BIN_COUNT
    )
with delimiter_container:
    feature_delimter = st.text_input("Feature Delimiter", DEFAULT_DELIMITER, 1)
binned_user_features = [
    bin_feature(steam_dataset[col], bins=feature_bins, delimiter=feature_delimter)
    for col in user_features_cols
]
binned_item_features = [
    bin_feature(steam_dataset[col], bins=feature_bins, delimiter=feature_delimter)
    for col in item_features_cols
]
user_features = (
    (
        np.concatenate([k for k, v in binned_user_features]),
        np.vstack([v.values for k, v in binned_user_features]).sum(axis=0).tolist(),
    )
    if binned_user_features
    else None
)
item_features = (
    (
        np.concatenate([k for k, v in binned_item_features]),
        np.vstack([v.values for k, v in binned_item_features]).sum(axis=0).tolist(),
    )
    if binned_item_features
    else None
)

# Display data
st.subheader("Review Data")
st.write(df_reviews.head())
for df_featureset in df_features_datasets:
    st.subheader("App Data")
    st.write(df_featureset.head())

# EDA
if eda_view:
    st.header("Exploratory Data Analysis")

    st.write("Steam app data description")
    st.write(fetch_dataframe_description(df_features_datasets[0]))
    st.write("Steam review data description")
    st.write(fetch_dataframe_description(df_reviews))

    # Top categories
    with st.expander("Top Categories", expanded=False):
        st.write("Unique game categories")
        st.write(unique_categories(df_features_datasets[0]["categories"]))
        plot_categories(
            df_features_datasets[0]["categories"],
            name="Category",
            delimiter=feature_delimter,
        )

    with st.expander("Top Steamspy Tags", expanded=False):
        st.write("Unique Steamspy tags")
        st.write(unique_categories(df_features_datasets[0]["steamspy_tags"]))
        plot_categories(
            df_features_datasets[0]["steamspy_tags"],
            name="Tag",
            delimiter=feature_delimter,
        )

    with st.expander("Top Genres", expanded=False):
        st.write("Unique game genres")
        st.write(unique_categories(df_features_datasets[0]["genres"]))
        plot_categories(
            df_features_datasets[0]["genres"], name="Genre", delimiter=feature_delimter
        )

    with st.expander("Top Platforms", expanded=False):
        st.write("Unique game platforms")
        st.write(unique_categories(df_features_datasets[0]["platforms"]))
        plot_categories(
            df_features_datasets[0]["platforms"],
            name="Platform",
            delimiter=feature_delimter,
        )

    with st.expander("Top Developers", expanded=False):
        st.write("Unique game developers")
        st.write(unique_categories(df_features_datasets[0]["developer"]))
        plot_categories(df_features_datasets[0]["developer"], name="Developer")

    with st.expander("Top Publishers", expanded=False):
        st.write("Unique game publishers")
        st.write(unique_categories(df_features_datasets[0]["publisher"]))
        plot_categories(df_features_datasets[0]["publisher"], name="Publisher")

    with st.expander("Most Expensive Games", expanded=False):
        interesting_columns = ["name", "price"]
        st.write(
            df_features_datasets[0]
            .sort_values("price", ascending=False)[interesting_columns]
            .head()
        )
        st.subheader("distribution")
        plot_distribution(df_features_datasets[0]["price"])
        plot_boxplot(df_features_datasets[0]["price"])

    with st.expander("Most Played Games", expanded=False):
        interesting_columns = ["name", "average_playtime", "median_playtime"]
        st.write("Average playtime")
        st.write(
            df_features_datasets[0]
            .sort_values("average_playtime", ascending=False)[interesting_columns]
            .head()
        )
        st.write("Median playtime")
        st.write(
            df_features_datasets[0]
            .sort_values("median_playtime", ascending=False)[interesting_columns]
            .head()
        )
        st.subheader("log-scale distribution")
        st.write("Note: filtered out games with zero playtime")
        plot_distribution(
            [
                np.log(
                    df_features_datasets[0]["average_playtime"]
                    .where(lambda x: x != 0)
                    .dropna()
                ),
                np.log(
                    df_features_datasets[0]["median_playtime"]
                    .where(lambda x: x != 0)
                    .dropna()
                ),
            ]
        )
        st.subheader("Boxplot")
        plot_boxplot(
            [
                df_features_datasets[0]["average_playtime"],
                df_features_datasets[0]["median_playtime"],
            ]
        )

    with st.expander("Most Rated Games", expanded=False):
        interesting_columns = ["name", "positive_ratings", "negative_ratings"]
        st.write(
            df_features_datasets[0]
            .sort_values("positive_ratings", ascending=False)[interesting_columns]
            .head()
        )
        st.write(
            df_features_datasets[0]
            .sort_values("negative_ratings", ascending=False)[interesting_columns]
            .head()
        )
        st.subheader("log-scale distribution")
        st.write("Note: filtered out games with zero rating")
        plot_distribution(
            [
                np.log(
                    df_features_datasets[0]["positive_ratings"]
                    .where(lambda x: x != 0)
                    .dropna()
                ),
                np.log(
                    df_features_datasets[0]["negative_ratings"]
                    .where(lambda x: x != 0)
                    .dropna()
                ),
            ],
        )
        st.subheader("Boxplot")
        plot_boxplot(
            [
                df_features_datasets[0]["positive_ratings"],
                df_features_datasets[0]["negative_ratings"],
            ]
        )

# Recommendations
st.subheader("Recommendations")
train, test, config = model.split_data(
    steam_dataset,
    uir_cols=[user_col, item_col, rating_col],
    ratio=0.75,
    user_features=user_features,
    item_features=item_features,
)
st.write("Test Split Sample")
st.write(test.head())

if st.button(f"Fit [{model.name}] Model!"):
    model.fit(train, config=config)
    preds = model.predict(train, col_user=user_col, col_item=item_col, config=config)

    write_metrics(
        test,
        preds,
        top_k=top_k,
        col_user=user_col,
        col_item=item_col,
        col_rating=rating_col,
    )
