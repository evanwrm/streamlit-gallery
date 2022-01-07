import re

import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.graph_objects as go

from lib.plotting import (
    GREEN_COLOR_RANGE,
    MAP_PROVIDER,
    MAP_STYLE,
    plot_datashader,
    plot_pydeck,
)
from lib.clustering import (
    CLUSTERING_METHODS,
    PARTITION_CLUSTERING,
    DENSITY_CLUSTERING,
    PARTITION_METHOD,
    DENSITY_METHOD,
)

#
# Config
#

st.set_page_config(
    page_title="Streamlit | Tree Inventory",
    initial_sidebar_state="auto",
    layout="centered",
)
# Remove footer
style_streamlit = """<style>
    footer {visibility: hidden;}
</style>"""
st.markdown(style_streamlit, unsafe_allow_html=True)


@st.cache
def fetch_tree_data():
    # data = pd.read_csv("https://data.winnipeg.ca/resource/hfwk-jp4h.csv")
    data = pd.read_csv("data/Tree_Inventory.csv")
    # data = pd.read_csv(
    #     "https://data.winnipeg.ca/api/views/hfwk-jp4h/rows.csv?accessType=DOWNLOAD&api_foundry=true"
    # )
    return data


@st.cache
def fetch_tree_clusters_partition(ndarray, n_clusters):
    return PARTITION_METHOD.cluster_trees(
        ndarray,
        n_clusters=n_clusters,
    )


@st.cache
def fetch_tree_clusters_density(ndarray, eps, metric):
    return DENSITY_METHOD.cluster_trees(ndarray, eps=eps, metric=metric)


#
# Introduction
#

st.title("Winnipeg Tree Inventory")
st.write(
    """
This project tests out the features of [Streamlit](https://streamlit.io/), specifically the mapping features of [PyDeck](https://deckgl.readthedocs.io/en/latest/) on the Winnipeg [Tree Inventory](https://data.winnipeg.ca/Parks/Tree-Inventory/hfwk-jp4h) dataset. 
This is partially done to also test the Winnipeg Open data portal.
"""
)

# Tree data
df_tree_data = fetch_tree_data().copy()
df_tree_data.rename(
    {name: "_".join(name.lower().split()) for name in df_tree_data.columns},
    axis=1,
    inplace=True,
)

st.subheader("Exploratory Data Analysis")
st.write("Table head")
st.write(df_tree_data.head())

st.write("Table column information")
st.write(df_tree_data.describe(include="all").astype(str))

#
# Sidebar Config
#

st.sidebar.subheader("Map Views")
scatter_view = st.sidebar.checkbox("Scatter Map View", True)
hexagon_view = st.sidebar.checkbox("Hexagon Map View", True)
heat_view = st.sidebar.checkbox("Heat Map View", True)

st.sidebar.write("---")
st.sidebar.subheader("Clustering Map View")
clustering_view = st.sidebar.checkbox("Clustering Map View", True)
if clustering_view:
    clustering_method = st.sidebar.selectbox("Clustering Method", CLUSTERING_METHODS)

    if clustering_method == PARTITION_CLUSTERING:
        km_n_clusters = st.sidebar.number_input(
            f"{PARTITION_METHOD.name} Cluster Number", 1, 25, 3, 1
        )
        km_cluster_column = st.sidebar.selectbox(
            f"{PARTITION_METHOD.name} Clustering Column",
            ["diameter_at_breast_height"],
        )
    elif clustering_method == DENSITY_CLUSTERING:
        ds_eps = st.sidebar.number_input(
            f"{DENSITY_METHOD.name} Neighborhood Distance", 0.0, 25.0, 0.5
        )
        ds_cluster_column = st.sidebar.selectbox(
            f"{DENSITY_METHOD.name} Clustering Column",
            ["location"],
        )

st.sidebar.write("---")
st.sidebar.subheader("Shader View")
datashader_view = st.sidebar.checkbox("Shader Map View", True)

# Tree types
st.subheader("Top Tree Types")
tree_breakdown = df_tree_data["common_name"].value_counts()
n_largest_tree_breakdown = st.slider(
    "N Largest Tree Types", 1, int(tree_breakdown.count()), 10
)
tree_breakdown_filtered = tree_breakdown.nlargest(n_largest_tree_breakdown)

fig1 = go.Figure(
    data=go.Bar(
        x=tree_breakdown_filtered.index,
        y=tree_breakdown_filtered,
        marker=dict(color=tree_breakdown_filtered, colorscale="algae"),
    )
)
st.plotly_chart(fig1, use_container_width=True)

# Park types
st.subheader("Top Parks")
park_breakdown = df_tree_data["park"].value_counts()
n_largest_park_breakdown = st.slider(
    "N Largest Parks", 1, int(park_breakdown.count()), 10
)
park_breakdown_filtered = park_breakdown.nlargest(n_largest_park_breakdown)

fig2 = go.Figure(
    data=go.Bar(
        x=park_breakdown_filtered.index,
        y=park_breakdown_filtered,
        marker=dict(color=tree_breakdown_filtered, colorscale="deep"),
    )
)
st.plotly_chart(fig2, use_container_width=True)

# Translate location data in lat lon columns
df_loc = pd.DataFrame(
    df_tree_data["location"]
    .apply(lambda x: re.sub("[\(\)\n\r ]", "", x).split(",")[-2:])
    .tolist(),
    columns=["lat", "lon"],
).apply(pd.to_numeric)
df_tree_data[["lat", "lon"]] = df_loc
df_tree_data.drop(["x", "y", "ded_tag_number", "location"], axis=1, inplace=True)

df_map_data = df_tree_data[
    [
        "common_name",
        "botanical_name",
        "park",
        "lat",
        "lon",
    ]
]

map_layers = {
    "scatter": pdk.Layer(
        "ScatterplotLayer",
        data=df_map_data,
        get_position=["lon", "lat"],
        get_color="[80, 130, 0, 160]",
        get_line_color="[0, 0, 0]",
        opacity=0.8,
        get_radius=5,
        stroked=True,
        auto_highlight=True,
        pickable=True,
    ),
    "hexagon": pdk.Layer(
        "HexagonLayer",
        data=df_map_data,
        get_position=["lon", "lat"],
        radius=25,
        elevation_scale=3,
        auto_highlight=True,
        pickable=True,
        extruded=True,
        coverage=1,
    ),
    "heat": pdk.Layer(
        "HeatmapLayer",
        data=df_map_data,
        get_position=["lon", "lat"],
        color_range=GREEN_COLOR_RANGE,
        aggregation="SUM",
    ),
}

if any([scatter_view, hexagon_view, heat_view]):
    st.subheader("Map Views")
if scatter_view:
    plot_pydeck(
        pdk.Deck(
            layers=[map_layers["scatter"]],
            initial_view_state=pdk.ViewState(
                latitude=49.85,
                longitude=-97.15,
                zoom=11,
            ),
            map_provider=MAP_PROVIDER,
            map_style=MAP_STYLE,
            tooltip={
                "html": "<b>Name:</b> {common_name} <br/>"
                "<b>Botanical Name:</b> {botanical_name} <br/>"
                "<b>Park:</b> {park} <br/>"
                "<b>Lat:</b> {lat} <br/>"
                "<b>Lon:</b> {lon} <br/>"
            },
        )
    )
if hexagon_view:
    plot_pydeck(
        pdk.Deck(
            layers=[map_layers["hexagon"]],
            initial_view_state=pdk.ViewState(
                latitude=49.85,
                longitude=-97.15,
                zoom=11,
                pitch=50,
                bearing=-27,
            ),
            map_provider=MAP_PROVIDER,
            map_style=MAP_STYLE,
            tooltip={
                "html": "<b>colorValue:</b> {colorValue} <br/>"
                "<b>elevationValue:</b> {elevationValue} <br/>"
            },
        )
    )
if heat_view:
    plot_pydeck(
        pdk.Deck(
            layers=[map_layers["heat"]],
            initial_view_state=pdk.ViewState(
                latitude=49.85,
                longitude=-97.15,
                zoom=11,
            ),
            map_provider=MAP_PROVIDER,
            map_style=MAP_STYLE,
        )
    )

if clustering_view:
    if clustering_method == PARTITION_CLUSTERING:
        km_fit_labels = fetch_tree_clusters_partition(
            df_tree_data[km_cluster_column].to_numpy().reshape(-1, 1),
            n_clusters=km_n_clusters,
        )
    elif clustering_method == DENSITY_CLUSTERING:
        km_fit_labels = fetch_tree_clusters_density(
            df_tree_data[["lat", "lon"]].to_numpy(), eps=ds_eps, metric="haversine"
        )

    df_tree_data["cluster"] = km_fit_labels

    df_map_data = df_tree_data[
        [
            "common_name",
            "diameter_at_breast_height",
            "cluster",
            "lat",
            "lon",
        ]
    ]

    cluster_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map_data,
        get_position=["lon", "lat"],
        get_color="[((cluster + 1) * 20) % 255, ((cluster + 1) * 30) % 255, 0, ((cluster + 1) * 15) % 255]",
        get_line_color="[0, 0, 0]",
        opacity=0.8,
        get_radius=5,
        stroked=True,
        auto_highlight=True,
        pickable=True,
    )

    st.subheader("Clustering Map View")
    st.write(
        "We really don't need to cluster this, but for demonstration we can pretend :)"
    )
    plot_pydeck(
        pdk.Deck(
            layers=[cluster_layer],
            initial_view_state=pdk.ViewState(
                latitude=49.85,
                longitude=-97.15,
                zoom=11,
            ),
            map_provider=MAP_PROVIDER,
            map_style=MAP_STYLE,
            tooltip={
                "html": "<b>Name:</b> {common_name} <br/>"
                "<b>Diameter at Breast Height:</b> {diameter_at_breast_height} <br/>"
                "<b>Cluster:</b> {cluster} <br/>"
                "<b>Lat:</b> {lat} <br/>"
                "<b>Lon:</b> {lon} <br/>"
            },
        )
    )

if datashader_view:
    st.subheader("Shader View")
    st.write(
        "Is there a better way to integrate Datashader? [Discussion](https://discuss.streamlit.io/t/working-with-jupyter-notebooks/368/10)"
    )
    plot_datashader(df_loc)
