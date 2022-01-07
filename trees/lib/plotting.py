import numpy as np

import pydeck as pdk
import streamlit as st
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import aggregate


#
# Mapbox & Deck.gl
#

MAP_PROVIDER = "mapbox"
MAP_STYLE = "mapbox://styles/mapbox/dark-v10"
# https://colorbrewer2.org/#type=sequential&scheme=Greens&n=6
GREEN_COLOR_RANGE = [
    [237, 248, 233],
    [199, 233, 192],
    [161, 217, 155],
    [116, 196, 118],
    [49, 163, 84],
    [0, 109, 44],
]


def plot_pydeck(deck: pdk.Deck, height=500):
    # st.pydeck_chart(deck)
    st.components.v1.html(deck.to_html(as_string=True), height=height)


#
# Datashader
#

hv.extension("bokeh")
renderer = hv.renderer("bokeh")
renderer = renderer.instance(mode="server")

opts.defaults(
    opts.Curve(
        xaxis=None,
        yaxis=None,
        show_grid=False,
        show_frame=False,
        color="orangered",
        framewise=True,
        width=100,
    ),
    opts.Image(
        width=800,
        height=400,
        shared_axes=False,
        logz=True,
        colorbar=True,
        xaxis=None,
        yaxis=None,
        axiswise=True,
        bgcolor="black",
    ),
    opts.HLine(color="white", line_width=1),
    opts.Layout(shared_axes=False),
    opts.VLine(color="white", line_width=1),
)


def plot_datashader(tree_data):
    points = hv.Points(tree_data, kdims=["lon", "lat"])

    # Use datashader to rasterize and linked streams for interactivity
    agg = aggregate(points, link_inputs=True, x_sampling=0.0001, y_sampling=0.0001)
    pointerx = hv.streams.PointerX(x=np.mean(points.range("lon")), source=points)
    pointery = hv.streams.PointerY(y=np.mean(points.range("lat")), source=points)
    vline = hv.DynamicMap(lambda x: hv.VLine(x), streams=[pointerx])
    hline = hv.DynamicMap(lambda y: hv.HLine(y), streams=[pointery])

    sampled = hv.util.Dynamic(
        agg,
        operation=lambda obj, x: obj.sample(lon=x),
        streams=[pointerx],
        link_inputs=False,
    )

    hvobj = (agg * hline * vline) << sampled
    hvplot = renderer.get_plot(hvobj)

    st.bokeh_chart(hvplot.state)
