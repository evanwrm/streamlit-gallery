from typing import List, Union

import numpy as np
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go


def plot_distribution(
    dfs: Union[np.ndarray, List[np.ndarray]],
    group_names: Union[str, List[str]] = None,
    curve_type="normal",
    bin_size=1,
    show_rug=False,
    show_hist=False,
):
    if isinstance(dfs, list):
        fig = ff.create_distplot(
            [df.values for df in dfs],
            group_names or [df.name for df in dfs],
            curve_type=curve_type,
            bin_size=bin_size,
            show_rug=show_rug,
            show_hist=show_hist,
        )
    else:
        fig = ff.create_distplot(
            [dfs.values],
            [group_names or dfs.name],
            curve_type=curve_type,
            bin_size=bin_size,
            show_rug=show_rug,
            show_hist=show_hist,
        )

    st.plotly_chart(fig, use_container_width=True)


def plot_boxplot(
    dfs: Union[np.ndarray, List[np.ndarray]], group_names: Union[str, List[str]] = None
):
    dfs = [dfs] if not isinstance(dfs, list) else dfs

    fig = go.Figure(data=tuple(go.Box(y=df.values, name=df.name) for df in dfs))
    st.plotly_chart(fig, use_container_width=True)


def plot_categories(
    df: pd.Series,
    name="Category",
    delimiter=None,
    slider=True,
    slider_colorscale="thermal",
):
    if delimiter is not None:
        category_breakdown = df.str.split(delimiter).explode().value_counts()
    else:
        category_breakdown = df.value_counts()

    n_largest_category_breakdown = min(10, len(category_breakdown))
    if slider:
        n_largest_category_breakdown = st.slider(
            f"N Largest {name}s",
            1,
            int(category_breakdown.count()),
            n_largest_category_breakdown,
        )
    category_breakdown_filtered = category_breakdown.nlargest(
        n_largest_category_breakdown
    )
    fig = go.Figure(
        data=go.Bar(
            x=category_breakdown_filtered.index,
            y=category_breakdown_filtered,
            marker=dict(
                color=category_breakdown_filtered, colorscale=slider_colorscale
            ),
        )
    )
    st.plotly_chart(fig, use_container_width=True)
