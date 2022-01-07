import warnings
import glob
import datetime as dt
from typing import ForwardRef, List, Union
from dataclasses import Field, fields

import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype, is_string_dtype
from lib.constants import DEFAULT_BIN_COUNT, DEFAULT_DELIMITER

from lib.recommenders import RecommenderModel


def unix_timestamp(date: dt.datetime):
    """Converts datetime to unix timestamp"""
    epoch = dt.datetime.utcfromtimestamp(0)
    return (date - epoch).total_seconds()


def get_dataset_files(path, extension="csv"):
    return glob.glob(f"{path}/**/*.{extension}")


def unique_categories(df: pd.Series, delimiter=DEFAULT_DELIMITER):
    return df.str.split(delimiter).explode().unique()


def count_categories(df: pd.Series, delimiter=DEFAULT_DELIMITER):
    return df.str.split(delimiter).explode().value_counts()


def column_index(df: pd.DataFrame, args: List[str]):
    cols = df.columns.tolist()
    for arg in args:
        if arg in cols:
            return cols.index(arg)

    return cols[0]


def bin_feature(
    df: pd.Series, bins=DEFAULT_BIN_COUNT, delimiter=DEFAULT_DELIMITER, warn=False
):
    if is_string_dtype(df):
        return unique_categories(df, delimiter=delimiter), df.str.split(delimiter)
    if is_numeric_dtype(df):
        binned = df.name + pd.Series(pd.cut(df, bins, duplicates="drop")).astype(str)
        return binned.unique(), binned.map(lambda x: [x])

    if warn:
        warnings.warn(f"Failed to bin {df.dtype} for {df.name}!")
    else:
        raise TypeError(f"Failed to bin {df.dtype} for {df.name}!")


def type_to_streamlit_element(field: Field, sidebar=True, warn=True):
    name = field.name
    value = field.default or field.default_factory()

    if field.type is int or field.type is float:
        if sidebar:
            return st.sidebar.number_input(name, value=value)
        else:
            return st.number_input(name, value=value)
    if field.type is str:
        if sidebar:
            return st.sidebar.text_input(name, value=value)
        else:
            return st.text_input(name, value=value)
    if field.type is bool:
        if sidebar:
            return st.sidebar.checkbox(name, value=value)
        else:
            return st.checkbox(name, value=value)
    if field.type.__origin__ is Union:
        args = field.type.__args__
        options = []
        for arg in args:
            if type(arg) is ForwardRef:
                options.append(arg.__forward_arg__)
        if sidebar:
            return st.sidebar.selectbox(
                name, options=options, index=options.index(value)
            )
        else:
            return st.selectbox(name, options=options, index=options.index(value))

    if warn:
        warnings.warn(f"Type {field.type} for {name} not recognized!")
    else:
        raise TypeError(f"Type {field.type} for {name} not recognized!")


def get_model_parameters(model: RecommenderModel):
    return {
        field.name: type_to_streamlit_element(field)
        for field in fields(model)
        if field.name.isupper()
    }
