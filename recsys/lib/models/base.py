from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass

import pandas as pd
import streamlit as st
from recommenders.utils.timer import Timer
from recommenders.datasets.python_splitters import python_random_split

from lib.constants import DEFAULT_COL_ITEM, DEFAULT_COL_RATING, DEFAULT_COL_USER


@dataclass
class RecommenderModel:
    id: str
    name: str

    @abstractmethod
    def __post_init__(self):
        self._model = None

    @abstractmethod
    def fit(
        self,
        train,
        config: Union[Dict, None] = None,
        col_user: str = DEFAULT_COL_USER,
        col_item: str = DEFAULT_COL_ITEM,
        col_rating: str = DEFAULT_COL_RATING,
        verbose: bool = True,
    ) -> None:
        raise NotImplementedError(
            "RecommenderModel with ID {self.id} has no implemented fit method!"
        )

    @abstractmethod
    def predict(
        self,
        train,
        config: Union[Dict, None] = None,
        col_user: str = DEFAULT_COL_USER,
        col_item: str = DEFAULT_COL_ITEM,
        col_rating: str = DEFAULT_COL_RATING,
        verbose: bool = True,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "RecommenderModel with ID {self.id} has no implemented predict method!"
        )

    def split_data(
        self,
        data: pd.DataFrame,
        uir_cols: List[str],
        ratio: float = 0.75,
        user_features: Tuple[List, List] = None,
        item_features: Tuple[List, List] = None,
        verbose=True,
    ) -> Tuple[Any, Any, Union[Dict, None]]:
        """Random train-test splitter

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset to be split
        uir_cols : List[str]
            Columns that contain the user-item-rating information
        ratio : float
            Ratio of the training dataset to split
        col_user : str
            Column name of the user id
        col_item : str
            Column name of the item id
        col_rating : str
            Column name of the rating
        verbose : bool
            Whether to print out the time taken to split the training data

        Returns
        -------
        Tuple[Any, Any, Union[Dict, None]]
            A tuple of train and test dataframes, and (optionally)
            extra config information required for further training
        """

        with Timer() as t:
            train, test = python_random_split(data[uir_cols], ratio=ratio)

        if verbose:
            st.write(f"Took {t} seconds for preparing train/test split [{self.name}]")

        return train, test, None
