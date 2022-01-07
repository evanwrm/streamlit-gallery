from dataclasses import dataclass

import cornac
import pandas as pd
import streamlit as st
from recommenders.models.cornac import cornac_utils
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED

from lib.constants import BPR_ID, DEFAULT_COL_ITEM, DEFAULT_COL_USER
from lib.recommenders import RecommenderModel


@dataclass
class RecommenderBPR(RecommenderModel):
    id: str = BPR_ID
    name: str = "Bayesian Personalized Ranking (BPR)"

    # Model params
    NUM_FACTORS: int = 200
    NUM_EPOCHS: int = 100

    def __post_init__(self):
        # Model
        self._model = cornac.models.BPR(
            k=self.NUM_FACTORS,
            max_iter=self.NUM_EPOCHS,
            learning_rate=0.01,
            lambda_reg=0.001,
            verbose=True,
            seed=SEED,
        )

    def fit(self, train, verbose: bool = True) -> None:
        train_set = cornac.data.Dataset.from_uir(
            train.itertuples(index=False), seed=SEED
        )
        self._model.verbose = verbose

        # Fit
        with st.spinner("Fitting Model"):
            with Timer() as t:
                self._model.fit(train_set)

        if verbose:
            st.write(f"Took {t} seconds for training [{self.name}]")

    def predict(
        self,
        train,
        col_user: str = DEFAULT_COL_USER,
        col_item: str = DEFAULT_COL_ITEM,
        verbose: bool = True,
    ) -> pd.DataFrame:
        # Prediction
        with st.spinner("Making Predictions"):
            with Timer() as t:
                all_predictions = cornac_utils.predict_ranking(
                    self._model,
                    train,
                    usercol=col_user,
                    itemcol=col_item,
                    remove_seen=True,
                )

        if verbose:
            st.write(f"Took {t} seconds for prediction [{self.name}]")
            st.write(all_predictions.head())

        return all_predictions
