from typing import List, Union
from dataclasses import dataclass, field

import torch
import cornac
import pandas as pd
import streamlit as st
from recommenders.models.cornac import cornac_utils
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED

from lib.constants import (
    ACT_ELU,
    ACT_RELU,
    ACT_RELU6,
    ACT_SIGMOID,
    ACT_TANH,
    BIVAE_ID,
    DEFAULT_COL_ITEM,
    DEFAULT_COL_USER,
    LIKELIHOOD_BERN,
    LIKELIHOOD_GAUS,
    LIKELIHOOD_POIS,
)
from lib.recommenders import RecommenderModel


@dataclass
class RecommenderBiVAE(RecommenderModel):
    id: str = BIVAE_ID
    name: str = "Bilateral Variational Autoencoder (BiVAE)"

    # Model parameters
    LATENT_DIM: int = 50
    ENCODER_DIMS: List[int] = field(default_factory=lambda: [100])
    ACT_FUNC: Union[ACT_SIGMOID, ACT_TANH, ACT_ELU, ACT_RELU, ACT_RELU6] = ACT_TANH
    LIKELIHOOD: Union[
        LIKELIHOOD_BERN, LIKELIHOOD_GAUS, LIKELIHOOD_POIS
    ] = LIKELIHOOD_POIS
    NUM_EPOCHS: int = 500
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 0.001

    def __post_init__(self):
        # Model
        self._model = cornac.models.BiVAECF(
            k=self.LATENT_DIM,
            encoder_structure=self.ENCODER_DIMS,
            act_fn=self.ACT_FUNC,
            likelihood=self.LIKELIHOOD,
            n_epochs=self.NUM_EPOCHS,
            batch_size=self.BATCH_SIZE,
            learning_rate=self.LEARNING_RATE,
            seed=SEED,
            use_gpu=torch.cuda.is_available(),
            verbose=True,
        )

    def fit(self, train, verbose: bool = True):
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
