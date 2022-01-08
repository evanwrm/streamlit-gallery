from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

import lightfm
import numpy as np
import pandas as pd
import streamlit as st
from lightfm import cross_validation
from lightfm.data import Dataset
from recommenders.models.lightfm import lightfm_utils
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED

from lib.constants import (
    DEFAULT_COL_PREDICTIONS,
    LIGHTFM_ID,
    DEFAULT_COL_ITEM,
    DEFAULT_COL_USER,
    LOSS_BPR,
    LOSS_LOGISTIC,
    LOSS_WARP,
    LOSS_WARP_KOS,
    OPT_ADADELTA,
    OPT_ADAGRAD,
)
from lib.recommenders import RecommenderModel


@dataclass
class RecommenderLightFM(RecommenderModel):
    id: str = LIGHTFM_ID
    name: str = "LightFM"

    # Model parameters
    LOSS: Union[LOSS_LOGISTIC, LOSS_BPR, LOSS_WARP, LOSS_WARP_KOS] = LOSS_WARP
    LEARNING_SCHEDULE: Union[OPT_ADAGRAD, OPT_ADADELTA] = OPT_ADAGRAD
    LEARNING_RATE: float = 0.25
    NO_COMPONENTS: int = 20
    NO_EPOCHS: int = 20
    ITEM_ALPHA: float = 1e-6
    USER_ALPHA: float = 1e-6

    def __post_init__(self):
        # Model
        self._model = lightfm.lightfm.LightFM(
            loss=self.LOSS,
            learning_schedule=self.LEARNING_SCHEDULE,
            no_components=self.NO_COMPONENTS,
            learning_rate=self.LEARNING_RATE,
            random_state=np.random.RandomState(SEED),
        )

    def fit(
        self,
        train,
        config: Union[Dict, None] = None,
        verbose: bool = True,
    ) -> None:
        # Fit
        with st.spinner("Fitting Model"):
            with Timer() as t:
                self._model.fit(
                    interactions=train,
                    user_features=config["user_features"],
                    item_features=config["item_features"],
                    epochs=self.NO_EPOCHS,
                )

        if verbose:
            st.write(f"Took {t} seconds for training [{self.name}]")

    def predict(
        self,
        train,
        config: Union[Dict, None] = None,
        col_user: str = DEFAULT_COL_USER,
        col_item: str = DEFAULT_COL_ITEM,
        verbose: bool = True,
    ) -> pd.DataFrame:
        data, uid_map, iid_map = config["data"], config["uid_map"], config["iid_map"]
        data = data.rename({col_user: "userID", col_item: "itemID"}, axis=1)

        # Prediction
        with st.spinner("Making Predictions"):
            with Timer() as t:
                # Note: source does not convert to int (may want to monkey patch)
                all_predictions = lightfm_utils.prepare_all_predictions(
                    data,
                    uid_map,
                    iid_map,
                    interactions=train,
                    user_features=config["user_features"],
                    item_features=config["item_features"],
                    model=self._model,
                    num_threads=8,
                )

                all_predictions = all_predictions.rename(
                    {
                        "userID": col_user,
                        "itemID": col_item,
                        "prediction": DEFAULT_COL_PREDICTIONS,
                    },
                    axis=1,
                )

        if verbose:
            st.write(f"Took {t} seconds for prediction [{self.name}]")
            st.write(all_predictions.head())

        return all_predictions

    def similar_users(
        self, user_id: int, config: Union[Dict, None] = None
    ) -> pd.DataFrame:
        return lightfm_utils.similar_users(
            user_id=user_id, user_features=config["user_features"], model=self._model
        )

    def similar_items(
        self, item_id: int, config: Union[Dict, None] = None
    ) -> pd.DataFrame:
        return lightfm_utils.similar_items(
            itme_id=item_id, user_features=config["item_features"], model=self._model
        )

    def split_data(
        self,
        data: pd.DataFrame,
        uir_cols: List[str],
        ratio: float = 0.75,
        user_features=None,
        item_features=None,
        verbose=True,
    ):
        dataset = Dataset()

        # LightFM works slightly different as both train as test splits must have the same dimension
        dataset.fit(
            users=data[uir_cols[0]],
            items=data[uir_cols[1]],
            user_features=user_features[0] if user_features else None,
            item_features=item_features[0] if item_features else None,
        )
        if user_features is not None:
            user_features = dataset.build_user_features(
                (x, y) for x, y in zip(data[uir_cols[0]], user_features[1])
            )
        if item_features is not None:
            item_features = dataset.build_item_features(
                (x, y) for x, y in zip(data[uir_cols[1]], item_features[1])
            )
        interactions, weights = dataset.build_interactions(data[uir_cols].values)
        (
            train_interactions,
            test_interactions,
        ) = cross_validation.random_train_test_split(
            interactions,
            test_percentage=1 - ratio,
            random_state=np.random.RandomState(SEED),
        )

        uids, iids, interaction_data = cross_validation._shuffle(
            interactions.row,
            interactions.col,
            interactions.data,
            random_state=np.random.RandomState(SEED),
        )

        cutoff = int(ratio * len(uids))
        test_idx = slice(cutoff, None)
        uid_map, ufeature_map, iid_map, ifeature_map = dataset.mapping()

        with Timer() as t:
            test_df = lightfm_utils.prepare_test_df(
                test_idx, uids, iids, uid_map, iid_map, weights
            )
            test_df = test_df.rename(
                {"userID": uir_cols[0], "itemID": uir_cols[1], "rating": uir_cols[2]},
                axis=1,
            )

        if verbose:
            st.write(f"Took {t} seconds for preparing train/test split [{self.name}]")

        return (
            train_interactions,
            test_df,
            {
                "data": data,
                "uid_map": uid_map,
                "iid_map": iid_map,
                "user_features": user_features,
                "item_features": item_features,
            },
        )
