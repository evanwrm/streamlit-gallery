import os
from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass, field
import _pickle as cPickle

import pandas as pd
import streamlit as st
from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel
from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator
from recommenders.models.deeprec.deeprec_utils import create_hparams
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED


from lib.constants import (
    ACT_RELU,
    DATA_PATH,
    DEFAULT_COL_ITEM,
    DEFAULT_COL_TIMESTAMP,
    DEFAULT_COL_USER,
    INIT_TNORMAL,
    LOSS_CROSS_ENTROPY,
    LOSS_LOG,
    LOSS_SOFTMAX,
    LOSS_SQUARE,
    METRIC_AUC,
    METRIC_GROUP_AUC,
    METRIC_LOGLOSS,
    METRIC_MEAN_MRR,
    METRIC_NDCG,
    OPT_ADADELTA,
    OPT_ADAM,
    OPT_FTRL,
    OPT_GD,
    OPT_PADAGRAD,
    OPT_PGD,
    OPT_RMSPROP,
    OPT_SQD,
    SLI_REC_ID,
)
from lib.recommenders import RecommenderModel


@dataclass
class RecommenderSLI_REC(RecommenderModel):
    id: str = SLI_REC_ID
    name: str = "Short-term and Long-term Preference Integrated Recommender (SLi-Rec)"

    # Model params
    LAYER_SIZES: List[int] = field(default_factory=lambda: [100, 64])
    ATT_FCN_LAYER_SIZES: List[int] = field(default_factory=lambda: [80, 40])
    ACTIVATION: List[str] = field(default_factory=lambda: [ACT_RELU, ACT_RELU])
    USER_DROPOUT: bool = True
    DROPOUT: List[int] = field(default_factory=lambda: [0.3, 0.3])
    ITEM_EMEDDING_DIM: int = 32
    CATE_EMEDDING_DIM: int = 8
    USER_EMEDDING_DIM: int = 16

    INIT_METHOD: Union[INIT_TNORMAL, str] = INIT_TNORMAL
    INIT_VALUE: float = 0.01
    EMBED_L2: float = 0.0001
    EMBED_L1: float = 0.0000
    LAYER_L2: float = 0.0001
    LAYER_L1: float = 0.0000
    CROSS_L2: float = 0.0000
    CROSS_L1: float = 0.0000
    LEARNING_RATE: float = 0.001
    LOSS: Union[LOSS_LOG, LOSS_CROSS_ENTROPY, LOSS_SQUARE, LOSS_SOFTMAX] = LOSS_SOFTMAX
    OPTIMIZER: Union[
        OPT_ADAM,
        OPT_ADADELTA,
        OPT_SQD,
        OPT_FTRL,
        OPT_GD,
        OPT_PADAGRAD,
        OPT_PGD,
        OPT_RMSPROP,
    ] = OPT_ADAM
    NUM_EPOCHS: int = 10
    BATCH_SIZE: int = 400
    ENABLE_BN: bool = True
    EARLY_STOP: int = 10
    MAX_SEQ_LENGTH: int = 50
    HIDDEN_STATES: int = 40
    ATTENTION_SIZE: int = 40
    NEED_SAMPLE: bool = True
    TRAIN_NUM_NGS: int = 4
    VALID_NUM_NGS: int = 4
    TEST_NUM_NGS: int = 9

    SHOW_STEPS: int = 100
    SAVE_MODEL: bool = False
    SAVE_EPOCH: int = 1
    METRICS: List[str] = field(default_factory=lambda: [METRIC_AUC, METRIC_LOGLOSS])
    PAIRWISE_METRICS: List[str] = field(
        default_factory=lambda: [
            METRIC_MEAN_MRR,
            METRIC_NDCG + "@2;4;6",
            METRIC_GROUP_AUC,
        ]
    )
    WRITE_TFEVENTS: bool = True

    def __post_init__(self):
        self._train_file = os.path.join(DATA_PATH, "train_data")
        self._valid_file = os.path.join(DATA_PATH, "valid_data")
        self._test_file = os.path.join(DATA_PATH, "test_data")
        self._output_file = os.path.join(DATA_PATH, "output.txt")
        self._user_vocab = os.path.join(DATA_PATH, "user_vocab.pkl")
        self._item_vocab = os.path.join(DATA_PATH, "item_vocab.pkl")
        self._cate_vocab = os.path.join(DATA_PATH, "category_vocab.pkl")

        hparams = create_hparams(
            {
                "method": "classification",
                "model_type": SLI_REC_ID,
                "layer_sizes": self.LAYER_SIZES,
                "att_fcn_layer_sizes": self.ATT_FCN_LAYER_SIZES,
                "activation": self.ACTIVATION,
                "user_dropout": self.USER_DROPOUT,
                "dropout": self.DROPOUT,
                "item_embedding_dim": self.ITEM_EMEDDING_DIM,
                "cate_embedding_dim": self.CATE_EMEDDING_DIM,
                "user_embedding_dim": self.USER_EMEDDING_DIM,
                "init_method": self.INIT_METHOD,
                "init_value": self.INIT_VALUE,
                "embed_l2": self.EMBED_L2,
                "embed_l1": self.EMBED_L1,
                "layer_l2": self.LAYER_L2,
                "layer_l1": self.LAYER_L1,
                "cross_l2": self.CROSS_L2,
                "cross_l1": self.CROSS_L1,
                "learning_rate": self.LEARNING_RATE,
                "loss": self.LOSS,
                "optimizer": self.OPTIMIZER,
                "epochs": self.NUM_EPOCHS,
                "batch_size": self.BATCH_SIZE,
                "enable_BN": self.ENABLE_BN,
                "EARLY_STOP": self.EARLY_STOP,
                "max_seq_length": self.MAX_SEQ_LENGTH,
                "hidden_state": self.HIDDEN_STATES,
                "attention_size": self.ATTENTION_SIZE,
                "need_sample": self.NEED_SAMPLE,
                "train_num_ngs": self.TRAIN_NUM_NGS,
                "show_step": self.SHOW_STEPS,
                "save_model": self.SAVE_MODEL,
                "save_epoch": self.SAVE_EPOCH,
                "metrics": self.METRICS,
                "pairwise_metrics": self.PAIRWISE_METRICS,
                "MODEL_DIR": os.path.join(DATA_PATH, "model/"),
                "SUMMARIES_DIR": os.path.join(DATA_PATH, "summary/"),
                "write_tfevents": self.WRITE_TFEVENTS,
                "user_vocab": self._user_vocab,
                "item_vocab": self._item_vocab,
                "cate_vocab": self._cate_vocab,
            }
        )
        input_creator = SequentialIterator

        # Model
        self._model = SLI_RECModel(hparams, input_creator, seed=SEED)

    def fit(self, train, verbose: bool = True) -> None:
        # Fit
        with st.spinner("Fitting Model"):
            with Timer() as t:
                self._model.fit(
                    self._train_file, self._valid_file, valid_num_ngs=self.VALID_NUM_NGS
                )

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
                self._model.predict(self._test_file, self._output_file)

        all_predictions = pd.read_csv(self._output_file, sep="\t")

        if verbose:
            st.write(f"Took {t} seconds for prediction [{self.name}]")
            st.write(all_predictions.head())

        return all_predictions

    def split_data(
        self,
        data: pd.DataFrame,
        uir_cols: List[str],
        ratio: float = 0.75,
        col_timestamp: str = DEFAULT_COL_TIMESTAMP,
        user_features: Tuple[List, List] = None,
        item_features: Tuple[List, List] = None,
        verbose=True,
    ) -> Tuple[Any, Any, Union[Dict, None]]:
        # We need to split into train, validation and test set files here
        # we'll put the last in seq in test, 2nd last in val, and the rest in train

        with Timer() as t:
            # Construct users dict
            users_dict = {}
            items_dict = {}
            cates_dict = {}
            for row in data.iterrows():
                user_id = row[uir_cols[0]]
                if user_id not in users_dict:
                    users_dict[user_id] = []

                item_id = row[uir_cols[1]]
                if item_id not in items_dict:
                    items_dict[item_id] = []

                cate_id = "Games"
                if cate_id not in cates_dict:
                    cates_dict[cate_id] = []

                rating = row[uir_cols[2]]

                users_dict[user_id].append(
                    f"{rating}\t{user_id}\t{item_id}\t{cate_id}\t{data[col_timestamp]}\t"
                )
                items_dict[item_id].append(True)
                cates_dict[cate_id].append(True)

            # Add history ids and timestamps
            # Write train, validation, and test files
            with open(self._train_file, "w") as f_train, open(
                self._valid_file, "w"
            ) as f_valid, open(self._test_file, "w") as f_test:
                for user in users_dict:
                    sorted_user = sorted(user, key=lambda x: x[col_timestamp])
                    user_len = len(sorted_user)
                    history = []
                    for idx, interaction in enumerate(sorted_user):
                        (
                            label,
                            user_id,
                            item_id,
                            category_id,
                            timestamp,
                        ) = interaction.split("\t")
                        history_cate_ids = ",".join(map(lambda x: x[0]))
                        history_timestamps = ",".join(map(lambda x: x[1]))
                        history.append((item_id, timestamp))
                        line = f"{label}\t{user_id}\t{item_id}\t{category_id}\t{timestamp}\t{history_cate_ids}\t{history_timestamps}"
                        if idx < user_len - 2:
                            f_train.write(line)
                        elif idx < user_len - 1:
                            f_valid.write(line)
                        else:
                            f_test.write(line)

            sorted_user_dict = sorted(
                users_dict.items(), key=lambda x: len(x[1]), reverse=True
            )
            sorted_item_dict = sorted(
                items_dict.items(), key=lambda x: len(x[1]), reverse=True
            )
            sorted_cat_dict = sorted(
                cates_dict.items(), key=lambda x: len(x[1]), reverse=True
            )

            uid_voc = {}
            index = 0
            for key, _ in sorted_user_dict:
                uid_voc[key] = index
                index += 1

            mid_voc = {}
            mid_voc["default_mid"] = 0
            index = 1
            for key, _ in sorted_item_dict:
                mid_voc[key] = index
                index += 1

            cat_voc = {}
            cat_voc["default_cat"] = 0
            index = 1
            for key, _ in sorted_cat_dict:
                cat_voc[key] = index
                index += 1

            with open(self._user_vocab, "wb") as uv, open(
                self._item_vocab, "wb"
            ) as iv, open(self._cate_vocab, "wb") as cv:
                cPickle.dump(uid_voc, uv)
                cPickle.dump(mid_voc, iv)
                cPickle.dump(cat_voc, cv)

        if verbose:
            st.write(f"Took {t} seconds for preparing train/test split [{self.name}]")

        return None, None, None
