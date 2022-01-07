from typing import List

import streamlit as st
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

from lib.constants import (
    DEFAULT_COL_ITEM,
    DEFAULT_COL_PREDICTIONS,
    DEFAULT_COL_RATING,
    DEFAULT_COL_USER,
    DEFAULT_TOP_K,
    GRU4REC_ID,
    LIGHTGCN_ID,
    NRMS_ID,
    SLI_REC_ID,
    SUM_ID,
    VOWPAL_WABBIT_ID,
    WIDE_AND_DEEP_ID,
)
from lib.models.base import RecommenderModel
from lib.models.bivae import RecommenderBiVAE
from lib.models.bpr import RecommenderBPR
from lib.models.lightfm import RecommenderLightFM


MODELS: List[RecommenderModel] = [
    RecommenderBPR,
    RecommenderModel(
        id=NRMS_ID, name="Neural Recommendation with Multi-Head Self-Attention (NRMS)"
    ),
    RecommenderBiVAE,
    RecommenderModel(
        id=SLI_REC_ID,
        name="Short-term and Long-term Preference Integrated Recommender (SLi-Rec)",
    ),
    RecommenderModel(id=SUM_ID, name="Sequential User Modeling (SUM)"),
    RecommenderModel(id=GRU4REC_ID, name="Gru4Rec"),
    RecommenderLightFM,
    RecommenderModel(id=LIGHTGCN_ID, name="LightGCN"),
    RecommenderModel(id=VOWPAL_WABBIT_ID, name="Vowpal Wabbit"),
    RecommenderModel(id=WIDE_AND_DEEP_ID, name="Wide and Deep"),
]
MODELS_MAP = {m.id: m for m in MODELS}


def write_metrics(
    ratings_true,
    ratings_pred,
    top_k=DEFAULT_TOP_K,
    col_user=DEFAULT_COL_USER,
    col_item=DEFAULT_COL_ITEM,
    col_rating=DEFAULT_COL_RATING,
):
    with st.spinner("Collecting Metrics"):
        eval_map = map_at_k(
            ratings_true,
            ratings_pred,
            k=top_k,
            col_prediction=DEFAULT_COL_PREDICTIONS,
            col_user=col_user,
            col_item=col_item,
            col_rating=col_rating,
        )
        eval_ndcg = ndcg_at_k(
            ratings_true,
            ratings_pred,
            k=top_k,
            col_prediction=DEFAULT_COL_PREDICTIONS,
            col_user=col_user,
            col_item=col_item,
            col_rating=col_rating,
        )
        eval_precision = precision_at_k(
            ratings_true,
            ratings_pred,
            k=top_k,
            col_prediction=DEFAULT_COL_PREDICTIONS,
            col_user=col_user,
            col_item=col_item,
            col_rating=col_rating,
        )
        eval_recall = recall_at_k(
            ratings_true,
            ratings_pred,
            k=top_k,
            col_prediction=DEFAULT_COL_PREDICTIONS,
            col_user=col_user,
            col_item=col_item,
            col_rating=col_rating,
        )

    st.write(
        f"""
            **MAP**:\t{eval_map}  
            **NDCG**:\t{eval_ndcg}  
            **Precision@{top_k}**:\t{eval_precision}  
            **Recall@{top_k}**:\t{eval_recall}  
        """
    )
