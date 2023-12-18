import numpy as np
import pandas as pd


def mrr(
    pred: pd.Series, label: pd.Series, dimension: str = "datetime"
) -> float:
    """Calculate MRR."""
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "`dimension` must be either 'datetime' or 'instrument'"
        )

    mrr_scores = []

    for _, sub_pred in pred.groupby(level=dimension):
        sub_label = label.loc[sub_pred.index]
        pred_argsorted = sub_pred.argsort()[::-1]
        label_argsorted = sub_label.argsort()[::-1]
        top1_stock_index = label_argsorted[0]
        pos_in_pred = pred_argsorted.tolist().index(top1_stock_index) + 1
        reciprocal_rank = 1 / pos_in_pred
        mrr_scores.append(reciprocal_rank)

    return np.mean(mrr_scores)


def precision_at_k(
    pred: pd.Series, label: pd.Series, k: int, dimension: str = "datetime"
) -> float:
    """Calculate precision at k."""
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "`dimension` must be either 'datetime' or 'instrument'"
        )

    prec_scores = []

    for _, sub_pred in pred.groupby(level=dimension):
        sub_label = label.loc[sub_pred.index]
        pred_argsorted = sub_pred.argsort()[::-1]
        label_argsorted = sub_label.argsort()[::-1]
        pred_topk = pred_argsorted[:k]
        label_topk = label_argsorted[:k]
        intersect_cardinality = len(
            set(pred_topk).intersection(set(label_topk))
        )
        precision = intersect_cardinality / k
        prec_scores.append(precision)

    return np.mean(prec_scores)


def ndcg_at_k(
    pred: pd.Series, label: pd.Series, k: int, dimension: str = "datetime"
) -> float:
    """Calculate ndcg at k."""
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "`dimension` must be either 'datetime' or 'instrument'"
        )

    ndcg_scores = []

    for _, sub_pred in pred.groupby(level=dimension):
        sub_label = label.loc[sub_pred.index]
        pred_argsorted = sub_pred.argsort()[::-1]
        label_argsorted = sub_label.argsort()[::-1]
        pred_topk = pred_argsorted[:k]
        label_topk = label_argsorted[:k]
        dcg = (pred_topk == label_topk).sum() / np.log2(np.arange(2, k + 2))
        idcg = (np.ones(k) / np.log2(np.arange(2, k + 2))).sum()
        ndcg_scores.append(dcg / idcg)

    return np.mean(ndcg_scores)


def map_at_k(
    pred: pd.Series, label: pd.Series, k: int, dimension: str = "datetime"
) -> float:
    """Calculate map at k."""
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "`dimension` must be either 'datetime' or 'instrument'"
        )

    map_scores = []

    for _, sub_pred in pred.groupby(level=dimension):
        sub_label = label.loc[sub_pred.index]
        pred_argsorted = sub_pred.argsort()[::-1]
        pred_topk = pred_argsorted[:k]
        ap = (pred_topk == sub_label).mean()
        map_scores.append(ap)

    return np.mean(map_scores)


def recall_at_k(
    pred: pd.Series, label: pd.Series, k: int, dimension: str = "datetime"
) -> float:
    """Calculate recall@k."""
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "`dimension` must be either 'datetime' or 'instrument'"
        )

    recall_scores = []

    for _, sub_pred in pred.groupby(level=dimension):
        sub_label = label.loc[sub_pred.index]
        pred_argsorted = sub_pred.argsort()[::-1][:k]
        label_argsorted = sub_label.argsort()[::-1][:k]
        hits = len(set(pred_argsorted.index) & set(label_argsorted.index))
        recall = hits / len(label_argsorted)
        recall_scores.append(recall)

    return np.mean(recall_scores)


def f1_at_k(
    pred: pd.Series, label: pd.Series, k: int, dimension: str = "datetime"
) -> float:
    """Calculate F1@k."""
    if dimension not in ["datetime", "instrument"]:
        raise ValueError(
            "`dimension` must be either 'datetime' or 'instrument'"
        )

    f1_scores = []

    for _, sub_pred in pred.groupby(level=dimension):
        sub_label = label.loc[sub_pred.index]
        pred_argsorted = sub_pred.argsort()[::-1][:k]
        label_argsorted = sub_label.argsort()[::-1][:k]
        hits = len(set(pred_argsorted.index) & set(label_argsorted.index))
        precision = hits / len(pred_argsorted)
        recall = hits / len(label_argsorted)
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )
        f1_scores.append(f1)

    return np.mean(f1_scores)
