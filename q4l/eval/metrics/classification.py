"""NOTES from ChatGPT:

Please note that for multiclass classification tasks, roc_auc_score expects the labels to be binarized (one-hot encoded), and log_loss expects the predictions to be probabilities for each class. The other functions expect the predictions to be the predicted class labels. Adjust the input as necessary for your specific use case.

Also, note that these functions use "weighted" average for multiclass tasks. This means that the metric is calculated for each class, taking into account class imbalance by averaging their scores weighted by the number of true instances for each class. You can change this to "micro" or "macro" as per your needs.

These functions assume that your data is not already one-hot encoded. If your data is one-hot encoded, you'll need to use np.argmax to convert predictions and labels to class labels.

"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def accuracy(pred: pd.Series, label: pd.Series, dimension: str = "datetime") -> float:
    """Calculate Accuracy."""
    if dimension not in ["datetime", "instrument"]:
        raise ValueError("`dimension` must be either 'datetime' or 'instrument'")

    accuracy_scores = []

    for _, sub_pred in pred.groupby(level=dimension):
        sub_label = label.loc[sub_pred.index]
        accuracy_scores.append(accuracy_score(sub_label, sub_pred))

    return np.mean(accuracy_scores)


def precision(pred: pd.Series, label: pd.Series, dimension: str = "datetime") -> float:
    """Calculate Precision."""
    if dimension not in ["datetime", "instrument"]:
        raise ValueError("`dimension` must be either 'datetime' or 'instrument'")

    precision_scores = []

    for _, sub_pred in pred.groupby(level=dimension):
        sub_label = label.loc[sub_pred.index]
        precision_scores.append(precision_score(sub_label, sub_pred, average="weighted"))

    return np.mean(precision_scores)


def recall(pred: pd.Series, label: pd.Series, dimension: str = "datetime") -> float:
    """Calculate Recall."""
    if dimension not in ["datetime", "instrument"]:
        raise ValueError("`dimension` must be either 'datetime' or 'instrument'")

    recall_scores = []

    for _, sub_pred in pred.groupby(level=dimension):
        sub_label = label.loc[sub_pred.index]
        recall_scores.append(recall_score(sub_label, sub_pred, average="weighted"))

    return np.mean(recall_scores)


def f1(pred: pd.Series, label: pd.Series, dimension: str = "datetime") -> float:
    """Calculate F1-score."""
    if dimension not in ["datetime", "instrument"]:
        raise ValueError("`dimension` must be either 'datetime' or 'instrument'")

    f1_scores = []

    for _, sub_pred in pred.groupby(level=dimension):
        sub_label = label.loc[sub_pred.index]
        f1_scores.append(f1_score(sub_label, sub_pred, average="weighted"))

    return np.mean(f1_scores)


def roc_auc(pred: pd.Series, label: pd.Series, dimension: str = "datetime") -> float:
    """Calculate ROC AUC score."""
    if dimension not in ["datetime", "instrument"]:
        raise ValueError("`dimension` must be either 'datetime' or 'instrument'")

    roc_auc_scores = []

    for _, sub_pred in pred.groupby(level=dimension):
        sub_label = label.loc[sub_pred.index]
        roc_auc_scores.append(roc_auc_score(sub_label, sub_pred, average="weighted"))

    return np.mean(roc_auc_scores)


def logloss(pred: pd.Series, label: pd.Series, dimension: str = "datetime") -> float:
    """Calculate Log-loss."""
    if dimension not in ["datetime", "instrument"]:
        raise ValueError("`dimension` must be either 'datetime' or 'instrument'")

    logloss_scores = []

    for _, sub_pred in pred.groupby(level=dimension):
        sub_label = label.loc[sub_pred.index]
        logloss_scores.append(log_loss(sub_label, sub_pred))

    return np.mean(logloss_scores)
