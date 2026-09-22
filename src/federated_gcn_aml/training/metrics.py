import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


THRESHOLD_SELECTION_RULE = "validation_f1_grid_mcc_tiebreak"
RANKING_METRIC_KEYS = ("AUPR", "AUROC")
THRESHOLDED_METRIC_KEYS = ("F1", "MCC", "precision", "recall", "TP", "FP", "TN", "FN")


def logits_to_probabilities(logits: torch.Tensor) -> np.ndarray:
    """Convert raw logits to detached probabilities."""
    return torch.sigmoid(logits).detach().cpu().numpy()


def _labels_to_numpy(y_true: torch.Tensor) -> np.ndarray:
    return y_true.detach().cpu().numpy().astype(np.int64)


def _compute_ranking_metrics_np(y_true_np: np.ndarray, prob_np: np.ndarray) -> dict:
    aupr = average_precision_score(y_true_np, prob_np)
    auroc = None
    if np.unique(y_true_np).size > 1:
        auroc = float(roc_auc_score(y_true_np, prob_np))
    return {"AUPR": float(aupr), "AUROC": auroc}


def _compute_threshold_metrics_np(y_true_np: np.ndarray, prob_np: np.ndarray, threshold: float) -> dict:
    pred_np = (prob_np >= threshold).astype(np.int64)
    mcc = matthews_corrcoef(y_true_np, pred_np)
    f1 = f1_score(y_true_np, pred_np, zero_division=0)
    precision = precision_score(y_true_np, pred_np, zero_division=0)
    recall = recall_score(y_true_np, pred_np, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true_np, pred_np, labels=[0, 1]).ravel()
    return {
        "MCC": float(mcc),
        "F1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def compute_ranking_metrics(y_true: torch.Tensor, prob_np: np.ndarray) -> dict:
    """Compute threshold-independent AML reporting metrics."""
    return _compute_ranking_metrics_np(_labels_to_numpy(y_true), prob_np)


def compute_threshold_metrics(y_true: torch.Tensor, prob_np: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute selected-threshold AML reporting metrics."""
    return _compute_threshold_metrics_np(_labels_to_numpy(y_true), prob_np, threshold)


def compute_metrics_from_probabilities(y_true: torch.Tensor, prob_np: np.ndarray, threshold: float = 0.5):
    """Compute AML classification metrics from probabilities at a chosen threshold."""
    y_true_np = _labels_to_numpy(y_true)
    ranking = _compute_ranking_metrics_np(y_true_np, prob_np)
    thresholded = _compute_threshold_metrics_np(y_true_np, prob_np, threshold)
    return {**ranking, **thresholded}


def combine_ranking_and_threshold_metrics(ranking_metrics: dict, threshold_metrics: dict) -> dict:
    """Use threshold-independent ranking metrics plus selected-threshold reporting metrics."""
    out = {key: ranking_metrics[key] for key in RANKING_METRIC_KEYS}
    out.update({key: threshold_metrics[key] for key in THRESHOLDED_METRIC_KEYS})
    return out


def select_threshold_from_probabilities(y_true: torch.Tensor, prob_np: np.ndarray, candidates=None) -> tuple[float, dict]:
    """Pick a validation threshold for reporting F1/MCC without affecting model selection."""
    if candidates is None:
        candidates = np.linspace(0.05, 0.95, 19)

    y_true_np = _labels_to_numpy(y_true)
    best_threshold = 0.5
    best_metrics = _compute_threshold_metrics_np(y_true_np, prob_np, threshold=best_threshold)

    for candidate in candidates:
        metrics = _compute_threshold_metrics_np(y_true_np, prob_np, threshold=float(candidate))
        if (
            metrics["F1"] > best_metrics["F1"]
            or (
                metrics["F1"] == best_metrics["F1"]
                and metrics["MCC"] > best_metrics["MCC"]
            )
        ):
            best_threshold = float(candidate)
            best_metrics = metrics

    return best_threshold, best_metrics


@torch.no_grad()
def select_threshold_by_f1(y_true: torch.Tensor, logits: torch.Tensor, candidates=None) -> tuple[float, dict]:
    """Pick a validation threshold from logits for reporting F1/MCC without changing AUPR model selection."""
    prob_np = logits_to_probabilities(logits)
    return select_threshold_from_probabilities(y_true, prob_np, candidates=candidates)


@torch.no_grad()
def compute_metrics(y_true: torch.Tensor, logits: torch.Tensor, threshold: float = 0.5):
    """
    y_true: shape [N] int {0,1}
    logits: shape [N] raw logits
    """
    prob_np = logits_to_probabilities(logits)
    return compute_metrics_from_probabilities(y_true, prob_np, threshold=threshold)
