import torch

from federated_gcn_aml.training import metrics as metrics_module
from federated_gcn_aml.training.metrics import (
    compute_metrics_from_probabilities,
    compute_ranking_metrics,
    compute_threshold_metrics,
    select_threshold_from_probabilities,
)


def test_reporting_metrics_include_auroc_precision_recall_and_confusion_counts():
    y = torch.tensor([0, 1, 1, 0])
    prob = torch.tensor([0.1, 0.8, 0.4, 0.7]).numpy()

    metrics = compute_metrics_from_probabilities(y, prob, threshold=0.5)

    assert metrics["AUROC"] == 0.75
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["F1"] == 0.5
    assert metrics["MCC"] == 0.0
    assert metrics["TP"] == 1
    assert metrics["FP"] == 1
    assert metrics["TN"] == 1
    assert metrics["FN"] == 1


def test_single_class_target_records_undefined_auroc_as_none():
    y = torch.tensor([0, 0, 0])
    prob = torch.tensor([0.1, 0.2, 0.3]).numpy()

    metrics = compute_metrics_from_probabilities(y, prob, threshold=0.5)

    assert metrics["AUROC"] is None
    assert metrics["TP"] == 0
    assert metrics["FP"] == 0
    assert metrics["TN"] == 3
    assert metrics["FN"] == 0


def test_thresholded_reporting_fields_use_passed_threshold():
    y = torch.tensor([0, 1, 1, 0])
    prob = torch.tensor([0.1, 0.8, 0.4, 0.7]).numpy()

    low_threshold = compute_metrics_from_probabilities(y, prob, threshold=0.5)
    high_threshold = compute_metrics_from_probabilities(y, prob, threshold=0.75)

    assert low_threshold["FP"] == 1
    assert high_threshold["FP"] == 0
    assert low_threshold["recall"] == 0.5
    assert high_threshold["recall"] == 0.5


def test_ranking_and_threshold_helpers_recombine_to_full_metric_set():
    y = torch.tensor([0, 1, 1, 0])
    prob = torch.tensor([0.1, 0.8, 0.4, 0.7]).numpy()

    full = compute_metrics_from_probabilities(y, prob, threshold=0.5)
    ranking = compute_ranking_metrics(y, prob)
    thresholded = compute_threshold_metrics(y, prob, threshold=0.5)

    assert full == {**ranking, **thresholded}


def test_threshold_selection_uses_thresholded_metrics_only(monkeypatch):
    y = torch.tensor([0, 1, 1, 0])
    prob = torch.tensor([0.1, 0.8, 0.4, 0.7]).numpy()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("ranking metrics should not be needed during threshold search")

    monkeypatch.setattr(metrics_module, "average_precision_score", fail_if_called)
    monkeypatch.setattr(metrics_module, "roc_auc_score", fail_if_called)
    threshold, selected_metrics = select_threshold_from_probabilities(
        y,
        prob,
        candidates=[0.5, 0.75],
    )

    assert threshold == 0.75
    assert selected_metrics["TP"] == 1
    assert selected_metrics["FP"] == 0
    assert selected_metrics["TN"] == 2
    assert selected_metrics["FN"] == 1


def test_selected_threshold_matches_f1_grid_mcc_tiebreak():
    y = torch.tensor([0, 1, 1, 0])
    prob = torch.tensor([0.1, 0.8, 0.4, 0.7]).numpy()

    threshold, selected_metrics = select_threshold_from_probabilities(
        y,
        prob,
        candidates=[0.5, 0.75],
    )

    assert threshold == 0.75
    assert selected_metrics["F1"] > compute_threshold_metrics(y, prob, threshold=0.5)["F1"]
