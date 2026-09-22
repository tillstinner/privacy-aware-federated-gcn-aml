import pytest
import torch

from federated_gcn_aml.federated.evaluator import GlobalOwnedNodeEvaluator
from federated_gcn_aml.federated.messages import ClientPredictionPacket, PredictionSplitPacket


def _packet(client_id, val_ids, test_ids, val_y=None, test_y=None):
    val_y = [0, 1][: len(val_ids)] if val_y is None else val_y
    test_y = [1, 0][: len(test_ids)] if test_y is None else test_y
    return ClientPredictionPacket(
        client_id=client_id,
        val=PredictionSplitPacket("val", torch.tensor(val_ids), torch.tensor(val_y), torch.tensor([-1.0, 2.0][: len(val_ids)])),
        test=PredictionSplitPacket("test", torch.tensor(test_ids), torch.tensor(test_y), torch.tensor([2.0, -1.0][: len(test_ids)])),
    )


def test_evaluator_checks_counts_and_duplicates_and_uses_global_packets():
    evaluator = GlobalOwnedNodeEvaluator(expected_val_count=2, expected_test_count=2)
    metrics, predictions = evaluator.evaluate([
        _packet("a", [3], [4], val_y=[1], test_y=[0]),
        _packet("b", [1], [2], val_y=[0], test_y=[1]),
    ])
    assert predictions["val"]["global_node_indices"].tolist() == [1, 3]
    assert metrics["actual_val_count"] == 2
    assert "selected_threshold" in metrics
    assert metrics["threshold_selection_rule"] == "validation_f1_grid_mcc_tiebreak"
    for split in ("val", "test"):
        for key in ("AUPR", "AUROC", "F1", "MCC", "precision", "recall", "TP", "FP", "TN", "FN"):
            assert key in metrics[split]
    with pytest.raises(ValueError, match="Duplicate"):
        evaluator.evaluate([_packet("a", [1], [2]), _packet("b", [1], [3])])
    with pytest.raises(ValueError, match="Expected 2 val"):
        evaluator.evaluate([_packet("a", [1], [2])])
