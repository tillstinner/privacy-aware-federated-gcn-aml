from __future__ import annotations

import torch

from federated_gcn_aml.federated.messages import ClientPredictionPacket, PredictionSplitPacket
from federated_gcn_aml.training.metrics import (
    THRESHOLD_SELECTION_RULE,
    combine_ranking_and_threshold_metrics,
    compute_metrics,
    select_threshold_by_f1,
)


class GlobalOwnedNodeEvaluator:
    def __init__(self, expected_val_count: int, expected_test_count: int) -> None:
        self.expected_counts = {"val": int(expected_val_count), "test": int(expected_test_count)}

    def _concat(self, packets: list[ClientPredictionPacket], split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        split_packets: list[PredictionSplitPacket] = [getattr(packet, split) for packet in packets]
        if split_packets:
            node_ids = torch.cat([packet.global_node_indices.detach().cpu().long() for packet in split_packets])
            y = torch.cat([packet.y.detach().cpu() for packet in split_packets])
            logits = torch.cat([packet.logits.detach().cpu() for packet in split_packets])
        else:
            node_ids = torch.empty(0, dtype=torch.long)
            y = torch.empty(0)
            logits = torch.empty(0)
        expected = self.expected_counts[split]
        if int(node_ids.numel()) != expected:
            raise ValueError(f"Expected {expected} {split} predictions, got {int(node_ids.numel())}.")
        order = torch.argsort(node_ids)
        node_ids = node_ids[order]
        if node_ids.unique().numel() != node_ids.numel():
            raise ValueError(f"Duplicate global node predictions detected for split {split}.")
        return node_ids, y[order], logits[order]

    def evaluate(self, packets: list[ClientPredictionPacket]) -> tuple[dict, dict]:
        val_nodes, y_val, logits_val = self._concat(packets, "val")
        test_nodes, y_test, logits_test = self._concat(packets, "test")

        selected_threshold, val_threshold_metrics = select_threshold_by_f1(y_val, logits_val)
        val_aupr_metrics = compute_metrics(y_val, logits_val, threshold=0.5)
        test_aupr_metrics = compute_metrics(y_test, logits_test, threshold=0.5)
        test_threshold_metrics = compute_metrics(y_test, logits_test, threshold=selected_threshold)

        metrics = {
            "selected_threshold": float(selected_threshold),
            "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
            "val": combine_ranking_and_threshold_metrics(val_aupr_metrics, val_threshold_metrics),
            "test": combine_ranking_and_threshold_metrics(test_aupr_metrics, test_threshold_metrics),
            "expected_val_count": self.expected_counts["val"],
            "actual_val_count": int(val_nodes.numel()),
            "expected_test_count": self.expected_counts["test"],
            "actual_test_count": int(test_nodes.numel()),
        }
        predictions = {
            "val": {"global_node_indices": val_nodes, "y": y_val, "logits": logits_val},
            "test": {"global_node_indices": test_nodes, "y": y_test, "logits": logits_test},
        }
        return metrics, predictions
