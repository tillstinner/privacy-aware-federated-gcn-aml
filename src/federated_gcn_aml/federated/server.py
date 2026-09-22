from __future__ import annotations

import logging
import time

import torch

from federated_gcn_aml.federated.client import FederatedClient
from federated_gcn_aml.federated.evaluator import GlobalOwnedNodeEvaluator
from federated_gcn_aml.federated.protocols import BasePretrainProtocol
from federated_gcn_aml.federated.results import FederatedRunResult
from federated_gcn_aml.federated.strategies import FedAvgStrategy


logger = logging.getLogger(__name__)


class FederatedServer:
    def __init__(
        self,
        initial_state: dict[str, torch.Tensor],
        clients: list[FederatedClient],
        evaluator: GlobalOwnedNodeEvaluator,
        strategy: FedAvgStrategy,
        protocol: BasePretrainProtocol,
    ) -> None:
        self.global_state = {key: value.detach().cpu().clone() for key, value in initial_state.items()}
        self.clients = list(clients)
        self.evaluator = evaluator
        self.strategy = strategy
        self.protocol = protocol
        self.pretrain_diagnostics: dict = {}

    def run_pretraining_protocol(self) -> None:
        t0 = time.perf_counter()
        contributions = self.protocol.collect_contributions(self.clients)
        self.protocol.aggregate_on_server(contributions)
        for client in self.clients:
            payload = self.protocol.build_payload_for_client(client.client_id)
            self.protocol.install_payload(client, payload)
        self.pretrain_diagnostics = self.protocol.diagnostics()
        self.pretrain_diagnostics["total_pretraining_protocol_seconds"] = time.perf_counter() - t0

    def run(self, rounds: int, local_epochs: int) -> FederatedRunResult:
        best_state = None
        best_metrics = None
        best_predictions = None
        best_val_aupr = -1.0
        round_metrics = []

        for round_idx in range(1, rounds + 1):
            aggregation_plan = self.strategy.build_round_plan(
                [(client.client_id, client.train_node_count()) for client in self.clients],
                self.global_state,
                round_idx,
            )
            updates = [
                client.train_local_model_update(
                    self.global_state,
                    local_epochs=local_epochs,
                    aggregation_plan=aggregation_plan,
                )
                for client in self.clients
            ]
            self.global_state = self.strategy.aggregate_model_updates(
                updates,
                current_state=self.global_state,
                round_plan=aggregation_plan,
            )
            packets = [client.predict_owned(self.global_state) for client in self.clients]
            metrics, predictions = self.evaluator.evaluate(packets)
            mean_loss_values = [update.loss for update in updates if update.loss is not None]
            row = {
                "round": round_idx,
                "mean_train_loss": None if not mean_loss_values else float(sum(mean_loss_values) / len(mean_loss_values)),
                "val_AUPR": metrics["val"]["AUPR"],
                "val_AUROC": metrics["val"]["AUROC"],
                "val_F1": metrics["val"]["F1"],
                "val_MCC": metrics["val"]["MCC"],
                "val_precision": metrics["val"]["precision"],
                "val_recall": metrics["val"]["recall"],
                "val_TP": metrics["val"]["TP"],
                "val_FP": metrics["val"]["FP"],
                "val_TN": metrics["val"]["TN"],
                "val_FN": metrics["val"]["FN"],
                "test_AUPR": metrics["test"]["AUPR"],
                "test_AUROC": metrics["test"]["AUROC"],
                "test_F1": metrics["test"]["F1"],
                "test_MCC": metrics["test"]["MCC"],
                "test_precision": metrics["test"]["precision"],
                "test_recall": metrics["test"]["recall"],
                "test_TP": metrics["test"]["TP"],
                "test_FP": metrics["test"]["FP"],
                "test_TN": metrics["test"]["TN"],
                "test_FN": metrics["test"]["FN"],
                "selected_threshold": metrics["selected_threshold"],
            }
            round_metrics.append(row)
            logger.info(
                "Round %03d | mean_loss=%s | val AUPR=%.4f F1=%.4f MCC=%.4f | test AUPR=%.4f F1=%.4f MCC=%.4f",
                round_idx,
                "nan" if row["mean_train_loss"] is None else f"{row['mean_train_loss']:.4f}",
                row["val_AUPR"],
                row["val_F1"],
                row["val_MCC"],
                row["test_AUPR"],
                row["test_F1"],
                row["test_MCC"],
            )
            if metrics["val"]["AUPR"] > best_val_aupr:
                best_val_aupr = metrics["val"]["AUPR"]
                best_state = {key: value.detach().cpu().clone() for key, value in self.global_state.items()}
                best_metrics = metrics
                best_predictions = predictions
                best_metrics["best_round"] = round_idx
                best_metrics["best_val_aupr"] = best_val_aupr

        if best_state is None or best_metrics is None or best_predictions is None:
            raise RuntimeError("Federated class training finished without a best checkpoint.")

        return FederatedRunResult(
            model_state=best_state,
            metrics=best_metrics,
            round_metrics=round_metrics,
            predictions=best_predictions,
            diagnostics={
                "pretrain_protocol": self.pretrain_diagnostics,
                "clients": [client.diagnostics() for client in self.clients],
                "aggregation_weighting": self.strategy.weighting,
                "model_update_aggregation": self.strategy.diagnostics(),
            },
        )
