import logging
from dataclasses import dataclass

import torch
from federated_gcn_aml.training.metrics import (
    THRESHOLD_SELECTION_RULE,
    combine_ranking_and_threshold_metrics,
    compute_metrics,
    select_threshold_by_f1,
)
from torch_geometric.data import Data
from federated_gcn_aml.models.gcn import GCN


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingRunResult:
    model: GCN
    metrics: dict
    best_val_aupr: float


def make_pos_weight(y: torch.Tensor, mask: torch.Tensor):
    """
    return a positive weight class for use in BCEWithLogitsLoss
    """
    y_m = y[mask]
    pos = (y_m == 1).sum().item()
    neg = (y_m == 0).sum().item()
    pos = max(pos, 1)
    return torch.tensor([neg / pos], dtype=torch.float, device=y.device)


def train_full_batch(model: GCN, data: Data, lr=1e-2, weight_decay=5e-4,
                     epochs=200, threshold=0.5, device="cpu"):
    logger.info(
        "Starting full-batch training: nodes=%d edges=%d x_dim=%d train_nodes=%d",
        data.num_nodes,
        data.num_edges,
        data.x.size(1),
        int(data.train_mask.sum().item()),
    )

    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_weight = make_pos_weight(data.y, data.train_mask)

    # binary cross-entropy with logits: expects logits not probs
    # pos_weight boosts loss contribution from positive samples
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # track model with best val AURP
    best_val_aupr = -1.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(data)

                val_metrics = compute_metrics(
                    y_true=data.y[data.val_mask],
                    logits=logits[data.val_mask],
                    threshold=threshold)
                test_metrics = compute_metrics(
                    y_true=data.y[data.test_mask],
                    logits=logits[data.test_mask],
                    threshold=threshold)

                if val_metrics["AUPR"] > best_val_aupr:
                    best_val_aupr = val_metrics["AUPR"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch

            logger.info(
                "Epoch %03d | loss=%.4f | val AUPR=%.4f F1=%.4f MCC=%.4f | test AUPR=%.4f F1=%.4f MCC=%.4f",
                epoch,
                loss.item(),
                val_metrics["AUPR"],
                val_metrics["F1"],
                val_metrics["MCC"],
                test_metrics["AUPR"],
                test_metrics["F1"],
                test_metrics["MCC"],
            )

    # restore best
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        logits = model(data)
        best_threshold, val_threshold_metrics = select_threshold_by_f1(
            y_true=data.y[data.val_mask],
            logits=logits[data.val_mask],
        )
        test_threshold_metrics = compute_metrics(
            y_true=data.y[data.test_mask],
            logits=logits[data.test_mask],
            threshold=best_threshold,
        )
        val_aupr_metrics = compute_metrics(
            y_true=data.y[data.val_mask],
            logits=logits[data.val_mask],
            threshold=threshold,
        )
        test_aupr_metrics = compute_metrics(
            y_true=data.y[data.test_mask],
            logits=logits[data.test_mask],
            threshold=threshold,
        )

    metrics = {
        "best_val_aupr": float(best_val_aupr),
        "best_epoch": int(best_epoch),
        "selected_threshold": float(best_threshold),
        "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
        "val": combine_ranking_and_threshold_metrics(val_aupr_metrics, val_threshold_metrics),
        "test": combine_ranking_and_threshold_metrics(test_aupr_metrics, test_threshold_metrics),
    }
    logger.info(
        "Best checkpoint restored from epoch %d with val AUPR %.4f; threshold %.2f gives val F1 %.4f and test F1 %.4f",
        best_epoch,
        best_val_aupr,
        best_threshold,
        metrics["val"]["F1"],
        metrics["test"]["F1"],
    )

    return TrainingRunResult(model=model, metrics=metrics, best_val_aupr=best_val_aupr)
