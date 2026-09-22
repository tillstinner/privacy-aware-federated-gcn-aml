"""Audit AMLSim non-IID datasets for simple shortcut signals.

This script is intentionally diagnostic. It does not train graph models. It
checks whether bank ownership, a single engineered feature, or split artifacts
can explain unusually strong centralized performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef

from federated_gcn_aml.data.build_graph import build_amlsim_graph, derive_amlsim_targets, load_raw_amlsim
from federated_gcn_aml.experiments.common_amlsim import create_amlsim_node_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit AMLSim non-IID shortcut signals.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representation-version", default="pass2", choices=("pass1", "pass2"))
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def _rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _mask_indices(mask: torch.Tensor) -> np.ndarray:
    return mask.detach().cpu().numpy().astype(bool)


def _threshold_metrics(y_val: np.ndarray, val_scores: np.ndarray, y_test: np.ndarray, test_scores: np.ndarray) -> dict[str, Any]:
    thresholds = np.linspace(0.0, 1.0, 101)
    best = {"threshold": 0.5, "val_f1": -1.0}
    for threshold in thresholds:
        pred = (val_scores >= threshold).astype(int)
        score = f1_score(y_val, pred, zero_division=0)
        if score > best["val_f1"]:
            best = {"threshold": float(threshold), "val_f1": float(score)}

    test_pred = (test_scores >= best["threshold"]).astype(int)
    return {
        "selected_threshold": best["threshold"],
        "val_f1": best["val_f1"],
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_mcc": float(matthews_corrcoef(y_test, test_pred)),
    }


def _aupr(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.0
    return float(average_precision_score(y, scores))


def _bank_column(accounts: pd.DataFrame) -> str:
    if "bank_id" in accounts.columns:
        return "bank_id"
    if "BANK_ID" in accounts.columns:
        return "BANK_ID"
    raise ValueError("accounts.csv has no bank_id/BANK_ID column")


def _account_column(accounts: pd.DataFrame) -> str:
    if "acct_id" in accounts.columns:
        return "acct_id"
    if "ACCOUNT_ID" in accounts.columns:
        return "ACCOUNT_ID"
    raise ValueError("accounts.csv has no acct_id/ACCOUNT_ID column")


def _bank_split_table(accounts: pd.DataFrame, account_ids: tuple[str, ...], y: np.ndarray, masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    bank_col = _bank_column(accounts)
    acct_col = _account_column(accounts)
    bank_by_id = accounts.assign(_acct=accounts[acct_col].astype(str).str.strip()).set_index("_acct")[bank_col].astype(str)
    banks = bank_by_id.reindex(list(account_ids)).fillna("__MISSING__").to_numpy()

    rows: list[dict[str, Any]] = []
    for bank in sorted(pd.Series(banks).unique()):
        bank_mask = banks == bank
        row: dict[str, Any] = {
            "bank_id": str(bank),
            "nodes": int(bank_mask.sum()),
            "positives": int(y[bank_mask].sum()),
            "label_rate": _rate(int(y[bank_mask].sum()), int(bank_mask.sum())),
        }
        for split_name, split_mask in masks.items():
            both = bank_mask & split_mask
            row[f"{split_name}_nodes"] = int(both.sum())
            row[f"{split_name}_positives"] = int(y[both].sum())
            row[f"{split_name}_label_rate"] = _rate(int(y[both].sum()), int(both.sum()))
        rows.append(row)
    return rows


def _bank_only_baseline(bank_rows: list[dict[str, Any]], accounts: pd.DataFrame, account_ids: tuple[str, ...], y: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    bank_col = _bank_column(accounts)
    acct_col = _account_column(accounts)
    bank_by_id = accounts.assign(_acct=accounts[acct_col].astype(str).str.strip()).set_index("_acct")[bank_col].astype(str)
    banks = bank_by_id.reindex(list(account_ids)).fillna("__MISSING__").to_numpy()
    train_rates = {row["bank_id"]: row["train_label_rate"] for row in bank_rows}
    scores = np.array([train_rates.get(str(bank), 0.0) for bank in banks], dtype=np.float64)

    val_mask = masks["val"]
    test_mask = masks["test"]
    result = {
        "val_aupr": _aupr(y[val_mask], scores[val_mask]),
        "test_aupr": _aupr(y[test_mask], scores[test_mask]),
    }
    result.update(_threshold_metrics(y[val_mask], scores[val_mask], y[test_mask], scores[test_mask]))
    return result


def _feature_audits(x: np.ndarray, y: np.ndarray, masks: dict[str, np.ndarray], feature_names: list[str], top_k: int) -> dict[str, Any]:
    val_mask = masks["val"]
    test_mask = masks["test"]
    rows: list[dict[str, Any]] = []
    exact_label_like: list[dict[str, Any]] = []

    for idx, name in enumerate(feature_names):
        values = x[:, idx].astype(np.float64)
        finite = np.isfinite(values)
        if not finite.all():
            values = np.nan_to_num(values, nan=0.0, posinf=np.nanmax(values[finite]), neginf=np.nanmin(values[finite]))

        pos_aupr = _aupr(y[test_mask], values[test_mask])
        neg_aupr = _aupr(y[test_mask], -values[test_mask])
        direction = "positive" if pos_aupr >= neg_aupr else "negative"
        best_scores = values if direction == "positive" else -values

        train_pos = values[masks["train"] & (y == 1)]
        train_neg = values[masks["train"] & (y == 0)]
        denom = float(np.std(values[masks["train"]]) or 1.0)
        mean_gap_std = float((np.mean(train_pos) - np.mean(train_neg)) / denom) if len(train_pos) and len(train_neg) else 0.0

        rows.append(
            {
                "feature": name,
                "direction": direction,
                "test_aupr": max(pos_aupr, neg_aupr),
                "val_aupr": _aupr(y[val_mask], best_scores[val_mask]),
                "mean_gap_train_std_units": mean_gap_std,
                "unique_values": int(pd.Series(values).nunique(dropna=False)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )

        unique_values = set(np.unique(values))
        if unique_values.issubset({0.0, 1.0}):
            agreement = float(np.mean(values == y))
            inverse_agreement = float(np.mean(1.0 - values == y))
            if max(agreement, inverse_agreement) > 0.95:
                exact_label_like.append(
                    {
                        "feature": name,
                        "agreement": agreement,
                        "inverse_agreement": inverse_agreement,
                    }
                )

    rows.sort(key=lambda row: row["test_aupr"], reverse=True)
    return {
        "top_univariate_features": rows[:top_k],
        "exact_binary_label_like_features": exact_label_like,
    }


def _bank_feature_shift(x: np.ndarray, accounts: pd.DataFrame, account_ids: tuple[str, ...], feature_names: list[str], top_k: int) -> list[dict[str, Any]]:
    bank_col = _bank_column(accounts)
    acct_col = _account_column(accounts)
    bank_by_id = accounts.assign(_acct=accounts[acct_col].astype(str).str.strip()).set_index("_acct")[bank_col].astype(str)
    banks = bank_by_id.reindex(list(account_ids)).fillna("__MISSING__").to_numpy()
    global_mean = x.mean(axis=0)
    global_std = x.std(axis=0)
    global_std[global_std == 0] = 1.0

    rows = []
    for idx, name in enumerate(feature_names):
        means = {}
        max_gap = 0.0
        for bank in sorted(pd.Series(banks).unique()):
            bank_values = x[banks == bank, idx]
            mean = float(bank_values.mean()) if bank_values.size else 0.0
            means[str(bank)] = mean
        bank_means = list(means.values())
        if bank_means:
            max_gap = float((max(bank_means) - min(bank_means)) / global_std[idx])
        rows.append({"feature": name, "max_bank_mean_gap_std_units": max_gap, "bank_means": means})
    rows.sort(key=lambda row: abs(row["max_bank_mean_gap_std_units"]), reverse=True)
    return rows[:top_k]


def _integrity_checks(accounts: pd.DataFrame, account_ids: tuple[str, ...], y: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    acct_col = _account_column(accounts)
    normalized = accounts[acct_col].astype(str).str.strip()
    account_set = set(normalized)
    graph_set = set(account_ids)
    split_total = int(sum(mask.sum() for mask in masks.values()))
    split_overlap = int((masks["train"] & masks["val"]).sum() + (masks["train"] & masks["test"]).sum() + (masks["val"] & masks["test"]).sum())
    return {
        "account_rows": int(len(accounts)),
        "unique_account_ids": int(normalized.nunique()),
        "duplicate_account_rows": int(len(accounts) - normalized.nunique()),
        "graph_nodes": int(len(account_ids)),
        "accounts_missing_from_graph": int(len(account_set - graph_set)),
        "graph_nodes_missing_from_accounts": int(len(graph_set - account_set)),
        "label_positive_count": int(y.sum()),
        "split_total_nodes": split_total,
        "split_overlap_pairs": split_overlap,
        "split_covers_all_nodes": split_total == len(account_ids),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_amlsim(data_root)
    targets, label_source = derive_amlsim_targets(raw.sar_accounts, raw.alert_accounts)
    graph = build_amlsim_graph(raw.accounts, raw.transactions, targets, representation_version=args.representation_version)
    data = graph.data
    masks_obj = create_amlsim_node_masks(data, 0.6, 0.2, 0.2, args.seed)
    masks = {
        "train": _mask_indices(masks_obj.train_mask),
        "val": _mask_indices(masks_obj.val_mask),
        "test": _mask_indices(masks_obj.test_mask),
    }

    x = data.x.detach().cpu().numpy()
    y = data.y.detach().cpu().numpy().astype(int)
    feature_names = [str(name) for name in graph.feature_columns]
    account_ids = tuple(str(account_id).strip() for account_id in graph.account_ids)

    bank_rows = _bank_split_table(raw.accounts, account_ids, y, masks)
    report = {
        "data_root": str(data_root),
        "representation_version": args.representation_version,
        "label_source": label_source,
        "seed": args.seed,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(data.num_node_features),
        "positive_labels": int(y.sum()),
        "positive_rate": _rate(int(y.sum()), int(data.num_nodes)),
        "integrity": _integrity_checks(raw.accounts, account_ids, y, masks),
        "bank_split_table": bank_rows,
        "bank_only_baseline": _bank_only_baseline(bank_rows, raw.accounts, account_ids, y, masks),
        "feature_shortcuts": _feature_audits(x, y, masks, feature_names, args.top_k),
        "bank_feature_shift": _bank_feature_shift(x, raw.accounts, account_ids, feature_names, args.top_k),
        "warnings": [],
    }

    if report["integrity"]["duplicate_account_rows"]:
        report["warnings"].append("Duplicate account rows detected.")
    if report["integrity"]["accounts_missing_from_graph"] or report["integrity"]["graph_nodes_missing_from_accounts"]:
        report["warnings"].append("Graph/account table mismatch detected.")
    if report["integrity"]["split_overlap_pairs"]:
        report["warnings"].append("Train/val/test split masks overlap.")
    if report["bank_only_baseline"]["test_aupr"] > 0.25:
        report["warnings"].append("Bank-only baseline has high AUPR; bank membership is a strong shortcut.")
    top_feature = report["feature_shortcuts"]["top_univariate_features"][0]
    if top_feature["test_aupr"] > 0.60:
        report["warnings"].append(
            f"Single feature {top_feature['feature']} has high test AUPR {top_feature['test_aupr']:.4f}."
        )
    if report["feature_shortcuts"]["exact_binary_label_like_features"]:
        report["warnings"].append("Binary feature nearly matches labels.")

    out_json = output_dir / "shortcut_audit.json"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = [
        f"# Shortcut Audit: {data_root.name}",
        "",
        f"- nodes: {report['num_nodes']:,}",
        f"- edges: {report['num_edges']:,}",
        f"- positives: {report['positive_labels']:,} ({report['positive_rate']:.4%})",
        f"- bank-only test AUPR: {report['bank_only_baseline']['test_aupr']:.4f}",
        f"- warnings: {len(report['warnings'])}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in report["warnings"]] or ["- none"])
    lines.extend(["", "## Top Univariate Feature AUPR", ""])
    for row in report["feature_shortcuts"]["top_univariate_features"]:
        lines.append(
            f"- `{row['feature']}` ({row['direction']}): test AUPR {row['test_aupr']:.4f}, "
            f"val AUPR {row['val_aupr']:.4f}, train mean gap {row['mean_gap_train_std_units']:.3f} std"
        )
    lines.extend(["", "## Bank Split Table", ""])
    for row in bank_rows:
        lines.append(
            f"- `{row['bank_id']}`: nodes {row['nodes']:,}, positives {row['positives']:,}, "
            f"rate {row['label_rate']:.4%}, train/val/test pos "
            f"{row['train_positives']:,}/{row['val_positives']:,}/{row['test_positives']:,}"
        )
    (output_dir / "shortcut_audit.md").write_text("\n".join(lines) + "\n")
    print(out_json)


if __name__ == "__main__":
    main()
