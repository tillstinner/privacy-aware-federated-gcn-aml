import json
import logging
import torch
import pandas as pd
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from federated_gcn_aml.data.build_graph import(
    build_amlsim_graph,
    derive_amlsim_targets,
    load_raw_amlsim,
    AMLSimGraphArtifacts,
    AMLSimRawTables
)
from federated_gcn_aml.experiments.common_amlsim import (
    AMLSimNodeMasks,
    create_amlsim_node_masks,
    parse_args,
    resolve_amlsim_context,
    resolve_device,
)

logger = logging.getLogger(__name__)


def _graph_data(graph: Any) -> Any:
    """Return the PyG-style data object from either graph artifacts or data itself."""
    return getattr(graph, "data", graph)


def _graph_account_ids(graph: Any) -> tuple[str, ...] | None:
    """Return graph account IDs when the artifact exposes them."""
    account_ids = getattr(graph, "account_ids", None)
    if account_ids is None:
        return None
    return tuple(str(account_id).strip() for account_id in account_ids)


def _graph_id2idx(graph: Any) -> dict[str, int] | None:
    """Return an account-ID to node-index mapping when recoverable."""
    id2idx = getattr(graph, "id2idx", None)
    if id2idx is not None:
        return {str(account_id).strip(): int(idx) for account_id, idx in id2idx.items()}

    account_ids = _graph_account_ids(graph)
    if account_ids is None:
        return None
    return {account_id: idx for idx, account_id in enumerate(account_ids)}


def _feature_columns(graph: Any, num_features: int) -> list[str]:
    """Return feature names from graph artifacts, or stable fallback column names."""
    columns = getattr(graph, "feature_columns", None)
    if columns is None:
        return [f"feature_{idx}" for idx in range(num_features)]
    return [str(column) for column in columns]


def _to_tensor(value: Any) -> torch.Tensor | None:
    """Convert tensor-like inputs to a detached tensor when possible."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach()
    try:
        return torch.as_tensor(value).detach()
    except Exception:
        return None


def _numeric_summary(values: torch.Tensor) -> dict[str, float | int | None]:
    """Compute simple summary statistics for a one-dimensional tensor."""
    values = values.detach().flatten()
    if values.numel() == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "p95": None,
            "p99": None,
        }

    values = values.float()
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "median": float(torch.quantile(values, 0.50).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    """Return numerator / denominator, or None when the denominator is zero."""
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _first_existing(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column present in columns."""
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _bool_series(series: pd.Series) -> pd.Series:
    """Parse common boolean spellings from a pandas series."""
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _series_summary(series: pd.Series, percentiles: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)) -> dict[str, Any]:
    """Return JSON-safe descriptive stats for a numeric series."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "percentiles": {},
        }

    return {
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "std": float(numeric.std(ddof=0)),
        "percentiles": {f"p{int(p * 100)}": float(numeric.quantile(p)) for p in percentiles},
    }


def _integer_distribution(series: pd.Series, max_exact_value: int = 20) -> dict[str, int]:
    """Bucket integer counts with exact low values and compact high-tail bins."""
    values = pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")
    exact_counts = values.loc[values <= max_exact_value].value_counts().sort_index()
    tail_values = values.loc[values > max_exact_value]

    distribution = {str(int(k)): int(v) for k, v in exact_counts.items()}
    if not tail_values.empty:
        bins = [max_exact_value, 50, 100, 250, 500, 1000, 5000, 10000, float("inf")]
        labels = [
            f"{max_exact_value + 1}-50",
            "51-100",
            "101-250",
            "251-500",
            "501-1000",
            "1001-5000",
            "5001-10000",
            "10001+",
        ]
        tail_bins = pd.cut(tail_values, bins=bins, labels=labels, right=True)
        distribution.update({str(k): int(v) for k, v in tail_bins.value_counts().sort_index().items() if v})

    return distribution


def _component_summary(sizes: list[int], total_nodes: int) -> dict[str, Any]:
    """Summarize connected component sizes."""
    if not sizes:
        return {
            "count": 0,
            "largest": 0,
            "largest_ratio": 0.0,
            "top_10": [],
            "size_summary": _series_summary(pd.Series(dtype="int64")),
            "size_distribution": {},
        }

    sizes_series = pd.Series(sizes, dtype="int64")
    largest = int(sizes_series.max())
    return {
        "count": int(len(sizes)),
        "largest": largest,
        "largest_ratio": _rate(largest, total_nodes) or 0.0,
        "top_10": [int(v) for v in sizes_series.sort_values(ascending=False).head(10).tolist()],
        "size_summary": _series_summary(sizes_series),
        "size_distribution": _integer_distribution(sizes_series),
    }


def _log_warnings(audit_name: str, warnings: list[str]) -> None:
    """Log audit warnings using a consistent compact format."""
    if not warnings:
        return
    logging.warning("%s warnings:", audit_name)
    for warning in warnings:
        logging.warning("  - %s", warning)


def target_audit(data_root: Path, label: str) -> dict[str, Any]:
    logging.info("------ initializing target audit of raw data ------")

    data_root = data_root.expanduser().resolve()
    logging.info("data_root: %s", data_root)

    sar_path = data_root / "sar_accounts.csv"
    alert_path = data_root / "alert_accounts.csv"

    sar_exists = sar_path.exists()
    alert_exists = alert_path.exists()

    logging.info("sar_accounts exist: %s", sar_exists)
    logging.info("alert_accounts exist: %s", alert_exists)

    warnings: list[str] = []

    df_sar_accs = pd.read_csv(sar_path, usecols=["ACCOUNT_ID"]) if sar_exists else None
    df_alert_accs = pd.read_csv(alert_path, usecols=["acct_id"]) if alert_exists else None

    def extract_ids(df: pd.DataFrame | None, id_col: str) -> tuple[set[str], int]:
        if df is None:
            return set(), 0

        series = df[id_col].dropna().astype(str).str.strip()
        series = series[series != ""]

        raw_count = len(series)
        unique_ids = set(series.tolist())
        duplicate_count = raw_count - len(unique_ids)

        return unique_ids, duplicate_count

    sar_ids, sar_duplicate_count = extract_ids(df_sar_accs, "ACCOUNT_ID")
    alert_ids, alert_duplicate_count = extract_ids(df_alert_accs, "acct_id")

    if label not in {"sar", "alert"}:
        warnings.append(f"Unexpected label mode: {label!r}. Expected 'sar' or 'alert'.")

    if label == "alert":
        current_target_source = "alert_accounts.csv"
        current_target_ids = alert_ids
        logging.info("Using alert_accounts for account classification")
    else:
        current_target_source = "sar_accounts.csv"
        current_target_ids = sar_ids
        logging.info("Using sar_accounts for account classification")

    overlap_ids = sar_ids & alert_ids
    sar_only_ids = sar_ids - alert_ids
    alert_only_ids = alert_ids - sar_ids

    if current_target_source == "sar_accounts.csv" and not sar_exists:
        warnings.append("Configured target source is sar_accounts.csv, but the file does not exist.")
    if current_target_source == "alert_accounts.csv" and not alert_exists:
        warnings.append("Configured target source is alert_accounts.csv, but the file does not exist.")

    if len(current_target_ids) == 0:
        warnings.append(f"Current target source {current_target_source} contains zero usable account IDs.")

    if sar_duplicate_count > 0:
        warnings.append(f"sar_accounts.csv contains {sar_duplicate_count} duplicate ACCOUNT_ID rows.")
    if alert_duplicate_count > 0:
        warnings.append(f"alert_accounts.csv contains {alert_duplicate_count} duplicate acct_id rows.")

    report = {
        "audit_type": "target_audit",
        "data_root": str(data_root),
        "configured_label_mode": label,
        "current_target_source": current_target_source,
        "files": {
            "sar_accounts.csv": sar_exists,
            "alert_accounts.csv": alert_exists,
        },
        "counts": {
            "sar_unique_accounts": len(sar_ids),
            "alert_unique_accounts": len(alert_ids),
            "overlap_accounts": len(overlap_ids),
            "sar_only_accounts": len(sar_only_ids),
            "alert_only_accounts": len(alert_only_ids),
            "current_target_unique_accounts": len(current_target_ids),
        },
        "duplicates": {
            "sar_duplicate_rows": sar_duplicate_count,
            "alert_duplicate_rows": alert_duplicate_count,
        },
        "warnings": warnings,
    }

    logging.info("Target audit summary:")
    logging.info("  current_target_source: %s", current_target_source)
    logging.info("  current_target_unique_accounts: %d", len(current_target_ids))
    logging.info("  sar_unique_accounts: %d", len(sar_ids))
    logging.info("  alert_unique_accounts: %d", len(alert_ids))
    logging.info("  overlap_accounts: %d", len(overlap_ids))
    logging.info("  sar_only_accounts: %d", len(sar_only_ids))
    logging.info("  alert_only_accounts: %d", len(alert_only_ids))

    if warnings:
        logging.warning("Target audit warnings:")
        for warning in warnings:
            logging.warning("  - %s", warning)

    return report


def label_alignment_audit(raw: AMLSimRawTables, sar_set: set[str], graph: AMLSimGraphArtifacts | Any) -> dict[str, Any]:
    """Audit whether raw positive account IDs align with graph nodes and labels."""
    warnings: list[str] = []
    data = _graph_data(graph)
    y = _to_tensor(getattr(data, "y", None))
    id2idx = _graph_id2idx(graph)
    account_ids = _graph_account_ids(graph)

    normalized_sar_set = {str(account_id).strip() for account_id in sar_set if str(account_id).strip()}
    raw_account_count = None
    if hasattr(raw, "accounts") and "acct_id" in raw.accounts.columns:
        raw_account_count = int(raw.accounts["acct_id"].astype(str).str.strip().nunique())

    report: dict[str, Any] = {
        "audit_type": "label_alignment_audit",
        "counts": {
            "raw_positive_ids": len(normalized_sar_set),
            "raw_unique_accounts": raw_account_count,
            "graph_nodes": int(getattr(data, "num_nodes", 0) or 0),
            "graph_positive_labels": int((y == 1).sum().item()) if y is not None else None,
        },
        "alignment": {
            "raw_positive_ids_in_graph": None,
            "raw_positive_ids_missing_from_graph": None,
            "expected_positive_nodes_labeled_positive": None,
            "expected_positive_nodes_labeled_non_positive": None,
            "graph_positive_labels_not_in_raw_positive_ids": None,
        },
        "warnings": warnings,
    }

    if id2idx is None:
        warnings.append("Graph does not expose account_ids or id2idx; label alignment cannot be checked.")
    if y is None:
        warnings.append("Graph data does not expose y labels; label alignment cannot be checked.")

    if id2idx is not None:
        present_ids = normalized_sar_set & set(id2idx)
        missing_ids = normalized_sar_set - set(id2idx)
        report["alignment"]["raw_positive_ids_in_graph"] = len(present_ids)
        report["alignment"]["raw_positive_ids_missing_from_graph"] = len(missing_ids)

        if missing_ids:
            warnings.append(f"{len(missing_ids)} raw positive account IDs are missing after graph construction.")

        if y is not None:
            present_indices = torch.tensor(
                [id2idx[account_id] for account_id in present_ids],
                dtype=torch.long,
                device=y.device,
            )
            present_labels = (
                y[present_indices]
                if present_indices.numel() > 0
                else torch.empty(0, dtype=y.dtype, device=y.device)
            )
            labeled_positive = int((present_labels == 1).sum().item())
            labeled_non_positive = int(present_labels.numel() - labeled_positive)

            report["alignment"]["expected_positive_nodes_labeled_positive"] = labeled_positive
            report["alignment"]["expected_positive_nodes_labeled_non_positive"] = labeled_non_positive

            if labeled_non_positive:
                warnings.append(
                    f"{labeled_non_positive} graph nodes expected to be positive are not labeled positive."
                )

            if account_ids is not None:
                graph_positive_ids = {
                    account_ids[idx]
                    for idx in torch.where(y == 1)[0].tolist()
                    if idx < len(account_ids)
                }
                unexplained_positive_ids = graph_positive_ids - normalized_sar_set
                report["alignment"]["graph_positive_labels_not_in_raw_positive_ids"] = len(unexplained_positive_ids)
                if unexplained_positive_ids:
                    warnings.append(
                        f"{len(unexplained_positive_ids)} graph positive labels are not explained by sar_set."
                    )
            else:
                warnings.append("Graph account_ids are unavailable; unexplained graph positives cannot be counted.")

    logging.info(
        "Label alignment audit: raw_pos=%d, in_graph=%s, missing=%s, mismatches=%s, graph_pos=%s",
        report["counts"]["raw_positive_ids"],
        report["alignment"]["raw_positive_ids_in_graph"],
        report["alignment"]["raw_positive_ids_missing_from_graph"],
        report["alignment"]["expected_positive_nodes_labeled_non_positive"],
        report["counts"]["graph_positive_labels"],
    )
    _log_warnings("Label alignment audit", warnings)
    return report


def _extract_sar_ids(df: pd.DataFrame | None, account_candidates: tuple[str, ...], is_sar_candidates: tuple[str, ...]) -> set[str]:
    """Extract SAR/alert account IDs from a dataframe with flexible AMLSim column names."""
    if df is None:
        return set()
    account_col = _first_existing(df.columns, account_candidates)
    if account_col is None:
        return set()

    source = df
    is_sar_col = _first_existing(df.columns, is_sar_candidates)
    if is_sar_col is not None:
        source = source.loc[_bool_series(source[is_sar_col])]

    return set(source[account_col].dropna().astype(str).str.strip())


def sar_grouping_audit(accounts: pd.DataFrame, grouping_df: pd.DataFrame | None) -> dict[str, Any]:
    """Summarize whether alert/SAR group memberships are internal or cross-bank."""
    if grouping_df is None:
        return {
            "total": 0,
            "fully_internal": 0,
            "cross_bank": 0,
            "unknown_bank": 0,
            "fully_internal_ratio": 0.0,
            "cross_bank_ratio": 0.0,
            "fully_internal_by_bank": {},
            "by_alert_type": {},
        }

    alert_id_col = _first_existing(grouping_df.columns, ("ALERT_ID", "alert_id"))
    account_col = _first_existing(grouping_df.columns, ("ACCOUNT_ID", "acct_id"))
    alert_type_col = _first_existing(grouping_df.columns, ("ALERT_TYPE", "alert_type"))
    is_sar_col = _first_existing(grouping_df.columns, ("IS_SAR", "is_sar"))
    bank_col = _first_existing(grouping_df.columns, ("bank_id", "BANK_ID", "bankID"))

    if alert_id_col is None or account_col is None:
        return {
            "total": 0,
            "fully_internal": 0,
            "cross_bank": 0,
            "unknown_bank": 0,
            "fully_internal_ratio": 0.0,
            "cross_bank_ratio": 0.0,
            "fully_internal_by_bank": {},
            "by_alert_type": {},
        }

    sar_members = grouping_df.copy()
    if is_sar_col is not None:
        sar_members = sar_members.loc[_bool_series(sar_members[is_sar_col])]

    if sar_members.empty:
        return {
            "total": 0,
            "fully_internal": 0,
            "cross_bank": 0,
            "unknown_bank": 0,
            "fully_internal_ratio": 0.0,
            "cross_bank_ratio": 0.0,
            "fully_internal_by_bank": {},
            "by_alert_type": {},
        }

    acct_to_bank = accounts.set_index("acct_id")["bank_id"].to_dict()
    sar_members = sar_members.assign(
        alert_id=sar_members[alert_id_col].astype(str),
        alert_type=sar_members[alert_type_col].astype(str) if alert_type_col is not None else "unknown",
        member_account=sar_members[account_col].astype(str),
        member_bank=(
            sar_members[bank_col].astype(str)
            if bank_col is not None
            else sar_members[account_col].astype(str).map(acct_to_bank)
        ),
    )

    group_stats = (
        sar_members.groupby("alert_id")
        .agg(
            alert_type=("alert_type", "first"),
            n_accounts=("member_account", "nunique"),
            n_banks=("member_bank", "nunique"),
            banks=("member_bank", lambda banks: sorted(str(bank) for bank in banks.dropna().unique())),
        )
        .reset_index()
    )
    group_stats["scope"] = group_stats["n_banks"].map(
        lambda n_banks: "unknown_bank" if n_banks == 0 else "fully_internal" if n_banks == 1 else "cross_bank"
    )

    total = int(len(group_stats))
    scope_counts = group_stats["scope"].value_counts()
    fully_internal = int(scope_counts.get("fully_internal", 0))
    cross_bank = int(scope_counts.get("cross_bank", 0))
    unknown_bank = int(scope_counts.get("unknown_bank", 0))

    internal_by_bank = (
        group_stats.loc[group_stats["scope"] == "fully_internal"]
        .assign(bank=lambda df: df["banks"].map(lambda banks: banks[0] if banks else "unknown"))
        ["bank"]
        .value_counts()
        .sort_index()
    )

    by_alert_type: dict[str, Any] = {}
    for alert_type, sub_df in group_stats.groupby("alert_type"):
        sub_counts = sub_df["scope"].value_counts()
        sub_total = int(len(sub_df))
        sub_internal = int(sub_counts.get("fully_internal", 0))
        sub_cross = int(sub_counts.get("cross_bank", 0))
        by_alert_type[str(alert_type)] = {
            "total": sub_total,
            "fully_internal": sub_internal,
            "cross_bank": sub_cross,
            "unknown_bank": int(sub_counts.get("unknown_bank", 0)),
            "fully_internal_ratio": _rate(sub_internal, sub_total) or 0.0,
            "cross_bank_ratio": _rate(sub_cross, sub_total) or 0.0,
        }

    return {
        "total": total,
        "fully_internal": fully_internal,
        "cross_bank": cross_bank,
        "unknown_bank": unknown_bank,
        "fully_internal_ratio": _rate(fully_internal, total) or 0.0,
        "cross_bank_ratio": _rate(cross_bank, total) or 0.0,
        "fully_internal_by_bank": {str(k): int(v) for k, v in internal_by_bank.items()},
        "by_alert_type": by_alert_type,
    }


def raw_structure_audit(raw: AMLSimRawTables, label_ids: set[str], label_source: str) -> dict[str, Any]:
    """Audit raw AMLSim CSV structure with viz-pipeline-style non-plot metrics."""
    warnings: list[str] = []
    accounts = raw.accounts.copy()
    tx = raw.transactions.copy()

    accounts["acct_id"] = accounts["acct_id"].astype(str).str.strip()
    accounts["bank_id"] = accounts["bank_id"].astype(str).str.strip()
    tx["orig_acct"] = tx["orig_acct"].astype(str).str.strip()
    tx["bene_acct"] = tx["bene_acct"].astype(str).str.strip()

    sar_file_ids = _extract_sar_ids(raw.sar_accounts, ("ACCOUNT_ID", "acct_id"), ("IS_SAR", "is_sar"))
    alert_ids = _extract_sar_ids(raw.alert_accounts, ("acct_id", "ACCOUNT_ID"), ("is_sar", "IS_SAR"))
    normalized_label_ids = {str(account_id).strip() for account_id in label_ids if str(account_id).strip()}

    total_accounts = int(len(accounts))
    total_tx = int(len(tx))
    acct_to_bank = accounts.set_index("acct_id")["bank_id"].to_dict()
    tx["orig_bank"] = tx["orig_acct"].map(acct_to_bank)
    tx["bene_bank"] = tx["bene_acct"].map(acct_to_bank)
    unknown_bank_rows = int((tx["orig_bank"].isna() | tx["bene_bank"].isna()).sum())
    tx["cross_bank"] = tx["orig_bank"] != tx["bene_bank"]

    if "is_sar" in tx.columns:
        sar_tx = int(_bool_series(tx["is_sar"]).sum())
    elif "alert_id" in tx.columns:
        sar_tx = int((tx["alert_id"].astype(str) != "-1").sum())
    else:
        sar_tx = 0

    bank_account_counts = accounts["bank_id"].value_counts().sort_index()
    sar_per_bank = (
        accounts.assign(is_sar=accounts["acct_id"].isin(normalized_label_ids))
        .groupby("bank_id")["is_sar"]
        .sum()
        .sort_index()
    )

    internal_count = int((~tx["cross_bank"]).sum())
    cross_count = int(tx["cross_bank"].sum())
    interbank_matrix = (
        tx.groupby(["orig_bank", "bene_bank"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    interbank_rows = {
        str(orig_bank): {str(bene_bank): int(count) for bene_bank, count in row.items()}
        for orig_bank, row in interbank_matrix.to_dict(orient="index").items()
    }

    out_tx_counts = tx.groupby("orig_acct").size()
    in_tx_counts = tx.groupby("bene_acct").size()
    directed_pair_counts = tx.groupby(["orig_acct", "bene_acct"]).size()
    unique_directed_pairs = int(len(directed_pair_counts))
    repeat_rows = int(total_tx - unique_directed_pairs)

    node_stats = pd.DataFrame({"acct_id": accounts["acct_id"], "bank_id": accounts["bank_id"]})
    node_stats["out_tx_count"] = node_stats["acct_id"].map(out_tx_counts).fillna(0).astype("int64")
    node_stats["in_tx_count"] = node_stats["acct_id"].map(in_tx_counts).fillna(0).astype("int64")
    node_stats["total_tx_count"] = node_stats["in_tx_count"] + node_stats["out_tx_count"]
    node_stats["out_degree"] = node_stats["acct_id"].map(directed_pair_counts.groupby(level=0).size()).fillna(0).astype("int64")
    node_stats["in_degree"] = node_stats["acct_id"].map(directed_pair_counts.groupby(level=1).size()).fillna(0).astype("int64")
    node_stats["degree"] = node_stats["in_degree"] + node_stats["out_degree"]

    pair_index = directed_pair_counts.index
    reverse_pairs = pd.MultiIndex.from_arrays([pair_index.get_level_values(1), pair_index.get_level_values(0)])
    reverse_edge_rate = _rate(int(pair_index.isin(reverse_pairs).sum()), unique_directed_pairs) or 0.0

    parent = {account_id: account_id for account_id in accounts["acct_id"]}
    component_size = {account_id: 1 for account_id in accounts["acct_id"]}

    def find(account_id: str) -> str:
        parent.setdefault(account_id, account_id)
        component_size.setdefault(account_id, 1)
        while parent[account_id] != account_id:
            parent[account_id] = parent[parent[account_id]]
            account_id = parent[account_id]
        return account_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if component_size[left_root] < component_size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        component_size[left_root] += component_size[right_root]

    for orig, bene in pair_index:
        if orig in parent and bene in parent:
            union(str(orig), str(bene))

    weak_component_counts = Counter(find(account_id) for account_id in accounts["acct_id"])
    weak_component_sizes = sorted((int(size) for size in weak_component_counts.values()), reverse=True)

    per_bank_activity = (
        node_stats.groupby("bank_id")
        .agg(
            n_accounts=("acct_id", "count"),
            avg_in_tx_per_account=("in_tx_count", "mean"),
            avg_out_tx_per_account=("out_tx_count", "mean"),
            avg_total_tx_per_account=("total_tx_count", "mean"),
            median_total_tx_per_account=("total_tx_count", "median"),
            max_total_tx_per_account=("total_tx_count", "max"),
            avg_degree=("degree", "mean"),
            median_degree=("degree", "median"),
            max_degree=("degree", "max"),
        )
        .round(6)
    )

    if unknown_bank_rows:
        warnings.append(f"{unknown_bank_rows} transaction rows contain an account without a known bank_id.")
    if repeat_rows:
        warnings.append(f"{repeat_rows} raw transaction rows repeat an already observed directed account pair.")

    report = {
        "audit_type": "raw_structure_audit",
        "counts": {
            "n_accounts": total_accounts,
            "n_transactions": total_tx,
            "n_label_accounts": len(normalized_label_ids),
            "label_source": label_source,
            "n_sar_accounts_from_sar_file": len(sar_file_ids),
            "n_alert_accounts": len(alert_ids),
            "n_sar_accounts_only_in_label_source": len(normalized_label_ids - sar_file_ids),
            "n_sar_accounts_only_in_sar_file": len(sar_file_ids - normalized_label_ids),
            "n_sar_transactions": sar_tx,
            "graph_edges_aggregated": unique_directed_pairs,
            "repeated_transaction_rows": repeat_rows,
        },
        "rates": {
            "sar_account_ratio": _rate(len(normalized_label_ids), total_accounts),
            "sar_file_account_coverage_of_label_source": _rate(len(normalized_label_ids & sar_file_ids), len(normalized_label_ids)),
            "sar_transaction_account_coverage": _rate(len(normalized_label_ids & sar_file_ids), len(normalized_label_ids)),
            "sar_transaction_ratio": _rate(sar_tx, total_tx),
            "repeat_transaction_row_rate": _rate(repeat_rows, total_tx),
        },
        "bank_account_counts": {str(k): int(v) for k, v in bank_account_counts.items()},
        "sar_accounts_per_bank": {str(k): int(v) for k, v in sar_per_bank.items()},
        "bank_flows": {
            "internal_transactions": internal_count,
            "cross_bank_transactions": cross_count,
            "internal_ratio": _rate(internal_count, total_tx),
            "cross_bank_ratio": _rate(cross_count, total_tx),
            "interbank_matrix": interbank_rows,
        },
        "sar_groupings": sar_grouping_audit(accounts, raw.alert_accounts if raw.alert_accounts is not None else raw.sar_accounts),
        "transactions_per_account": {
            "incoming": _series_summary(node_stats["in_tx_count"]),
            "incoming_distribution": _integer_distribution(node_stats["in_tx_count"]),
            "outgoing": _series_summary(node_stats["out_tx_count"]),
            "outgoing_distribution": _integer_distribution(node_stats["out_tx_count"]),
            "total": _series_summary(node_stats["total_tx_count"]),
            "total_distribution": _integer_distribution(node_stats["total_tx_count"]),
            "zero_transaction_accounts": int((node_stats["total_tx_count"] == 0).sum()),
            "zero_transaction_account_ratio": _rate(int((node_stats["total_tx_count"] == 0).sum()), total_accounts),
        },
        "per_bank_activity": {
            str(bank): {
                key: int(value) if key in {"n_accounts", "max_total_tx_per_account", "max_degree"} else float(value)
                for key, value in row.items()
            }
            for bank, row in per_bank_activity.to_dict(orient="index").items()
        },
        "aggregated_graph_topology": {
            "density": _rate(unique_directed_pairs, total_accounts * (total_accounts - 1)),
            "reciprocity": reverse_edge_rate,
            "weak_components": _component_summary(weak_component_sizes, total_accounts),
            "degree": {
                "in": _series_summary(node_stats["in_degree"]),
                "out": _series_summary(node_stats["out_degree"]),
                "total": _series_summary(node_stats["degree"]),
                "total_distribution": _integer_distribution(node_stats["degree"]),
            },
            "average_in_degree": _rate(unique_directed_pairs, total_accounts),
            "average_out_degree": _rate(unique_directed_pairs, total_accounts),
            "average_total_degree": _rate(2 * unique_directed_pairs, total_accounts),
        },
        "warnings": warnings,
    }

    logging.info(
        "Raw structure audit: rows=%d, unique_pairs=%d, cross_bank=%d (%.4f), weak_largest=%s",
        total_tx,
        unique_directed_pairs,
        cross_count,
        report["bank_flows"]["cross_bank_ratio"] or 0.0,
        report["aggregated_graph_topology"]["weak_components"]["largest"],
    )
    _log_warnings("Raw structure audit", warnings)
    return report


def graph_audit(graph: AMLSimGraphArtifacts | Any) -> dict[str, Any]:
    """Audit the topology of the graph consumed by the GCN."""
    warnings: list[str] = []
    data = _graph_data(graph)
    edge_index = _to_tensor(getattr(data, "edge_index", None))
    y = _to_tensor(getattr(data, "y", None))
    num_nodes = int(getattr(data, "num_nodes", 0) or (y.numel() if y is not None else 0))

    report: dict[str, Any] = {
        "audit_type": "graph_audit",
        "counts": {
            "num_nodes": num_nodes,
            "num_edges": None,
            "unique_directed_pairs": None,
            "unique_undirected_pairs": None,
            "duplicate_edges": None,
            "self_loops": None,
        },
        "rates": {
            "reverse_edge_rate": None,
            "isolated_node_rate": None,
            "positive_rate_isolated": None,
            "positive_rate_non_isolated": None,
        },
        "degree": {
            "in": None,
            "out": None,
            "total": None,
            "isolated_nodes": None,
        },
        "warnings": warnings,
    }

    if edge_index is None or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        warnings.append("Graph data does not expose a valid edge_index with shape [2, num_edges].")
        logging.info("Graph audit: nodes=%d, edge_index unavailable", num_nodes)
        _log_warnings("Graph audit", warnings)
        return report

    edge_index = edge_index.long()
    num_edges = int(edge_index.shape[1])
    report["counts"]["num_edges"] = num_edges

    if num_nodes == 0:
        warnings.append("Graph has zero nodes; topology rates cannot be computed.")
        logging.info("Graph audit: nodes=0, edges=%d", num_edges)
        _log_warnings("Graph audit", warnings)
        return report

    src = edge_index[0]
    dst = edge_index[1]
    invalid_edges = int(((src < 0) | (dst < 0) | (src >= num_nodes) | (dst >= num_nodes)).sum().item())
    if invalid_edges:
        warnings.append(f"{invalid_edges} edges reference node indices outside [0, num_nodes).")

    valid_edge_mask = (src >= 0) & (dst >= 0) & (src < num_nodes) & (dst < num_nodes)
    src = src[valid_edge_mask]
    dst = dst[valid_edge_mask]

    pair_codes = src * num_nodes + dst
    unique_pair_codes = torch.unique(pair_codes)
    reverse_codes = dst * num_nodes + src
    non_self_mask = src != dst
    unique_directed_pairs = int(unique_pair_codes.numel())
    report["counts"]["unique_directed_pairs"] = unique_directed_pairs
    report["counts"]["duplicate_edges"] = int(src.numel() - unique_directed_pairs)
    report["counts"]["self_loops"] = int((src == dst).sum().item())

    undirected_a = torch.minimum(src, dst)
    undirected_b = torch.maximum(src, dst)
    report["counts"]["unique_undirected_pairs"] = int(torch.unique(undirected_a * num_nodes + undirected_b).numel())

    if non_self_mask.any():
        unique_non_self = torch.unique(pair_codes[non_self_mask])
        reverse_unique = torch.unique(reverse_codes[non_self_mask])
        reversible = int(torch.isin(unique_non_self, reverse_unique).sum().item())
        report["rates"]["reverse_edge_rate"] = _rate(reversible, int(unique_non_self.numel()))
    else:
        warnings.append("Graph has no non-self edges; reverse-edge rate is undefined.")

    out_degree = torch.bincount(src, minlength=num_nodes)
    in_degree = torch.bincount(dst, minlength=num_nodes)
    total_degree = in_degree + out_degree
    isolated_mask = total_degree == 0
    isolated_count = int(isolated_mask.sum().item())

    report["degree"]["in"] = _numeric_summary(in_degree)
    report["degree"]["out"] = _numeric_summary(out_degree)
    report["degree"]["total"] = _numeric_summary(total_degree)
    report["degree"]["isolated_nodes"] = isolated_count
    report["rates"]["isolated_node_rate"] = _rate(isolated_count, num_nodes)

    if y is not None and y.numel() == num_nodes:
        positive = y == 1
        report["rates"]["positive_rate_isolated"] = _rate(
            int((positive & isolated_mask).sum().item()),
            isolated_count,
        )
        report["rates"]["positive_rate_non_isolated"] = _rate(
            int((positive & ~isolated_mask).sum().item()),
            int((~isolated_mask).sum().item()),
        )
    elif y is not None:
        warnings.append("Label tensor length does not match num_nodes; isolated positive rates cannot be computed.")

    if isolated_count:
        warnings.append(f"{isolated_count} isolated nodes have no incoming or outgoing edges.")
    if report["counts"]["duplicate_edges"]:
        warnings.append(f"{report['counts']['duplicate_edges']} duplicate directed edges are present.")

    logging.info(
        "Graph audit: nodes=%d, edges=%d, unique_directed=%d, duplicate_edges=%d, isolated=%d, reverse_rate=%s",
        num_nodes,
        num_edges,
        report["counts"]["unique_directed_pairs"],
        report["counts"]["duplicate_edges"],
        isolated_count,
        report["rates"]["reverse_edge_rate"],
    )
    _log_warnings("Graph audit", warnings)
    return report


def feature_audit(graph: AMLSimGraphArtifacts | Any) -> dict[str, Any]:
    """Audit node feature values without mutating or normalizing them."""
    warnings: list[str] = []
    data = _graph_data(graph)
    x = _to_tensor(getattr(data, "x", None))
    y = _to_tensor(getattr(data, "y", None))

    report: dict[str, Any] = {
        "audit_type": "feature_audit",
        "shape": None,
        "features": {},
        "class_conditional": {},
        "warnings": warnings,
    }

    if x is None or x.ndim != 2:
        warnings.append("Graph data does not expose a valid two-dimensional node feature matrix x.")
        logging.info("Feature audit: x unavailable")
        _log_warnings("Feature audit", warnings)
        return report

    x = x.float()
    num_nodes, num_features = int(x.shape[0]), int(x.shape[1])
    columns = _feature_columns(graph, num_features)
    if len(columns) != num_features:
        warnings.append(
            f"Feature column metadata has {len(columns)} names but x has {num_features} columns; using fallback names."
        )
        columns = [f"feature_{idx}" for idx in range(num_features)]

    report["shape"] = {"num_nodes": num_nodes, "num_features": num_features}

    for idx, column in enumerate(columns):
        values = x[:, idx]
        nan_count = int(torch.isnan(values).sum().item())
        inf_count = int(torch.isinf(values).sum().item())
        finite_mask = torch.isfinite(values)
        finite_values = values[finite_mask]

        feature_report = {
            **_numeric_summary(finite_values),
            "zero_fraction": _rate(int((values == 0).sum().item()), int(values.numel())),
            "nan_count": nan_count,
            "inf_count": inf_count,
        }
        report["features"][column] = feature_report

        if nan_count or inf_count:
            warnings.append(f"Feature {column!r} contains {nan_count} NaN and {inf_count} infinite values.")
        if finite_values.numel() > 0:
            minimum = feature_report["min"]
            maximum = feature_report["max"]
            p99 = feature_report["p99"]
            median = feature_report["median"]
            if minimum is not None and minimum < 0:
                warnings.append(f"Feature {column!r} contains negative values.")
            if p99 not in (None, 0.0) and maximum is not None and maximum > 10 * p99:
                warnings.append(f"Feature {column!r} has max more than 10x its 99th percentile.")
            if median not in (None, 0.0) and p99 is not None and p99 > 100 * abs(median):
                warnings.append(f"Feature {column!r} has p99 more than 100x its median magnitude.")
        else:
            warnings.append(f"Feature {column!r} has no finite values.")

    if y is not None and y.numel() == num_nodes:
        for class_value in (0, 1):
            class_mask = y == class_value
            class_count = int(class_mask.sum().item())
            report["class_conditional"][f"class_{class_value}"] = {
                "count": class_count,
                "features": {
                    column: _numeric_summary(x[class_mask, idx][torch.isfinite(x[class_mask, idx])])
                    for idx, column in enumerate(columns)
                },
            }
    elif y is not None:
        warnings.append("Label tensor length does not match x rows; class-conditional feature summaries skipped.")

    total_nan = sum(feature["nan_count"] for feature in report["features"].values())
    total_inf = sum(feature["inf_count"] for feature in report["features"].values())
    logging.info(
        "Feature audit: nodes=%d, features=%d, total_nan=%d, total_inf=%d",
        num_nodes,
        num_features,
        total_nan,
        total_inf,
    )
    _log_warnings("Feature audit", warnings)
    return report


def split_audit(data: Any, masks: AMLSimNodeMasks | Mapping[str, Any] | None) -> dict[str, Any]:
    """Audit train/validation/test masks for coverage, overlap, and class balance."""
    warnings: list[str] = []
    y = _to_tensor(getattr(data, "y", None))
    num_nodes = int(getattr(data, "num_nodes", 0) or (y.numel() if y is not None else 0))
    mask_device = y.device if y is not None else None

    def normalize_mask(value: Any) -> torch.Tensor | None:
        tensor = _to_tensor(value)
        if tensor is None:
            return None
        tensor = tensor.bool().flatten()
        if mask_device is not None and tensor.device != mask_device:
            tensor = tensor.to(mask_device)
        return tensor

    def get_mask(name: str) -> torch.Tensor | None:
        if masks is not None:
            if isinstance(masks, Mapping):
                value = masks.get(name)
            else:
                value = getattr(masks, name, None)
            tensor = normalize_mask(value)
            if tensor is not None:
                return tensor
        value = getattr(data, name, None)
        return normalize_mask(value)

    mask_names = ("train_mask", "val_mask", "test_mask")
    split_masks = {name: get_mask(name) for name in mask_names}

    report: dict[str, Any] = {
        "audit_type": "split_audit",
        "num_nodes": num_nodes,
        "splits": {},
        "overlap": {},
        "coverage": {
            "covered_nodes": None,
            "uncovered_nodes": None,
            "covers_all_nodes": None,
        },
        "class_balance": {
            "global_positive_rate": None,
            "max_split_positive_rate_delta": None,
        },
        "warnings": warnings,
    }

    missing = [name for name, mask in split_masks.items() if mask is None]
    if missing:
        warnings.append(f"Required masks are missing: {', '.join(missing)}.")
        logging.info("Split audit: missing masks=%s", missing)
        _log_warnings("Split audit", warnings)
        return report

    for name, mask in split_masks.items():
        if mask.numel() != num_nodes:
            warnings.append(f"{name} length {mask.numel()} does not match num_nodes {num_nodes}.")

    if any(mask.numel() != num_nodes for mask in split_masks.values()):
        logging.info("Split audit: mask length mismatch; overlap and coverage skipped")
        _log_warnings("Split audit", warnings)
        return report

    if y is not None and y.numel() != num_nodes:
        warnings.append("Label tensor length does not match num_nodes; class counts cannot be computed.")
        y = None

    split_positive_rates: list[float] = []
    for name, mask in split_masks.items():
        count = int(mask.sum().item())
        positive_count = int(((y == 1) & mask).sum().item()) if y is not None else None
        negative_count = int(((y == 0) & mask).sum().item()) if y is not None else None
        positive_rate = _rate(positive_count, count) if positive_count is not None else None
        if positive_rate is not None:
            split_positive_rates.append(positive_rate)

        report["splits"][name.removesuffix("_mask")] = {
            "count": count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_rate": positive_rate,
        }

        if count == 0:
            warnings.append(f"{name} is empty.")

    train_mask = split_masks["train_mask"]
    val_mask = split_masks["val_mask"]
    test_mask = split_masks["test_mask"]
    pairwise_overlaps = {
        "train_val": int((train_mask & val_mask).sum().item()),
        "train_test": int((train_mask & test_mask).sum().item()),
        "val_test": int((val_mask & test_mask).sum().item()),
    }
    report["overlap"] = pairwise_overlaps
    for pair_name, overlap_count in pairwise_overlaps.items():
        if overlap_count:
            warnings.append(f"{pair_name} masks overlap on {overlap_count} nodes.")

    union_mask = train_mask | val_mask | test_mask
    covered_nodes = int(union_mask.sum().item())
    uncovered_nodes = int(num_nodes - covered_nodes)
    report["coverage"] = {
        "covered_nodes": covered_nodes,
        "uncovered_nodes": uncovered_nodes,
        "covers_all_nodes": uncovered_nodes == 0,
    }
    if uncovered_nodes:
        warnings.append(f"{uncovered_nodes} nodes are not covered by any split mask.")

    if y is not None:
        global_positive_rate = _rate(int((y == 1).sum().item()), num_nodes)
        max_delta = None
        if global_positive_rate is not None and split_positive_rates:
            max_delta = max(abs(rate - global_positive_rate) for rate in split_positive_rates)
            if max_delta > 0.05:
                warnings.append(
                    f"At least one split positive rate differs from global positive rate by more than 5pp ({max_delta:.4f})."
                )
        report["class_balance"] = {
            "global_positive_rate": global_positive_rate,
            "max_split_positive_rate_delta": max_delta,
        }

    logging.info(
        "Split audit: train=%d, val=%d, test=%d, uncovered=%d, overlaps=%s",
        report["splits"]["train"]["count"],
        report["splits"]["val"]["count"],
        report["splits"]["test"]["count"],
        uncovered_nodes,
        pairwise_overlaps,
    )
    _log_warnings("Split audit", warnings)
    return report


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args(description="Audit the centralized AMLSim graph benchmark before training.")
    torch.manual_seed(args.seed)

    context = resolve_amlsim_context(args.data_root, args.output_dir)
    device = resolve_device(args.device)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    logging.info("audit device: %s", device)

    raw = load_raw_amlsim(context.data_root)
    sar_set, label = derive_amlsim_targets(raw.sar_accounts, raw.alert_accounts)

    target_report = target_audit(context.data_root, label)
    raw_structure_report = raw_structure_audit(raw, sar_set, label)

    graph = build_amlsim_graph(raw.accounts, raw.transactions, sar_set)
    data = graph.data.to(device)
    graph = AMLSimGraphArtifacts(
        data=data,
        account_ids=graph.account_ids,
        id2idx=graph.id2idx,
        feature_columns=graph.feature_columns,
        feature_groups=graph.feature_groups,
        log1p_columns=graph.log1p_columns,
        feature_semantics=graph.feature_semantics,
    )

    label_alignment_report = label_alignment_audit(raw, sar_set, graph)
    graph_report = graph_audit(graph)
    feature_report = feature_audit(graph)

    masks = create_amlsim_node_masks(
        data=data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    masks = AMLSimNodeMasks(
        train_mask=masks.train_mask.to(device),
        val_mask=masks.val_mask.to(device),
        test_mask=masks.test_mask.to(device),
    )
    split_report = split_audit(data, masks)

    audit_report = {
        "audit_type": "amlsim_dataset_audit",
        "data_root": str(context.data_root),
        "output_dir": str(context.output_dir),
        "requested_device": args.device,
        "device": device,
        "seed": args.seed,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "target": target_report,
        "raw_structure": raw_structure_report,
        "label_alignment": label_alignment_report,
        "graph": graph_report,
        "features": feature_report,
        "split": split_report,
        "warnings": (
            target_report["warnings"]
            + raw_structure_report["warnings"]
            + label_alignment_report["warnings"]
            + graph_report["warnings"]
            + feature_report["warnings"]
            + split_report["warnings"]
        ),
    }

    report_path = context.output_dir / "audit_report.json"
    report_json = json.dumps(audit_report, indent=2)
    with report_path.open("w") as f:
        f.write(report_json)
        f.write("\n")

    print(report_json)

    logger.info("AMLSim dataset audit completed. Report written to %s", report_path)


if __name__ == "__main__":
    main()
