from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AMLSIM_ROOT = PROJECT_ROOT / "datasets" / "outputs" / "1m_3b_diag_i66_x33_combined_smoke_v07"
DEFAULT_IBMAML_ROOT = PROJECT_ROOT / "datasets" / "ibmaml" / "datasets"
AMLSIM_FEATURE_COLUMNS = (
    "initial_deposit",
    "in_tx_count",
    "out_tx_count",
    "in_sum",
    "out_sum",
    "in_mean",
    "out_mean",
    "in_max",
    "out_max",
    "in_std",
    "out_std",
    "in_median",
    "out_median",
    "in_unique_counterparties",
    "out_unique_counterparties",
    "total_unique_counterparties",
    "in_out_tx_count_ratio",
    "in_out_sum_ratio",
    "net_flow",
    "net_flow_relative",
)
AMLSIM_PASS2_EXTRA_FEATURE_COLUMNS = (
    "agg_in_degree",
    "agg_out_degree",
    "agg_total_degree",
    "weighted_in_degree",
    "weighted_out_degree",
    "weighted_total_degree",
    "repeat_intensity",
    "in_avg_tx_per_counterparty",
    "out_avg_tx_per_counterparty",
    "cross_bank_tx_share",
    "cross_bank_counterparty_share",
    "distinct_counterparty_banks",
    "neighbor_mean_total_tx_count",
    "neighbor_max_total_tx_count",
    "neighbor_mean_agg_total_degree",
    "neighbor_max_agg_total_degree",
)
AMLSIM_PASS2_FEATURE_COLUMNS = AMLSIM_FEATURE_COLUMNS + AMLSIM_PASS2_EXTRA_FEATURE_COLUMNS
AMLSIM_PRIVACY_CLEAN_REMOVED_FEATURE_COLUMNS = (
    "neighbor_mean_total_tx_count",
    "neighbor_max_total_tx_count",
    "neighbor_mean_agg_total_degree",
    "neighbor_max_agg_total_degree",
)
AMLSIM_FEATURE_BOUNDARIES = ("oracle", "privacy_clean")
AMLSIM_FEATURE_BOUNDARY_ASSUMPTION = (
    "direct transaction endpoints and counterparty bank IDs are assumed observable; "
    "topology privacy is not claimed."
)
AMLSIM_LOG1P_COLUMNS = (
    "initial_deposit",
    "in_tx_count",
    "out_tx_count",
    "in_sum",
    "out_sum",
    "in_mean",
    "out_mean",
    "in_max",
    "out_max",
    "in_unique_counterparties",
    "out_unique_counterparties",
    "total_unique_counterparties",
)
AMLSIM_PASS2_LOG1P_COLUMNS = AMLSIM_LOG1P_COLUMNS + (
    "agg_in_degree",
    "agg_out_degree",
    "agg_total_degree",
    "weighted_in_degree",
    "weighted_out_degree",
    "weighted_total_degree",
    "repeat_intensity",
    "in_avg_tx_per_counterparty",
    "out_avg_tx_per_counterparty",
    "distinct_counterparty_banks",
    "neighbor_mean_total_tx_count",
    "neighbor_max_total_tx_count",
    "neighbor_mean_agg_total_degree",
    "neighbor_max_agg_total_degree",
)
AMLSIM_FEATURE_GROUPS = {
    "account": ("initial_deposit",),
    "direct_flow_stats": (
        "in_tx_count",
        "out_tx_count",
        "in_sum",
        "out_sum",
        "in_mean",
        "out_mean",
        "in_max",
        "out_max",
        "in_std",
        "out_std",
        "in_median",
        "out_median",
    ),
    "counterparty_breadth": (
        "in_unique_counterparties",
        "out_unique_counterparties",
        "total_unique_counterparties",
    ),
    "flow_balance": (
        "in_out_tx_count_ratio",
        "in_out_sum_ratio",
        "net_flow",
        "net_flow_relative",
    ),
}
AMLSIM_PASS2_EXTRA_FEATURE_GROUPS = {
    "aggregated_topology": (
        "agg_in_degree",
        "agg_out_degree",
        "agg_total_degree",
        "weighted_in_degree",
        "weighted_out_degree",
        "weighted_total_degree",
    ),
    "repeat_intensity": (
        "repeat_intensity",
        "in_avg_tx_per_counterparty",
        "out_avg_tx_per_counterparty",
    ),
    "bank_mixing": (
        "cross_bank_tx_share",
        "cross_bank_counterparty_share",
        "distinct_counterparty_banks",
    ),
    "neighbor_activity": (
        "neighbor_mean_total_tx_count",
        "neighbor_max_total_tx_count",
        "neighbor_mean_agg_total_degree",
        "neighbor_max_agg_total_degree",
    ),
}
AMLSIM_PASS2_FEATURE_GROUPS = {
    **AMLSIM_FEATURE_GROUPS,
    **AMLSIM_PASS2_EXTRA_FEATURE_GROUPS,
}
AMLSIM_FEATURE_SEMANTICS = {
    "initial_deposit": "Initial account deposit from accounts.csv; retained for comparability and not derived from labels.",
    "in_tx_count": "Raw inbound transaction-row count; repeated directed edges are preserved.",
    "out_tx_count": "Raw outbound transaction-row count; repeated directed edges are preserved.",
    "in_sum": "Sum of inbound transaction base_amt values.",
    "out_sum": "Sum of outbound transaction base_amt values.",
    "in_mean": "Mean inbound transaction base_amt.",
    "out_mean": "Mean outbound transaction base_amt.",
    "in_max": "Maximum inbound transaction base_amt.",
    "out_max": "Maximum outbound transaction base_amt.",
    "in_std": "Population standard deviation of inbound transaction base_amt.",
    "out_std": "Population standard deviation of outbound transaction base_amt.",
    "in_median": "Median inbound transaction base_amt.",
    "out_median": "Median outbound transaction base_amt.",
    "in_unique_counterparties": "Distinct outbound-origin counterparties that send funds to the node; captures inbound neighborhood breadth.",
    "out_unique_counterparties": "Distinct inbound-destination counterparties that receive funds from the node; captures outbound neighborhood breadth.",
    "total_unique_counterparties": "Distinct union of inbound and outbound counterparties.",
    "in_out_tx_count_ratio": "Inbound/outbound transaction-count ratio with safe denominator handling.",
    "in_out_sum_ratio": "Inbound/outbound transaction-sum ratio with safe denominator handling.",
    "net_flow": "Inbound sum minus outbound sum.",
    "net_flow_relative": "Net flow normalized by total flow magnitude using a safe denominator.",
}
AMLSIM_PASS2_EXTRA_FEATURE_SEMANTICS = {
    "agg_in_degree": "Number of unique inbound counterparties in the aggregated directed transaction graph.",
    "agg_out_degree": "Number of unique outbound counterparties in the aggregated directed transaction graph.",
    "agg_total_degree": "Total unique directed in/out counterparty degree in the aggregated transaction graph.",
    "weighted_in_degree": "Inbound aggregated degree weighted by repeated raw transaction-row count.",
    "weighted_out_degree": "Outbound aggregated degree weighted by repeated raw transaction-row count.",
    "weighted_total_degree": "Total in/out aggregated degree weighted by repeated raw transaction-row count.",
    "repeat_intensity": "Total raw transaction rows divided by total unique counterparties, with safe denominator handling.",
    "in_avg_tx_per_counterparty": "Inbound raw transaction rows per unique inbound counterparty.",
    "out_avg_tx_per_counterparty": "Outbound raw transaction rows per unique outbound counterparty.",
    "cross_bank_tx_share": "Share of in/out raw transaction rows whose counterparty belongs to another bank.",
    "cross_bank_counterparty_share": "Share of in/out unique directed counterparties from another bank.",
    "distinct_counterparty_banks": "Number of distinct banks represented among inbound and outbound counterparties.",
    "neighbor_mean_total_tx_count": "Mean total raw transaction-row count of aggregated-graph neighbors.",
    "neighbor_max_total_tx_count": "Maximum total raw transaction-row count of aggregated-graph neighbors.",
    "neighbor_mean_agg_total_degree": "Mean aggregated total degree of aggregated-graph neighbors.",
    "neighbor_max_agg_total_degree": "Maximum aggregated total degree of aggregated-graph neighbors.",
}
AMLSIM_PASS2_FEATURE_SEMANTICS = {
    **AMLSIM_FEATURE_SEMANTICS,
    **AMLSIM_PASS2_EXTRA_FEATURE_SEMANTICS,
}


@dataclass(frozen=True)
class AMLSimRawTables:
    accounts: pd.DataFrame
    transactions: pd.DataFrame
    sar_accounts: pd.DataFrame
    alert_accounts: pd.DataFrame | None = None


@dataclass(frozen=True)
class AMLSimGraphArtifacts:
    data: Data
    account_ids: tuple[str, ...]
    id2idx: dict[str, int]
    feature_columns: tuple[str, ...]
    feature_groups: dict[str, tuple[str, ...]]
    log1p_columns: tuple[str, ...]
    feature_semantics: dict[str, str]
    representation_version: str = "pass1"
    graph_semantics: dict | None = None
    feature_boundary: str = "oracle"
    configured_removed_feature_columns: tuple[str, ...] = ()
    actually_removed_feature_columns: tuple[str, ...] = ()
    feature_columns_before_boundary: tuple[str, ...] = ()
    feature_columns_after_boundary: tuple[str, ...] = ()
    feature_count_before_boundary: int = 0
    feature_count_after_boundary: int = 0
    feature_boundary_assumption: str = AMLSIM_FEATURE_BOUNDARY_ASSUMPTION


def load_raw_amlsim(root=DEFAULT_AMLSIM_ROOT) -> AMLSimRawTables:
    root = Path(root).expanduser()

    accounts = pd.read_csv(root / "accounts.csv")
    transactions = pd.read_csv(root / "transactions.csv")
    sar_accounts = pd.read_csv(root / "sar_accounts.csv")

    alert_accounts_path = root / "alert_accounts.csv"
    alert_accounts = pd.read_csv(alert_accounts_path) if alert_accounts_path.exists() else None

    return AMLSimRawTables(
        accounts=accounts,
        transactions=transactions,
        sar_accounts=sar_accounts,
        alert_accounts=alert_accounts,
    )


def _sar_account_set(sar, alert_accounts=None):
    if alert_accounts is not None:
        is_sar = alert_accounts["is_sar"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
        return set(alert_accounts.loc[is_sar, "acct_id"].astype(str).values), "alert"

    is_sar = sar["IS_SAR"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
    return set(sar.loc[is_sar, "ACCOUNT_ID"].astype(str).values), "sar"


def derive_amlsim_targets(sar_accounts, alert_accounts=None) -> tuple[set[str], str]:
    return _sar_account_set(sar_accounts, alert_accounts)


def _safe_log1p_columns(features: pd.DataFrame, columns: tuple[str, ...]) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Apply log1p only to non-negative columns so feature construction stays inspectable."""
    transformed = features.copy()
    applied: list[str] = []
    for column in columns:
        if column not in transformed.columns:
            continue
        values = transformed[column].to_numpy(dtype=np.float64, copy=False)
        if np.any(values < 0):
            continue
        transformed[column] = np.log1p(values)
        applied.append(column)
    return transformed, tuple(applied)


def _base_amlsim_account_features(accounts: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe account and direct transaction aggregate features."""
    # Pass 1 intentionally uses only safe transaction/account inputs and excludes
    # leakage-prone columns such as SAR flags, alert IDs, and prior_sar_count.
    out_tx = tx.groupby("orig_acct").agg(
        out_tx_count=("base_amt", "count"),
        out_sum=("base_amt", "sum")
    )
    in_tx = tx.groupby("bene_acct").agg(
        in_tx_count=("base_amt", "count"),
        in_sum=("base_amt", "sum"),
    )
    out_stats = tx.groupby("orig_acct").agg(
        out_mean=("base_amt", "mean"),
        out_max=("base_amt", "max"),
        out_std=("base_amt", lambda values: float(np.std(values, ddof=0))),
        out_median=("base_amt", "median"),
    )
    in_stats = tx.groupby("bene_acct").agg(
        in_mean=("base_amt", "mean"),
        in_max=("base_amt", "max"),
        in_std=("base_amt", lambda values: float(np.std(values, ddof=0))),
        in_median=("base_amt", "median"),
    )
    out_unique = tx.groupby("orig_acct").agg(
        out_unique_counterparties=("bene_acct", "nunique"),
    )
    in_unique = tx.groupby("bene_acct").agg(
        in_unique_counterparties=("orig_acct", "nunique"),
    )

    counterparties = pd.concat(
        [
            tx.loc[:, ["orig_acct", "bene_acct"]].rename(
                columns={"orig_acct": "acct_id", "bene_acct": "counterparty"}
            ),
            tx.loc[:, ["bene_acct", "orig_acct"]].rename(
                columns={"bene_acct": "acct_id", "orig_acct": "counterparty"}
            ),
        ],
        ignore_index=True,
    )
    total_unique = counterparties.groupby("acct_id").agg(
        total_unique_counterparties=("counterparty", "nunique"),
    )

    account_features = accounts.set_index("acct_id")
    account_features = (
        account_features.join(out_tx)
        .join(in_tx)
        .join(out_stats)
        .join(in_stats)
        .join(out_unique)
        .join(in_unique)
        .join(total_unique)
        .fillna(0.0)
    )

    eps = 1e-6
    account_features["in_out_tx_count_ratio"] = (
        account_features["in_tx_count"] / np.maximum(account_features["out_tx_count"], 1.0)
    )
    account_features["in_out_sum_ratio"] = (
        account_features["in_sum"] / np.maximum(account_features["out_sum"], eps)
    )
    account_features["net_flow"] = account_features["in_sum"] - account_features["out_sum"]
    total_flow = account_features["in_sum"] + account_features["out_sum"]
    account_features["net_flow_relative"] = account_features["net_flow"] / np.maximum(total_flow, eps)
    return account_features


def _add_pass2_topology_features(accounts: pd.DataFrame, tx_pairs: pd.DataFrame) -> pd.DataFrame:
    """Add topology-aware features from the aggregated directed transaction graph."""
    pair_cols = ["orig_acct", "bene_acct", "tx_count"]
    pairs = tx_pairs.loc[:, pair_cols].copy()

    out_degree = pairs.groupby("orig_acct").agg(
        agg_out_degree=("bene_acct", "count"),
        weighted_out_degree=("tx_count", "sum"),
    )
    in_degree = pairs.groupby("bene_acct").agg(
        agg_in_degree=("orig_acct", "count"),
        weighted_in_degree=("tx_count", "sum"),
    )
    accounts = accounts.join(out_degree).join(in_degree).fillna(0.0)
    accounts["agg_total_degree"] = accounts["agg_in_degree"] + accounts["agg_out_degree"]
    accounts["weighted_total_degree"] = accounts["weighted_in_degree"] + accounts["weighted_out_degree"]

    accounts["repeat_intensity"] = (
        (accounts["in_tx_count"] + accounts["out_tx_count"])
        / np.maximum(accounts["total_unique_counterparties"], 1.0)
    )
    accounts["in_avg_tx_per_counterparty"] = (
        accounts["in_tx_count"] / np.maximum(accounts["in_unique_counterparties"], 1.0)
    )
    accounts["out_avg_tx_per_counterparty"] = (
        accounts["out_tx_count"] / np.maximum(accounts["out_unique_counterparties"], 1.0)
    )

    bank_by_account = accounts["bank_id"].astype(str)
    pairs["orig_bank"] = pairs["orig_acct"].map(bank_by_account)
    pairs["bene_bank"] = pairs["bene_acct"].map(bank_by_account)
    pairs["is_cross_bank"] = pairs["orig_bank"] != pairs["bene_bank"]
    pairs["cross_tx_count"] = pairs["tx_count"].where(pairs["is_cross_bank"], 0)

    out_cross = pairs.groupby("orig_acct").agg(
        out_cross_tx_count=("cross_tx_count", "sum"),
        out_cross_counterparties=("is_cross_bank", "sum"),
    )
    in_cross = pairs.groupby("bene_acct").agg(
        in_cross_tx_count=("cross_tx_count", "sum"),
        in_cross_counterparties=("is_cross_bank", "sum"),
    )
    accounts = accounts.join(out_cross).join(in_cross).fillna(0.0)
    cross_tx_count = accounts["out_cross_tx_count"] + accounts["in_cross_tx_count"]
    total_tx_count = accounts["out_tx_count"] + accounts["in_tx_count"]
    cross_counterparties = accounts["out_cross_counterparties"] + accounts["in_cross_counterparties"]
    directed_counterparties = accounts["agg_out_degree"] + accounts["agg_in_degree"]
    accounts["cross_bank_tx_share"] = cross_tx_count / np.maximum(total_tx_count, 1.0)
    accounts["cross_bank_counterparty_share"] = cross_counterparties / np.maximum(directed_counterparties, 1.0)

    counterparty_banks = pd.concat(
        [
            pairs.loc[:, ["orig_acct", "bene_bank"]].rename(
                columns={"orig_acct": "acct_id", "bene_bank": "counterparty_bank"}
            ),
            pairs.loc[:, ["bene_acct", "orig_bank"]].rename(
                columns={"bene_acct": "acct_id", "orig_bank": "counterparty_bank"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    distinct_banks = counterparty_banks.groupby("acct_id").agg(
        distinct_counterparty_banks=("counterparty_bank", "nunique"),
    )
    accounts = accounts.join(distinct_banks).fillna(0.0)

    accounts["_total_tx_count"] = total_tx_count
    neighbor_source = accounts.loc[:, ["_total_tx_count", "agg_total_degree"]]
    neighbor_rows = pd.concat(
        [
            pairs.loc[:, ["orig_acct", "bene_acct"]].rename(
                columns={"orig_acct": "acct_id", "bene_acct": "neighbor_acct"}
            ),
            pairs.loc[:, ["bene_acct", "orig_acct"]].rename(
                columns={"bene_acct": "acct_id", "orig_acct": "neighbor_acct"}
            ),
        ],
        ignore_index=True,
    )
    neighbor_rows = neighbor_rows.join(neighbor_source, on="neighbor_acct")
    neighbor_stats = neighbor_rows.groupby("acct_id").agg(
        neighbor_mean_total_tx_count=("_total_tx_count", "mean"),
        neighbor_max_total_tx_count=("_total_tx_count", "max"),
        neighbor_mean_agg_total_degree=("agg_total_degree", "mean"),
        neighbor_max_agg_total_degree=("agg_total_degree", "max"),
    )
    accounts = accounts.join(neighbor_stats).fillna(0.0)
    accounts = accounts.drop(columns=["_total_tx_count"])
    return accounts


def _build_amlsim_edge_index(id2idx: dict[str, int], tx: pd.DataFrame, representation_version: str):
    if representation_version == "pass1":
        src = tx["orig_acct"].map(id2idx).to_numpy(dtype=np.int64)
        dst = tx["bene_acct"].map(id2idx).to_numpy(dtype=np.int64)
        edge_index = torch.from_numpy(np.stack([src, dst]))
        return edge_index, None, None

    tx_pairs = tx.groupby(["orig_acct", "bene_acct"], sort=False).agg(
        tx_count=("base_amt", "count"),
        amount_sum=("base_amt", "sum"),
    ).reset_index()
    src = tx_pairs["orig_acct"].map(id2idx).to_numpy(dtype=np.int64)
    dst = tx_pairs["bene_acct"].map(id2idx).to_numpy(dtype=np.int64)
    edge_index = torch.from_numpy(np.stack([src, dst]))
    edge_weight = torch.from_numpy(np.log1p(tx_pairs["tx_count"].to_numpy(dtype=np.float32)))
    return edge_index, edge_weight, tx_pairs


def _apply_feature_boundary(
    feature_columns: tuple[str, ...],
    feature_groups: dict[str, tuple[str, ...]],
    log1p_columns: tuple[str, ...],
    feature_semantics: dict[str, str],
    feature_boundary: str,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], tuple[str, ...], dict[str, str], tuple[str, ...], tuple[str, ...]]:
    if feature_boundary not in AMLSIM_FEATURE_BOUNDARIES:
        raise ValueError(f"feature_boundary must be one of: {', '.join(AMLSIM_FEATURE_BOUNDARIES)}")

    configured_removed = (
        AMLSIM_PRIVACY_CLEAN_REMOVED_FEATURE_COLUMNS if feature_boundary == "privacy_clean" else ()
    )
    configured_removed_set = set(configured_removed)
    actually_removed = tuple(column for column in feature_columns if column in configured_removed_set)
    active_columns = tuple(column for column in feature_columns if column not in configured_removed_set)
    active_column_set = set(active_columns)
    active_groups = {
        name: tuple(column for column in columns if column in active_column_set)
        for name, columns in feature_groups.items()
    }
    active_groups = {name: columns for name, columns in active_groups.items() if columns}
    active_log1p = tuple(column for column in log1p_columns if column in active_column_set)
    active_semantics = {
        column: description for column, description in feature_semantics.items() if column in active_column_set
    }
    return active_columns, active_groups, active_log1p, active_semantics, configured_removed, actually_removed


def build_amlsim_graph(
    accounts,
    transactions,
    sar_set,
    representation_version="pass1",
    feature_boundary="oracle",
) -> AMLSimGraphArtifacts:
    """Build the centralized AMLSim graph with engineered node features."""
    if representation_version not in {"pass1", "pass2"}:
        raise ValueError("representation_version must be one of: pass1, pass2")

    accounts = accounts.copy()
    tx = transactions.copy()
    accounts["acct_id"] = accounts["acct_id"].astype(str)
    tx["orig_acct"] = tx["orig_acct"].astype(str)
    tx["bene_acct"] = tx["bene_acct"].astype(str)

    account_ids = tuple(accounts["acct_id"])
    id2idx = {acc: i for i, acc in enumerate(account_ids)}

    edge_index, edge_weight, tx_pairs = _build_amlsim_edge_index(id2idx, tx, representation_version)

    accounts = _base_amlsim_account_features(accounts, tx)
    if representation_version == "pass2":
        accounts = _add_pass2_topology_features(accounts, tx_pairs)
        feature_columns = AMLSIM_PASS2_FEATURE_COLUMNS
        feature_groups = AMLSIM_PASS2_FEATURE_GROUPS
        log1p_columns = AMLSIM_PASS2_LOG1P_COLUMNS
        feature_semantics = AMLSIM_PASS2_FEATURE_SEMANTICS
    else:
        feature_columns = AMLSIM_FEATURE_COLUMNS
        feature_groups = AMLSIM_FEATURE_GROUPS
        log1p_columns = AMLSIM_LOG1P_COLUMNS
        feature_semantics = AMLSIM_FEATURE_SEMANTICS
    feature_columns_before_boundary = tuple(feature_columns)
    (
        feature_columns,
        feature_groups,
        log1p_columns,
        feature_semantics,
        configured_removed,
        actually_removed,
    ) = _apply_feature_boundary(
        feature_columns,
        feature_groups,
        log1p_columns,
        feature_semantics,
        feature_boundary,
    )

    features = accounts.loc[:, list(feature_columns)].astype(np.float32)
    features, applied_log1p = _safe_log1p_columns(features, log1p_columns)

    x = torch.tensor(
        features.to_numpy(dtype=np.float32),
        dtype=torch.float
    )

    y = torch.tensor(
        accounts.index.isin(sar_set).astype(int),
        dtype=torch.long
    )

    data = Data(x=x, edge_index=edge_index, y=y)
    if edge_weight is not None:
        data.edge_weight = edge_weight

    return AMLSimGraphArtifacts(
        data=data,
        account_ids=account_ids,
        id2idx=id2idx,
        feature_columns=feature_columns,
        feature_groups={name: tuple(columns) for name, columns in feature_groups.items()},
        log1p_columns=applied_log1p,
        feature_semantics=dict(feature_semantics),
        representation_version=representation_version,
        graph_semantics={
            "representation_version": representation_version,
            "edge_source": "raw transaction rows" if representation_version == "pass1" else "aggregated directed account pairs",
            "edge_weight": None if representation_version == "pass1" else "log1p(raw transaction count per directed pair)",
        },
        feature_boundary=feature_boundary,
        configured_removed_feature_columns=configured_removed,
        actually_removed_feature_columns=actually_removed,
        feature_columns_before_boundary=feature_columns_before_boundary,
        feature_columns_after_boundary=feature_columns,
        feature_count_before_boundary=len(feature_columns_before_boundary),
        feature_count_after_boundary=len(feature_columns),
        feature_boundary_assumption=AMLSIM_FEATURE_BOUNDARY_ASSUMPTION,
    )


def build_graph_amlsim(root=DEFAULT_AMLSIM_ROOT, representation_version="pass1", feature_boundary="oracle"):
    raw = load_raw_amlsim(root=root)
    sar_set, _ = derive_amlsim_targets(raw.sar_accounts, raw.alert_accounts)
    return build_amlsim_graph(
        raw.accounts,
        raw.transactions,
        sar_set,
        representation_version=representation_version,
        feature_boundary=feature_boundary,
    ).data


def load_amlsim(root=DEFAULT_AMLSIM_ROOT):
    return build_graph_amlsim(root=root)

# ----------------------------------------------------
# ---------------------IBMAML-------------------------
# ----------------------------------------------------

def _acct_key(bank_id, acct_num) -> str:
    # Bank ID can be int-like or string; normalize
    return f"{int(bank_id)}::{str(acct_num).strip()}"

def _parse_patterns_txt(patterns_path):
    """
    Parses patterns.txt and returns:
      - laundering_accts: set of account keys (bank_id::acct_num) involved in laundering attempts
      - laundering_edges: set of (src_key, dst_key) for laundering transactions (optional)
    Expected line format inside attempts:
      timestamp,from_bank,from_acct,to_bank,to_acct,amt_recv,curr_recv,amt_paid,curr_paid,payment_type,1
    """
    laundering_accts = set()
    laundering_edges = set()

    with patterns_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("BEGIN") or line.startswith("END"):
                continue

            # Split safely: there seem to be no quoted commas

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 11:
                continue

            try:
                from_bank = parts[1]
                from_acct = parts[2]
                to_bank   = parts[3]
                to_acct   = parts[4]
            except Exception:
                continue

            src_key = _acct_key(from_bank, from_acct)
            dst_key = _acct_key(to_bank, to_acct)

            laundering_accts.add(src_key)
            laundering_accts.add(dst_key)
            laundering_edges.add((src_key, dst_key))

    return laundering_accts, laundering_edges


def load_ibmaml(root=DEFAULT_IBMAML_ROOT, version=2):
    root = Path(root).expanduser()
    prefixes = {
        0: "HI-Large_",
        1: "HI-Medium_",
        2: "HI-Small_",
        3: "LI-Large_",
        4: "LI-Medium_",
        5: "LI-Small_",
    }
    prefix = prefixes[version]

    accs = pd.read_csv(root / f"{prefix}accounts.csv")
    tx = pd.read_csv(root / f"{prefix}Trans.csv")
    patterns_path = root / f"{prefix}Patterns.txt"

    laundering_accs, _ = _parse_patterns_txt(patterns_path)

    accs["acct_key"] = [
        _acct_key(b, a) for b, a in zip(accs["Bank ID"].values, accs["Account Number"].values)
    ]

    id2idx = {k: i for i, k in enumerate(accs["acct_key"].values)}

    tx_src_key = [_acct_key(b, a) for b, a in zip(tx["From Bank"].values, tx["Account"].values)]
    tx_dst_key = [_acct_key(b, a) for b, a in zip(tx["To Bank"], tx["Account.1"].values)]

    # Filter transactions that reference unknown accounts (just in case)
    src_idx = []
    dst_idx = []
    keep_rows = []
    for i, (s, d) in enumerate(zip(tx_src_key, tx_dst_key)):
        if s in id2idx and d in id2idx:
            src_idx.append(id2idx[s])
            dst_idx.append(id2idx[d])
            keep_rows.append(i)

    tx = tx.iloc[keep_rows].reset_index(drop=True)
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

    # We use Amount Paid (outgoing) and Amount Received (incoming) if present.
    # If only one is present, fall back gracefully.
    amt_out_col = "Amount Paid" if "Amount Paid" in tx.columns else None
    amt_in_col  = "Amount Received" if "Amount Received" in tx.columns else None

    # Use log1p to stabilize
    if amt_out_col is not None:
        tx["_amt_out"] = np.log1p(tx[amt_out_col].astype(float).values)
    else:
        tx["_amt_out"] = 0.0

    if amt_in_col is not None:
        tx["_amt_in"] = np.log1p(tx[amt_in_col].astype(float).values)
    else:
        tx["_amt_in"] = 0.0

    # Aggregate outgoing by (From Bank, Account)
    out_df = tx.groupby(["From Bank", "Account"]).agg(
        out_count=("_amt_out", "count"),
        out_sum=("_amt_out", "sum"),
        out_mean=("_amt_out", "mean"),
        out_std=("_amt_out", "std"),
    ).reset_index()
    out_df["acct_key"] = [_acct_key(b, a) for b, a in zip(out_df["From Bank"], out_df["Account"])]

    # Aggregate incoming by (To Bank, Account.1)
    in_df = tx.groupby(["To Bank", "Account.1"]).agg(
        in_count=("_amt_in", "count"),
        in_sum=("_amt_in", "sum"),
        in_mean=("_amt_in", "mean"),
        in_std=("_amt_in", "std"),
    ).reset_index()
    in_df["acct_key"] = [_acct_key(b, a) for b, a in zip(in_df["To Bank"], in_df["Account.1"])]

    # feat = accs[["acct_key", "Entity Name"]].copy()

    feat = accs[["acct_key"]].copy()

    # Entity type one-hot (low-cardinality, meaningful)
    # ent_type = feat["Entity Name"].astype(str).str.split().str[0]
    # ent_oh = pd.get_dummies(ent_type, prefix="ent")

    # feat = pd.concat([feat[["acct_key"]], ent_oh], axis=1)

    feat = feat.merge(out_df[["acct_key","out_count","out_sum","out_mean","out_std"]], on="acct_key", how="left")
    feat = feat.merge(in_df[["acct_key","in_count","in_sum","in_mean","in_std"]], on="acct_key", how="left")
    feat = feat.fillna(0.0)

    feature_cols = [c for c in feat.columns if c != "acct_key"]
    x = torch.tensor(feat[feature_cols].to_numpy(dtype=np.float32), dtype=torch.float)
    y = torch.tensor([1 if k in laundering_accs else 0 for k in accs["acct_key"].values], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    return data

def viz_subgraph(data, node_idx, hops=1, max_nodes=200):
    import matplotlib.pyplot as plt
    import networkx as nx
    import torch_geometric.utils as pyg_utils

    G = pyg_utils.to_networkx(data, to_undirected=True)

    nodes = nx.single_source_shortest_path_length(G, node_idx, hops).keys()

    nodes = list(nodes)[:max_nodes]
    H = G.subgraph(nodes)

    print(f"Visualizing {H.number_of_nodes()} nodes and {H.number_of_edges()} edges")

    plt.figure(figsize=(8,8))

    pos = nx.spring_layout(H, k=0.15, seed=42)

    nx.draw(
        H,
        pos,
        node_size=30,
        node_color="skyblue",
        edge_color="gray",
        linewidths=0.5,
        alpha=0.8,
        with_labels=False
    )

    plt.show()

def main():

    data = build_graph_amlsim()
    print(data)
    print(data.num_nodes)
    print(data.num_edges)

    viz_subgraph(data, node_idx=10, hops=2)



if __name__ == "__main__":
    main()
