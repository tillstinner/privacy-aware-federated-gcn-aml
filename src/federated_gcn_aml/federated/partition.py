from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch


@dataclass(frozen=True)
class BankPartition:
    bank_ids: tuple[str, ...]
    client_node_indices: dict[str, torch.Tensor]
    node_to_client: tuple[str, ...]
    diagnostics: dict


def _validate_bank_ids(accounts: pd.DataFrame) -> pd.DataFrame:
    required = {"acct_id", "bank_id"}
    missing = required.difference(accounts.columns)
    if missing:
        raise ValueError(f"accounts is missing required columns: {sorted(missing)}")

    account_bank = accounts.loc[:, ["acct_id", "bank_id"]].copy()
    if account_bank["bank_id"].isna().any():
        bad = account_bank.loc[account_bank["bank_id"].isna(), "acct_id"].head(5).tolist()
        raise ValueError(f"accounts contains null bank_id values; first acct_id values: {bad}")
    bank_as_str = account_bank["bank_id"].astype(str).str.strip()
    if (bank_as_str == "").any():
        bad = account_bank.loc[bank_as_str == "", "acct_id"].head(5).tolist()
        raise ValueError(f"accounts contains empty bank_id values; first acct_id values: {bad}")
    account_bank["acct_id"] = account_bank["acct_id"].astype(str)
    account_bank["bank_id"] = bank_as_str
    return account_bank


def build_bank_partition(
    accounts: pd.DataFrame,
    account_ids: tuple[str, ...],
    id2idx: dict[str, int],
    y: torch.Tensor | None = None,
    masks: dict[str, torch.Tensor] | None = None,
) -> BankPartition:
    """Build a deterministic one-bank-per-client partition for AMLSim graph nodes."""

    account_bank = _validate_bank_ids(accounts)
    if not account_bank["acct_id"].is_unique:
        duplicate_accounts = account_bank["acct_id"].duplicated(keep=False)
        preview = ", ".join(account_bank.loc[duplicate_accounts, "acct_id"].drop_duplicates().head(5))
        raise ValueError(f"accounts contains duplicate acct_id values; first: {preview}")

    account_ids_str = tuple(str(account_id) for account_id in account_ids)
    if len(set(account_ids_str)) != len(account_ids_str):
        raise ValueError("graph account_ids contains duplicate account identifiers")
    for account_id in account_ids_str:
        if account_id not in id2idx:
            raise ValueError(f"account_id {account_id!r} is missing from id2idx")
        if int(id2idx[account_id]) < 0 or int(id2idx[account_id]) >= len(account_ids_str):
            raise ValueError(f"id2idx for account_id {account_id!r} is outside graph node range")

    account_to_bank = account_bank.set_index("acct_id")["bank_id"].to_dict()
    missing_accounts = [account_id for account_id in account_ids_str if account_id not in account_to_bank]
    if missing_accounts:
        preview = ", ".join(missing_accounts[:5])
        raise ValueError(f"{len(missing_accounts)} graph accounts have no bank_id mapping; first: {preview}")

    node_to_client: list[str | None] = [None] * len(account_ids_str)
    client_nodes: dict[str, list[int]] = {}
    for account_id in account_ids_str:
        node_idx = int(id2idx[account_id])
        if node_to_client[node_idx] is not None:
            raise ValueError(f"graph node index {node_idx} was assigned more than once")
        bank_id = account_to_bank[account_id]
        node_to_client[node_idx] = bank_id
        client_nodes.setdefault(bank_id, []).append(node_idx)

    unassigned = [idx for idx, bank_id in enumerate(node_to_client) if bank_id is None]
    if unassigned:
        raise ValueError(f"{len(unassigned)} graph nodes were not assigned to any bank.")

    extra_accounts = sorted(set(account_to_bank).difference(account_ids_str))
    bank_ids = tuple(sorted(client_nodes))
    client_node_indices = {
        bank_id: torch.tensor(sorted(nodes), dtype=torch.long)
        for bank_id, nodes in client_nodes.items()
    }

    diagnostics = {
        "num_clients": len(bank_ids),
        "bank_ids": list(bank_ids),
        "accounts_not_in_graph": {"count": len(extra_accounts), "first": extra_accounts[:5]},
        "clients": {},
    }
    total_split_counts = {split: 0 for split in (masks or {})}
    for bank_id in bank_ids:
        nodes = client_node_indices[bank_id]
        client_diag = {"owned_nodes": int(nodes.numel())}
        if y is not None:
            positives = int(y[nodes].sum().item())
            client_diag["positive_labels"] = positives
            client_diag["label_rate"] = float(positives / max(int(nodes.numel()), 1))
        if masks is not None:
            client_diag["split_counts"] = {}
            for split_name, mask in masks.items():
                split_nodes = mask[nodes]
                count = int(split_nodes.sum().item())
                total_split_counts[split_name] += count
                split_diag = {"count": count}
                if y is not None:
                    positives = int(y[nodes][split_nodes].sum().item())
                    split_diag["positive_labels"] = positives
                    split_diag["label_rate"] = float(positives / max(count, 1))
                client_diag["split_counts"][split_name] = split_diag
        diagnostics["clients"][bank_id] = client_diag
    if masks is not None:
        diagnostics["split_totals"] = {
            split_name: {"client_sum": total, "global_count": int(masks[split_name].sum().item())}
            for split_name, total in total_split_counts.items()
        }

    return BankPartition(
        bank_ids=bank_ids,
        client_node_indices=client_node_indices,
        node_to_client=tuple(str(bank_id) for bank_id in node_to_client),
        diagnostics=diagnostics,
    )
