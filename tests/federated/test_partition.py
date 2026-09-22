import pandas as pd
import pytest
import torch

from federated_gcn_aml.federated.partition import build_bank_partition


def test_partition_assigns_every_node_once_and_preserves_splits():
    accounts = pd.DataFrame({"acct_id": ["a", "b", "c", "d"], "bank_id": ["B2", "B1", "B2", "B1"]})
    y = torch.tensor([0, 1, 0, 1])
    masks = {
        "train": torch.tensor([True, True, False, False]),
        "val": torch.tensor([False, False, True, False]),
        "test": torch.tensor([False, False, False, True]),
    }
    partition = build_bank_partition(accounts, ("a", "b", "c", "d"), {"a": 0, "b": 1, "c": 2, "d": 3}, y, masks)
    assert partition.bank_ids == ("B1", "B2")
    assert sorted(torch.cat(list(partition.client_node_indices.values())).tolist()) == [0, 1, 2, 3]
    assert partition.diagnostics["split_totals"]["train"]["client_sum"] == 2
    assert partition.diagnostics["split_totals"]["val"]["client_sum"] == 1
    assert partition.diagnostics["split_totals"]["test"]["client_sum"] == 1


def test_partition_rejects_null_empty_missing_and_duplicate_accounts():
    base = pd.DataFrame({"acct_id": ["a", "b"], "bank_id": ["B1", "B2"]})
    with pytest.raises(ValueError, match="null bank_id"):
        build_bank_partition(pd.DataFrame({"acct_id": ["a"], "bank_id": [None]}), ("a",), {"a": 0})
    with pytest.raises(ValueError, match="empty bank_id"):
        build_bank_partition(pd.DataFrame({"acct_id": ["a"], "bank_id": [" "]}), ("a",), {"a": 0})
    with pytest.raises(ValueError, match="no bank_id mapping"):
        build_bank_partition(base, ("a", "b", "c"), {"a": 0, "b": 1, "c": 2})
    with pytest.raises(ValueError, match="duplicate acct_id"):
        build_bank_partition(pd.DataFrame({"acct_id": ["a", "a"], "bank_id": ["B1", "B1"]}), ("a",), {"a": 0})
