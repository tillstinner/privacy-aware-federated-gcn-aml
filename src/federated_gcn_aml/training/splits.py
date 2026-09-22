import torch
import numpy as np
import pandas as pd

def random_node_split_stratified(y: torch.Tensor, train=0.6, val=0.2, test=0.2, seed=42):
    """
    Stratified node split to not end up with 0 positives in val/test.
    """

    assert abs(train + val + test - 1.0) < 1e-9

    rng = np.random.default_rng(seed)

    # Split y into indices of pos / neg samples
    y_np = y.cpu().numpy()
    idx_pos = np.where(y_np == 1)[0]
    idx_neg = np.where(y_np == 0)[0]

    # Shuffle indices to prevent any bias due to order
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    # Splits indices based on given rations
    def split_indices(idx):
        n = len(idx)
        n_train = int(n * train)
        n_val = int(n * val)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        return train_idx, val_idx, test_idx

    pos_tr, pos_va, pos_te = split_indices(idx_pos)
    neg_tr, neg_va, neg_te = split_indices(idx_neg)

    # Merge splits across both classes
    train_idx = np.concatenate([pos_tr, neg_tr])
    val_idx = np.concatenate([pos_va, neg_va])
    test_idx = np.concatenate([pos_te, neg_te])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    # Setup boolean masks
    N = y.shape[0]
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    train_mask[torch.from_numpy(train_idx)] = True
    val_mask[torch.from_numpy(val_idx)] = True
    test_mask[torch.from_numpy(test_idx)] = True

    return train_mask, val_mask, test_mask

def temporal_edge_split_by_timestamp(tx_df, id2idx, train=0.6, val=0.2, test=0.2, time_col="timestamp"):
    """
    Create edge masks by sorting transactions by timestamp
    """

    assert time_col in tx_df.columns

    tx_sorted = tx_df.sort_values(time_col)

    m = len(tx_sorted)
    m_tr = int(train * m)
    m_va = int(val * m)

    tr = tx_sorted[:m_tr]
    va = tx_sorted[m_tr:m_tr + m_va]
    te = tx_sorted[m_tr + m_va:]

    return tr, va ,te
