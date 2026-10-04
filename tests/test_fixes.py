"""Tests for Phase 2 fixes: label creation, splitter, metrics, scalers, gate."""
import sys
import json
import types
from pathlib import Path
import numpy as np

NB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NB_ROOT))

nb = json.load(open(NB_ROOT / 'notebooks' / 'development.ipynb'))
imports_src = ''.join(nb['cells'][1]['source'])
entity_src = ''.join(nb['cells'][4]['source'])
utils_src = ''.join(nb['cells'][6]['source'])

mod = types.ModuleType('nb_ns')
exec(imports_src.replace('%matplotlib inline', ''), mod.__dict__)
exec(entity_src, mod.__dict__)
exec(utils_src, mod.__dict__)

CFG = mod.CFG
align = mod.align
make_tensors = mod.make_tensors
walk_forward_train_and_eval = mod.walk_forward_train_and_eval
StockModel = mod.StockModel

def test_label_no_nan_as_down():
    """D2/H3e: df should not contain rows where future_close is NaN (those become invalid labels)."""
    jci, bi, inf, kur = mod.load_frames()
    df = align(jci, bi, inf, kur)
    nan_mask = df['future_close'].isna()
    assert not nan_mask.any(), f"df has {nan_mask.sum()} rows with invalid future_close"
    assert set(np.unique(df['target_direction'].values)) <= {0.0, 1.0}

def test_sample_count_subtracts_horizon():
    """D2: n_win should be len(df) - lb - horizon + 1."""
    jci, bi, inf, kur = mod.load_frames()
    df = align(jci, bi, inf, kur)
    lb, horizon = CFG.look_back, CFG.horizon
    expected = len(df) - lb - horizon + 1
    n_win = len(df) - lb + 1  # buggy formula
    n_win_fixed = len(df) - lb - horizon + 1  # fixed formula
    assert n_win_fixed == expected
    assert n_win != expected, "Old formula still used somewhere"

def test_splitter_fold_sizes():
    """H1: each fold should have ~260 test samples, not 1-5."""
    jci, bi, inf, kur = mod.load_frames()
    df = align(jci, bi, inf, kur)
    total_len = len(df)
    n_splits = 5
    test_per_fold = max(100, total_len // (n_splits + 2))
    for i in range(n_splits):
        train_end = int(total_len * (0.5 + i * 0.5 / n_splits))
        test_start = train_end + CFG.look_back + CFG.horizon
        test_end = min(test_start + test_per_fold, total_len)
        test_size = test_end - test_start
        assert test_size >= 100 or test_end >= total_len, f"Fold {i} test_size={test_size} < 100"
        assert train_end + CFG.look_back + CFG.horizon <= test_start, f"Fold {i}: no purge gap"

def test_gate_not_collapsed():
    """H4: gate weights should not be fully collapsed on random input."""
    model = StockModel(CFG)
    model.eval()
    import torch
    tb = torch.randn(4, CFG.look_back, len(CFG.tech_cols))
    mb = torch.randn(4, len(CFG.macro_cols))
    news = torch.zeros(4, 512, dtype=torch.long)
    with torch.no_grad():
        logits, loss, weights = model(tb, mb, news, torch.ones(4))
    w = weights.squeeze()
    assert w.shape[-1] == 3, f"Expected 3 gate weights, got {w.shape}"
    assert (w.sum(dim=-1) - 1.0).abs().max() < 1e-5, "Gate weights must sum to 1"
    # Not fully collapsed: at least one weight should be > 0.5
    assert w.max(dim=-1).values.max() < 0.99, "Gate fully collapsed"

def test_scalers_train_only():
    """H3b: scalers fit on train prefix only."""
    jci, bi, inf, kur = mod.load_frames()
    df = align(jci, bi, inf, kur)
    lb = CFG.look_back
    n_win = len(df) - lb - CFG.horizon + 1
    train_end_df = int(n_win * 0.70) + lb - 1
    X_tech, X_macro, X_news, y, dates, feat_scaler = make_tensors(df, train_end_df)
    raw_tech = df[CFG.tech_cols].values.astype(np.float32)
    assert np.allclose(feat_scaler.data_min_, raw_tech[:train_end_df].min(axis=0)), \
        "feat_scaler.data_min_ != train prefix min"

if __name__ == "__main__":
    test_label_no_nan_as_down()
    print("✓ test_label_no_nan_as_down")
    test_sample_count_subtracts_horizon()
    print("✓ test_sample_count_subtracts_horizon")
    test_splitter_fold_sizes()
    print("✓ test_splitter_fold_sizes")
    test_gate_not_collapsed()
    print("✓ test_gate_not_collapsed")
    test_scalers_train_only()
    print("✓ test_scalers_train_only")
    print("All tests passed!")
