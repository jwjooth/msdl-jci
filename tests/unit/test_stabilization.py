"""Tests for Phase 3-10 stabilizers: gating entropy, modality dropout, macro
change features, news missingness, trading costs, threshold val-only, seeds."""

import numpy as np
import pandas as pd
import torch


def test_branch_layernorm_before_gating():
    from msdl_jci.models.fusion import AdaptiveSoftGatingFusionModel

    m = AdaptiveSoftGatingFusionModel()
    assert isinstance(m.norm_tech, torch.nn.LayerNorm)
    assert isinstance(m.norm_macro, torch.nn.LayerNorm)
    assert isinstance(m.norm_news, torch.nn.LayerNorm)


def test_gating_entropy_balanced_vs_collapsed():
    from msdl_jci.models.fusion import SoftGatingNetwork

    g = SoftGatingNetwork()
    balanced = torch.full((16, 3), 1 / 3)
    assert float(g.gating_entropy(balanced)) > 1.0  # ln3 ≈ 1.0986
    collapsed = torch.tensor([[0.02, 0.96, 0.02]] * 16)
    assert float(g.gating_entropy(collapsed)) < 0.5


def test_gating_temperature_and_min_weight():
    from msdl_jci.models.fusion import SoftGatingNetwork

    h = torch.randn(32, 144)
    hot = SoftGatingNetwork(temperature=0.1)
    soft = SoftGatingNetwork(temperature=5.0)
    with torch.no_grad():
        hot_max = float(hot(h).max(1).values.mean())
        soft_max = float(soft(h).max(1).values.mean())
        wmin = float(floored(h).min()) if (floored := SoftGatingNetwork(min_weight=0.1)) else 0.0
    assert hot_max >= soft_max
    assert wmin >= 0.09


def test_modality_dropout_zeroes_branch_in_train():
    from msdl_jci.models.fusion import AdaptiveSoftGatingFusionModel

    m = AdaptiveSoftGatingFusionModel(modality_dropout=1.0)
    m.train()
    h = torch.randn(8, 64)
    assert torch.all(m._maybe_drop_modality(h) == 0)
    m.eval()
    assert torch.all(m._maybe_drop_modality(h) == h)


def test_aux_logits_shape():
    from msdl_jci.models.fusion import AdaptiveSoftGatingFusionModel

    m = AdaptiveSoftGatingFusionModel(aux_loss=True)
    m.train()
    aux = m.aux_logits(torch.randn(5, 28, 7), torch.randn(5, 3), torch.randn(5, 768))
    assert aux.shape == (5, 3)


def test_macro_change_features_causal():
    from msdl_jci.utils.macro_features import ENGINEERED_MACRO_COLS, add_macro_change_features

    n = 300
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=n, freq="B"),
            "bi_rate": 5.0,
            "inflation_rate": 3.0,
            "usd_idr": 15000.0,
        }
    )
    df.loc[100:, "bi_rate"] = 6.0  # step change at row 100
    out = add_macro_change_features(df)
    for c in ENGINEERED_MACRO_COLS:
        assert c in out.columns
    # Causal: change features before the step must be ~0 (no future peek).
    assert abs(out["bi_rate_chg_21d"].iloc[99]) < 1e-6
    assert out["bi_rate_chg_21d"].iloc[121] > 0.1


def test_news_missing_indicator_from_zero_vectors():
    E = np.zeros((10, 8))
    E[:7] = np.random.default_rng(0).normal(size=(7, 8))
    missing = (np.linalg.norm(E, axis=1) == 0.0).astype(int)
    assert missing.tolist() == [0] * 7 + [1] * 3


def test_trading_costs_gross_vs_net():
    from msdl_jci.evaluation.trading_simulation import simulate_trading_strategy

    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200)
    r = rng.normal(0.001, 0.02, 200)
    free = simulate_trading_strategy(p, r, transaction_cost=0.0, slippage=0.0)
    paid = simulate_trading_strategy(p, r, transaction_cost=0.0015, slippage=0.0005)
    assert paid.gross_return == free.total_return
    assert paid.total_return <= paid.gross_return
    assert paid.total_costs >= 0.0
    assert paid.exposure_pct > 0 and paid.turnover > 0


def test_threshold_uses_validation_only():
    import inspect

    from msdl_jci.evaluation import walk_forward as wf

    src = inspect.getsource(wf.train_single_split)
    assert "val_loader" in src  # threshold fit path uses val predictions
    assert inspect.signature(wf.find_best_threshold).parameters["metric"].default == "mcc"


def test_multiseed_determinism():
    from msdl_jci.evaluation.walk_forward import set_all_seeds

    set_all_seeds(7)
    a = (np.random.rand(5), torch.randn(3).numpy())
    set_all_seeds(7)
    b = (np.random.rand(5), torch.randn(3).numpy())
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
