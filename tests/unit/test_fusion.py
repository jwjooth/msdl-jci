"""Unit tests for Multi-Source Deep Learning architectures and Soft Gating."""

import torch

from msdl_jci.models.fusion import (
    AdaptiveSoftGatingFusionModel,
    LSTMMacroModel,
    LSTMNewsModel,
    MacroMLPBranch,
    NewsProjectionBranch,
    PureLSTMModel,
    SoftGatingNetwork,
    StaticFusionModel,
    TechnicalLSTMBranch,
)


def test_technical_lstm_branch():
    B, seq_len, in_dim = 4, 28, 7
    branch = TechnicalLSTMBranch(input_dim=in_dim, hidden_dim=64, dropout=0.2)
    x = torch.randn(B, seq_len, in_dim)
    out = branch(x)
    assert out.shape == (B, 64)


def test_macro_mlp_branch():
    B, in_dim = 4, 3
    branch = MacroMLPBranch(input_dim=in_dim, latent_dim=16)
    x = torch.randn(B, in_dim)
    out = branch(x)
    assert out.shape == (B, 16)


def test_news_projection_branch():
    B, in_dim = 4, 768
    branch = NewsProjectionBranch(input_dim=in_dim, proj_dim=64)
    x = torch.randn(B, in_dim)
    out = branch(x)
    assert out.shape == (B, 64)


def test_soft_gating_network_weights_sum_to_one():
    B, in_dim = 5, 144
    gating = SoftGatingNetwork(input_dim=in_dim, num_experts=3)
    x = torch.randn(B, in_dim)
    weights = gating(x)
    assert weights.shape == (B, 3)
    # Each row must sum to 1.0 (Softmax constraint)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B), atol=1e-5)
    # Each weight must be non-negative
    assert (weights >= 0.0).all()


def test_adaptive_soft_gating_fusion_forward_backward():
    B = 4
    model = AdaptiveSoftGatingFusionModel()
    x_tech = torch.randn(B, 28, 7)
    x_macro = torch.randn(B, 3)
    x_news = torch.randn(B, 768)

    logits, weights = model(x_tech, x_macro, x_news)
    assert logits.shape == (B, 1)
    assert weights.shape == (B, 3)

    target = torch.ones_like(logits)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    assert model.tech_branch.lstm.weight_ih_l0.grad is not None
    assert model.soft_gating.gating[0].weight.grad is not None


def test_ablation_baseline_models():
    B = 4
    x_tech = torch.randn(B, 28, 7)
    x_macro = torch.randn(B, 3)
    x_news = torch.randn(B, 768)

    # 1. Pure LSTM
    m1 = PureLSTMModel()
    out1, w1 = m1(x_tech)
    assert out1.shape == (B, 1)
    assert w1 is None

    # 2. LSTM + Macro
    m2 = LSTMMacroModel()
    out2, w2 = m2(x_tech, x_macro)
    assert out2.shape == (B, 1)
    assert w2 is None

    # 3. LSTM + News
    m3 = LSTMNewsModel()
    out3, w3 = m3(x_tech, None, x_news)
    assert out3.shape == (B, 1)
    assert w3 is None

    # 4. Static Fusion
    m4 = StaticFusionModel()
    out4, w4 = m4(x_tech, x_macro, x_news)
    assert out4.shape == (B, 1)
    assert w4 is None
