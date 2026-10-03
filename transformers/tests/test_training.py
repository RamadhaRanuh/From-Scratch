import pytest

from model import build_transformer, checkpoint_shares_weights
from train import learning_rate

D_MODEL, WARMUP = 512, 4000


def test_learning_rate_warms_up_linearly():
    assert learning_rate(2000, D_MODEL, WARMUP) == pytest.approx(learning_rate(4000, D_MODEL, WARMUP) / 2)
    assert learning_rate(1, D_MODEL, WARMUP) == pytest.approx(learning_rate(4000, D_MODEL, WARMUP) / 4000)


def test_learning_rate_peaks_at_warmup_then_decays_as_inverse_sqrt():
    peak = learning_rate(WARMUP, D_MODEL, WARMUP)
    assert peak == pytest.approx(D_MODEL ** -0.5 * WARMUP ** -0.5)  # ~7.0e-4 for the base model
    assert peak > learning_rate(WARMUP - 1, D_MODEL, WARMUP)
    assert peak > learning_rate(WARMUP + 1, D_MODEL, WARMUP)
    # 4x the steps after warmup -> half the learning rate
    assert learning_rate(4 * WARMUP, D_MODEL, WARMUP) == pytest.approx(peak / 2)


def test_learning_rate_factor_scales_schedule():
    assert learning_rate(100, D_MODEL, WARMUP, factor=2.0) == pytest.approx(2 * learning_rate(100, D_MODEL, WARMUP))


def tiny(share_weights):
    return build_transformer(50, 60, 10, 10, d_model=32, N=1, h=4, d_ff=64, share_weights=share_weights)


def test_shared_weights_are_one_parameter():
    model = tiny(share_weights=True)
    assert model.projection_layer.proj.weight is model.tgt_embed.embedding.weight
    assert model.src_embed.embedding.weight is not model.tgt_embed.embedding.weight

    untied = tiny(share_weights=False)
    n_tied = sum(p.numel() for p in model.parameters())
    n_untied = sum(p.numel() for p in untied.parameters())
    assert n_untied - n_tied == 60 * 32  # one (tgt_vocab, d_model) matrix fewer


def test_shared_weights_receive_gradient_from_both_uses():
    import torch

    model = tiny(share_weights=True)
    tokens = torch.tensor([[1, 2, 3]])
    logits = model.project(model.tgt_embed(tokens))
    logits.sum().backward()
    grad = model.tgt_embed.embedding.weight.grad
    # Rows of tokens never embedded still get gradient through the projection
    assert grad is not None and grad[10].abs().sum() > 0


def test_checkpoint_shares_weights_detects_tying():
    assert checkpoint_shares_weights(tiny(share_weights=True).state_dict())
    assert not checkpoint_shares_weights(tiny(share_weights=False).state_dict())
