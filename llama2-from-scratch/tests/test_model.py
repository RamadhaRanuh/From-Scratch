import pytest
import torch

from model import (
    FeedForward,
    ModelArgs,
    RMSNorm,
    Transformer,
    apply_rotary_embeddings,
    precompute_theta_pos_frequencies,
    repeat_kv,
)


def tiny_args(**overrides):
    # dim 32 / 4 heads -> head_dim 8; 2 KV heads -> grouped-query attention with 2 queries per KV head
    args = dict(dim=32, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=50, multiple_of=16, max_batch_size=2, max_seq_len=16)
    args.update(overrides)
    return ModelArgs(**args)


def tiny_model(**overrides):
    torch.manual_seed(0)
    return Transformer(tiny_args(**overrides)).eval()


def test_forward_returns_logits_for_every_position():
    model = tiny_model()
    tokens = torch.randint(0, 50, (2, 5))
    logits = model(tokens, start_pos=0)
    assert logits.shape == (2, 5, 50)
    assert logits.dtype == torch.float32


def test_kv_cache_incremental_decoding_matches_full_forward():
    model = tiny_model()
    tokens = torch.randint(0, 50, (2, 8))
    full = model(tokens, start_pos=0)

    # Prefill the first 3 tokens at once, then feed the rest one at a time through the KV cache
    steps = [model(tokens[:, :3], start_pos=0)]
    for pos in range(3, 8):
        steps.append(model(tokens[:, pos:pos + 1], start_pos=pos))
    incremental = torch.cat(steps, dim=1)

    assert torch.allclose(full, incremental, atol=1e-5)


def test_causal_mask_hides_future_tokens():
    model = tiny_model()
    tokens = torch.randint(0, 50, (1, 6))
    changed = tokens.clone()
    changed[0, 4:] = (changed[0, 4:] + 1) % 50
    a, b = model(tokens, start_pos=0), model(changed, start_pos=0)
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-6)
    assert not torch.allclose(a[:, 4:], b[:, 4:])


def test_theta_follows_paper_formula():
    head_dim = 8
    freqs = precompute_theta_pos_frequencies(head_dim, seq_len=3)
    assert freqs.shape == (3, head_dim // 2)
    # Position m = 1 rotates pair i by theta_i = 10000^(-2(i-1)/d), i = 1..d/2
    expected = torch.tensor([10000 ** (-2 * (i - 1) / head_dim) for i in range(1, head_dim // 2 + 1)])
    assert torch.allclose(torch.angle(freqs[1]), expected, atol=1e-6)
    assert torch.allclose(freqs.abs(), torch.ones_like(freqs.abs()))


def test_rope_preserves_norm_and_depends_only_on_relative_position():
    head_dim = 8
    freqs = precompute_theta_pos_frequencies(head_dim, seq_len=20)
    torch.manual_seed(0)
    q, k = torch.randn(head_dim), torch.randn(head_dim)

    def rotate(v, pos):
        # (1, 1, 1, Head_Dim) at a single position
        return apply_rotary_embeddings(v.view(1, 1, 1, head_dim), freqs[pos:pos + 1]).view(head_dim)

    assert torch.allclose(rotate(q, 7).norm(), q.norm(), atol=1e-5)
    # <R_m q, R_n k> depends only on m - n: (5, 2) and (15, 12) give the same score
    assert torch.allclose(rotate(q, 5) @ rotate(k, 2), rotate(q, 15) @ rotate(k, 12), atol=1e-5)


def test_repeat_kv_duplicates_each_kv_head_for_its_query_group():
    x = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).view(2, 3, 2, 4)  # (B, Seq, H_KV=2, Head_Dim)
    out = repeat_kv(x, n_rep=3)
    assert out.shape == (2, 3, 6, 4)
    for head in range(6):
        assert torch.equal(out[:, :, head], x[:, :, head // 3])
    assert repeat_kv(x, 1) is x


def test_rmsnorm_matches_formula():
    norm = RMSNorm(4, eps=1e-5)
    norm.weight.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    rms = torch.sqrt((x ** 2).mean() + 1e-5)
    assert torch.allclose(norm(x), x / rms * norm.weight)


def test_grouped_query_attention_projection_shapes():
    attention = tiny_model().layers[0].attention
    assert attention.wq.weight.shape == (4 * 8, 32)
    assert attention.wk.weight.shape == (2 * 8, 32)
    assert attention.wv.weight.shape == (2 * 8, 32)
    assert attention.wo.weight.shape == (32, 4 * 8)
    # n_kv_heads=None means plain multi-head attention, as in Llama 2 7B
    assert tiny_model(n_kv_heads=None).layers[0].attention.wk.weight.shape == (32, 32)


@pytest.mark.parametrize("dim, multiple_of, ffn_dim_multiplier, hidden", [
    (4096, 256, None, 11008),  # Llama 2 7B
    (8192, 4096, 1.3, 28672),  # Llama 2 70B
])
def test_swiglu_hidden_size_matches_released_models(dim, multiple_of, ffn_dim_multiplier, hidden):
    with torch.device("meta"):  # shapes only, no memory allocated
        ffn = FeedForward(ModelArgs(dim=dim, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier))
    assert ffn.w1.weight.shape == (hidden, dim)
    assert ffn.w2.weight.shape == (dim, hidden)


def test_state_dict_keys_match_meta_checkpoint():
    keys = set(tiny_model().state_dict())
    expected_layer = {"attention.wq.weight", "attention.wk.weight", "attention.wv.weight", "attention.wo.weight",
                      "feed_forward.w1.weight", "feed_forward.w2.weight", "feed_forward.w3.weight",
                      "attention_norm.weight", "ffn_norm.weight"}
    expected = {"tok_embeddings.weight", "norm.weight", "output.weight"}
    expected |= {f"layers.{n}.{k}" for n in range(2) for k in expected_layer}
    # KV cache and RoPE table are non-persistent buffers, so they are not in the checkpoint
    assert keys == expected


def test_model_args_accepts_meta_params_json():
    params = {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": -1}
    args = ModelArgs(max_seq_len=512, max_batch_size=1, **params)
    assert args.norm_eps == 1e-5 and args.n_kv_heads is None
