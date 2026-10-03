import pytest
import torch

from model import build_transformer
from translate import encode_source, greedy_decode, load_model, translate_sentence

SEQ_LEN = 12


def tiny_model(tokenizer_src, tokenizer_tgt, d_model=16, share_weights=True, **kwargs):
    torch.manual_seed(0)
    return build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), SEQ_LEN, SEQ_LEN,
                             d_model=d_model, share_weights=share_weights, **kwargs).eval()


def test_encode_source_matches_dataset_layout(tokenizer_src):
    source, mask = encode_source("the cat sits", tokenizer_src, SEQ_LEN)
    assert source.shape == (1, SEQ_LEN) and mask.shape == (1, 1, 1, SEQ_LEN)
    assert source[0, 0].item() == tokenizer_src.token_to_id('[SOS]')
    assert source[0, 4].item() == tokenizer_src.token_to_id('[EOS]')
    assert mask.sum().item() == 5  # [SOS] + 3 words + [EOS]
    with pytest.raises(ValueError):
        encode_source("the cat sits", tokenizer_src, seq_len=4)


def test_greedy_decode_starts_with_sos_and_respects_max_len(tokenizer_src, tokenizer_tgt):
    model = tiny_model(tokenizer_src, tokenizer_tgt, N=1, h=4, d_ff=32)
    source, mask = encode_source("i eat rice", tokenizer_src, SEQ_LEN)
    sos, eos = tokenizer_tgt.token_to_id('[SOS]'), tokenizer_tgt.token_to_id('[EOS]')
    with torch.no_grad():
        out = greedy_decode(model, source, mask, sos, eos, SEQ_LEN, 'cpu')
    assert out[0].item() == sos
    assert 2 <= out.numel() <= SEQ_LEN
    # Generation stops at the first [EOS]
    assert eos not in out[:-1].tolist()


def test_greedy_decode_is_deterministic(tokenizer_src, tokenizer_tgt):
    model = tiny_model(tokenizer_src, tokenizer_tgt, N=1, h=4, d_ff=32)
    first = translate_sentence(model, "the dog runs fast", tokenizer_src, tokenizer_tgt, SEQ_LEN, 'cpu')
    second = translate_sentence(model, "the dog runs fast", tokenizer_src, tokenizer_tgt, SEQ_LEN, 'cpu')
    assert isinstance(first, str) and first == second


@pytest.mark.parametrize("share_weights", [True, False])
def test_load_model_restores_tied_and_untied_checkpoints(tmp_path, tokenizer_src, tokenizer_tgt, share_weights):
    # load_model uses the default depth/heads/d_ff and reads seq_len + d_model from the checkpoint
    model = tiny_model(tokenizer_src, tokenizer_tgt, d_model=16, share_weights=share_weights)
    path = tmp_path / "checkpoint.pt"
    torch.save({'model_state_dict': model.state_dict()}, path)

    loaded, seq_len = load_model(path, {}, tokenizer_src, tokenizer_tgt, 'cpu')
    assert seq_len == SEQ_LEN
    assert (loaded.projection_layer.proj.weight is loaded.tgt_embed.embedding.weight) == share_weights
    for (name, a), b in zip(model.state_dict().items(), loaded.state_dict().values()):
        assert torch.equal(a, b), name
