import pytest
import torch

from dataset import BilingualDataset, causal_mask

SEQ_LEN = 8


def test_causal_mask_is_lower_triangular():
    mask = causal_mask(4)
    assert mask.shape == (1, 4, 4)
    assert torch.equal(mask[0], torch.tril(torch.ones(4, 4, dtype=torch.bool)))


def test_item_layout_and_masks(pairs, tokenizer_src, tokenizer_tgt):
    ds = BilingualDataset(pairs, tokenizer_src, tokenizer_tgt, "en", "id", SEQ_LEN)
    item = ds[0]  # "the cat sits" -> "kucing itu duduk"
    sos, eos, pad = (tokenizer_tgt.token_to_id(t) for t in ("[SOS]", "[EOS]", "[PAD]"))
    src_ids = tokenizer_src.encode("the cat sits").ids
    tgt_ids = tokenizer_tgt.encode("kucing itu duduk").ids

    # Encoder: [SOS] sentence [EOS] [PAD]...
    assert item["encoder_input"].tolist() == [sos, *src_ids, eos] + [pad] * (SEQ_LEN - len(src_ids) - 2)
    # Decoder input: [SOS] sentence [PAD]...  /  label: sentence [EOS] [PAD]...  (shifted by one)
    assert item["decoder_input"].tolist() == [sos, *tgt_ids] + [pad] * (SEQ_LEN - len(tgt_ids) - 1)
    assert item["label"].tolist() == [*tgt_ids, eos] + [pad] * (SEQ_LEN - len(tgt_ids) - 1)

    assert item["encoder_mask"].shape == (1, 1, SEQ_LEN)
    assert item["encoder_mask"][0, 0].tolist() == [1] * (len(src_ids) + 2) + [0] * (SEQ_LEN - len(src_ids) - 2)

    decoder_mask = item["decoder_mask"]
    assert decoder_mask.shape == (1, SEQ_LEN, SEQ_LEN)
    n_real = len(tgt_ids) + 1
    # Row i may attend to columns <= i, and never to padding columns
    assert decoder_mask[0, 2].tolist() == [1, 1, 1] + [0] * (SEQ_LEN - 3)
    assert decoder_mask[0, -1].tolist() == [1] * n_real + [0] * (SEQ_LEN - n_real)


def test_too_long_sentence_raises(pairs, tokenizer_src, tokenizer_tgt):
    ds = BilingualDataset(pairs, tokenizer_src, tokenizer_tgt, "en", "id", seq_len=4)
    with pytest.raises(ValueError):
        ds[2]  # "the dog runs fast" needs 4 + 2 positions
