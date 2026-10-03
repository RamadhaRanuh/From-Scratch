import torch

from dataset import causal_mask
from model import build_transformer

SRC_VOCAB, TGT_VOCAB, SEQ_LEN, D_MODEL = 50, 60, 10, 32


def tiny_transformer():
    torch.manual_seed(0)
    model = build_transformer(SRC_VOCAB, TGT_VOCAB, SEQ_LEN, SEQ_LEN, d_model=D_MODEL, N=2, h=4, dropout=0.0, d_ff=64)
    return model.eval()


def test_encode_decode_project_shapes():
    model = tiny_transformer()
    src = torch.randint(0, SRC_VOCAB, (2, SEQ_LEN))
    tgt = torch.randint(0, TGT_VOCAB, (2, SEQ_LEN))
    src_mask = torch.ones(2, 1, 1, SEQ_LEN, dtype=torch.int)
    tgt_mask = causal_mask(SEQ_LEN).int().unsqueeze(0)  # (1, 1, seq_len, seq_len)

    encoder_output = model.encode(src, src_mask)
    assert encoder_output.shape == (2, SEQ_LEN, D_MODEL)

    decoder_output = model.decode(encoder_output, src_mask, tgt, tgt_mask)
    assert decoder_output.shape == (2, SEQ_LEN, D_MODEL)

    logits = model.project(decoder_output)
    assert logits.shape == (2, SEQ_LEN, TGT_VOCAB)


def test_causal_mask_hides_future_target_tokens():
    model = tiny_transformer()
    src = torch.randint(0, SRC_VOCAB, (1, SEQ_LEN))
    src_mask = torch.ones(1, 1, 1, SEQ_LEN, dtype=torch.int)
    tgt_mask = causal_mask(SEQ_LEN).int().unsqueeze(0)
    tgt = torch.randint(0, TGT_VOCAB, (1, SEQ_LEN))
    tgt_changed = tgt.clone()
    tgt_changed[0, 5:] = (tgt_changed[0, 5:] + 1) % TGT_VOCAB  # change only positions 5..end

    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
        out = model.decode(encoder_output, src_mask, tgt, tgt_mask)
        out_changed = model.decode(encoder_output, src_mask, tgt_changed, tgt_mask)

    # Positions before 5 cannot see the changed tokens, so their outputs are identical
    assert torch.allclose(out[:, :5], out_changed[:, :5], atol=1e-6)
    assert not torch.allclose(out[:, 5:], out_changed[:, 5:])


def test_padding_mask_hides_padded_source_tokens():
    model = tiny_transformer()
    src = torch.randint(0, SRC_VOCAB, (1, SEQ_LEN))
    src_padded_differently = src.clone()
    src_padded_differently[0, 6:] = (src[0, 6:] + 1) % SRC_VOCAB
    src_mask = torch.ones(1, 1, 1, SEQ_LEN, dtype=torch.int)
    src_mask[..., 6:] = 0  # treat positions 6..end as padding

    with torch.no_grad():
        out = model.encode(src, src_mask)
        out_other = model.encode(src_padded_differently, src_mask)

    # Real (unmasked) positions never attend to padding, so they are unaffected by its content
    assert torch.allclose(out[:, :6], out_other[:, :6], atol=1e-6)
