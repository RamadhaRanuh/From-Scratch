import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import get_config
from dataset import causal_mask
from model import build_transformer, checkpoint_shares_weights


def greedy_decode(model, source, source_mask, sos_id: int, eos_id: int, max_len: int, device):
    # Encode the source once; the decoder then re-reads it through cross-attention at every step
    encoder_output = model.encode(source, source_mask) # (1, seq_len, d_model)
    # Start the target with [SOS] and append the most likely next token until [EOS] or max_len
    decoder_input = torch.full((1, 1), sos_id, dtype=source.dtype, device=device) # (1, 1)
    while decoder_input.size(1) < max_len:
        # Causal mask over the tokens generated so far: (1, cur_len, cur_len)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        decoder_output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask) # (1, cur_len, d_model)
        # Only the last position predicts the next token: (1, d_model) -> (1, tgt_vocab_size) -> (1,)
        next_token = model.project(decoder_output[:, -1]).argmax(dim=-1)
        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
        if next_token.item() == eos_id:
            break
    return decoder_input.squeeze(0) # (generated_len,)


def encode_source(text: str, tokenizer_src, seq_len: int):
    # Same layout as BilingualDataset: [SOS] tokens [EOS] [PAD]... and a (1, 1, 1, seq_len) padding mask
    sos, eos, pad = (tokenizer_src.token_to_id(t) for t in ('[SOS]', '[EOS]', '[PAD]'))
    ids = tokenizer_src.encode(text).ids
    num_padding = seq_len - len(ids) - 2
    if num_padding < 0:
        raise ValueError(f'Sentence has {len(ids)} tokens; at most {seq_len - 2} fit in seq_len={seq_len}')
    source = torch.tensor([[sos, *ids, eos] + [pad] * num_padding], dtype=torch.int64) # (1, seq_len)
    source_mask = (source != pad).unsqueeze(0).unsqueeze(0).int() # (1, 1, 1, seq_len)
    return source, source_mask


@torch.no_grad()
def translate_sentence(model, text: str, tokenizer_src, tokenizer_tgt, seq_len: int, device) -> str:
    model.eval()
    source, source_mask = encode_source(text, tokenizer_src, seq_len)
    output_ids = greedy_decode(model, source.to(device), source_mask.to(device),
                               tokenizer_tgt.token_to_id('[SOS]'), tokenizer_tgt.token_to_id('[EOS]'), seq_len, device)
    # decode() drops the special tokens ([SOS], [EOS])
    return tokenizer_tgt.decode(output_ids.tolist())


def load_model(checkpoint_path, config, tokenizer_src, tokenizer_tgt, device):
    state = torch.load(checkpoint_path, map_location=device)
    model_state = state['model_state_dict']
    # Read the shape-defining settings from the checkpoint itself, so older checkpoints still load
    seq_len = model_state['src_pos.pe'].shape[1]
    d_model = model_state['src_pos.pe'].shape[2]
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), seq_len, seq_len, d_model,
                              share_weights=checkpoint_shares_weights(model_state)).to(device)
    model.load_state_dict(model_state)
    return model, seq_len


def latest_checkpoint(config):
    checkpoints = sorted(Path(config['model_folder']).glob(f"{config['model_filename']}*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {config['model_folder']}/; train first or pass --checkpoint")
    return checkpoints[-1]


def main():
    config = get_config()
    parser = argparse.ArgumentParser(description=f"Translate {config['lang_src']} -> {config['lang_tgt']} with a trained Transformer")
    parser.add_argument('text', help='sentence to translate')
    parser.add_argument('--checkpoint', help='path to a .pt checkpoint (default: last one in the weights folder)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))
    checkpoint = args.checkpoint or latest_checkpoint(config)
    model, seq_len = load_model(checkpoint, config, tokenizer_src, tokenizer_tgt, device)

    print(f'Checkpoint: {checkpoint}')
    print(f'SOURCE:    {args.text}')
    print(f'PREDICTED: {translate_sentence(model, args.text, tokenizer_src, tokenizer_tgt, seq_len, device)}')


if __name__ == '__main__':
    main()
