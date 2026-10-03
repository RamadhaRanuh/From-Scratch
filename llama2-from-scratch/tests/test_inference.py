import json

import torch
from sentencepiece import SentencePieceProcessor

from inference import LLaMA, sample_top_p
from model import ModelArgs, Transformer

PARAMS = {"dim": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "multiple_of": 16, "norm_eps": 1e-5, "vocab_size": -1}
MAX_SEQ_LEN = 24


def tiny_llama(tokenizer_path, max_batch_size=2):
    tokenizer = SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    args = ModelArgs(max_seq_len=MAX_SEQ_LEN, max_batch_size=max_batch_size, **{**PARAMS, "vocab_size": tokenizer.vocab_size()})
    torch.manual_seed(0)
    return LLaMA(Transformer(args).eval(), tokenizer, args)


def naive_greedy(model, prompt, max_gen_len, eos_id):
    # Reference decoder without incremental decoding: re-run the whole sequence from position 0 every step
    tokens = list(prompt)
    for _ in range(max_gen_len):
        logits = model(torch.tensor([tokens]), start_pos=0)
        next_token = logits[0, -1].argmax().item()
        if next_token == eos_id:
            break
        tokens.append(next_token)
    return tokens[len(prompt):]


def test_sample_top_p_only_samples_from_the_nucleus():
    torch.manual_seed(0)
    probs = torch.tensor([[0.05, 0.5, 0.15, 0.3]])
    # Sorted: 0.5 (id 1), 0.3 (id 3), 0.15 (id 2), 0.05 (id 0). With p=0.7 the nucleus is {1, 3}:
    # the mass before id 3 is 0.5 <= 0.7 (kept), the mass before id 2 is 0.8 > 0.7 (dropped)
    samples = {sample_top_p(probs.clone(), p=0.7).item() for _ in range(200)}
    assert samples == {1, 3}
    # p=0 keeps only the most likely token
    assert {sample_top_p(probs.clone(), p=0.0).item() for _ in range(20)} == {1}


def test_greedy_batched_generation_matches_naive_decoding(tokenizer_path):
    llama = tiny_llama(tokenizer_path)
    # Prompts of different lengths exercise left-aligned padding and keeping prompt tokens during prefill
    prompts = [llama.tokenizer.encode("the quick brown fox", add_bos=True),
               llama.tokenizer.encode("keys and values", add_bos=True)]
    assert len(prompts[0]) != len(prompts[1])

    generated = llama.generate(prompts, temperature=0, max_gen_len=8)

    for prompt, out in zip(prompts, generated):
        assert out == naive_greedy(llama.model, prompt, 8, llama.tokenizer.eos_id())


def test_text_completion_returns_text_within_max_gen_len(tokenizer_path):
    llama = tiny_llama(tokenizer_path)
    torch.manual_seed(1)
    out_tokens, out_text = llama.text_completion(["the kv cache", "a transformer predicts"], temperature=0.8, top_p=0.9, max_gen_len=5)
    assert len(out_tokens) == len(out_text) == 2
    for tokens, text in zip(out_tokens, out_text):
        assert len(tokens) <= 5
        assert llama.tokenizer.eos_id() not in tokens
        assert isinstance(text, str)


def test_build_loads_a_meta_style_checkpoint_folder(tmp_path, tokenizer_path):
    reference = tiny_llama(tokenizer_path, max_batch_size=1)
    # Same layout as Meta's download: consolidated.00.pth + params.json, with the extra rope.freqs tensor
    state = reference.model.state_dict()
    state["rope.freqs"] = torch.zeros(4)
    torch.save(state, tmp_path / "consolidated.00.pth")
    (tmp_path / "params.json").write_text(json.dumps(PARAMS))

    llama = LLaMA.build(str(tmp_path), tokenizer_path, load_model=True, max_seq_len=MAX_SEQ_LEN, max_batch_size=1, device="cpu")

    assert llama.args.vocab_size == reference.tokenizer.vocab_size()
    assert next(llama.model.parameters()).dtype == torch.bfloat16  # half precision on CPU
    for name, tensor in reference.model.state_dict().items():
        assert torch.equal(llama.model.state_dict()[name], tensor.to(torch.bfloat16)), name
