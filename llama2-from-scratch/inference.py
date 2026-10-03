import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    # Nucleus (top-p) sampling: sample only from the smallest set of tokens whose probability mass exceeds p
    # (B, Vocab_Size) -> (B, Vocab_Size), sorted from most to least likely
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # (B, Vocab_Size) running total of the sorted probabilities
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # Drop a token when the mass of the tokens before it already exceeds p.
    # Subtracting probs_sort keeps the token that crosses p, so the most likely token is always kept.
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    # Renormalize the kept probabilities so they sum to 1
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # (B, 1) index into the sorted list
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Map back to the vocabulary id: (B, 1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"No checkpoint files found in {checkpoints_dir}"
            # The 7B model ships as a single shard (consolidated.00.pth); larger models are split across GPUs
            assert len(checkpoints) == 1, "Only single-shard checkpoints (e.g. Llama 2 7B) are supported"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            # mmap avoids reading the whole ~13 GB file into RAM up front
            checkpoint = torch.load(ckpt_path, map_location="cpu", mmap=True)
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        # params.json keys match ModelArgs field names; batch size and context length are chosen at inference time
        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        # Meta's weights are stored in fp16. Build the model directly on the target device in half precision
        # (bf16 on CPU, where fp16 matmuls are poorly supported) so the 7B model never exists in fp32.
        dtype = torch.float16 if device == "cuda" else torch.bfloat16
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device(device):
                model = Transformer(model_args)
        finally:
            torch.set_default_dtype(default_dtype)

        if load_model:
            # The checkpoint carries a precomputed RoPE table that this model computes itself
            checkpoint.pop("rope.freqs", None)
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

        return LLaMA(model, tokenizer, model_args)

    def generate(self, prompt_tokens: List[List[int]], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None) -> List[List[int]]:
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        min_prompt_len = min(len(prompt) for prompt in prompt_tokens)
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens.
        # Prompts are left-aligned; the rest of each row is filled with pad_id until a token is generated there.
        pad_id = self.tokenizer.pad_id()
        device = self.model.tok_embeddings.weight.device
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise
        prev_pos = 0
        cur_iterator = tqdm(range(min_prompt_len, total_len), desc="Generating tokens", disable=None)
        for cur_pos in cur_iterator:
            # The first step feeds all tokens up to the shortest prompt's end at once (prefill);
            # every later step feeds only the newest token and reuses the KV cache for the rest.
            # (B, cur_pos - prev_pos) -> (B, cur_pos - prev_pos, Vocab_Size)
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                # The temperature is applied before the softmax: < 1 sharpens the distribution, > 1 flattens it
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the max probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token: rows whose prompt is longer keep their prompt token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Keep only the completion: drop the prompt and cut at max_gen_len
            prompt_len = len(prompt_tokens[prompt_index])
            current_prompt_tokens = current_prompt_tokens[prompt_len : prompt_len + max_gen_len]
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
        return out_tokens

    def text_completion(self, prompts: List[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        # Convert each prompt into tokens, starting with BOS as during pre-training
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        out_tokens = self.generate(prompt_tokens, temperature, top_p, max_gen_len)
        out_text = [self.tokenizer.decode(tokens) for tokens in out_tokens]
        return out_tokens, out_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with Meta's Llama 2 weights and this from-scratch model")
    parser.add_argument("--ckpt-dir", default="llama-2-7b", help="folder with consolidated.00.pth and params.json")
    parser.add_argument("--tokenizer", default="tokenizer.model", help="path to Meta's SentencePiece tokenizer.model")
    parser.add_argument("--prompt", action="append", help="prompt to complete (repeat for a batch)")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="context length to allocate for the KV cache")
    parser.add_argument("--max-gen-len", type=int, default=64, help="maximum number of new tokens per prompt")
    parser.add_argument("--temperature", type=float, default=0.6, help="0 = greedy decoding")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # Generated text can contain characters the console encoding (e.g. cp1252 on Windows) cannot show
    sys.stdout.reconfigure(errors="replace")
    torch.manual_seed(args.seed)
    prompts = args.prompt or [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
    ]
    model = LLaMA.build(
        checkpoints_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer,
        load_model=True,
        max_seq_len=args.max_seq_len,
        max_batch_size=len(prompts),
        device=args.device,
    )
    _, out_texts = model.text_completion(prompts, temperature=args.temperature, top_p=args.top_p, max_gen_len=args.max_gen_len)
    for prompt, completion in zip(prompts, out_texts):
        print(f"{prompt}{completion}")
        print("-" * 50)


if __name__ == "__main__":
    main()
