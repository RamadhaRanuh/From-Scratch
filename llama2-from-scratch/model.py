import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    # Field names match Meta's params.json, so ModelArgs(**params) works directly. Defaults are Llama 2 7B.
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for the queries
    n_kv_heads: Optional[int] = None # Number of heads for the K and V (None = same as n_heads, i.e. no GQA)
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str = None, theta: float = 10000.0):
    # As written in the paper, the dimension of the embedding must be even.
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameters
    # According to the formula theta_i = 10000 ^ (-2(i - 1)/dim) for i = [1,2,...,dim/2]
    # 2(i - 1) for i = 1..dim/2 is the sequence [0, 2, 4, ..., dim - 2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim))
    # Construct the positions (the "m" parameter)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product
    # Shape: (Seq_Len) outer_product (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(i * m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor):
    # Group consecutive pairs of dimensions into one complex number: (x0, x1) -> x0 + i*x1
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (Seq_Len, Head_Dim / 2) -> (1, Seq_Len, 1, Head_Dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiplying by exp(i * m * theta) rotates each pair by the angle m * theta
    # (B, Seq_Len, H, Head_Dim / 2) * (1, Seq_Len, 1, Head_Dim / 2) = (B, Seq_Len, H, Head_Dim / 2)
    x_rotated = x_complex * freqs_complex
    # (B, Seq_Len, H, Head_Dim / 2) -> (B, Seq_Len, H, Head_Dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim / 2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # KV cache. Non-persistent buffers follow the model across .to(device) / .half(), but are not saved
        # in the state_dict, so Meta's checkpoints load without them.
        self.register_buffer("cache_k", torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)), persistent=False)
        self.register_buffer("cache_v", torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor, mask: Optional[torch.Tensor]):
        batch_size, seq_len, _ = x.shape # (B, Seq_Len, Dim); Seq_Len is the prompt length on the first call, then 1

        # (B, Seq_Len, Dim) -> (B, Seq_Len, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_Dim)
        xv = self.wv(x)

        # (B, Seq_Len, H_Q * Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # RoPE is applied to the queries and keys only, never to the values
        # (B, Seq_Len, H_Q, Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex)
        # (B, Seq_Len, H_KV, Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex)

        # Replace the entry in cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # (B, Seq_Len_KV, H_KV, Head_Dim), where Seq_Len_KV = start_pos + Seq_Len
        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, :start_pos + seq_len]

        # Repeat the K and V heads for every Q in the same group
        # (B, Seq_Len_KV, H_KV, Head_Dim) -> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) -> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # (B, Seq_Len, H_Q, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # (B, H_Q, Seq_Len, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, Seq_Len, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            # (Seq_Len, Seq_Len_KV) additive mask: 0 where attention is allowed, -inf for future positions
            scores = scores + mask
        # (B, H_Q, Seq_Len, Seq_Len_KV) -> (B, H_Q, Seq_Len, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, Seq_Len, Seq_Len_KV) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, Seq_Len, Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim) -> (B, Seq_Len, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)

class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        hidden_dim = 4 * args.dim
        # SwiGLU has three weight matrices instead of two, so 2/3 of 4*dim keeps the parameter count the same
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim up to the next multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x

class TransformerBlock(nn.Module):
    # One decoder layer: Llama is decoder-only, so there is no cross-attention
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the self attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor, mask: Optional[torch.Tensor]):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex, mask
        )
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # Precomputed rotations for every position (twice max_seq_len, as in Meta's code). A non-persistent buffer
        # moves with the model to its device and is not part of the state_dict.
        freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device, theta=self.args.rope_theta)
        self.register_buffer("freqs_complex", freqs_complex, persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape

        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Feeding several tokens at once (the prompt) needs a causal mask. Row i is absolute position start_pos + i:
        # it may see every cached position plus the new positions up to itself.
        # A single new token may attend to everything in the cache, so it needs no mask.
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            # (Seq_Len, Seq_Len) -> (Seq_Len, start_pos + Seq_Len)
            mask = torch.hstack([torch.zeros((seq_len, start_pos), device=tokens.device), mask]).type_as(h)

        # Consecutively apply all the decoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex, mask)

        h = self.norm(h)
        # (B, Seq_Len, Dim) -> (B, Seq_Len, Vocab_Size)
        output = self.output(h).float()
        return output
