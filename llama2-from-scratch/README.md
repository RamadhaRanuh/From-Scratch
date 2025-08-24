# Llama 2 from Scratch
<img width="373" height="433" alt="1_CQs4ceLpN8tIN8QyezL2Ag" src="https://github.com/user-attachments/assets/d10e61de-3f9e-4d33-9697-d30333a36358" />

This repository contains a from-scratch implementation of the Llama 2 language model architecture in PyTorch. The focus is on providing a clear, concise, and technically detailed codebase that explains the inner workings of the model. This documentation serves as a guide to the `model.py` script, breaking down each component of the Llama 2 architecture as implemented.

## Model Architecture Overview

The implementation is a decoder-only transformer model, staying true to the Llama 2 design. It leverages several key architectural innovations for improved performance and efficiency:

  * **Pre-normalization** using **RMSNorm** for training stability.
  * **Rotary Positional Embeddings (RoPE)** to inject positional information.
  * **Grouped-Query Attention (GQA)** for efficient inference.
  * **SwiGLU** activation function in the feed-forward network.
  * **Key-Value (KV) Caching** for fast autoregressive generation.

-----

## Core Components (`model.py`)

### `ModelArgs`

The model's configuration is managed through the `ModelArgs` dataclass. This allows for easy definition and modification of the model's hyperparameters.

```python
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[float] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_ops: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None
```

  * **`dim`**: The embedding dimension of the model.
  * **`n_layers`**: The number of transformer blocks (EncoderBlock).
  * **`n_heads`**: The number of attention heads for the query.
  * **`n_kv_heads`**: The number of attention heads for the key and value, enabling Grouped-Query Attention. If `None`, it defaults to `n_heads`.
  * **`vocab_size`**: The size of the vocabulary. It is initialized to -1 and set during tokenizer loading.
  * **`ffn_dim_multiplier`**: A multiplier to adjust the hidden dimension in the FeedForward network.
  * **`norm_ops`**: The epsilon value used in RMSNorm for numerical stability.
  * **`max_batch_size`** and **`max_seq_len`**: Pre-allocated dimensions for the KV cache.

-----

### Rotary Positional Embeddings (RoPE)

Instead of traditional absolute or learned positional encodings, Llama 2 uses RoPE, which applies rotations to the query and key vectors to encode relative positional information.

#### `precompute_theta_pos_frequencies`

This function precomputes the complex numbers (`freqs_complex`) required for the rotary embeddings. The frequencies are calculated based on a chosen `theta` value (defaulting to 10000.0) and the head dimension.

  * It first calculates the `theta_i` values according to the formula: `theta_i = 10000^(-2(i-1)/dim)`.
  * An outer product is then performed between the sequence positions (`m`) and the `theta` values to get the phase angles (`m * theta_i`).
  * These angles are converted into complex numbers in polar form `R * exp(i * m * theta)`, where R=1. These are stored and reused for efficiency.

#### `apply_rotary_embeddings`

This function applies the precomputed rotary embeddings to the query (`xq`) and key (`xk`) tensors.

1.  The input tensor `x` (shape `B, Seq_Len, H, Head_Dim`) is treated as a sequence of complex numbers by reshaping it to `B, Seq_Len, H, Head_Dim/2, 2` and viewing it as a complex tensor.
2.  The precomputed `freqs_complex` are broadcasted to match the input tensor's shape.
3.  The complex multiplication `x_complex * freqs_complex` performs the rotation in the complex plane.
4.  The result is converted back to a real tensor and reshaped to its original dimensions.

-----

### Normalization: `RMSNorm`

Llama 2 uses RMSNorm instead of the standard LayerNorm. RMSNorm is computationally simpler and has been shown to be effective. It normalizes the input by its Root Mean Square and then scales it by a learnable `gamma` parameter (`weight`).

The normalization is defined as:
`x_norm = x / sqrt(mean(x^2) + epsilon)`
And the forward pass is:
`output = gamma * x_norm`

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensors):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)
```

-----

### Self-Attention with GQA and KV Cache

#### `SelfAttention`

This module implements the multi-head attention mechanism. It includes optimizations like Grouped-Query Attention (GQA) and a Key-Value (KV) cache.

1.  **GQA Setup**: The number of heads for Query (`n_heads_q`) can be different from the number of heads for Key and Value (`n_kv_heads`). The ratio `n_rep = n_heads_q // n_kv_heads` determines how many query heads share a single key/value head.

2.  **Projections**: The input tensor `x` is projected into query (`xq`), key (`xk`), and value (`xv`) tensors using linear layers (`wq`, `wk`, `wv`).

3.  **RoPE Application**: Rotary embeddings are applied to `xq` and `xk`.

4.  **KV Caching**: For efficient auto-regressive decoding, the computed `xk` and `xv` for the current token are cached. The cache stores keys and values up to `max_seq_len`. In each forward pass, the new key/value is appended to the cache, and the full sequence of keys/values is used for attention calculation.

    ```python
    self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
    self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv
    keys = self.cache_k[:batch_size, :start_pos + seq_len]
    values = self.cache_v[:batch_size, :start_pos + seq_len]
    ```

5.  **Repeating KV Heads**: To perform attention, the KV heads are repeated `n_rep` times to match the number of query heads. The `repeat_kv` function handles this expansion efficiently.

6.  **Scaled Dot-Product Attention**: The standard attention mechanism is applied:

      * Scores are computed: `scores = (xq @ K.T) / sqrt(d_k)`.
      * Scores are passed through a `softmax` function.
      * The output is a weighted sum of the values: `output = scores @ V`.

7.  **Final Projection**: The concatenated outputs from all heads are passed through a final linear layer (`wo`).

-----

### `FeedForward` Network

The feed-forward network in Llama 2 uses a SwiGLU (Swish-Gated Linear Unit) activation function, which has shown better performance than standard ReLU.

The network consists of three linear transformations (`w1`, `w2`, `w3`):
`output = w2(Swish(w1(x)) * w3(x))`

  * **`w1`** and **`w3`** project the input `x` into a hidden dimension.
  * The `Swish` activation (`F.silu`) is applied to the output of `w1`.
  * The result is element-wise multiplied by the output of `w3`.
  * **`w2`** projects the result back to the original embedding dimension.

The hidden dimension is calculated based on a multiple of the input dimension, as specified in the paper.

-----

### `EncoderBlock`

This class combines the Self-Attention and FeedForward modules into a single transformer block. The architecture employs pre-normalization, where the input is normalized *before* being passed to the attention and feed-forward layers. Residual connections are used around both sub-layers.

The forward pass is as follows:

1.  `h = x + self.attention(self.attention_norm(x))`
2.  `out = h + self.feed_forward(self.ffn_norm(h))`

-----

### `Transformer`

This is the main class that assembles the entire model.

1.  **Initialization**:

      * It initializes the token embedding layer (`tok_embeddings`).
      * It creates a stack of `n_layers` `EncoderBlock` modules.
      * It initializes the final normalization layer (`norm`) and the output linear layer (`output`) that maps the final embeddings to vocabulary logits.
      * Crucially, it precomputes the RoPE frequencies (`freqs_complex`) once during initialization for the maximum sequence length, saving computation during training and inference.

2.  **Forward Pass**:

      * It takes a batch of tokens (`tokens`) and a starting position (`start_pos`) for the KV cache as input.
      * The tokens are converted to embeddings.
      * The appropriate slice of precomputed RoPE frequencies is selected.
      * The embeddings are passed sequentially through all `EncoderBlock` layers.
      * The output of the final layer is normalized.
      * The final linear layer produces the logits over the vocabulary.
