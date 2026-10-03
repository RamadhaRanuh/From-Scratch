import sys
from pathlib import Path

import pytest
import sentencepiece as spm

# Make the project modules (model.py, inference.py) importable from the tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a transformer predicts the next token from the previous tokens",
    "rotary embeddings rotate queries and keys by their position",
    "grouped query attention shares keys and values across heads",
    "the kv cache stores keys and values of past tokens",
]


@pytest.fixture(scope="session")
def tokenizer_path(tmp_path_factory):
    # A tiny SentencePiece model with Llama's special-token layout: <unk>=0, <s>=1, </s>=2, no pad (pad_id=-1)
    folder = tmp_path_factory.mktemp("tokenizer")
    corpus = folder / "corpus.txt"
    corpus.write_text("\n".join(CORPUS * 4), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(folder / "tokenizer"), vocab_size=64, model_type="bpe",
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1, hard_vocab_limit=False, minloglevel=2,
    )
    return str(folder / "tokenizer.model")
