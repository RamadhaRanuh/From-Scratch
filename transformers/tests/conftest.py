import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

# Make the project modules (model.py, dataset.py, ...) importable from the tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PAIRS = [
    {"translation": {"en": "the cat sits", "id": "kucing itu duduk"}},
    {"translation": {"en": "i eat rice", "id": "saya makan nasi"}},
    {"translation": {"en": "the dog runs fast", "id": "anjing itu berlari cepat"}},
]


def _build_tokenizer(lang):
    # Same recipe as train.get_or_build_tokenizer, trained on a toy corpus
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=1)
    tokenizer.train_from_iterator((p["translation"][lang] for p in PAIRS), trainer=trainer)
    return tokenizer


@pytest.fixture
def pairs():
    return PAIRS


@pytest.fixture
def tokenizer_src():
    return _build_tokenizer("en")


@pytest.fixture
def tokenizer_tgt():
    return _build_tokenizer("id")
