import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_weight_file_path, get_config

from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pathlib import Path
import warnings
import logging
from datasets import logging as datasets_logging

datasets_logging.set_verbosity_info()
logging.basicConfig(level=logging.INFO)

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def load_raw_dataset(config):
    lang_pair = f"{config['lang_src']}-{config['lang_tgt']}"
    dataset_path = Path(config['dataset_cache'])
    if dataset_path.exists():
        print(f"--> Loading cached dataset from {dataset_path}...")
        ds_full = load_from_disk(str(dataset_path))
    else:
        # First run: download once from the Hugging Face Hub, then cache in Arrow format for fast reloads
        print(f"--> Downloading '{config['datasource']}' ({lang_pair}) and caching to {dataset_path}...")
        ds_full = load_dataset(config['datasource'], lang_pair)
        ds_full.save_to_disk(str(dataset_path))
    return ds_full['train']

def get_ds(config):
    ds_raw = load_raw_dataset(config)
    print(f"--> Dataset loaded with {len(ds_raw)} samples.")
    # Build Tokenizers

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Drop pairs that cannot fit in seq_len: the encoder needs room for [SOS] + [EOS], the decoder for [SOS] (or [EOS] in the label)
    src_lengths = [len(e.ids) for e in tokenizer_src.encode_batch([item[config['lang_src']] for item in ds_raw['translation']])]
    tgt_lengths = [len(e.ids) for e in tokenizer_tgt.encode_batch([item[config['lang_tgt']] for item in ds_raw['translation']])]
    keep = [i for i, (s, t) in enumerate(zip(src_lengths, tgt_lengths)) if s + 2 <= config['seq_len'] and t + 1 <= config['seq_len']]
    print(f"--> Dropped {len(ds_raw) - len(keep)} pairs longer than seq_len={config['seq_len']}.")
    print(f'Max length of source sentence: {max(src_lengths)}')
    print(f'Max length of target sentence: {max(tgt_lengths)}')
    ds_raw = ds_raw.select(keep)

    # keep 90% training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])


    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])


    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model



def train_model(config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    print('testing model folder:', config['model_folder'])
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    print(train_dataloader)
    # Tensorboard
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    print('model written')
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps=1e-9)
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weight_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load.state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (B, seq_len)

            # (B, seq_len, tgt_vocab_size) -> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagation the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # save the model at the end of every epoch
        model_filename = get_weight_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    config = get_config()
    train_model(config)
