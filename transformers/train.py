import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer, checkpoint_shares_weights
from translate import greedy_decode

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
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'], share_weights=config['share_weights'])
    return model


def learning_rate(step: int, d_model: int, warmup_steps: int, factor: float = 1.0) -> float:
    # Paper Eq. 3: lrate = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
    # Rises linearly for warmup_steps, then decays proportionally to 1/sqrt(step). Steps are 1-based.
    step = max(step, 1)
    return factor * d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)



def run_validation(model, val_dataloader, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    # Translate a few held-out sentences with greedy decoding so progress is visible beyond the loss curve
    model.eval()
    sos_id, eos_id = tokenizer_tgt.token_to_id('[SOS]'), tokenizer_tgt.token_to_id('[EOS]')
    with torch.no_grad():
        for count, batch in enumerate(val_dataloader, start=1):
            encoder_input = batch['encoder_input'].to(device) # (1, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (1, 1, 1, seq_len)
            model_out = greedy_decode(model, encoder_input, encoder_mask, sos_id, eos_id, max_len, device)
            predicted = tokenizer_tgt.decode(model_out.tolist())

            print_msg('-' * 80)
            print_msg(f"SOURCE:    {batch['src_text'][0]}")
            print_msg(f"TARGET:    {batch['tgt_text'][0]}")
            print_msg(f"PREDICTED: {predicted}")
            # TensorBoard renders text as Markdown; two trailing spaces + newline make a line break
            text = f"**source:** {batch['src_text'][0]}  \n**target:** {batch['tgt_text'][0]}  \n**predicted:** {predicted}"
            writer.add_text(f'validation/example_{count}', text, global_step)
            if count == num_examples:
                break
    writer.flush()


def train_model(config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    # Tensorboard
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    writer = SummaryWriter(config['experiment_name'])

    # The learning rate is set every step from the Eq. 3 schedule, so the value passed here is only a placeholder
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(1, config['d_model'], config['warmup_steps'], config['lr_factor']), betas=config['adam_betas'], eps=config['adam_eps'])
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weight_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        if checkpoint_shares_weights(state['model_state_dict']) != config['share_weights']:
            raise ValueError(f"{model_filename} was trained with share_weights={not config['share_weights']}; set config['share_weights'] to match")
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        # Checkpoints from before the paper schedule saved other Adam settings; the config is the source of truth
        for group in optimizer.param_groups:
            group['betas'] = config['adam_betas']
            group['eps'] = config['adam_eps']
        global_step = state['global_step']

    # Labels are target-language ids, so padding must be ignored using the target tokenizer's [PAD] id
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

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

            # Update the weights with the scheduled learning rate (global_step + 1 is the 1-based step number)
            lr = learning_rate(global_step + 1, config['d_model'], config['warmup_steps'], config['lr_factor'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            writer.add_scalar('learning rate', lr, global_step)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len'], device, batch_iterator.write, global_step, writer)

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
