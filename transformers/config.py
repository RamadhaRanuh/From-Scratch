from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        # Paper section 5.3: Adam(beta1=0.9, beta2=0.98, eps=1e-9) with warmup + inverse-sqrt decay (Eq. 3)
        "warmup_steps": 4000,
        "lr_factor": 1.0, # multiplies the Eq. 3 learning rate; 1.0 reproduces the paper
        "adam_betas": (0.9, 0.98),
        "adam_eps": 1e-9,
        "seq_len": 350,
        "d_model": 512,
        "share_weights": True, # paper section 3.4: tie target embedding and pre-softmax projection
        "lang_src": "en",
        "lang_tgt": "id",
        "datasource": "Helsinki-NLP/opus-100",
        "dataset_cache": "opus-100-fast-cache",
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weight_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_filename']
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)
