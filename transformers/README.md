# Transformer from Scratch

This project is a comprehensive implementation of the Transformer model from the ground up, as detailed in the seminal paper "Attention Is All You Need." The entire architecture is built using PyTorch, with a focus on clarity and a modular structure. The primary application demonstrated in this codebase is Neural Machine Translation (NMT), specifically for translating from English to Indonesian.

## Features

* **Complete Transformer Architecture**: All core components, including Multi-Head Self-Attention, Positional Encoding, Encoder/Decoder stacks, and Layer Normalization, are implemented from scratch in `model.py`.
* **Neural Machine Translation Task**: The model is set up and trained on the Helsinki-NLP/opus-100 dataset for English-to-Indonesian translation.
* **Efficient Data Handling**: Utilizes Hugging Face's `datasets` library for loading data and `tokenizers` for building custom Word-Level tokenizers. It includes a `BilingualDataset` class to preprocess, tokenize, and batch the data efficiently.
* **Checkpointing**: The training script saves the model's state after each epoch, allowing for easy resumption of training.
* **Configuration-Driven**: All hyperparameters and settings are centralized in `config.py` for easy modification and experimentation.
* **TensorBoard Integration**: Logs the training loss for real-time monitoring and visualization of the training process.

## File Structure

The project is organized into several key files:

| File                | Description                                                                                                                   |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `train.py`          | The main executable script that handles data loading, model building, and the entire training and validation process.      |
| `model.py`          | Contains the complete implementation of the Transformer model architecture, including all its sub-layers and components. |
| `dataset.py`        | Defines the `BilingualDataset` class responsible for tokenizing, padding, and creating masks for the source and target sentences. |
| `config.py`         | A centralized file for all hyperparameters, model settings, and file paths.                                               |
| `requirements.txt`  | Lists all the Python dependencies required to run the project.                                                          |
| `tokenizer_en.json` | The saved tokenizer file for the English language.                                                                      |
| `tokenizer_id.json` | The saved tokenizer file for the Indonesian language.                                                                     |

## How to Run

Follow these steps to set up and run the training process.

### 1. Install Dependencies

First, ensure you have Python installed. Then, install all the required libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset and Tokenizers

The `train.py` script is designed to automate data and tokenizer preparation. The first time you run it, it will:
1.  Download the `Helsinki-NLP/opus-100` dataset for the `en-id` language pair.
2.  Cache the dataset in a local directory named `./opus-100-fast-cache` for faster subsequent loads.
3.  Train custom `en` and `id` tokenizers from the dataset and save them as `tokenizer_en.json` and `tokenizer_id.json`.

### 3. Start Training

To begin training the model, simply run the `train.py` script:

```bash
python train.py
```

The script will use the configuration from `config.py`, initialize the model, and start the training loop. You will see a progress bar from `tqdm` for each epoch.

### 4. Monitor with TensorBoard

You can monitor the training loss in real-time using TensorBoard. Run the following command in a separate terminal:

```bash
tensorboard --logdir=runs
```

### 5. Resume Training

The script saves a checkpoint file in the `weights` directory after each epoch. To resume training from a specific checkpoint, modify the `preload` value in `config.py`:

```python
# in config.py
def get_config():
    return {
        # ...
        "preload": "10", # epoch number to load
        # ...
    }
```
