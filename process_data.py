from datasets import load_dataset, load_from_disk
import time
import os

DATASET_NAME = 'Helsinki-NLP/opus-100'
LANG_PAIR = 'en-id'
FAST_DATASET_PATH = './opus-100-fast-cache'

if not os.path.exists(FAST_DATASET_PATH):
    print("Dataset not found in fast cache. Processing and saving for the first time...")

    # This is the slow load, which we only do ONCE.
    start_time = time.time()
    ds = load_dataset(DATASET_NAME, LANG_PAIR)
    print(f"Initial slow load took {time.time() - start_time:.2f} seconds.")

    # Save it to a new location in a super-fast format.
    print("Saving dataset to a faster format...")
    start_time = time.time()
    ds.save_to_disk(FAST_DATASET_PATH)
    print(f"Saving took {time.time() - start_time:.2f} seconds.")
else:
    print("Fast dataset cache already exists.")

# Now, test the fast loading speed.
print("\n--- Testing FAST loading speed ---")
start_time = time.time()
reloaded_ds = load_from_disk(FAST_DATASET_PATH)
print(f"SUCCESS! Loading from fast cache took only {time.time() - start_time:.2f} seconds.")
print("Dataset splits available:", list(reloaded_ds.keys()))