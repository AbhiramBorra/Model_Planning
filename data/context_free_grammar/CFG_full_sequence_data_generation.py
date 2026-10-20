import random
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from tqdm import tqdm
import os
import numpy as np

from CFG_data_generation import CFG

import json
import os
import numpy as np, os, pickle, mmap

num_sequences = 4000000

cfg_directory = "./cfg_s1444-64-_rd3456_rl23_4000k"
cfg_instance_path = cfg_directory +"/cfg_instance.pkl"
save_as = "bin"
save_directory = cfg_directory 

cfg = pickle.load(open(cfg_instance_path, "rb"))
PAD_TOKEN = cfg.PAD_TOKEN
BOS_TOKEN = cfg.BOS_TOKEN

history = False
start_symbol = None
max_workers = os.cpu_count()-1
sequences, expansion_histories = cfg.generate_multiple_sequences_parallel(num_sequences=num_sequences, start_symbol=start_symbol, history=history, max_workers=max_workers)

lengths = [len(sublist) for sublist in sequences]
total_tokens = sum(lengths)
shortest_length = min(lengths)
longest_length = max(lengths)
shortest_length_with_bos = shortest_length + 1
longest_length_with_bos = longest_length + 1

# Pad sequences to max length
data = []
for sequence in sequences:
    padded_sequence = [PAD_TOKEN] * (longest_length - len(sequence)) + [BOS_TOKEN] + sequence
    data.append(padded_sequence)

# Process test data
sequences_test, expansion_histories_text = cfg.generate_multiple_sequences_parallel(num_sequences=num_sequences//100, start_symbol=start_symbol, history=history, max_workers=max_workers)

data_test = []
for sequence in sequences_test:
    padded_sequence = [PAD_TOKEN] * (longest_length - len(sequence)) + [BOS_TOKEN] + sequence
    data_test.append(padded_sequence)

# Save sequence metadata
sequence_meta = {
    'padding_token': PAD_TOKEN,
    'max_sequence_length': longest_length_with_bos,
    'min_sequence_length': shortest_length_with_bos,
}

# Save sequence meta.pkl
sequence_meta_path = os.path.join(save_directory, 'sequence_meta_full.pkl')
with open(sequence_meta_path, 'wb') as f:
    pickle.dump(sequence_meta, f)

if not os.path.isdir(save_directory):
    os.makedirs(save_directory)
    print("Directory created successfully!")
else:
    print("Directory already exists!")

if save_as=="JSON":
    # Saving sequences to a JSON file
    with open(os.path.join(save_directory, f'train_full_seq_maxLength{longest_length_with_bos}.json'), 'w') as f:
        json.dump(sequences, f)

    with open(os.path.join(save_directory, f'test_full_seq_maxLength{longest_length_with_bos}.json'), 'w') as f:
        json.dump(sequences_test, f)

elif save_as=="bin":
    dtype = np.uint8

    # Convert list to NumPy array
    data_np = np.asarray(data, dtype=dtype)
    bin_path = os.path.join(save_directory, f"train_full_seq_maxLength{longest_length_with_bos}.bin")
    memmap = np.memmap(bin_path, dtype=dtype, mode="w+", shape=data_np.shape)
    memmap[:] = data_np[:]
    memmap.flush()

    data_np_test = np.asarray(data_test, dtype=dtype)
    bin_path = os.path.join(save_directory, f"val_full_seq_maxLength{longest_length_with_bos}.bin")
    memmap_test = np.memmap(bin_path, dtype=dtype, mode="w+", shape=data_np_test.shape)
    memmap_test[:] = data_np_test[:]
    memmap_test.flush()

