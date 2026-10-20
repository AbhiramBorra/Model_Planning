import random
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from tqdm import tqdm
import os
import numpy as np

from CFG_data_generation import CFG

import json
import os
import numpy as np, os, pickle, mmap

num_sequences = 15000000


cfg_directory = "cfg_s1444-64-_rd3456_rl23_4000k"

cfg_instance_path = cfg_directory +"/cfg_instance.pkl"
save_as = "bin"

# Parameters for clipped sequences
min_prefix_length_with_bos = 8  # Minimum prefix length to sample
max_prefix_length_with_bos = 24  # Maximum prefix length to sample
min_prefix_length = min_prefix_length_with_bos - 1
max_prefix_length = max_prefix_length_with_bos - 1

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
temp =[temp for temp in cfg_directory.split('_') if temp.startswith('s')][0]
last = temp[-1]
if last == '-':
    last = temp[:-1].split('-')[-1]
eos_token_id = int(last) + 1


# assert max_prefix_length < shortest_length, "max_prefix_length must be less than shortest_length={}".format(shortest_length)
# No need to assert max_prefix_length < shortest_length anymore
# We'll handle long prefix lengths by capping them at sequence_length - 1 

data = []
for sequence in sequences:
    prefix_length= random.randint(min_prefix_length, max_prefix_length)
    if prefix_length > len(sequence) - 1:
        prefix_length = len(sequence) - 1
    padded_sequence = [PAD_TOKEN] * (max_prefix_length - prefix_length) + [BOS_TOKEN] + sequence[:prefix_length] 
    data.append(padded_sequence)
    

# Process test data
sequences_test, expansion_histories_text = cfg.generate_multiple_sequences_parallel(num_sequences=num_sequences//100, start_symbol=start_symbol, history=history, max_workers=max_workers)

data_test = []
for sequence in sequences_test:
    prefix_length= random.randint(min_prefix_length, max_prefix_length)
    if prefix_length > len(sequence) - 1:
        prefix_length = len(sequence) - 1
    padded_sequence = [PAD_TOKEN] * (max_prefix_length - prefix_length) + [BOS_TOKEN] + sequence[:prefix_length] 
    data_test.append(padded_sequence)

# Save prefix-specific metadata
prefix_meta = {
    'prefix_padding': PAD_TOKEN,
    'min_prefix_length': min_prefix_length_with_bos,
    'max_prefix_length': max_prefix_length_with_bos,
    'longest_length': longest_length,
    'shortest_length': shortest_length,
    'total_tokens': total_tokens,
    "eos_token_id": eos_token_id
}

# Save prefix meta.pkl
prefix_meta_path = os.path.join(save_directory, f'prefix_meta_{min_prefix_length_with_bos}_{max_prefix_length_with_bos}.pkl')
with open(prefix_meta_path, 'wb') as f:
    pickle.dump(prefix_meta, f)

if not os.path.isdir(save_directory):
    os.makedirs(save_directory)
    print("Directory created successfully!")
else:
    print("Directory already exists!")


if save_as=="JSON":
    # Saving sequences to a JSON file
    with open(os.path.join(save_directory,'train.json'), 'w') as f:
        json.dump(sequences, f)

    with open(os.path.join(save_directory,'test.json'), 'w') as f:
        json.dump(sequences_test, f)

elif save_as=="bin":
    # Determine the smallest unsigned‑integer dtype that can hold every token
    #max_token = max(processed_data)
    #if max_token < 256:
    #    dtype = np.uint8
    #elif max_token < 65536:
    #    dtype = np.uint16
    #else:
    #    dtype = np.uint32
    dtype = np.uint8

    # Convert list to NumPy array
    data_np = np.asarray(data, dtype=dtype)
    bin_path = os.path.join(save_directory, f"train_prefixes{min_prefix_length_with_bos}_{max_prefix_length_with_bos}.bin")
    memmap = np.memmap(bin_path, dtype=dtype, mode="w+", shape=data_np.shape)
    memmap[:] = data_np[:]
    memmap.flush()


    data_np_test = np.asarray(data_test, dtype=dtype)
    bin_path = os.path.join(save_directory, f"val_prefixes{min_prefix_length_with_bos}_{max_prefix_length_with_bos}.bin")
    memmap_test = np.memmap(bin_path, dtype=dtype, mode="w+", shape=data_np_test.shape)
    memmap_test[:] = data_np_test[:]
    memmap_test.flush()

