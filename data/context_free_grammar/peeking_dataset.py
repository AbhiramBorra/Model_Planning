import numpy as np
from pathlib import Path
import os

# data_file_path = "cfg_s14448_rd3456_rl234_4000k/train.bin"
data_file_path = "cfg_s1444-64-_rd3456_rl23_4000k/train_prefixes8_24.bin"
plot_length_distribution = True

# --- path to your binary file -------------------------------------------------
bin_path = Path(data_file_path )   # update if yours lives elsewhere


# --- plot length distribution -------------------------------------------------

if not data_file_path.endswith('train.bin') and not data_file_path.endswith('val.bin'):
    plot_length_distribution = False

# --- choose the same dtype used when the file was written ---------------------
#   * nanoGPT OpenWebText uses uint16
#   * your own CFG generator might have used uint8, uint16, or uint32
dtype = np.uint8                               # change if needed

# --- memory‑map (zero‑copy) read ---------------------------------------------
tokens = np.memmap(bin_path, dtype=dtype, mode="r")

# --- inspect the first few entries -------------------------------------------
N = 1200  # how many you want to peek at
print(f"First {N} tokens in {bin_path.name}:")
for i in range(N):
    print(tokens[i], end=' ')
print()

min = 999999999999
max = 0
for token in tokens[:900]:

    if token > max:
        max = token
    if token < min:
        min = token


print("min:", min)
print("max:", max)

print(len(tokens))

if plot_length_distribution:
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from collections import defaultdict, Counter
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    def analyze_chunk(chunk):
        chunk_lengths = []
        current_length = 0
        in_sequence = False
        
        for token in chunk:
            if token == min:
                if in_sequence:
                    current_length = 0
                in_sequence = True
                current_length = 0
            elif token == max and in_sequence:
                chunk_lengths.append(current_length)
                in_sequence = False
            elif in_sequence:
                current_length += 1
                
        return chunk_lengths

    # Split tokens into chunks for parallel processing
    num_cores = multiprocessing.cpu_count()
    chunk_size = len(tokens) // num_cores
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    # Process chunks in parallel
    sequence_lengths = Counter()
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(analyze_chunk, chunk) for chunk in chunks]
        
        # Collect results with progress bar
        for future in tqdm(futures, desc="Processing chunks"):
            chunk_lengths = future.result()
            sequence_lengths.update(Counter(chunk_lengths))

    # Plot histogram of sequence lengths
    lengths = sorted(sequence_lengths.keys())
    counts = [sequence_lengths[l] for l in lengths]

    plt.figure(figsize=(10,6))
    plt.bar(lengths, counts)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    total_seqs = sum(sequence_lengths.values())
    plt.title(f'Distribution of Sequence Lengths\n(min={np.min(lengths)}, max={np.max(lengths)}, total={total_seqs:,} sequences)')
    plt.grid(True)
    plt.savefig(os.path.join(bin_path.parent, 'sequence_length_distribution.png'))
    plt.close()

    print("\nSequence length distribution:")
    for length, count in sorted(sequence_lengths.items()):
        print(f"Length {length}: {count} sequences")
