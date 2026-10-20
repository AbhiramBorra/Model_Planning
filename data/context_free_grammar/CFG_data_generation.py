import random
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from tqdm import tqdm
import os
import numpy as np
import json
import os
import numpy as np, os, pickle, mmap
import matplotlib.pyplot as plt

class CFG():
    def __init__(self, size, rule_degrees, rule_lengths) -> None:
        """
        size: (|NT1|, |NT2|, . . . , |NTL|)
        """
        self.size = size
        self.rule_lengths = rule_lengths
        self.rule_degrees = rule_degrees

        self.BOS_TOKEN = 0 
        self.EOS_TOKEN = size[-1]+1
        self.PAD_TOKEN = self.EOS_TOKEN + 1 

        # A poor way to track the max sequence length, but it's ok for now
        self.max_seq_length = 0 # will be updated during generation
        # TODO: find a better way to calculate the max theoretical sequence length
        # TODO: it should be found after rules are generated
        # TODO: for now, I am skipping this
        self.max_theoretical_seq_length = 0

        self.symbols = []
        self.rules = dict()

        self._generate_symbols()
        self._generate_rules()
        self._prepare_fast_parser()

    def _generate_symbols(self):
        c = 1
        for idx, s in enumerate(self.size[::-1]):
            l = []
            for j in range(s):
                l.append(c)
                c += 1
            self.symbols.insert(0, l)

    def _generate_rules(self):
        for layer, symbls in enumerate(self.symbols[:-1]):
            used_rules = []
            for sym in symbls:
                d = dict()
                for j in range(random.choice(self.rule_degrees)):
                    l = random.choice(self.rule_lengths)
                    rule = random.choices(self.symbols[layer + 1], k = l)
                    while rule in used_rules:
                        rule = random.choices(self.symbols[layer + 1], k = l)
                    d[j] = rule
                    used_rules.append(rule)
                self.rules[sym] = d

    def generate_sequence(self, start_symbol = None, history=False):
        """
        Generates a sequence by expanding the start symbol according to the CFG rules.
        If history is True, returns additional information needed for tree plotting.
        """
        if start_symbol!=None:
            if start_symbol not in self.symbols[0]:
                raise Exception("Incorrect root is chosen")
        else:
            start_symbol = random.choice(self.symbols[0])

        sequence = []
        expansion_history = []

        if history:
            self._expand_symbol(start_symbol, sequence, expansion_history)
        else:
            self._expand_symbol(start_symbol, sequence, None)
        
        if len(sequence) > self.max_seq_length:
            self.max_seq_length = len(sequence)

        return sequence, expansion_history
        
    def _expand_symbol(self, symbol, sequence, expansion_history):
        """
        Recursively expands a symbol and appends the result to the sequence.
        """
        if symbol not in self.rules:
            sequence.append(symbol)  # If it's a terminal symbol, add it directly to the sequence
        else:
            rule_options = self.rules[symbol].values()
            chosen_expansion = random.choice(list(rule_options))  # Choose one expansion randomly
            
            if expansion_history != None:
                expansion_history.append((symbol, chosen_expansion))  # Save the expansion history for visualization
            
            for sym in chosen_expansion:
                self._expand_symbol(sym, sequence, expansion_history)  # Recursively expand each symbol in the chosen expansion

    def _generate_single_sequence(self, start_symbol=None, history=False):
        """
        A wrapper function for generating a single sequence, used for parallel execution.
        """
        return self.generate_sequence(start_symbol=start_symbol, history=history)

    def generate_multiple_sequences_parallel(self, num_sequences, start_symbol=None, history=False, max_workers=None):
        """
        Generates multiple sequences in parallel using ThreadPoolExecutor for efficiency.
        If history is True, returns additional information needed for tree plotting.
        """
        sequences = []
        expansion_histories = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._generate_single_sequence, start_symbol, history)
                for _ in tqdm(range(num_sequences))
            ]
            
            for future in tqdm(as_completed(futures), total=num_sequences):
                sequence, expansion_history = future.result()
                sequences.append(sequence)
                
                if history:
                    expansion_histories.append(expansion_history)

        
        return sequences, expansion_histories
        
    def is_valid_sequence_slow(self, sequence, start_symbol=None):
        """
        Return True iff the list of terminal symbols `sequence` can be
        derived from this grammar.

        Parameters
        ----------
        sequence : List[int]
            The terminals (no BOS/EOS tokens) to validate.
        start_symbol : Optional[int]
            Root non‑terminal.  If None, accept any symbol in layer‑1.

        Notes
        -----
        • Works for rule lengths 1, 2 or 3 (exactly what this generator produces).
        • Uses an O(n³) CYK‑style dynamic programme.
        """
        n = len(sequence)
        if n == 0:
            return False

        # ---------- bucket rules by RHS length ---------------------------
        rules_len1, rules_len2, rules_len3 = [], [], []  # tuples of ints
        for A, exp_dict in self.rules.items():
            for rhs in exp_dict.values():
                if len(rhs) == 1:
                    rules_len1.append((A, rhs[0]))
                elif len(rhs) == 2:
                    rules_len2.append((A, rhs[0], rhs[1]))
                elif len(rhs) == 3:
                    rules_len3.append((A, rhs[0], rhs[1], rhs[2]))

        # ---------- CYK table: dp[i][j] = NTs => sequence[i:j+1] ----------
        dp = [[set() for _ in range(n)] for _ in range(n)]

        # length‑1 spans ---------------------------------------------------
        for i, tok in enumerate(sequence):
            dp[i][i].add(tok)                # terminal derives itself
            for A, b in rules_len1:          # unary rules A → b
                if tok == b:
                    dp[i][i].add(A)

        # longer spans -----------------------------------------------------
        for span in range(2, n + 1):         # span length
            for i in range(n - span + 1):
                j = i + span - 1

                # length‑2 rules A → B C
                for k in range(i, j):
                    left, right = dp[i][k], dp[k + 1][j]
                    if not left or not right:
                        continue
                    for A, B, C in rules_len2:
                        if B in left and C in right:
                            dp[i][j].add(A)

                # length‑3 rules A → B C D
                for k1 in range(i + 1, j):
                    for k2 in range(k1, j):
                        s1, s2, s3 = dp[i][k1 - 1], dp[k1][k2], dp[k2 + 1][j]
                        if not s1 or not s2 or not s3:
                            continue
                        for A, B, C, D in rules_len3:
                            if B in s1 and C in s2 and D in s3:
                                dp[i][j].add(A)

        # ---------- accept / reject --------------------------------------
        roots = {start_symbol} if start_symbol is not None else set(self.symbols[0])
        return bool(dp[0][n - 1] & roots)
            
    # ============================================================
    #  Fast CKY tables (built once per CFG instance)
    # ============================================================
    def _prepare_fast_parser(self):
        """
        Build lookup tables used by the O(n³) CKY membership test.

        After this call the object has:
          • self._sym2id  :  dict(symbol -> dense int id)
          • self._unary   :  dict(term_id -> set(nonterm_id))
          • self._binary  :  dict((B_id,C_id) -> set(A_id))
          • self._roots   :  set(ids of layer-1 non-terminals)
        Any rule length > 2 is binarised ONCE here.
        """
        # 1) dense ids for every existing symbol
        self._sym2id = {}
        cur = 0
        for layer in self.symbols:
            for s in layer:
                self._sym2id[s] = cur
                cur += 1
        next_id = cur                     # fresh ids for binarisation

        unary  = {}                       # terminal -> {A}
        binary = {}                       # (B,C)   -> {A}

        # 2) iterate through original rules
        for A, exp_dict in self.rules.items():
            A_id = self._sym2id[A]
            for rhs in exp_dict.values():
                if len(rhs) == 1:         # unary  A → a
                    term_id = self._sym2id[rhs[0]]
                    unary.setdefault(term_id, set()).add(A_id)

                elif len(rhs) == 2:       # binary A → B C
                    B_id, C_id = map(self._sym2id.__getitem__, rhs)
                    binary.setdefault((B_id, C_id), set()).add(A_id)

                elif len(rhs) > 2:       # length >2  A → B C D ...
                    # general binarization for arbitrary length
                    symbols_ids = [self._sym2id[sym] for sym in rhs]
                    prev_id = symbols_ids[0]
                    for sym_id in symbols_ids[1:-1]:
                        X_id = next_id
                        next_id += 1
                        binary.setdefault((prev_id, sym_id), set()).add(X_id)
                        prev_id = X_id
                    last_id = symbols_ids[-1]
                    binary.setdefault((prev_id, last_id), set()).add(A_id)

        self._unary  = unary
        self._binary = binary
        self._roots  = {self._sym2id[s] for s in self.symbols[0]}
        self._n_sym  = next_id            # total # ids

    # ------------------------------------------------------------
    #  Fast O(n³) CKY membership test
    # ------------------------------------------------------------
    def is_valid_sequence(self, sequence, start_symbol=None):
        """
        Return True iff the token list `sequence` (no BOS/EOS) can be
        derived from the grammar.  Runs cubic-time CKY on NumPy sets.
        Supports arbitrary rule lengths via binarization.
        """
        if not hasattr(self, "_sym2id"):
            self._prepare_fast_parser()

        n = len(sequence)
        if n == 0:
            return False

        # dp[i][j]  <- python set of NT-ids
        dp = [[set() for _ in range(n)] for _ in range(n)]

        # ---- length-1 spans (terminals + unary closures) ----------
        for i, tok in enumerate(sequence):
            tok_id = self._sym2id.get(tok)
            if tok_id is None:
                return False              # unknown token
            cell = dp[i][i]
            cell.add(tok_id)
            for A in self._unary.get(tok_id, ()):
                cell.add(A)

        # ---- longer spans ----------------------------------------
        for span in range(2, n + 1):
            for i in range(n - span + 1):
                j = i + span - 1
                cell = dp[i][j]
                for k in range(i, j):
                    left, right = dp[i][k], dp[k + 1][j]
                    if not left or not right:
                        continue
                    for B in left:
                        for C in right:
                            for A in self._binary.get((B, C), ()):
                                cell.add(A)

        roots = ({self._sym2id[start_symbol]} if start_symbol is not None
                 else self._roots)
        return bool(dp[0][n - 1] & roots)

    def new_is_valid_sequence_slow(self, sequence, start_symbol=None):
        """
        General validation for arbitrary-length rules:
        Binarize grammar on-the-fly and then run fast CKY.
        """
        # Ensure the fast parser tables are built (with general binarization)
        if not hasattr(self, "_sym2id"):
            self._prepare_fast_parser()
        # Delegate to the fast O(n^3) CKY membership test
        return self.is_valid_sequence(sequence, start_symbol)

    def generate_and_save_in_batches(self, total_sequences, batch_size, save_directory, start_symbol=None, history=False, max_workers=None):
        """
        Generate sequences in batches and write directly to binary files to avoid OOM.
        Preserves the same file structure as the original implementation.
        """
        os.makedirs(save_directory, exist_ok=True)
        print(f"Directory ready: {save_directory}")

        # Calculate total tokens needed (estimate)
        # We'll resize files as needed during processing
        dtype = np.uint8
        
        # Create memory-mapped files that we'll write to incrementally
        train_bin_path = os.path.join(save_directory, "train.bin")
        val_bin_path = os.path.join(save_directory, "val.bin")
        
        # Initialize files
        train_file = open(train_bin_path, 'wb')
        val_file = open(val_bin_path, 'wb')
        
        train_tokens_written = 0
        val_tokens_written = 0
        
        # Statistics tracking
        all_lengths = []
        total_tokens = 0
        
        print(f"Generating {total_sequences} sequences in batches of {batch_size}...")
        
        # Generate training data in batches
        num_batches = (total_sequences + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Training batches"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_sequences)
            current_batch_size = batch_end - batch_start
            
            # Generate batch
            batch_sequences, _ = self.generate_multiple_sequences_parallel(
                num_sequences=current_batch_size, 
                start_symbol=start_symbol, 
                history=history, 
                max_workers=max_workers
            )
            
            # Process and write batch
            batch_processed = []
            for sequence in batch_sequences:
                processed_sequence = [0] + sequence + [self.size[-1] + 1]  # BOS + sequence + EOS
                batch_processed.extend(processed_sequence)
                all_lengths.append(len(sequence))
                total_tokens += len(sequence)
            
            # Write to train file
            batch_array = np.array(batch_processed, dtype=dtype)
            train_file.write(batch_array.tobytes())
            train_tokens_written += len(batch_processed)
            
            # Clear memory
            del batch_sequences, batch_processed, batch_array
            
        # Generate validation data (smaller, can do in one batch)
        val_sequences_count = total_sequences // 100
        print(f"Generating {val_sequences_count} validation sequences...")
        
        val_batch_size = min(batch_size, val_sequences_count)
        val_num_batches = (val_sequences_count + val_batch_size - 1) // val_batch_size
        
        for batch_idx in tqdm(range(val_num_batches), desc="Validation batches"):
            batch_start = batch_idx * val_batch_size
            batch_end = min(batch_start + val_batch_size, val_sequences_count)
            current_batch_size = batch_end - batch_start
            
            # Generate validation batch
            val_batch_sequences, _ = self.generate_multiple_sequences_parallel(
                num_sequences=current_batch_size,
                start_symbol=start_symbol,
                history=history,
                max_workers=max_workers
            )
            
            # Process and write validation batch
            val_batch_processed = []
            for sequence in val_batch_sequences:
                processed_sequence = [0] + sequence + [self.size[-1] + 1]  # BOS + sequence + EOS
                val_batch_processed.extend(processed_sequence)
            
            # Write to val file
            val_batch_array = np.array(val_batch_processed, dtype=dtype)
            val_file.write(val_batch_array.tobytes())
            val_tokens_written += len(val_batch_processed)
            
            # Clear memory
            del val_batch_sequences, val_batch_processed, val_batch_array
        
        # Close files
        train_file.close()
        val_file.close()
        
        # Calculate statistics
        shortest_length = min(all_lengths)
        longest_length = max(all_lengths)
        
        print(f"Training tokens written: {train_tokens_written}")
        print(f"Validation tokens written: {val_tokens_written}")
        print(f"Shortest sequence length: {shortest_length}")
        print(f"Longest sequence length: {longest_length}")
        print(f"Total tokens (without BOS, EOS): {total_tokens}")
        
        return {
            'shortest_length': shortest_length,
            'longest_length': longest_length,
            'total_tokens': total_tokens,
            'train_tokens': train_tokens_written,
            'val_tokens': val_tokens_written
        }
    
# %% TREE VISUALIZATION %%%%
def plot_simple_tree(expansion_history):
    """
    DOES NOT WORK AS INTENTED. I NEED TO FIX THE ERRORS LATER WHEN I NEED
    """
    # Initialize positions and tracking variables
    levels = {}
    y_level = 0
    
    # Define the distance between nodes
    x_distance = 2
    y_distance = 2

    def plot_node(parent, children, parent_pos, level):
        nonlocal levels

        # Calculate the y position for the current level
        y_pos = -level * y_distance

        if level not in levels:
            levels[level] = []
        
        # Calculate x positions for children
        start_x = parent_pos[0] - (len(children) - 1) * x_distance / 2
        child_positions = [(start_x + i * x_distance, y_pos) for i in range(len(children))]
        levels[level].extend(child_positions)
        
        for i, child_pos in enumerate(child_positions):
            # Draw line from parent to child
            plt.plot([parent_pos[0], child_pos[0]], [parent_pos[1], child_pos[1]], 'k-', lw=2)
            
            # Draw the child node
            plt.text(child_pos[0], child_pos[1], str(children[i]), fontsize=12, ha='center', va='center',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle'))
        
        return child_positions

    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Initialize the root position
    root = expansion_history[0][0]
    root_pos = (0, 0)
    
    # Draw the root node
    plt.text(root_pos[0], root_pos[1], str(root), fontsize=12, ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle'))

    # Plot the tree based on expansion history
    parent_positions = [root_pos]
    for level, (parent, children) in enumerate(expansion_history):
        parent_pos = parent_positions.pop(0)
        child_positions = plot_node(parent, children, parent_pos, level + 1)
        parent_positions.extend(child_positions)
    
    # Remove axes and show the plot
    plt.axis('off')
    plt.show()

def main():
    num_sequences = 4000000
    size = (1, 4, 4, 4, 64)
    BOS_TOKEN = 0 
    EOS_TOKEN = size[-1]+1
    PAD_TOKEN = EOS_TOKEN + 1
    rule_degrees = [3, 4, 5, 6]
    rule_lengths = [2, 3]
    save_as = "bin"
    
    # Batch size to control memory usage - adjust based on your system
    batch_size = 50000  # Generate 50k sequences at a time

    save_directory = os.path.join(os.path.dirname(__file__), "cfg_s"+''.join(map(lambda x: f"-{x}-" if x >= 10 else str(x), size))+"_rd"+''.join(map(lambda x: f"-{x}-" if x >= 10 else str(x), rule_degrees))+"_rl"+''.join(map(lambda x: f"-{x}-" if x >= 10 else str(x), rule_lengths))+"_"+str(int(num_sequences/1000))+"k")

    cfg = CFG(size=size, rule_degrees=rule_degrees, rule_lengths=rule_lengths)

    history = False
    start_symbol = None
    # Reduce max_workers to control memory usage
    max_workers = min(256, os.cpu_count())  # Limit to 8 workers max
    
    print(f"Using batch processing with batch_size={batch_size} and max_workers={max_workers}")
    
    # Use batch processing instead of loading everything into memory
    stats = cfg.generate_and_save_in_batches(
        total_sequences=num_sequences,
        batch_size=batch_size,
        save_directory=save_directory,
        start_symbol=start_symbol,
        history=history,
        max_workers=max_workers
    )

    print(f"Max theoretical sequence length: {cfg.max_theoretical_seq_length}")
    print(f"Max sequence length: {cfg.max_seq_length}")
    if cfg.max_seq_length == cfg.max_theoretical_seq_length:
        print("Max sequence length is equal to the max theoretical sequence length")
    else:
        print("Max sequence length is not equal to the max theoretical sequence length")

    # --------------------------------------------------------------
    # Save dataset metadata expected by the training pipeline
    # --------------------------------------------------------------
    print("Saving metadata...")
    dtype = np.uint8
    meta = {"vocab_size": EOS_TOKEN+2, "dtype": dtype, "symbols": cfg.symbols, "rules": cfg.rules}  # +1 because indices start at 0 and +1 because of PAD token 

    print("Saving meta.pkl...")
    with open(os.path.join(save_directory, "meta.pkl"), "wb") as f_meta:
        pickle.dump(meta, f_meta)

    print("Saving cfg_instance.pkl...")
    with open(os.path.join(save_directory,"cfg_instance.pkl"), "wb") as f:
        pickle.dump(cfg, f)

    print("Saving train_info.txt...")
    with open(os.path.join(save_directory,'train_info.txt'), 'w') as f:
        f.write(f"num_squences = {num_sequences}\n")
        f.write(f"size = {size}\n")
        f.write(f"rule_degreees = {rule_degrees}\n")
        f.write(f"rule_lengths = {rule_lengths}\n")
        f.write(f"history = {history}\n")
        f.write(f"start_symbol = {start_symbol}\n")
        f.write(f"batch_size = {batch_size}\n")
        f.write(f"max_workers = {max_workers}\n")

        f.write(f"\nshortest sentence length: {stats['shortest_length']}\n")
        f.write(f"longest sentence length: {stats['longest_length']}\n")
        f.write(f"total tokens (without BOS, EOS): {stats['total_tokens']}\n")
        f.write(f"train tokens written: {stats['train_tokens']}\n")
        f.write(f"val tokens written: {stats['val_tokens']}\n")

        f.write(f"CFG symbols: {cfg.symbols}\n")
        f.write(f"CFG rules: {cfg.rules}\n")

if __name__ == "__main__":
    main()
