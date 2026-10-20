import math
import json
import os
import pickle
import torch
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def save_training_metadata(config, model_args, out_dir):
    """
    Saves model configuration and training parameters in a human-readable format.
    
    Args:
        config (dict): Training configuration parameters
        model_args (dict): Model architecture parameters
        out_dir (str): Output directory where metadata will be saved
    """
    # Create metadata dictionary
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_architecture': {
            'n_layer': model_args.get('n_layer'),
            'n_head': model_args.get('n_head'),
            'n_embd': model_args.get('n_embd'),
            'block_size': model_args.get('block_size'),
            'vocab_size': model_args.get('vocab_size'),
            'bias': model_args.get('bias'),
            'dropout': model_args.get('dropout')
        },
        'training_parameters': {
            'batch_size': config.get('batch_size'),
            'gradient_accumulation_steps': config.get('gradient_accumulation_steps'),
            'learning_rate': config.get('learning_rate'),
            'max_iters': config.get('max_iters'),
            'weight_decay': config.get('weight_decay'),
            'beta1': config.get('beta1'),
            'beta2': config.get('beta2'),
            'grad_clip': config.get('grad_clip'),
            'warmup_iters': config.get('warmup_iters'),
            'lr_decay_iters': config.get('lr_decay_iters'),
            'min_lr': config.get('min_lr')
        },
        'system_config': {
            'device': config.get('device'),
            'dtype': config.get('dtype'),
            'compile': config.get('compile'),
            'backend': config.get('backend')
        }
    }
    
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Save as JSON file
    metadata_path = os.path.join(out_dir, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Also save as a human-readable text file
    text_path = os.path.join(out_dir, 'training_metadata.txt')
    with open(text_path, 'w') as f:
        f.write(f"Training Metadata - {metadata['timestamp']}\n")
        f.write("\nModel Architecture:\n")
        for key, value in metadata['model_architecture'].items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nTraining Parameters:\n")
        for key, value in metadata['training_parameters'].items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nSystem Configuration:\n")
        for key, value in metadata['system_config'].items():
            f.write(f"{key}: {value}\n")

@torch.no_grad()
def evaluate_cfg_accuracy(model, cfg, device, num_samples=500, temperature=1.0, batch_size=32, save_samples=True):
    """
    Evaluate the model's ability to generate valid sequences according to a CFG,
    without knowing the sequence length in advance.
    
    Args:
        model: The GPT model to evaluate
        cfg: The context-free grammar object
        device: Device to run the model on
        num_samples: Number of samples to evaluate
        temperature: Temperature for sampling
        batch_size: Batch size for model inference
        save_samples: Whether to save sample sequences for logging
        
    Returns:
        dict: Dictionary containing zero_cut and half_cut accuracy values and sample sequences if save_samples=True
    """
    model.eval()
    BOS, EOS = 0, cfg.symbols[-1][-1] + 1  # Data convention
    
    # Get the max sequence length from CFG
    max_seq_length = cfg.max_seq_length
    
    # Step 1: Generate ground truth sequences in parallel
    print("Generating ground truth sequences in parallel...")
    
    def generate_sequences_batch(num_to_generate):
        return [cfg.generate_sequence()[0] for _ in range(num_to_generate)]
    
    # Use ThreadPoolExecutor to generate sequences in parallel
    all_sequences = []
    with ThreadPoolExecutor(max_workers=max(os.cpu_count(), 16)) as executor:
        # Split the work into chunks for each worker
        chunk_size = max(1, num_samples // max(os.cpu_count(), 16))
        futures = []
        
        for i in range(0, num_samples, chunk_size):
            n = min(chunk_size, num_samples - i)
            futures.append(executor.submit(generate_sequences_batch, n))
        
        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating sequences"):
            all_sequences.extend(future.result())
    
    # Make sure we have exactly num_samples sequences
    all_sequences = all_sequences[:num_samples]
    
    # Function to complete prefixes until EOS token or max_seq_length
    def complete_prefixes_batch_until_eos(prefix_list):
        """
        Generate completions for multiple prefixes in a batch until EOS token or max_seq_length.
        
        Args:
            prefix_list: List of token sequences (prefixes)
            
        Returns:
            List of generated sequences and raw outputs (with BOS/EOS)
        """
        # Create input tensors (prepend BOS to each if not already included)
        batch_size = len(prefix_list)
        input_ids = torch.zeros((batch_size, max(len(p) for p in prefix_list) + 1), dtype=torch.long, device=device)
        
        # Fill in the input tensor with BOS and prefixes
        for i, prefix in enumerate(prefix_list):
            input_ids[i, 0] = BOS
            if len(prefix) > 0:
                input_ids[i, 1:len(prefix)+1] = torch.tensor(prefix, dtype=torch.long, device=device)
        
        # Generate completions until EOS or max_seq_length
        outputs = model.generate(input_ids, max_new_tokens=max_seq_length, temperature=temperature, 
                               top_k=None, eos_token=EOS)
        
        # Process the generated sequences
        results = []
        raw_results = []  # Store the raw outputs (including BOS and any EOS)
        
        for output in outputs:
            # Store the raw output
            raw_results.append(output.tolist())
            
            # Convert to list and remove BOS token
            full_seq = output.tolist()[1:]
            
            # If EOS is in the sequence, trim everything after it
            if EOS in full_seq:
                eos_idx = full_seq.index(EOS)
                full_seq = full_seq[:eos_idx]  # Exclude EOS
                
            results.append(full_seq)
        
        return results, raw_results
    
    # Process batches for zero-cut evaluation
    print("Evaluating zero-cut accuracy...")
    zero_cut_results = []
    zero_cut_raw_results = []
    for i in range(0, num_samples, batch_size):
        batch_sequences = all_sequences[i:i+batch_size]
        
        # Empty prefixes for zero-cut
        empty_prefixes = [[] for _ in batch_sequences]
        
        generated, raw_generated = complete_prefixes_batch_until_eos(empty_prefixes)
        zero_cut_results.extend(generated)
        zero_cut_raw_results.extend(raw_generated)
    
    # Process batches for half-cut evaluation
    print("Evaluating half-cut accuracy...")
    half_cut_results = []
    half_cut_raw_results = []
    for i in range(0, num_samples, batch_size):
        batch_sequences = all_sequences[i:i+batch_size]
        
        # Calculate half-cut length using min(CFG_max_length/2, min_length_in_batch)
        min_length_in_batch = min(len(seq) for seq in batch_sequences)
        half_max_length = cfg.max_seq_length // 2
        cut_length = min(half_max_length, min_length_in_batch)
        
        # Create prefixes (first half of each sequence, using the calculated cut_length)
        prefixes = [seq[:cut_length] for seq in batch_sequences]
        
        generated, raw_generated = complete_prefixes_batch_until_eos(prefixes)
        half_cut_results.extend(generated)
        half_cut_raw_results.extend(raw_generated)
    
    # Check validity in parallel
    print("Checking sequence validity in parallel...")
    def check_validity_batch(sequences):
        return [cfg.is_valid_sequence(seq) for seq in sequences]
    
    # Check validity of all sequences
    valid_zero_cut = []
    valid_half_cut = []
    
    with ThreadPoolExecutor(max_workers=max(os.cpu_count(), 16)) as executor:
        # Process zero-cut results
        chunk_size = max(1, len(zero_cut_results) // max(os.cpu_count(), 16))
        futures = []
        
        for i in range(0, len(zero_cut_results), chunk_size):
            chunk = zero_cut_results[i:i+chunk_size]
            futures.append(executor.submit(check_validity_batch, chunk))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating zero-cut"):
            valid_zero_cut.extend(future.result())
        
        # Process half-cut results
        futures = []
        for i in range(0, len(half_cut_results), chunk_size):
            chunk = half_cut_results[i:i+chunk_size]
            futures.append(executor.submit(check_validity_batch, chunk))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating half-cut"):
            valid_half_cut.extend(future.result())
    
    # Calculate final accuracies
    zero_cut_acc = sum(valid_zero_cut) / len(valid_zero_cut)
    half_cut_acc = sum(valid_half_cut) / len(valid_half_cut)
    
    model.train()
    
    result = {
        "zero_cut": zero_cut_acc,
        "half_cut": half_cut_acc
    }
    
    # Add sample sequences to the result if requested
    if save_samples:
        result.update({
            "ground_truth": all_sequences[:20],  # Save 20 ground truth samples
            "zero_cut_samples": zero_cut_results[:20],  # Save 20 generated samples
            "half_cut_samples": half_cut_results[:20],  # Save 20 generated samples
            "zero_cut_raw": zero_cut_raw_results[:20],  # Raw outputs with BOS/EOS
            "half_cut_raw": half_cut_raw_results[:20],  # Raw outputs with BOS/EOS
            "zero_cut_valid": valid_zero_cut[:20],  # Validity of samples
            "half_cut_valid": valid_half_cut[:20]  # Validity of samples
        })
    
    return result

def save_cfg_accuracy_log(out_dir, iter_num, accuracy_results, cfg=None):
    """
    Save CFG accuracy evaluation results to a log file.
    
    Args:
        out_dir: Output directory
        iter_num: Current iteration number
        accuracy_results: Results from accuracy evaluation
        cfg: The CFG instance (optional)
    """
    # Create directory if it doesn't exist
    log_dir = os.path.join(out_dir, "cfg_acc_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file name with iteration number
    log_file = os.path.join(log_dir, f"cfg_accuracy_iter_{iter_num}.txt")
    
    with open(log_file, "w") as f:
        f.write(f"CFG Accuracy Evaluation at Iteration {iter_num}\n")
        f.write("=" * 50 + "\n\n")
        
        # Write CFG max length information if available
        if cfg is not None:
            f.write(f"CFG max sequence length: {cfg.max_seq_length}\n\n")
        
        # Get ground truth, generated samples, and validity
        ground_truth = accuracy_results.get("ground_truth", [])
        zero_cut_samples = accuracy_results.get("zero_cut_samples", [])
        half_cut_samples = accuracy_results.get("half_cut_samples", [])
        zero_cut_raw = accuracy_results.get("zero_cut_raw", [])
        half_cut_raw = accuracy_results.get("half_cut_raw", [])
        zero_cut_valid = accuracy_results.get("zero_cut_valid", [])
        half_cut_valid = accuracy_results.get("half_cut_valid", [])
        
        # Write accuracy results
        f.write(f"Zero-cut accuracy: {accuracy_results['zero_cut']*100:.2f}%\n")
        f.write(f"Half-cut accuracy: {accuracy_results['half_cut']*100:.2f}%\n\n")
        
        # Write sample comparisons
        f.write("SAMPLE COMPARISONS:\n")
        for i in range(min(20, len(ground_truth))):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  Ground truth: {ground_truth[i]}\n")
            
            if i < len(zero_cut_samples):
                valid_str = "VALID" if zero_cut_valid[i] else "INVALID"
                f.write(f"  Zero-cut generated (processed): {zero_cut_samples[i]} ({valid_str})\n")
                if i < len(zero_cut_raw):
                    f.write(f"  Zero-cut generated (raw with BOS/EOS): {zero_cut_raw[i]}\n")
            
            if i < len(half_cut_samples):
                valid_str = "VALID" if half_cut_valid[i] else "INVALID"
                f.write(f"  Half-cut generated (processed): {half_cut_samples[i]} ({valid_str})\n")
                if i < len(half_cut_raw):
                    f.write(f"  Half-cut generated (raw with BOS/EOS): {half_cut_raw[i]}\n")
            
            f.write("\n")

def load_cfg_from_dataset(dataset_path):
    """
    Load the CFG instance from the dataset directory.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        CFG: The loaded CFG instance
    """
    # Extract the CFG pickle path from the dataset path
    if not dataset_path.startswith('data/'):
        dataset_path = os.path.join('data', dataset_path)
    
    cfg_pickle_path = os.path.join(dataset_path, 'cfg_instance.pkl')
    
    # Make sure to import CFG class before loading pickle
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(cfg_pickle_path)))
    from data.context_free_grammar.CFG_data_generation import CFG
    
    # Load the CFG instance
    with open(cfg_pickle_path, 'rb') as f:
        cfg = pickle.load(f)
    
    return cfg
