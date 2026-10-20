# %% 
"""
Utility functions for training the VQ-VAE-like model for LLM hidden state representation learning

"""
# %%
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
import inspect
import h5py
import math
import pickle
import json
from model import GPT, GPTConfig
from torch.nn import functional as F


def get_logger(save_logs_flag=True, print_logs=True, experiment_dir=''):
    """Configure logging to write to a local file and log to the console."""
    import os, logging
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    # Ensure experiment_dir exists if saving logs to a file
    if save_logs_flag and experiment_dir:
        os.makedirs(experiment_dir, exist_ok=True)
        log_file_path = os.path.join(experiment_dir, 'exp_log.log')
        # Clear the log file if it exists
        if os.path.exists(log_file_path):
            with open(log_file_path, 'w'):
                pass
        # File handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('[%(levelname)s][%(asctime)s]: %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(file_handler)
        
    if print_logs:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('[%(levelname)s][%(asctime)s]: %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(console_handler)
        
    logger.setLevel(logging.INFO)
    return logger

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer

class nanoLLM:
    """
    Custom LLM wrapper class that loads checkpoints from CFG-trained models.
    This is designed to work with the minimal_VQVAE1 training code.
    """
    def __init__(self, model_name=None, base_dir='LLMout', data_dir='data/context_free_grammar'):
        """
        Initialize the model by loading from a checkpoint.
        
        Args:
            model_name: String identifier of the model to load. This should match 
                       one of the folder names in the base_dir directory. Or, it should be an existing 
                       directory path with a checkpoint file in it.
            base_dir: Base directory where model checkpoints are stored. Default is 'out'.
            data_dir: Directory where CFG dataset is stored. Default is 'data/context_free_grammar'.
        """
        self.model_name = model_name
        
        # Check if model_name is provided
        if model_name is None:
            raise ValueError("model_name must be provided")
        """
        # Load the tokenizer from the CFG dataset
        try:
            self.cfg_folders = [f for f in os.listdir(data_dir) if f.startswith('cfg_s') and os.path.isdir(os.path.join(data_dir, f))]
            
            if not self.cfg_folders:
                raise ValueError(f"No CFG dataset folders found in {data_dir}")
            
            # Use the first CFG folder by default
            cfg_folder = self.cfg_folders[0]
            cfg_path = os.path.join(data_dir, cfg_folder)
            
            # Load metadata about the CFG dataset
            meta_path = os.path.join(cfg_path, 'meta.pkl')
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    self.meta = pickle.load(f)
                
                # Create a simple tokenizer from the CFG vocabulary
                self.vocab_size = self.meta.get('vocab_size', 5)  # Default to 5 if not specified
            else:
                logging.warning(f"No meta.pkl found in {cfg_path}, using default vocab_size=5")
                self.vocab_size = 5
                self.meta = {"vocab_size": 5}
        except (FileNotFoundError, OSError) as e:
            logging.warning(f"Could not access CFG data directory: {e}. Using default vocab_size=5")
            self.vocab_size = 5
            self.meta = {"vocab_size": 5}
        """   
        # Load the checkpoint from the specified directory
        if os.path.exists(model_name):
            model_dir = model_name
        else:
            model_dir = os.path.join(base_dir, model_name)
        checkpoint_path = os.path.join(model_dir, 'ckpt.pt')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        # Load model metadata
        metadata_path = os.path.join(model_dir, 'training_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Extract model architecture parameters
            model_arch = self.metadata.get('model_architecture', {})
            self.config = GPTConfig(
                block_size=model_arch.get('block_size', 512),
                vocab_size=model_arch.get('vocab_size', None),
                n_layer=model_arch.get('n_layer', 4),
                n_head=model_arch.get('n_head', 4),
                n_embd=model_arch.get('n_embd', 64),
                dropout=model_arch.get('dropout', 0.0),
                bias=model_arch.get('bias', False)
            )
        else:
            # Default config if no metadata file exists
            logging.warning(f"No metadata found at {metadata_path}, using default configuration")
            raise ValueError(f"No metadata found at {metadata_path}, using default configuration")
        
        # Initialize the model architecture
        self.model = GPT(self.config)
        self.vocab_size = self.config.vocab_size
        
        # Load the checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    # If the checkpoint has a 'model' key, it's from the training script
                    state_dict = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    # Alternative key sometimes used
                    state_dict = checkpoint['model_state_dict']
                else:
                    raise ValueError(f"No valid model state dict found in checkpoint: {checkpoint}")
            else:
                # Otherwise assume it's just the model state dict
                state_dict = checkpoint

            # Remove _orig_mod prefix from state dict keys if present
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # Load the modified state dict
            self.model.load_state_dict(new_state_dict, strict=True)
            
            # Verify the model loaded correctly
            logging.info(f"Loaded model {model_name} with {self.model.get_num_params()} parameters")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from checkpoint: {e}")
        
        # Set model to evaluation mode by default
        self.model.eval()

    def compile(self):
        """Compile the model using torch.compile() if available."""
        try:
            self.model = torch.compile(self.model)
            logging.info("Model compilation successful.")
        except Exception as e:
            logging.warning(f"Model compilation failed: {e}")
        return self

    def to(self, device):
        """Move the model to the specified device."""
        self.model.to(device)
        return self

    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()
        return self

    def train(self):
        """Set the model to training mode."""
        self.model.train()
        return self

    def generate_prefix_hidden_states(self, input_ids, attention_mask=None):
        """
        Generate hidden states for input sequences.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing token IDs
            attention_mask: Tensor of shape (batch_size, seq_len) with 1 for valid tokens, 0 for padding
            
        Returns:
            Tensor of shape (batch_size, n_layer+1, seq_len, n_embd) containing hidden states for each layer
            (+1 because we include the final layer norm output)
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        batch_size, seq_len = input_ids.size()

        # Replace padding tokens (61) with 0 before passing to model !!!!!!
        model_input_ids = input_ids.clone()
        # TODO: it shouldn't hurt to turn off this since I removed 61 from CFG tasks
        # model_input_ids[input_ids == 61] = 0
        
        # Set up storage for collecting hidden states
        all_hidden_states = []
        
        # Run the model in eval mode with no gradients
        with torch.no_grad():
            # Forward the GPT model
            tok_emb = self.model.transformer.wte(model_input_ids)
            x = self.model.transformer.drop(tok_emb)
            
            # Track hidden states for each layer
            for i, block in enumerate(self.model.transformer.h):
                x = block(x, attention_mask=attention_mask)
                all_hidden_states.append(x)
                
            # Final layer norm
            x = self.model.transformer.ln_f(x)
            all_hidden_states.append(x)
        
        # Stack hidden states from all layers [batch_size, n_layer+1, seq_len, n_embd]
        hidden_states = torch.stack(all_hidden_states, dim=1)
        
        return hidden_states

    def generate(self, input_ids, max_new_tokens=1, temperature=1.0, top_k=None, attention_mask=None, eos_token=None):
        """
        Generate text using the model.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing token IDs
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: If specified, only sample from the top k most likely tokens
            attention_mask: Tensor of shape (batch_size, seq_len) with 1 for valid tokens, 0 for padding
            
        Returns:
            Tensor of shape (batch_size, seq_len + max_new_tokens) with generated token IDs
        """
        return self.model.generate(input_ids, max_new_tokens, temperature, top_k, attention_mask=attention_mask, eos_token=eos_token)


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def get_beta(it, warmup_iters, beta_start, beta_max):
    """
    Linearly increase beta from beta_start to beta_max over warmup_iters steps.
    
    Args:
        it (int): Current iteration.
        warmup_iters (int): Number of iterations to increase beta linearly.
        beta_start (float): Starting beta value.
        beta_max (float): Maximum beta value after warmup.
    
    Returns:
        float: The current beta value.
    """
    if it < warmup_iters:
        return beta_start + (beta_max - beta_start) * (it / warmup_iters)
    return beta_max

def compute_masked_nrmse(H, output, attention_mask, eps=1e-8):
    """
    Compute the normalized RMSE (||H - output|| / ||H||) for each sample in a batch,
    but only considering indices where attention_mask == 1.
    
    Args:
        H: Tensor of shape (B, L, seq_len, dim) or (B, seq_len, dim)
        output: Tensor of shape (B, L, seq_len, dim) or (B, seq_len, dim)
        attention_mask: Tensor of shape (B, seq_len) with 1 for active tokens, 0 for padding.
        eps: Small constant to avoid division by zero.
        
    Returns:
        nrmse: Tensor of shape (B,) with the nRMSE for each sample.
    """
    # Handle 3D input case by adding L dimension
    if len(H.shape) == 3:
        H = H.unsqueeze(1)
        output = output.unsqueeze(1)
    
    # Expand the attention mask so it can broadcast over layers and dimensions.
    # New shape: (B, 1, seq_len, 1)
    mask = attention_mask.unsqueeze(1).unsqueeze(-1).to(H.dtype)
    
    # Compute the squared difference, but only for the active tokens.
    diff_squared = ((H - output) * mask) ** 2
    # Sum over the layer, sequence, and dimension axes
    numerator = torch.sqrt(diff_squared.sum(dim=(1, 2, 3)))
    
    # Compute the squared norm of H only for active tokens.
    H_squared = (H * mask) ** 2
    denominator = torch.sqrt(H_squared.sum(dim=(1, 2, 3)))
    
    # Compute normalized RMSE, adding eps to denominator to avoid division by zero.
    nrmse = numerator / (denominator + eps)
    return nrmse

def compute_masked_nrmse_per_vector(H, output, attention_mask, eps=1e-8):
    """
    Compute NRMSE along the final dimension for each vector, then average across all valid positions.
    
    Args:
        H: Tensor of shape (B, L, seq_len, dim) or (B, seq_len, dim)
        output: Tensor of shape (B, L, seq_len, dim) or (B, seq_len, dim)
        attention_mask: Tensor of shape (B, seq_len) with 1 for active tokens, 0 for padding.
        eps: Small constant to avoid division by zero.
        
    Returns:
        nrmse: Tensor of shape (B,) with the averaged vector-wise nRMSE for each sample.
    """
    # Handle 3D input case by adding L dimension
    if len(H.shape) == 3:
        H = H.unsqueeze(1)
        output = output.unsqueeze(1)
    
    # Calculate NRMSE for each vector along the final dimension
    # H and output shape: (B, L, seq_len, dim)
    L = H.shape[1] # number of layers
    diff_norm = torch.norm(H - output, dim=-1)  # (B, L, seq_len)
    H_norm = torch.norm(H, dim=-1)  # (B, L, seq_len)
    
    # Compute vector-wise NRMSE
    vector_nrmse = diff_norm / (H_norm + eps)  # (B, L, seq_len)
    
    # Expand attention mask to match shape: (B, 1, seq_len)
    mask = attention_mask.unsqueeze(1).to(H.dtype)
    
    # Apply mask and compute mean across valid positions
    masked_nrmse = vector_nrmse * mask  # (B, L, seq_len)
    valid_count = L*mask.sum(dim=(1, 2))  # (B,) - total valid positions per batch
    
    # Sum and average across all valid positions
    nrmse = masked_nrmse.sum(dim=(1, 2)) / (valid_count + eps)  # (B,)
    
    return nrmse

def compute_masked_nrmse_per_element(H, output, attention_mask, eps=1e-8):
    """
    Compute NRMSE for each individual element, then average across all valid positions.
    
    Args:
        H: Tensor of shape (B, L, seq_len, dim) or (B, seq_len, dim)
        output: Tensor of shape (B, L, seq_len, dim) or (B, seq_len, dim)
        attention_mask: Tensor of shape (B, seq_len) with 1 for active tokens, 0 for padding.
        eps: Small constant to avoid division by zero.
        
    Returns:
        nrmse: Tensor of shape (B,) with the averaged element-wise nRMSE for each sample.
    """
    # Handle 3D input case by adding L dimension
    if len(H.shape) == 3:
        H = H.unsqueeze(1)
        output = output.unsqueeze(1)
    
    L = H.shape[1] # number of layers
    dim = H.shape[3] # number of dimensions
    
    # Calculate element-wise NRMSE
    # H and output shape: (B, L, seq_len, dim)
    element_diff = torch.abs(H - output)  # (B, L, seq_len, dim)
    element_H = torch.abs(H) + eps  # (B, L, seq_len, dim)
    
    # Compute element-wise relative error
    element_nrmse = element_diff / element_H  # (B, L, seq_len, dim)
    
    # Expand attention mask to match shape: (B, 1, seq_len, 1)
    mask = attention_mask.unsqueeze(1).unsqueeze(-1).to(H.dtype)
    
    # Apply mask and compute mean across valid positions
    masked_nrmse = element_nrmse * mask  # (B, L, seq_len, dim)
    valid_count = L * dim * mask.sum(dim=(1, 2, 3))  # (B,) - total valid elements per batch
    
    # Sum and average across all valid elements
    nrmse = masked_nrmse.sum(dim=(1, 2, 3)) / (valid_count + eps)  # (B,)
    
    return nrmse

def checkModelLoadCorrect(model, param_dict):
    # Check that all keys in encoder_param_dict match the keys in model.state_dict()
    model_state = model.state_dict()
    ckpt_keys = set(param_dict.keys())
    model_keys = set(model_state.keys())

    missing_in_ckpt = model_keys - ckpt_keys
    extra_in_ckpt = ckpt_keys - model_keys

    if missing_in_ckpt:
        print("Warning: The following keys are missing in the checkpoint:")
        for key in missing_in_ckpt:
            print("  ", key)
    else:
        print("All keys from model exist in the checkpoint.")

    if extra_in_ckpt:
        print("Warning: The following keys in the checkpoint do not exist in model:")
        for key in extra_in_ckpt:
            print("  ", key)
    else:
        print("All keys in the checkpoint match those in model.")

    # Additionally, check that the parameter values for matching keys are equal.
    # Handle device mismatches by moving both tensors to CPU for comparison
    mismatched_params = []
    for key in model_keys.intersection(ckpt_keys):
        # Move both tensors to CPU to handle potential device mismatches
        model_param = model_state[key].cpu()
        ckpt_param = param_dict[key].cpu() if hasattr(param_dict[key], 'cpu') else param_dict[key]
        
        if not torch.allclose(model_param, ckpt_param, atol=1e-6):
            mismatched_params.append(key)
    if mismatched_params:
        print("Warning: The following parameters do not match between model and checkpoint:")
        for key in mismatched_params:
            print("  ", key)
    else:
        print("All parameter values match between model and the checkpoint.")

class CFGDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading context-free grammar data from binary files.
    Each sequence in the data is of fixed length with padding tokens (61) at the start.
    The data is stored as int8 but converted to int64 for PyTorch compatibility.
    """
    def __init__(self, data_path, seq_length=7, pad_token_id=61):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to the binary data file
            seq_length (int): Length of each sequence in the data
            pad_token_id (int): Token ID used for padding (default: 61)
        """
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id
        
        # Load the data using numpy memmap with int8 dtype since that's how it's stored
        self.data = np.memmap(data_path, dtype=np.int8, mode='r')
        
        # Calculate number of sequences
        self.num_sequences = len(self.data) // seq_length
        
    def __len__(self):
        """Return the total number of sequences in the dataset."""
        return self.num_sequences
    
    def __getitem__(self, idx):
        """
        Get a single sequence from the dataset.
        
        Args:
            idx (int): Index of the sequence to retrieve
            
        Returns:
            tuple: (sequence, attention_mask)
                - sequence: Tensor of shape (seq_length,) containing token IDs as int64
                - attention_mask: Tensor of shape (seq_length,) with 1 for valid tokens, 0 for padding
        """
        # Calculate start and end indices for the sequence
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        
        # Get the sequence as int8 and convert to int64 for PyTorch compatibility
        sequence = torch.from_numpy(self.data[start_idx:end_idx].copy().astype(np.int64))
        
        # Create attention mask: 0 for padding tokens, 1 for valid tokens
        attention_mask = (sequence != self.pad_token_id).long()
        
        return sequence, attention_mask

def create_cfg_dataloader(data_path, batch_size, seq_length=7, pad_token_id=61, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the CFG dataset.
    
    Args:
        data_path (str): Path to the binary data file
        batch_size (int): Number of sequences per batch
        seq_length (int): Length of each sequence
        pad_token_id (int): Token ID used for padding
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader for the CFG dataset
    """
    dataset = CFGDataset(data_path, seq_length, pad_token_id)
    
    def collate_fn(batch):
        """
        Custom collate function to handle the sequences and attention masks.
        
        Args:
            batch: List of (sequence, attention_mask) tuples
            
        Returns:
            tuple: (sequences, attention_masks)
                - sequences: Tensor of shape (batch_size, seq_length)
                - attention_masks: Tensor of shape (batch_size, seq_length)
        """
        sequences, attention_masks = zip(*batch)
        return torch.stack(sequences), torch.stack(attention_masks)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

def convert_left_to_right_padding(x, mask):
    """
    Convert left-padded sequences and their attention masks to right-padded versions,
    preserving the original padded token values.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T, d) with left padding.
        mask (torch.Tensor): Attention mask of shape (B, T) with zeros in padded positions 
                             (at the beginning) and ones for valid tokens (at the end).
                             
    Returns:
        new_x (torch.Tensor): Tensor with valid tokens at the beginning and original padded tokens
                              appended on the right, of shape (B, T, d).
        new_mask (torch.Tensor): New attention mask of shape (B, T) with ones for valid tokens
                                 at the beginning and zeros for padded tokens at the end.
    """
    B, T, d = x.shape
    new_x_list = []
    new_mask_list = []
    
    for i in range(B):
        valid_count = int(mask[i].sum().item())  # number of valid (ones) tokens
        # Extract valid tokens (originally at the end)
        valid_tokens = x[i, T - valid_count:, :]  # shape: (valid_count, d)
        # Extract padded tokens (original left padded values)
        padded_tokens = x[i, :T - valid_count, :]   # shape: (T - valid_count, d)
        # Concatenate: valid tokens first, padded tokens afterwards
        new_x_i = torch.cat([valid_tokens, padded_tokens], dim=0)  # shape: (T, d)
        new_x_list.append(new_x_i)
        
        # Build new mask: ones for the valid tokens then zeros for padded tokens
        new_mask_i = torch.cat([
            torch.ones(valid_count, dtype=mask.dtype, device=mask.device),
            torch.zeros(T - valid_count, dtype=mask.dtype, device=mask.device)
        ], dim=0)
        new_mask_list.append(new_mask_i)
    
    new_x = torch.stack(new_x_list, dim=0)
    new_mask = torch.stack(new_mask_list, dim=0)
    
    return new_x, new_mask

def convert_right_to_left_padding(x, mask):
    """
    Convert right-padded sequences and their attention masks to left-padded versions,
    preserving the original padded token values.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T, d) with right padding.
        mask (torch.Tensor): Attention mask of shape (B, T) with ones for valid tokens
                             (at the beginning) and zeros in padded positions (at the end).
                             
    Returns:
        new_x (torch.Tensor): Tensor with valid tokens at the end and original padded tokens
                              prepended on the left, of shape (B, T, d).
        new_mask (torch.Tensor): New attention mask of shape (B, T) with zeros for padded tokens
                                 at the beginning and ones for valid tokens at the end.
    """
    B, T, d = x.shape
    new_x_list = []
    new_mask_list = []
    
    for i in range(B):
        valid_count = int(mask[i].sum().item())  # number of valid (ones) tokens
        # Extract valid tokens (originally at the beginning)
        valid_tokens = x[i, :valid_count, :]  # shape: (valid_count, d)
        # Extract padded tokens (original right padded values)
        padded_tokens = x[i, valid_count:, :]  # shape: (T - valid_count, d)
        # Concatenate: padded tokens first, valid tokens afterwards
        new_x_i = torch.cat([padded_tokens, valid_tokens], dim=0)  # shape: (T, d)
        new_x_list.append(new_x_i)
        
        # Build new mask: zeros for padded tokens then ones for valid tokens
        new_mask_i = torch.cat([
            torch.zeros(T - valid_count, dtype=mask.dtype, device=mask.device),
            torch.ones(valid_count, dtype=mask.dtype, device=mask.device)
        ], dim=0)
        new_mask_list.append(new_mask_i)
    
    new_x = torch.stack(new_x_list, dim=0)
    new_mask = torch.stack(new_mask_list, dim=0)
    
    return new_x, new_mask

def test_padding_conversions():
    """
    Test function to verify the padding conversion functions work correctly.
    Tests both convert_left_to_right_padding and convert_right_to_left_padding
    with various test cases.
    """
    # Create some test cases
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test Case 1: Simple batch with different lengths
    x1 = torch.tensor([
        [[0, 0], [0, 0], [1, 1], [2, 2]], # 2 padding, 2 valid
        [[0, 0], [1, 1], [2, 2], [3, 3]]  # 1 padding, 3 valid
    ], dtype=torch.float32, device=device)
    
    mask1 = torch.tensor([
        [0, 0, 1, 1],
        [0, 1, 1, 1]
    ], dtype=torch.long, device=device)
    
    # Test Case 2: All tokens valid
    x2 = torch.tensor([
        [[1, 1], [2, 2], [3, 3], [4, 4]]
    ], dtype=torch.float32, device=device)
    
    mask2 = torch.tensor([
        [1, 1, 1, 1]
    ], dtype=torch.long, device=device)
    
    # Test Case 3: All tokens padded
    x3 = torch.tensor([
        [[0, 0], [0, 0], [0, 0], [0, 0]]
    ], dtype=torch.float32, device=device)
    
    mask3 = torch.tensor([
        [0, 0, 0, 0]
    ], dtype=torch.long, device=device)
    
    test_cases = [
        (x1, mask1, "Different lengths"),
        (x2, mask2, "All valid"),
        (x3, mask3, "All padded")
    ]
    
    for x, mask, desc in test_cases:
        print(f"\nTesting case: {desc}")
        print("Original shape:", x.shape)
        print("Original x:\n", x)
        print("Original mask:\n", mask)
        
        # Test left to right conversion
        right_x, right_mask = convert_left_to_right_padding(x, mask)
        print("\nAfter left->right conversion:")
        print("x:\n", right_x)
        print("mask:\n", right_mask)
        
        # Test right to left conversion
        left_x, left_mask = convert_right_to_left_padding(right_x, right_mask)
        print("\nAfter right->left conversion:")
        print("x:\n", left_x)
        print("mask:\n", left_mask)
        
        # Verify the conversions are reversible
        assert torch.allclose(x, left_x), f"Test case '{desc}': Conversion is not reversible for x"
        assert torch.all(mask == left_mask), f"Test case '{desc}': Conversion is not reversible for mask"
        print(f"✓ Test case '{desc}' passed: conversions are reversible")
        
        # Verify mask properties
        assert torch.all(mask.sum(dim=-1) == right_mask.sum(dim=-1)), \
            f"Test case '{desc}': Number of valid tokens changed during conversion"
        print(f"✓ Test case '{desc}' passed: number of valid tokens preserved")



def path_finding_generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token=None, attention_mask=None):
    """
    Autoregressively generate up to `max_new_tokens` tokens.
    Respects EOS per row: once a row hits EOS, it keeps appending *masked* steps
    so shapes remain consistent, but that row is forced to sample EOS again and
    subsequent positions are padded (masked) in `attention_mask`.
    """
    device = idx.device
    b, t = idx.size()

    # A row is considered finished if its last **valid** token (per attention_mask when given)
    # equals EOS. With left-padding masks (0...0 1...1), we must use the absolute index
    # of the rightmost 1 (not just count-of-ones - 1).
    """if eos_token is not None:
        if attention_mask is not None:
            # valid_counts: number of valid tokens per row
            valid_counts = attention_mask.sum(dim=1)  # (b,)

            # Find absolute index of the last 1 in each mask row:
            # flip -> argmax gives index of first 1 from the right; convert back to absolute
            flip_first_one = (attention_mask.flip(1) != 0).int().argmax(dim=1)  # (b,)
            last_pos_abs = attention_mask.size(1) - 1 - flip_first_one          # (b,)

            # For rows with no valid tokens, guard against bogus indices and force finished=False
            last_pos_abs = torch.where(valid_counts > 0, last_pos_abs, torch.zeros_like(last_pos_abs))

            last_tok = idx.gather(1, last_pos_abs.view(-1, 1)).squeeze(1)       # (b,)
            finished = (valid_counts > 0) & (last_tok == eos_token)
        else:
            finished = (idx[:, -1] == eos_token)
    else:"""
    finished = torch.zeros(b, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        if finished.all():
            break

        # Respect the model context window
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

        # Align/crop attention mask with the current context window
        mask_cond = None
        if attention_mask is not None:
            mask_cond = attention_mask if attention_mask.size(1) == idx_cond.size(1) else attention_mask[:, -idx_cond.size(1):]

        # Forward pass to get logits for next token
        logits, _ = model(idx_cond, attention_mask=mask_cond)
        logits = logits[:, -1, :] / temperature

        # Optional top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')

        # Snapshot current finished rows for this step
        finished_before = finished.clone()

        # For rows that were already finished BEFORE this step, force sampling EOS
        if eos_token is not None and finished_before.any():
            logits[finished_before] = -float('inf')
            logits[finished_before, eos_token] = 0.0

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # (b, 1)

        # Update finished flags AFTER sampling this step
        if eos_token is not None:
            just_finished = (idx_next.squeeze(1) == eos_token) & (~finished)
            finished = finished | just_finished

        # Append next token for every row to keep tensor shapes consistent
        idx = torch.cat((idx, idx_next), dim=1)

        # Extend the attention mask: keep padding (0) after EOS
        if attention_mask is not None:
            # New position is valid (1) for rows that were not already finished
            # and padding (0) for rows that were finished BEFORE this step
            pad = torch.ones((b, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            if finished_before.any():
                pad[finished_before] = 0
            attention_mask = torch.cat((attention_mask, pad), dim=1)

    return idx, attention_mask

@torch.no_grad()
def path_finding_generate_with_hidden_states(model,
                                             idx,
                                             max_new_tokens,
                                             temperature=1.0,
                                             top_k=None,
                                             eos_token=None,
                                             attention_mask=None):
    """
    Same sampling behavior as `path_finding_generate`, but additionally returns
    hidden states for:
      1) the *entire* generated sequence (prompt + new tokens), and
      2) the *generated tokens only*.

    Returns:
        generated_ids:            (B, T_total)
        generated_mask:           (B, T_total)
        all_hidden_states:        (B, L+1, T_total, D)
        gen_tokens_hidden_states: (B, L+1, T_new,   D)
        generated_mask_tokens:    (B, T_new)
    """
    device = idx.device
    B, T0 = idx.size()

    # If no mask is provided, treat all existing tokens as valid
    if attention_mask is None:
        attention_mask = torch.ones_like(idx, dtype=torch.long, device=device)

    # Track finished rows (EOS encountered in *this* generation loop)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    # Collect hidden states
    # 1) Prompt hidden states (may be length-limited by block_size, ensure it's within)
    if T0 > model.config.block_size:
        prompt_ids_cond = idx[:, -model.config.block_size:]
        prompt_mask_cond = attention_mask[:, -model.config.block_size:]
    else:
        prompt_ids_cond = idx
        prompt_mask_cond = attention_mask

    prompt_hidden = model.hidden_states(prompt_ids_cond, attention_mask=prompt_mask_cond)  # (B, L+1, T_prompt, D)

    # Keep a running list of per-step hidden states for the *new* token only
    gen_steps_h_list = []  # each element: (B, L+1, 1, D)

    # Autoregressive generation loop
    for _ in range(max_new_tokens):
        if finished.all():
            break

        # Respect model context window
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        mask_cond = attention_mask if attention_mask.size(1) == idx_cond.size(1) else attention_mask[:, -idx_cond.size(1):]

        # Forward for logits of last position
        logits, _ = model(idx_cond, attention_mask=mask_cond)

        # Choose next token based on temperature
        if temperature == 0.0:
            logits = logits[:, -1, :]
        else:
            logits = logits[:, -1, :] / temperature

        # Optional top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')

        finished_before = finished.clone()

        # Force EOS for rows that were already finished in the *previous* step
        if eos_token is not None and finished_before.any():
            logits[finished_before] = -float('inf')
            logits[finished_before, eos_token] = 0.0

        if temperature == 0.0:
            # Deterministic selection: choose argmax
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
        else:
            # Stochastic selection: use softmax sampling
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        # print(f"shapes of logits: {logits.shape}, idx_next: {idx_next.shape}, idx: {idx.shape}")

        # Update finished flags AFTER sampling this step
        if eos_token is not None:
            just_finished = (idx_next.squeeze(1) == eos_token) & (~finished)
            finished = finished | just_finished

        # Append new token
        idx = torch.cat((idx, idx_next), dim=1)

        # Extend the attention mask: keep padding (0) after EOS
        pad_col = torch.ones((B, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        if finished_before.any():
            pad_col[finished_before] = 0
        attention_mask = torch.cat((attention_mask, pad_col), dim=1)

        # === Capture hidden states for the *new* token ===
        idx_cond2 = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        mask_cond2 = attention_mask if attention_mask.size(1) == idx_cond2.size(1) else attention_mask[:, -idx_cond2.size(1):]

        h_full = model.hidden_states(idx_cond2, attention_mask=mask_cond2)  # (B, L+1, T_cond2, D)
        h_last = h_full[:, :, -1:, :]  # (B, L+1, 1, D)
        gen_steps_h_list.append(h_last)

    # Stack the per-step hidden states for generated tokens (or create empty if none)
    if len(gen_steps_h_list) > 0:
        gen_tokens_hidden_states = torch.cat(gen_steps_h_list, dim=2)  # (B, L+1, T_new, D)
    else:
        # No new tokens generated
        Lp1 = prompt_hidden.size(1)
        D = prompt_hidden.size(-1)
        gen_tokens_hidden_states = torch.empty(B, Lp1, 0, D, device=device, dtype=prompt_hidden.dtype)

    # Build the full-sequence hidden states by concatenating prompt + generated-token states
    all_hidden_states = torch.cat([prompt_hidden, gen_tokens_hidden_states], dim=2)  # (B, L+1, T_total, D)

    # Final outputs
    generated_ids = idx
    generated_mask = attention_mask

    # Masks aligned to *generated tokens only*
    T_new = generated_ids.size(1) - T0
    if T_new > 0:
        generated_mask_tokens = generated_mask[:, -T_new:]
    else:
        generated_mask_tokens = torch.empty(B, 0, device=device, dtype=generated_mask.dtype)

    return (
        generated_ids,
        generated_mask,
        all_hidden_states,
        gen_tokens_hidden_states,
        generated_mask_tokens,
    )

def log_and_plot_usage_for_model(model, model_key, output_dir, wandb_flag):
    try:
        _model = model.module if hasattr(model, 'module') else model
        if not hasattr(_model, 'get_usage_statistics'):
            logging.info(f"{model_key}: get_usage_statistics not available; skipping usage logging")
            return

        usage_stats = _model.get_usage_statistics()
        similarity_stats = None
        if hasattr(_model, 'compute_codebook_similarities'):
            similarity_stats = _model.compute_codebook_similarities()

        logging.info(
            f"{model_key} codebook usage: {usage_stats['unique_vectors_used']}/"
            f"{len(usage_stats['usage_counts'])} vectors used, Total vectors processed: "
            f"{usage_stats['total_vectors_processed']}"
        )

        # Plot usage histogram
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(usage_stats['usage_counts'])), usage_stats['usage_counts'].cpu().numpy())
        plt.title(
            f"{model_key} Codebook Vector Usage\n"
            f"Total vectors processed: {usage_stats['total_vectors_processed']}, "
            f"Unique vectors used: {usage_stats['unique_vectors_used']}/{len(usage_stats['usage_counts'])}"
        )
        plt.xlabel('Codebook Vector Index')
        plt.ylabel('Usage Count')
        plt.savefig(os.path.join(output_dir, 'usage_plots', f'{model_key}_codebook_usage.png'))
        plt.close()

        # Plot similarity heatmap if we have used vectors and stats are available
        if similarity_stats is not None and similarity_stats.get('similarities') is not None:
            plt.figure(figsize=(10, 10))
            plt.imshow(similarity_stats['similarities'].cpu().numpy(), cmap='viridis')
            plt.colorbar()
            plt.title(
                f"Cosine Similarities Between Used VQ-VAE1 Codebook Vectors\n"
                f"Used vectors: {similarity_stats['num_used_vectors']}"
            )
            plt.xlabel('Used Vector Index')
            plt.ylabel('Used Vector Index')
            plt.savefig(os.path.join(output_dir, 'usage_plots', f'{model_key}_codebook_similarities.png'))
            plt.close()

        if wandb_flag:
            try:
                import wandb
                wandb.log({
                    f'{model_key}/unique_vectors_used': usage_stats['unique_vectors_used'],
                    f'{model_key}/total_vectors_processed': usage_stats['total_vectors_processed'],
                    f'{model_key}/usage_percentage': usage_stats['unique_vectors_used'] / len(usage_stats['usage_counts']) * 100
                })
            except Exception:
                pass
    except Exception:
        logging.exception(f"Failed to log/plot usage for {model_key}")
# %%