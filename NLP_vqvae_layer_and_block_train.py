
# %%
import argparse
import os
import time
import math
import pickle
import shutil
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import logging
import gc
import json
import copy
import sys
import matplotlib.pyplot as plt
import random

# Configure logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stdout,
    force=True,
)

from minimal_VQVAEs import VQVAE3v2, Encoder3v2, Decoder3v2
from model import GPTConfig, GPT
from vqvae_utils import log_and_plot_usage_for_model, compute_masked_nrmse

# Load configuration
try:
    parser = argparse.ArgumentParser(description="Train VQVAE_layer model for LLM hidden states")
    parser.add_argument('--config', type=str, required=True,
                       help="Path to configuration file")
    args = parser.parse_args()
    config_path = args.config
except:
    config_path = 'NLP_openwebtext/vqvae_layer_config.json'

# Load configuration
with open(config_path, 'r') as f:
    config = json.load(f)

# %%
# Extract configuration parameters
output_dir = config['output_dir']
save_interval = config['save_interval']
log_interval = config['log_interval']
eval_interval = config['eval_interval']
usage_log_interval = config.get('usage_log_interval', 1000)

# Model configuration
T_max = config['T_max']  # Maximum sequence length
max_attention_window = config['max_attention_window']
assert T_max == max_attention_window, "T_max must equal max_attention_window"

delta_b = config['delta_b']
assert max_attention_window % delta_b == 0, f"max_attention_window {max_attention_window} must be divisible by delta_b {delta_b}"

# Generate b_l values
b_l_values = list(range(delta_b, max_attention_window, delta_b))  # Don't include max_attention_window
b_l_values.append(max_attention_window)  # Include T_max as a possible b value

# Training parameters
batch_size = config['batch_size']
gradient_accumulation_steps = config['gradient_accumulation_steps']
learning_rate = config['learning_rate']
max_iters = config['max_iters']
weight_decay = config['weight_decay']
beta1 = config['beta1']
beta2 = config['beta2']
grad_clip = config['grad_clip']
warmup_iters = config['warmup_iters']
lr_decay_iters = config['lr_decay_iters']
min_lr = config['min_lr']
decay_lr = config['decay_lr']

# Memory optimization flags
train_vqvae_last = config.get('train_vqvae_last', True)  # Default to True for backward compatibility
gradient_checkpointing = config.get('gradient_checkpointing', False)

# Device and dtype
device = config['device']
if device == 'cuda' and not torch.cuda.is_available():
    device = 'cpu'
    logging.warning("CUDA not available, using CPU")

dtype = config['dtype']
if dtype == 'bfloat16' and (device == 'cpu' or not torch.cuda.is_bf16_supported()):
    dtype = 'float32'
    logging.warning("bfloat16 not supported, using float32")

LM_compile = config['LM_compile']
vqvae_compile = config['vqvae_compile']

# Wandb setup
wandb_flag = config['wandb_flag']
wandb_project_name = config['wandb_project_name']
wandb_run_name = config['wandb_run_name']
wandb_group = config['wandb_group']
wandb_entity = config['wandb_entity']

# DDP settings
backend = config['backend']
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

world_size = torch.distributed.get_world_size() if ddp else 1

# Only master process should log to W&B
if not master_process and wandb_flag:
    wandb_flag = False

if master_process:
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
    file_handler.setFormatter(logging.Formatter('[%(levelname)s][%(asctime)s]: %(message)s', datefmt='%H:%M:%S'))
    logging.getLogger().addHandler(file_handler)
    
    # Copy configuration
    try:
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(output_dir, "config.used.json"))
        with open(os.path.join(output_dir, "resolved_config.json"), "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logging.warning(f"Couldn't save config: {e}")
    
    # Setup wandb
    if wandb_flag:
        try:
            import wandb
            os.environ.setdefault("WANDB_START_METHOD", "thread")
            
            run_config = dict(config)
            run_config.update({
                "device": str(device),
                "dtype": dtype,
                "ddp_world_size": ddp_world_size,
                "T_max": T_max,
                "b_l_values": b_l_values,
            })
            
            wandb.init(
                project=wandb_project_name,
                name=wandb_run_name,
                group=wandb_group,
                entity=wandb_entity,
                config=run_config,
                reinit=True,
            )
            logging.info(f"W&B initialized. Run: {getattr(getattr(wandb, 'run', None), 'url', 'N/A')}")
        except Exception as e:
            logging.exception("Failed to initialize W&B; disabling wandb_flag.")
            wandb_flag = False

# Set random seed
torch.manual_seed(config['base_seed'] + seed_offset)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# %%
# Load LLM model
if master_process:
    logging.info("Loading LLM model for hidden state extraction...")

llm_checkpoint_path = config['llm_checkpoint_path']
if not os.path.exists(llm_checkpoint_path):
    raise FileNotFoundError(f"LLM checkpoint not found at {llm_checkpoint_path}")

if master_process:
    logging.info(f"Loading LLM checkpoint from {llm_checkpoint_path}")

# Load checkpoint
checkpoint = torch.load(llm_checkpoint_path, map_location=device)
model_args = checkpoint['model_args']

# Update block_size
model_args_for_hidden_states = model_args.copy()
model_args_for_hidden_states['block_size'] = max_attention_window

# Create GPT model
gptconf = GPTConfig(**model_args_for_hidden_states)
llm_model = GPT(gptconf)

# Load weights
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

llm_model.load_state_dict(state_dict)
logging.info('LLM model state dict loaded')
del checkpoint
torch.cuda.empty_cache() if torch.cuda.is_available() else None

llm_model.to(device)
llm_model.eval()

if LM_compile:
    if master_process:
        logging.info("Compiling LLM model...")
    llm_model = torch.compile(llm_model)

# %%
# Data loading
dataset = config['dataset']
data_dir = os.path.join('data', dataset)

meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size', 50257)
    meta_dtype = meta.get('dtype', np.uint16)
else:
    meta_vocab_size = 50257
    meta_dtype = np.uint16

if master_process:
    logging.info(f"Vocab size: {meta_vocab_size}, dtype: {meta_dtype}")

def get_batch(split='train'):
    """Get a batch of data"""
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=meta_dtype, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=meta_dtype, mode='r')
    
    ix = torch.randint(len(data) - max_attention_window, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+max_attention_window]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    
    return x

# %%
# Learning rate scheduler
def get_lr(iter_num):
    """Learning rate decay scheduler"""
    if not decay_lr:
        return learning_rate
    
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    
    if iter_num > lr_decay_iters:
        return min_lr
    
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# %%
# Initialize VQVAE_layer model
if master_process:
    logging.info("Initializing VQVAE_layer model...")
    logging.info(f"T_max: {T_max}")
    logging.info(f"b_l values: {b_l_values}")

# Get first batch to determine dimensions
input_ids = get_batch('train')
with torch.no_grad():
    with ctx:
        hidden_states = llm_model.hidden_states(input_ids)  # (B, n_layer+1, T, n_embd)

B, num_layers_total, T, d = hidden_states.shape
n_layers = num_layers_total - 1  # Exclude final layer norm
n_embd = d

# Update config with actual dimensions
config['d'] = n_embd
config['n_layers'] = n_layers

if master_process:
    logging.info(f"Hidden state dimensions: B={B}, n_layers={n_layers}, T={T}, d={d}")
    logging.info(f"Will process {n_layers - 1} layers (excluding last layer)")

# Create VQVAE3v2_layer model
vqvae_layer_config = config['vqvae_layer_config'].copy()
vqvae_layer_config.update({
    'd': n_embd,
})

# Create encoder for VQVAE3v2_layer (L=1, T=T_max)
encoder_layer = Encoder3v2(
    L=vqvae_layer_config['L'],
    d=vqvae_layer_config['d'],
    d2=vqvae_layer_config['d2'],
    T=vqvae_layer_config['T'],
    num_layers_layerwise_stage=vqvae_layer_config['num_layers_layerwise_stage'],
    num_layers_aggregate_stage=vqvae_layer_config['num_layers_aggregate_stage'],
    config_layerwise_stage=vqvae_layer_config['config_layerwise_stage'],
    config_aggregate_stage=vqvae_layer_config['config_aggregate_stage']
)

# Create decoder for VQVAE3v2_layer
decoder_layer = Decoder3v2(
    L=vqvae_layer_config['L'],
    d=vqvae_layer_config['d'],
    d2=vqvae_layer_config['d2'],
    T=vqvae_layer_config['T'],
    num_layers_aggregate_stage=vqvae_layer_config['num_layers_aggregate_stage'],
    num_layers_layerwise_stage=vqvae_layer_config['num_layers_layerwise_stage'],
    config_aggregate_stage=vqvae_layer_config['config_aggregate_stage'],
    config_layerwise_stage=vqvae_layer_config['config_layerwise_stage']#,
    #tied_encoder_proj=encoder_layer.proj
)

# Create VQVAE3v2_layer model
vqvae3v2_layer = VQVAE3v2(encoder_layer, decoder_layer, vqvae_layer_config)
vqvae3v2_layer.to(device)

# Create VQVAE3v2_block model
vqvae_block_config = config['vqvae_block_config'].copy()
vqvae_block_config.update({
    'd': n_embd,
})

# Create encoder for VQVAE3v2_block (L=1, T=delta_b)
encoder_block = Encoder3v2(
    L=vqvae_block_config['L'],
    d=vqvae_block_config['d'],
    d2=vqvae_block_config['d2'],
    T=vqvae_block_config['T'],
    num_layers_layerwise_stage=vqvae_block_config['num_layers_layerwise_stage'],
    num_layers_aggregate_stage=vqvae_block_config['num_layers_aggregate_stage'],
    config_layerwise_stage=vqvae_block_config['config_layerwise_stage'],
    config_aggregate_stage=vqvae_block_config['config_aggregate_stage']
)

# Create decoder for VQVAE3v2_block
decoder_block = Decoder3v2(
    L=vqvae_block_config['L'],
    d=vqvae_block_config['d'],
    d2=vqvae_block_config['d2'],
    T=vqvae_block_config['T'],
    num_layers_aggregate_stage=vqvae_block_config['num_layers_aggregate_stage'],
    num_layers_layerwise_stage=vqvae_block_config['num_layers_layerwise_stage'],
    config_aggregate_stage=vqvae_block_config['config_aggregate_stage'],
    config_layerwise_stage=vqvae_block_config['config_layerwise_stage']#,
    #tied_encoder_proj=encoder_block.proj
)

# Create VQVAE3v2_block model
vqvae3v2_block = VQVAE3v2(encoder_block, decoder_block, vqvae_block_config)
vqvae3v2_block.to(device)

# Create optimizer for VQVAE3v2_layer
optimizer_layer = optim.AdamW(
    vqvae3v2_layer.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay
)

# Create optimizer for VQVAE3v2_block
optimizer_block = optim.AdamW(
    vqvae3v2_block.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay
)

# Compile if requested
if vqvae_compile:
    if master_process:
        logging.info("Compiling VQVAE models...")
    try:
        vqvae3v2_layer = torch.compile(vqvae3v2_layer)
        vqvae3v2_block = torch.compile(vqvae3v2_block)
    except Exception as e:
        logging.warning(f"Failed to compile VQVAE models: {e}")

# Wrap with DDP
if ddp:
    vqvae3v2_layer = DDP(vqvae3v2_layer, device_ids=[ddp_local_rank])
    vqvae3v2_block = DDP(vqvae3v2_block, device_ids=[ddp_local_rank])

if master_process:
    # Count and log VQVAE3v2_layer parameters
    layer_total_params = sum(p.numel() for p in vqvae3v2_layer.parameters())
    layer_trainable_params = sum(p.numel() for p in vqvae3v2_layer.parameters() if p.requires_grad)
    logging.info(f"VQVAE3v2_layer parameters: {layer_total_params:,} total, {layer_trainable_params:,} trainable ({layer_total_params*4/1e9:.2f} GB)")
    
    # Count and log VQVAE3v2_block parameters
    block_total_params = sum(p.numel() for p in vqvae3v2_block.parameters())
    block_trainable_params = sum(p.numel() for p in vqvae3v2_block.parameters() if p.requires_grad)
    logging.info(f"VQVAE3v2_block parameters: {block_total_params:,} total, {block_trainable_params:,} trainable ({block_total_params*4/1e9:.2f} GB)")
    
    # Log component breakdown for layer model
    layer_encoder_params = sum(p.numel() for p in vqvae3v2_layer.encoder.parameters())
    layer_decoder_params = sum(p.numel() for p in vqvae3v2_layer.decoder.parameters())
    layer_codebook_params = sum(p.numel() for p in vqvae3v2_layer.codebook.parameters()) if hasattr(vqvae3v2_layer, 'codebook') and vqvae3v2_layer.codebook is not None else 0
    logging.info(f"  Layer Encoder: {layer_encoder_params:,} params ({layer_encoder_params*4/1e6:.1f} MB)")
    logging.info(f"  Layer Decoder: {layer_decoder_params:,} params ({layer_decoder_params*4/1e9:.2f} GB)")
    logging.info(f"  Layer Codebook: {layer_codebook_params:,} params ({layer_codebook_params*4/1e6:.1f} MB)")
    
    # Log component breakdown for block model
    block_encoder_params = sum(p.numel() for p in vqvae3v2_block.encoder.parameters())
    block_decoder_params = sum(p.numel() for p in vqvae3v2_block.decoder.parameters())
    block_codebook_params = sum(p.numel() for p in vqvae3v2_block.codebook.parameters()) if hasattr(vqvae3v2_block, 'codebook') and vqvae3v2_block.codebook is not None else 0
    logging.info(f"  Block Encoder: {block_encoder_params:,} params ({block_encoder_params*4/1e6:.1f} MB)")
    logging.info(f"  Block Decoder: {block_decoder_params:,} params ({block_decoder_params*4/1e9:.2f} GB)")
    logging.info(f"  Block Codebook: {block_codebook_params:,} params ({block_codebook_params*4/1e6:.1f} MB)")
    
    logging.info("VQVAE3v2_layer and VQVAE3v2_block models initialized")

# %%
# Data processing functions
def process_hidden_states_for_vqvae3v2_layer(hidden_states):
    """
    Process hidden states for VQVAE3v2_layer training.
    
    Args:
        hidden_states: (B, n_layer+1, T_max, d)
        
    Returns:
        layer_data: (B*(n_layer-1), 1, T_max, d) for VQVAE3v2_layer
    """
    B, n_layer_plus_1, T_max, d = hidden_states.shape
    n_layer = n_layer_plus_1 - 1
    
    # Exclude last two layers (last layer and final layer norm)
    layer_data = hidden_states[:, :n_layer-1, :, :]  # (B, n_layer-1, T_max, d)
    
    # Reshape to treat layers as samples
    layer_data = layer_data.reshape(B * (n_layer - 1), T_max, d)  # (B*(n_layer-1), T_max, d)
    
    # Normalize by L2 norm (as done in original script)
    layer_data = layer_data / torch.norm(layer_data, dim=-1, keepdim=True)
    
    # Prepare data for VQVAE3v2_layer: (B*(n_layer-1), 1, T_max, d)
    layer_data_for_vqvae = layer_data.unsqueeze(1)  # Add L dimension
    
    return layer_data_for_vqvae

def process_hidden_states_for_vqvae3v2_block(hidden_states, delta_b):
    """
    Process hidden states for VQVAE3v2_block training.
    
    Args:
        hidden_states: (B, n_layer+1, T_max, d)
        delta_b: Block size for block model
        
    Returns:
        block_data: (B*(n_layer-1)*num_blocks, 1, delta_b, d) for VQVAE3v2_block
    """
    B, n_layer_plus_1, T_max, d = hidden_states.shape
    n_layer = n_layer_plus_1 - 1
    
    # Exclude last two layers (last layer and final layer norm)
    layer_data = hidden_states[:, :n_layer-1, :, :]  # (B, n_layer-1, T_max, d)
    
    # Reshape to treat layers as samples
    layer_data = layer_data.reshape(B * (n_layer - 1), T_max, d)  # (B*(n_layer-1), T_max, d)
    
    # Normalize by L2 norm (as done in original script)
    layer_data = layer_data / torch.norm(layer_data, dim=-1, keepdim=True)
    
    # Prepare data for VQVAE3v2_block
    num_blocks = T_max // delta_b
    block_data_list = []
    
    for i in range(layer_data.shape[0]):  # For each layer sample
        seq = layer_data[i]  # (T_max, d)
        
        # Split into blocks
        for block_idx in range(num_blocks):
            start_idx = block_idx * delta_b
            end_idx = start_idx + delta_b
            block = seq[start_idx:end_idx, :]  # (delta_b, d)
            block_data_list.append(block)
    
    # Stack block data: (B*(n_layer-1)*num_blocks, delta_b, d)
    block_data = torch.stack(block_data_list, dim=0)
    
    # Add L dimension for VQVAE3v2_block: (B*(n_layer-1)*num_blocks, 1, delta_b, d)
    block_data_for_vqvae = block_data.unsqueeze(1)
    
    return block_data_for_vqvae

# %%
# Training loop
iter_num = 0
best_loss = float('inf')

# Histories for VQVAE3v2_layer
layer_loss_history = []
layer_recon_loss_history = []
layer_codebook_loss_history = []
layer_commitment_loss_history = []
layer_cosine_push_loss_history = []
layer_entropy_loss_history = []
layer_nrmse_history = []
layer_unique_count_history = []

# Histories for VQVAE3v2_block
block_loss_history = []
block_recon_loss_history = []
block_codebook_loss_history = []
block_commitment_loss_history = []
block_cosine_push_loss_history = []
block_entropy_loss_history = []
block_nrmse_history = []
block_unique_count_history = []

if master_process:
    logging.info("Starting training...")

t0 = time.time()

while iter_num < max_iters:
    # Set learning rate
    lr = get_lr(iter_num)
    for param_group in optimizer_layer.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_block.param_groups:
        param_group['lr'] = lr
    
    # Zero gradients
    optimizer_layer.zero_grad(set_to_none=True)
    optimizer_block.zero_grad(set_to_none=True)
    
    # Accumulate gradients over multiple steps for VQVAE3v2_layer
    layer_total_loss_accum = 0.0
    layer_recon_loss_accum = 0.0
    layer_codebook_loss_accum = 0.0
    layer_commitment_loss_accum = 0.0
    layer_cosine_push_loss_accum = 0.0
    layer_entropy_loss_accum = 0.0
    layer_nrmse_accum = 0.0
    layer_unique_count_accum = 0.0
    layer_num_sequences_accum = 0
    
    # Accumulate gradients over multiple steps for VQVAE3v2_block
    block_total_loss_accum = 0.0
    block_recon_loss_accum = 0.0
    block_codebook_loss_accum = 0.0
    block_commitment_loss_accum = 0.0
    block_cosine_push_loss_accum = 0.0
    block_entropy_loss_accum = 0.0
    block_nrmse_accum = 0.0
    block_unique_count_accum = 0.0
    block_num_sequences_accum = 0
    
    for micro_step in range(gradient_accumulation_steps):
        # DDP gradient sync control
        if ddp:
            sync = (micro_step == gradient_accumulation_steps - 1)
            if hasattr(vqvae3v2_layer, "require_backward_grad_sync"):
                vqvae3v2_layer.require_backward_grad_sync = sync
            if hasattr(vqvae3v2_block, "require_backward_grad_sync"):
                vqvae3v2_block.require_backward_grad_sync = sync
        
        # Get batch
        input_ids = get_batch('train')
        
        # Get hidden states
        with torch.no_grad():
            with ctx:
                hidden_states = llm_model.hidden_states(input_ids)  # (B, n_layer+1, T, n_embd)
        
        # Get dimensions for sequence counting
        B, n_layer_plus_1, T_max, d = hidden_states.shape
        n_layer = n_layer_plus_1 - 1
        
        # Process hidden states for VQVAE3v2_layer first
        layer_data = process_hidden_states_for_vqvae3v2_layer(hidden_states)
        
        # Forward pass through VQVAE3v2_layer
        vqvae3v2_layer.train()
        with ctx:
            layer_x_recon, layer_total_loss, layer_recon_loss, layer_codebook_loss, layer_commitment_loss, layer_unique_count, layer_cosine_push_loss, layer_entropy_loss = vqvae3v2_layer(
                layer_data
            )
        
        # Scale loss for gradient accumulation
        layer_scaled_loss = layer_total_loss / gradient_accumulation_steps
        layer_scaled_loss.backward()
        
        # Compute NRMSE for layer model
        with torch.no_grad():
            # Create attention mask for full sequence (all ones)
            layer_attention_mask = torch.ones(layer_data.shape[0], layer_data.shape[2], device=layer_data.device)
            layer_nrmse = compute_masked_nrmse(layer_data, layer_x_recon, layer_attention_mask)
        
        # Clear layer data to free memory
        del layer_data, layer_x_recon
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Process hidden states for VQVAE3v2_block
        block_data = process_hidden_states_for_vqvae3v2_block(hidden_states, delta_b)
        
        # Forward pass through VQVAE3v2_block
        vqvae3v2_block.train()
        with ctx:
            block_x_recon, block_total_loss, block_recon_loss, block_codebook_loss, block_commitment_loss, block_unique_count, block_cosine_push_loss, block_entropy_loss = vqvae3v2_block(
                block_data
            )
        
        # Scale loss for gradient accumulation
        block_scaled_loss = block_total_loss / gradient_accumulation_steps
        block_scaled_loss.backward()
        
        # Compute NRMSE for block model
        with torch.no_grad():
            # Create attention mask for full sequence (all ones)
            block_attention_mask = torch.ones(block_data.shape[0], block_data.shape[2], device=block_data.device)
            block_nrmse = compute_masked_nrmse(block_data, block_x_recon, block_attention_mask)
        
        # Clear block data to free memory
        del block_data, block_x_recon
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Accumulate metrics for VQVAE3v2_layer
        layer_total_loss_accum += layer_total_loss.item()
        layer_recon_loss_accum += layer_recon_loss.item()
        layer_codebook_loss_accum += layer_codebook_loss.item()
        layer_commitment_loss_accum += layer_commitment_loss.item()
        layer_cosine_push_loss_accum += layer_cosine_push_loss.item()
        layer_entropy_loss_accum += layer_entropy_loss.item()
        layer_nrmse_accum += layer_nrmse.mean().item()
        layer_unique_count_accum += layer_unique_count
        layer_num_sequences_accum += B * (n_layer - 1)  # Number of layer samples
        
        # Accumulate metrics for VQVAE3v2_block
        block_total_loss_accum += block_total_loss.item()
        block_recon_loss_accum += block_recon_loss.item()
        block_codebook_loss_accum += block_codebook_loss.item()
        block_commitment_loss_accum += block_commitment_loss.item()
        block_cosine_push_loss_accum += block_cosine_push_loss.item()
        block_entropy_loss_accum += block_entropy_loss.item()
        block_nrmse_accum += block_nrmse.mean().item()
        block_unique_count_accum += block_unique_count
        block_num_sequences_accum += B * (n_layer - 1) * (T_max // delta_b)  # Number of block samples
    
    # Average metrics for VQVAE3v2_layer
    layer_avg_total_loss = layer_total_loss_accum / gradient_accumulation_steps
    layer_avg_recon_loss = layer_recon_loss_accum / gradient_accumulation_steps
    layer_avg_codebook_loss = layer_codebook_loss_accum / gradient_accumulation_steps
    layer_avg_commitment_loss = layer_commitment_loss_accum / gradient_accumulation_steps
    layer_avg_cosine_push_loss = layer_cosine_push_loss_accum / gradient_accumulation_steps
    layer_avg_entropy_loss = layer_entropy_loss_accum / gradient_accumulation_steps
    layer_avg_nrmse = layer_nrmse_accum / gradient_accumulation_steps
    layer_avg_unique_count = layer_unique_count_accum / gradient_accumulation_steps
    
    # Average metrics for VQVAE3v2_block
    block_avg_total_loss = block_total_loss_accum / gradient_accumulation_steps
    block_avg_recon_loss = block_recon_loss_accum / gradient_accumulation_steps
    block_avg_codebook_loss = block_codebook_loss_accum / gradient_accumulation_steps
    block_avg_commitment_loss = block_commitment_loss_accum / gradient_accumulation_steps
    block_avg_cosine_push_loss = block_cosine_push_loss_accum / gradient_accumulation_steps
    block_avg_entropy_loss = block_entropy_loss_accum / gradient_accumulation_steps
    block_avg_nrmse = block_nrmse_accum / gradient_accumulation_steps
    block_avg_unique_count = block_unique_count_accum / gradient_accumulation_steps
    
    # All-reduce metrics for DDP
    if ddp:
        # All-reduce VQVAE3v2_layer metrics
        for metric_name, metric_val in [
            ('layer_total_loss', layer_avg_total_loss),
            ('layer_recon_loss', layer_avg_recon_loss),
            ('layer_codebook_loss', layer_avg_codebook_loss),
            ('layer_commitment_loss', layer_avg_commitment_loss),
            ('layer_cosine_push_loss', layer_avg_cosine_push_loss),
            ('layer_entropy_loss', layer_avg_entropy_loss),
            ('layer_nrmse', layer_avg_nrmse),
            ('layer_unique_count', layer_avg_unique_count)
        ]:
            tensor = torch.tensor(metric_val, device=device)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            if metric_name == 'layer_total_loss':
                layer_avg_total_loss = tensor.item() / world_size
            elif metric_name == 'layer_recon_loss':
                layer_avg_recon_loss = tensor.item() / world_size
            elif metric_name == 'layer_codebook_loss':
                layer_avg_codebook_loss = tensor.item() / world_size
            elif metric_name == 'layer_commitment_loss':
                layer_avg_commitment_loss = tensor.item() / world_size
            elif metric_name == 'layer_cosine_push_loss':
                layer_avg_cosine_push_loss = tensor.item() / world_size
            elif metric_name == 'layer_entropy_loss':
                layer_avg_entropy_loss = tensor.item() / world_size
            elif metric_name == 'layer_nrmse':
                layer_avg_nrmse = tensor.item() / world_size
            elif metric_name == 'layer_unique_count':
                layer_avg_unique_count = tensor.item() / world_size
        
        # All-reduce VQVAE3v2_block metrics
        for metric_name, metric_val in [
            ('block_total_loss', block_avg_total_loss),
            ('block_recon_loss', block_avg_recon_loss),
            ('block_codebook_loss', block_avg_codebook_loss),
            ('block_commitment_loss', block_avg_commitment_loss),
            ('block_cosine_push_loss', block_avg_cosine_push_loss),
            ('block_entropy_loss', block_avg_entropy_loss),
            ('block_nrmse', block_avg_nrmse),
            ('block_unique_count', block_avg_unique_count)
        ]:
            tensor = torch.tensor(metric_val, device=device)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            if metric_name == 'block_total_loss':
                block_avg_total_loss = tensor.item() / world_size
            elif metric_name == 'block_recon_loss':
                block_avg_recon_loss = tensor.item() / world_size
            elif metric_name == 'block_codebook_loss':
                block_avg_codebook_loss = tensor.item() / world_size
            elif metric_name == 'block_commitment_loss':
                block_avg_commitment_loss = tensor.item() / world_size
            elif metric_name == 'block_cosine_push_loss':
                block_avg_cosine_push_loss = tensor.item() / world_size
            elif metric_name == 'block_entropy_loss':
                block_avg_entropy_loss = tensor.item() / world_size
            elif metric_name == 'block_nrmse':
                block_avg_nrmse = tensor.item() / world_size
            elif metric_name == 'block_unique_count':
                block_avg_unique_count = tensor.item() / world_size
    
    # Gradient clipping
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(vqvae3v2_layer.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(vqvae3v2_block.parameters(), grad_clip)
    
    # Optimizer steps
    optimizer_layer.step()
    optimizer_block.step()
    
    # Normalize codebook vectors if needed
    layer_to_normalize = vqvae3v2_layer.module if hasattr(vqvae3v2_layer, 'module') else vqvae3v2_layer
    if hasattr(layer_to_normalize, 'normalize_codebook_vectors'):
        layer_to_normalize.normalize_codebook_vectors()
    
    block_to_normalize = vqvae3v2_block.module if hasattr(vqvae3v2_block, 'module') else vqvae3v2_block
    if hasattr(block_to_normalize, 'normalize_codebook_vectors'):
        block_to_normalize.normalize_codebook_vectors()
    
    # Store history for VQVAE3v2_layer
    layer_loss_history.append(layer_avg_total_loss)
    layer_recon_loss_history.append(layer_avg_recon_loss)
    layer_codebook_loss_history.append(layer_avg_codebook_loss)
    layer_commitment_loss_history.append(layer_avg_commitment_loss)
    layer_cosine_push_loss_history.append(layer_avg_cosine_push_loss)
    layer_entropy_loss_history.append(layer_avg_entropy_loss)
    layer_nrmse_history.append(layer_avg_nrmse)
    layer_unique_count_history.append(layer_avg_unique_count)
    
    # Store history for VQVAE3v2_block
    block_loss_history.append(block_avg_total_loss)
    block_recon_loss_history.append(block_avg_recon_loss)
    block_codebook_loss_history.append(block_avg_codebook_loss)
    block_commitment_loss_history.append(block_avg_commitment_loss)
    block_cosine_push_loss_history.append(block_avg_cosine_push_loss)
    block_entropy_loss_history.append(block_avg_entropy_loss)
    block_nrmse_history.append(block_avg_nrmse)
    block_unique_count_history.append(block_avg_unique_count)
    
    # Logging
    if iter_num % log_interval == 0 and master_process:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        # Log VQVAE3v2_layer metrics
        logging.info(f"Iter {iter_num}: VQVAE3v2_layer loss={layer_avg_total_loss:.4f}, lr={lr:.6f}, time={dt:.2f}s")
        logging.info(f"  layer_recon={layer_avg_recon_loss:.4f}, layer_cb={layer_avg_codebook_loss:.4f}, layer_commit={layer_avg_commitment_loss:.4f}")
        logging.info(f"  layer_cos_push={layer_avg_cosine_push_loss:.4f}, layer_entropy={layer_avg_entropy_loss:.4f}")
        logging.info(f"  layer_nrmse={layer_avg_nrmse:.4f}, layer_unique_codes={layer_avg_unique_count:.1f}/{vqvae_layer_config['codebook_size']}")
        logging.info(f"  layer_sequences_per_step={layer_num_sequences_accum}, grad_accum={gradient_accumulation_steps}")
        
        # Log VQVAE3v2_block metrics
        logging.info(f"  VQVAE3v2_block loss={block_avg_total_loss:.4f}")
        logging.info(f"  block_recon={block_avg_recon_loss:.4f}, block_cb={block_avg_codebook_loss:.4f}, block_commit={block_avg_commitment_loss:.4f}")
        logging.info(f"  block_cos_push={block_avg_cosine_push_loss:.4f}, block_entropy={block_avg_entropy_loss:.4f}")
        logging.info(f"  block_nrmse={block_avg_nrmse:.4f}, block_unique_codes={block_avg_unique_count:.1f}/{vqvae_block_config['codebook_size']}")
        logging.info(f"  block_sequences_per_step={block_num_sequences_accum}")
        
        # Update best loss (use combined loss)
        combined_loss = layer_avg_total_loss + block_avg_total_loss
        if combined_loss < best_loss:
            best_loss = combined_loss
            logging.info(f"  New best combined loss: {best_loss:.4f}")
        
        # W&B logging
        if wandb_flag:
            log_data = {
                'iter': iter_num,
                'lr': lr,
                'time': dt,
                # VQVAE3v2_layer metrics
                'vqvae3v2_layer/total_loss': layer_avg_total_loss,
                'vqvae3v2_layer/recon_loss': layer_avg_recon_loss,
                'vqvae3v2_layer/codebook_loss': layer_avg_codebook_loss,
                'vqvae3v2_layer/commitment_loss': layer_avg_commitment_loss,
                'vqvae3v2_layer/cosine_push_loss': layer_avg_cosine_push_loss,
                'vqvae3v2_layer/entropy_loss': layer_avg_entropy_loss,
                'vqvae3v2_layer/nrmse': layer_avg_nrmse,
                'vqvae3v2_layer/unique_codes': layer_avg_unique_count,
                'vqvae3v2_layer/codebook_usage_ratio': layer_avg_unique_count / vqvae_layer_config['codebook_size'],
                # VQVAE3v2_block metrics
                'vqvae3v2_block/total_loss': block_avg_total_loss,
                'vqvae3v2_block/recon_loss': block_avg_recon_loss,
                'vqvae3v2_block/codebook_loss': block_avg_codebook_loss,
                'vqvae3v2_block/commitment_loss': block_avg_commitment_loss,
                'vqvae3v2_block/cosine_push_loss': block_avg_cosine_push_loss,
                'vqvae3v2_block/entropy_loss': block_avg_entropy_loss,
                'vqvae3v2_block/nrmse': block_avg_nrmse,
                'vqvae3v2_block/unique_codes': block_avg_unique_count,
                'vqvae3v2_block/codebook_usage_ratio': block_avg_unique_count / vqvae_block_config['codebook_size'],
                # General metrics
                'layer_sequences_per_step': layer_num_sequences_accum,
                'block_sequences_per_step': block_num_sequences_accum,
                'combined_loss': combined_loss,
            }
            
            if torch.cuda.is_available():
                log_data['gpu_memory_allocated_GB'] = torch.cuda.memory_allocated(device) / 1024**3
                log_data['gpu_memory_reserved_GB'] = torch.cuda.memory_reserved(device) / 1024**3
            wandb.log(log_data)
    
    # Codebook usage statistics
    if iter_num % usage_log_interval == 0 and master_process:
        if not os.path.exists(os.path.join(output_dir, 'usage_plots')):
            os.makedirs(os.path.join(output_dir, 'usage_plots'))
        
        # Log usage for VQVAE3v2_layer
        layer_to_log = vqvae3v2_layer.module if hasattr(vqvae3v2_layer, 'module') else vqvae3v2_layer
        log_and_plot_usage_for_model(layer_to_log, "vqvae3v2_layer", output_dir, wandb_flag)
        
        # Log usage for VQVAE3v2_block
        block_to_log = vqvae3v2_block.module if hasattr(vqvae3v2_block, 'module') else vqvae3v2_block
        log_and_plot_usage_for_model(block_to_log, "vqvae3v2_block", output_dir, wandb_flag)
    
    # Save checkpoint
    if iter_num % save_interval == 0 and iter_num > 0 and master_process:
        checkpoint = {
            'iter_num': iter_num,
            # VQVAE3v2_layer state
            'layer_model_state_dict': vqvae3v2_layer.state_dict(),
            'layer_optimizer_state_dict': optimizer_layer.state_dict(),
            'layer_loss_history': layer_loss_history,
            'layer_recon_loss_history': layer_recon_loss_history,
            'layer_codebook_loss_history': layer_codebook_loss_history,
            'layer_commitment_loss_history': layer_commitment_loss_history,
            'layer_cosine_push_loss_history': layer_cosine_push_loss_history,
            'layer_entropy_loss_history': layer_entropy_loss_history,
            'layer_nrmse_history': layer_nrmse_history,
            'layer_unique_count_history': layer_unique_count_history,
            # VQVAE3v2_block state
            'block_model_state_dict': vqvae3v2_block.state_dict(),
            'block_optimizer_state_dict': optimizer_block.state_dict(),
            'block_loss_history': block_loss_history,
            'block_recon_loss_history': block_recon_loss_history,
            'block_codebook_loss_history': block_codebook_loss_history,
            'block_commitment_loss_history': block_commitment_loss_history,
            'block_cosine_push_loss_history': block_cosine_push_loss_history,
            'block_entropy_loss_history': block_entropy_loss_history,
            'block_nrmse_history': block_nrmse_history,
            'block_unique_count_history': block_unique_count_history,
            # General
            'best_loss': best_loss,
            'config': config,
        }
        checkpoint_path = os.path.join(output_dir, f'checkpoint_iter_{iter_num}.pt')
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    iter_num += 1

# Final save
if master_process:
    checkpoint = {
        'iter_num': iter_num,
        # VQVAE3v2_layer state
        'layer_model_state_dict': vqvae3v2_layer.state_dict(),
        'layer_optimizer_state_dict': optimizer_layer.state_dict(),
        'layer_loss_history': layer_loss_history,
        'layer_recon_loss_history': layer_recon_loss_history,
        'layer_codebook_loss_history': layer_codebook_loss_history,
        'layer_commitment_loss_history': layer_commitment_loss_history,
        'layer_cosine_push_loss_history': layer_cosine_push_loss_history,
        'layer_entropy_loss_history': layer_entropy_loss_history,
        'layer_nrmse_history': layer_nrmse_history,
        'layer_unique_count_history': layer_unique_count_history,
        # VQVAE3v2_block state
        'block_model_state_dict': vqvae3v2_block.state_dict(),
        'block_optimizer_state_dict': optimizer_block.state_dict(),
        'block_loss_history': block_loss_history,
        'block_recon_loss_history': block_recon_loss_history,
        'block_codebook_loss_history': block_codebook_loss_history,
        'block_commitment_loss_history': block_commitment_loss_history,
        'block_cosine_push_loss_history': block_cosine_push_loss_history,
        'block_entropy_loss_history': block_entropy_loss_history,
        'block_nrmse_history': block_nrmse_history,
        'block_unique_count_history': block_unique_count_history,
        # General
        'best_loss': best_loss,
        'config': config,
    }
    
    final_path = os.path.join(output_dir, 'checkpoint_final.pt')
    torch.save(checkpoint, final_path)
    logging.info(f"Training completed! Final checkpoint saved to {final_path}")
    
    # Plot final loss curves
    plt.figure(figsize=(16, 12))
    
    # Layer model plots
    plt.subplot(3, 2, 1)
    plt.plot(layer_loss_history)
    plt.title('VQVAE3v2_layer Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(3, 2, 2)
    plt.plot(layer_recon_loss_history, label='Reconstruction')
    plt.plot(layer_codebook_loss_history, label='Codebook')
    plt.plot(layer_commitment_loss_history, label='Commitment')
    plt.legend()
    plt.title('VQVAE3v2_layer Loss Components')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(3, 2, 3)
    plt.plot(layer_nrmse_history)
    plt.title('VQVAE3v2_layer NRMSE')
    plt.xlabel('Iteration')
    plt.ylabel('NRMSE')
    
    plt.subplot(3, 2, 4)
    plt.plot(layer_unique_count_history)
    plt.axhline(y=vqvae_layer_config['codebook_size'], color='r', linestyle='--', label='Codebook Size')
    plt.title('VQVAE3v2_layer Unique Codes Used')
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    plt.legend()
    
    # Block model plots
    plt.subplot(3, 2, 5)
    plt.plot(block_loss_history)
    plt.title('VQVAE3v2_block Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(3, 2, 6)
    plt.plot(block_unique_count_history)
    plt.axhline(y=vqvae_block_config['codebook_size'], color='r', linestyle='--', label='Codebook Size')
    plt.title('VQVAE3v2_block Unique Codes Used')
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    # Usage statistics
    layer_to_stats = vqvae3v2_layer.module if hasattr(vqvae3v2_layer, 'module') else vqvae3v2_layer
    layer_usage_stats = layer_to_stats.get_usage_statistics()
    logging.info(f"Final VQVAE3v2_layer usage statistics:")
    logging.info(f"  Total vectors processed: {layer_usage_stats['total_vectors_processed']}")
    logging.info(f"  Unique vectors used: {layer_usage_stats['unique_vectors_used']}/{vqvae_layer_config['codebook_size']}")
    logging.info(f"  Usage ratio: {layer_usage_stats['unique_vectors_used']/vqvae_layer_config['codebook_size']*100:.1f}%")
    
    block_to_stats = vqvae3v2_block.module if hasattr(vqvae3v2_block, 'module') else vqvae3v2_block
    block_usage_stats = block_to_stats.get_usage_statistics()
    logging.info(f"Final VQVAE3v2_block usage statistics:")
    logging.info(f"  Total vectors processed: {block_usage_stats['total_vectors_processed']}")
    logging.info(f"  Unique vectors used: {block_usage_stats['unique_vectors_used']}/{vqvae_block_config['codebook_size']}")
    logging.info(f"  Usage ratio: {block_usage_stats['unique_vectors_used']/vqvae_block_config['codebook_size']*100:.1f}%")
    
    if wandb_flag:
        wandb.finish()

if ddp:
    destroy_process_group()
