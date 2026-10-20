

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

from minimal_VQVAEs import VQVAELastToken
from model import GPTConfig, GPT
from vqvae_utils import log_and_plot_usage_for_model, compute_masked_nrmse

# Load configuration
try:
    parser = argparse.ArgumentParser(description="Train VQVAE_last model for LLM hidden states")
    parser.add_argument('--config', type=str, required=True,
                       help="Path to configuration file")
    args = parser.parse_args()
    config_path = args.config
except:
    raise ValueError("No configuration file provided")

# Load configuration
with open(config_path, 'r') as f:
    config = json.load(f)

# %%
# Extract configuration parameters
output_dir = config['output_dir']
save_interval = config['save_interval']
log_interval = config['log_interval']
eval_interval = config['eval_interval']
usage_log_interval = config['usage_log_interval']

# Model configuration
T_max = config['T_max']  # Maximum sequence length
max_attention_window = config['max_attention_window']
assert T_max == max_attention_window, "T_max must equal max_attention_window"

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
gradient_checkpointing = config['gradient_checkpointing']

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
    meta_vocab_size = meta['vocab_size']
    meta_dtype = meta['dtype']
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
# Initialize VQVAE_last model
if master_process:
    logging.info("Initializing VQVAE_last model for last layer last token...")

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

# Load VQVAELastToken configuration
vqvae_last_config = config['vqvae_last_config']
# Override input_dim to match LLM's n_embd
vqvae_last_config['input_dim'] = n_embd

if master_process:
    logging.info(f"VQVAELastToken config: {vqvae_last_config}")

# Create VQVAELastToken model
vqvae_last = VQVAELastToken(
    input_dim=vqvae_last_config['input_dim'],
    hidden_dim=vqvae_last_config['hidden_dim'],
    codebook_size=vqvae_last_config['codebook_size'],
    beta=vqvae_last_config['beta'],
    config=vqvae_last_config
)
vqvae_last.to(device)

if master_process:
    logging.info(f"VQVAELastToken parameters: {sum(p.numel() for p in vqvae_last.parameters()):,}")

# Create optimizer for VQVAELastToken
optimizer = optim.AdamW(
    vqvae_last.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay
)

# Compile if requested
if vqvae_compile:
    if master_process:
        logging.info("Compiling VQVAE_last model...")
    try:
        vqvae_last = torch.compile(vqvae_last)
    except Exception as e:
        logging.warning(f"Failed to compile VQVAE_last model: {e}")

# Wrap with DDP
if ddp:
    vqvae_last = DDP(vqvae_last, device_ids=[ddp_local_rank])

if master_process:
    logging.info("VQVAE_last model initialized")

# %%
# Data processing function for VQVAE_last
def process_hidden_states_for_vqvae_last(hidden_states):
    """
    Process hidden states for VQVAE_last training.
    
    Args:
        hidden_states: (B, n_layer+1, T_max, d)
        
    Returns:
        last_token_data: (B, d) - last token from second-to-last layer
    """
    # Get last token from second-to-last layer (exclude final layer norm)
    last_token_data = hidden_states[:, -2, -1, :]  # (B, d)
    
    # Normalize by L2 norm
    last_token_data = last_token_data / torch.norm(last_token_data, dim=-1, keepdim=True)
    
    return last_token_data

# %%
# Training loop
iter_num = 0
best_loss = float('inf')

# Histories for VQVAE_last
loss_history = []
recon_loss_history = []
codebook_loss_history = []
commitment_loss_history = []
cosine_push_loss_history = []
entropy_loss_history = []
nrmse_history = []
unique_count_history = []

if master_process:
    logging.info("Starting training...")

t0 = time.time()

while iter_num < max_iters:
    # Set learning rate
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Zero gradients
    optimizer.zero_grad(set_to_none=True)
    
    # Accumulate gradients over multiple steps for VQVAE_last
    total_loss_accum = 0.0
    recon_loss_accum = 0.0
    codebook_loss_accum = 0.0
    commitment_loss_accum = 0.0
    cosine_push_loss_accum = 0.0
    entropy_loss_accum = 0.0
    nrmse_accum = 0.0
    unique_count_accum = 0.0
    
    for micro_step in range(gradient_accumulation_steps):
        # DDP gradient sync control
        if ddp:
            sync = (micro_step == gradient_accumulation_steps - 1)
            if hasattr(vqvae_last, "require_backward_grad_sync"):
                vqvae_last.require_backward_grad_sync = sync
        
        # Get batch
        input_ids = get_batch('train')
        
        # Get hidden states
        with torch.no_grad():
            with ctx:
                hidden_states = llm_model.hidden_states(input_ids)  # (B, n_layer+1, T, n_embd)
        
        # Process hidden states for VQVAE_last
        last_token_data = process_hidden_states_for_vqvae_last(hidden_states)
        
        # Forward pass through VQVAE_last
        vqvae_last.train()
        with ctx:
            last_recon, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count, cosine_push_loss, entropy_loss = vqvae_last(
                last_token_data
            )
        
        # Scale loss for gradient accumulation
        scaled_loss = total_loss / gradient_accumulation_steps
        scaled_loss.backward()
        
        # Compute NRMSE for last token
        with torch.no_grad():
            nrmse = compute_masked_nrmse(
                last_token_data.unsqueeze(1),  # Add sequence dimension
                last_recon.unsqueeze(1),
                torch.ones(last_token_data.shape[0], 1, device=last_token_data.device)
            )
        
        # Accumulate metrics for VQVAE_last
        total_loss_accum += total_loss.item()
        recon_loss_accum += recon_loss.item()
        codebook_loss_accum += codebook_loss.item()
        commitment_loss_accum += commitment_loss.item()
        cosine_push_loss_accum += cosine_push_loss.item()
        entropy_loss_accum += entropy_loss.item()
        nrmse_accum += nrmse.mean().item()
        unique_count_accum += unique_count
    
    # Average metrics for VQVAE_last
    avg_total_loss = total_loss_accum / gradient_accumulation_steps
    avg_recon_loss = recon_loss_accum / gradient_accumulation_steps
    avg_codebook_loss = codebook_loss_accum / gradient_accumulation_steps
    avg_commitment_loss = commitment_loss_accum / gradient_accumulation_steps
    avg_cosine_push_loss = cosine_push_loss_accum / gradient_accumulation_steps
    avg_entropy_loss = entropy_loss_accum / gradient_accumulation_steps
    avg_nrmse = nrmse_accum / gradient_accumulation_steps
    avg_unique_count = unique_count_accum / gradient_accumulation_steps
    
    # All-reduce metrics for DDP
    if ddp:
        for metric_name, metric_val in [
            ('total_loss', avg_total_loss),
            ('recon_loss', avg_recon_loss),
            ('codebook_loss', avg_codebook_loss),
            ('commitment_loss', avg_commitment_loss),
            ('cosine_push_loss', avg_cosine_push_loss),
            ('entropy_loss', avg_entropy_loss),
            ('nrmse', avg_nrmse),
            ('unique_count', avg_unique_count)
        ]:
            tensor = torch.tensor(metric_val, device=device)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            if metric_name == 'total_loss':
                avg_total_loss = tensor.item() / world_size
            elif metric_name == 'recon_loss':
                avg_recon_loss = tensor.item() / world_size
            elif metric_name == 'codebook_loss':
                avg_codebook_loss = tensor.item() / world_size
            elif metric_name == 'commitment_loss':
                avg_commitment_loss = tensor.item() / world_size
            elif metric_name == 'cosine_push_loss':
                avg_cosine_push_loss = tensor.item() / world_size
            elif metric_name == 'entropy_loss':
                avg_entropy_loss = tensor.item() / world_size
            elif metric_name == 'nrmse':
                avg_nrmse = tensor.item() / world_size
            elif metric_name == 'unique_count':
                avg_unique_count = tensor.item() / world_size
    
    # Gradient clipping
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(vqvae_last.parameters(), grad_clip)
    
    # Optimizer step
    optimizer.step()
    
    # Normalize codebook vectors if needed
    model_to_normalize = vqvae_last.module if hasattr(vqvae_last, 'module') else vqvae_last
    if hasattr(model_to_normalize, 'normalize_codebook_vectors'):
        model_to_normalize.normalize_codebook_vectors()
    
    # Store history for VQVAE_last
    loss_history.append(avg_total_loss)
    recon_loss_history.append(avg_recon_loss)
    codebook_loss_history.append(avg_codebook_loss)
    commitment_loss_history.append(avg_commitment_loss)
    cosine_push_loss_history.append(avg_cosine_push_loss)
    entropy_loss_history.append(avg_entropy_loss)
    nrmse_history.append(avg_nrmse)
    unique_count_history.append(avg_unique_count)
    
    # Logging
    if iter_num % log_interval == 0 and master_process:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        # Log VQVAE_last metrics
        logging.info(f"Iter {iter_num}: VQVAE_last loss={avg_total_loss:.4f}, lr={lr:.6f}, time={dt:.2f}s")
        logging.info(f"  recon={avg_recon_loss:.4f}, cb={avg_codebook_loss:.4f}, commit={avg_commitment_loss:.4f}")
        logging.info(f"  cos_push={avg_cosine_push_loss:.4f}, entropy={avg_entropy_loss:.4f}")
        logging.info(f"  nrmse={avg_nrmse:.4f}, unique_codes={avg_unique_count:.1f}/{vqvae_last_config['codebook_size']}")
        logging.info(f"  grad_accum={gradient_accumulation_steps}")
        
        # Update best loss
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            logging.info(f"  New best loss: {best_loss:.4f}")
        
        # W&B logging
        if wandb_flag:
            log_data = {
                'iter': iter_num,
                'lr': lr,
                'time': dt,
                # VQVAE_last metrics
                'vqvae_last/total_loss': avg_total_loss,
                'vqvae_last/recon_loss': avg_recon_loss,
                'vqvae_last/codebook_loss': avg_codebook_loss,
                'vqvae_last/commitment_loss': avg_commitment_loss,
                'vqvae_last/cosine_push_loss': avg_cosine_push_loss,
                'vqvae_last/entropy_loss': avg_entropy_loss,
                'vqvae_last/nrmse': avg_nrmse,
                'vqvae_last/unique_codes': avg_unique_count,
                'vqvae_last/codebook_usage_ratio': avg_unique_count / vqvae_last_config['codebook_size'],
            }
            
            if torch.cuda.is_available():
                log_data['gpu_memory_allocated_GB'] = torch.cuda.memory_allocated(device) / 1024**3
                log_data['gpu_memory_reserved_GB'] = torch.cuda.memory_reserved(device) / 1024**3
            wandb.log(log_data)
    
    # Codebook usage statistics
    if iter_num % usage_log_interval == 0 and master_process:
        if not os.path.exists(os.path.join(output_dir, 'usage_plots')):
            os.makedirs(os.path.join(output_dir, 'usage_plots'))
        
        # Log usage for VQVAE_last
        model_to_log = vqvae_last.module if hasattr(vqvae_last, 'module') else vqvae_last
        log_and_plot_usage_for_model(model_to_log, "vqvae_last", output_dir, wandb_flag)
    
    # Save checkpoint
    if iter_num % save_interval == 0 and iter_num > 0 and master_process:
        checkpoint = {
            'iter_num': iter_num,
            # VQVAE_last state
            'model_state_dict': vqvae_last.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history,
            'recon_loss_history': recon_loss_history,
            'codebook_loss_history': codebook_loss_history,
            'commitment_loss_history': commitment_loss_history,
            'cosine_push_loss_history': cosine_push_loss_history,
            'entropy_loss_history': entropy_loss_history,
            'nrmse_history': nrmse_history,
            'unique_count_history': unique_count_history,
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
        # VQVAE_last state
        'model_state_dict': vqvae_last.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'recon_loss_history': recon_loss_history,
        'codebook_loss_history': codebook_loss_history,
        'commitment_loss_history': commitment_loss_history,
        'cosine_push_loss_history': cosine_push_loss_history,
        'entropy_loss_history': entropy_loss_history,
        'nrmse_history': nrmse_history,
        'unique_count_history': unique_count_history,
        # General
        'best_loss': best_loss,
        'config': config,
    }
    
    final_path = os.path.join(output_dir, 'checkpoint_final.pt')
    torch.save(checkpoint, final_path)
    logging.info(f"Training completed! Final checkpoint saved to {final_path}")
    
    # Plot final loss curves
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(loss_history)
    plt.title('Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(recon_loss_history, label='Reconstruction')
    plt.plot(codebook_loss_history, label='Codebook')
    plt.plot(commitment_loss_history, label='Commitment')
    plt.legend()
    plt.title('Loss Components')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 3)
    plt.plot(nrmse_history)
    plt.title('NRMSE')
    plt.xlabel('Iteration')
    plt.ylabel('NRMSE')
    
    plt.subplot(2, 2, 4)
    plt.plot(unique_count_history)
    plt.axhline(y=vqvae_last_config['codebook_size'], color='r', linestyle='--', label='Codebook Size')
    plt.title('Unique Codes Used')
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    # Usage statistics
    model_to_stats = vqvae_last.module if hasattr(vqvae_last, 'module') else vqvae_last
    usage_stats = model_to_stats.get_usage_statistics()
    logging.info(f"Final usage statistics:")
    logging.info(f"  Total vectors processed: {usage_stats['total_vectors_processed']}")
    logging.info(f"  Unique vectors used: {usage_stats['unique_vectors_used']}/{vqvae_last_config['codebook_size']}")
    logging.info(f"  Usage ratio: {usage_stats['unique_vectors_used']/vqvae_last_config['codebook_size']*100:.1f}%")
    
    if wandb_flag:
        wandb.finish()

if ddp:
    destroy_process_group()
