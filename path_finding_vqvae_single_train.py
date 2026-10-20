
# %%
import argparse
import os
import time
import math
import pickle
from contextlib import nullcontext
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import logging
import gc # garbage collector

from vqvae_utils import * 
import json
import atexit
from minimal_VQVAEs import *
from path_finding_llm_utilities import PathFindingDataset, custom_collate_fn, crop_prompt_and_attention_mask
from torch.utils.data import DataLoader
from model import GPT, GPTConfig

"""
PF_15XL/LLMout/FINAL_15XL_block_size_128_num_samples_156254208_padding_avare_False
"""

# %%
# Load training configuration from a JSON file
try:
    parser = argparse.ArgumentParser(description="Train VQ-VAE-like model")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    config_path = args.config
except:
    config_path = '--' # './PF_15XL/vqvae_single_config_seed11.json'
    # raise RuntimeError("Failed to parse command line arguments for config file. Please provide --config argument.")

# %%

with open(config_path, 'r') as config_file:
    config = json.load(config_file)
    print('Loaded config from', config_path)

# Default values

# logging and output
output_dir = config.get('output_dir', './sample_output')
save_logs_flag = config.get('save_logs', True)
print_logs = config.get('print_logs', True)
log_interval = config.get('log_interval', 1)
save_interval = config.get('save_interval', 300)  # set <= 0 for no saves
plot_interval = config.get('plot_interval', -1)  # set <= 0 for no plot saves

# seed
base_seed = config.get('base_seed', 6461)

# Language model
language_model_ckpt_folder_path = config.get('language_model_ckpt_folder_path', None)
LM_compile = config.get('LM_compile', True)  # use PyTorch 2.0 to compile the model to be faster

# Data for training the model
gradient_accumulation_steps = config.get('gradient_accumulation_steps')  # used to simulate larger batch sizes
batch_size = config.get('batch_size')  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_new_tokens = config.get('max_new_tokens', 9)  # number of tokens generated in each sample

dataset_dir = config.get('dataset_dir', None)
# For path finding, we use the split from config (e.g., 'vqvae') to determine which JSONL file to use
split = config.get('split', 'vqvae')
max_seq_length = config.get('max_seq_length')  # Maximum sequence length for path finding
max_nodes = config.get('max_nodes')  # Maximum number of nodes for path finding
# training
init_from = config.get('init_from', 'scratch')  # 'scratch' or 'resume'

# Optimizer parameters
learning_rate = config.get('learning_rate', 5e-5)
max_iters = config.get('max_iters', 20)
weight_decay = config.get('weight_decay', 1e-1)
beta1 = config.get('beta1', 0.9)
beta2 = config.get('beta2', 0.95)
grad_clip = config.get('grad_clip', 1.0)

# learning rate decay settings
decay_lr = config.get('decay_lr', True)  # whether to decay the learning rate
warmup_iters = config.get('warmup_iters', 60)  # how many steps to warm up for
lr_decay_iters = config.get('lr_decay_iters', max_iters)  # should be ~= max_iters per Chinchilla
min_lr = config.get('min_lr', 5e-6)  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# commitment loss beta growth settings
vqvae_single_grow_beta = config['vqvae_single_config'].get('grow_beta', False)  # whether to grow the beta parameter
vqvae_single_betainit = config['vqvae_single_config'].get('betainit', 0.25)  # initial value of beta
vqvae_single_betafinal = config['vqvae_single_config'].get('betafinal', 1.0)  # final value of beta
vqvae_single_warmup_iters_beta = config['vqvae_single_config'].get('warmup_iters_beta', max_iters/3) #let's try this once  # how many steps to warm up beta for IN THE FUTURE WE MAY WANT TO CHANGE THIS TO BE DIFFERENT FROM warmup_iters

#VQ-VAE_single model parameters
vqvae_single_config = config.get('vqvae_single_config', {})
vqvae_single_codebook_size = vqvae_single_config.get('codebook_size', 64)
vqvae_single_codebook_reset_counter_multiplier = vqvae_single_config.get('codebook_reset_counter_multiplier', 10) # 0 for no reset
vqvae_single_beta = vqvae_single_config.get('beta', 0.25)
vqvae_single_d = vqvae_single_config.get('d', 128)
vqvae_single_hidden_dim = vqvae_single_config.get('hidden_dim', 128)
vqvae_single_cosine_push_weight = vqvae_single_config['cosine_push_weight']
vqvae_single_entropy_loss_weight = vqvae_single_config['entropy_loss_weight']
vqvae_single_entropy_temperature = vqvae_single_config['entropy_temperature']
vqvae_single_mask_prob = vqvae_single_config['mask_prob']
vqvae_single_usage_tracking_window = vqvae_single_config['usage_tracking_window']

# VQ-model compile
vq_compile = config.get('vq_compile', True)  # use PyTorch 2.0 to compile the model to be faster

# default wandb flag
wandb_flag = config.get('wandb_flag', False)  # whether to use wandb for logging
wandb_project_name = config.get('wandb_project_name', "VQ_VAE_default_project")  # This is experiment name
wandb_run_name = config.get('wandb_run_name', "VQ_VAE_default_project")  # This is run name within each experiment
wandb_group = config.get('wandb_group', "VQ_VAE_default_group")  # This is group name to collect the same runs with different seeds
wandb_entity = config.get('wandb_entity', "llm_analysis")  # This is entity name to collect the same runs with different seeds

# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.

# system
device = config.get('device', 'cuda')  # 'cuda' or 'cpu'
if device == 'cuda' and not torch.cuda.is_available():
    device = 'cpu'
    logging.info("Warning: CUDA is not available, using CPU instead.")

# Determine dtype based on config and device capabilities
if config.get('dtype') == 'bfloat16':
    if 'cuda' in device and torch.cuda.is_bf16_supported():
        dtype = 'bfloat16'
    else:
        dtype = 'float32'
        logging.info("Warning: bfloat16 not supported on this device, using float32 instead.")
else:
    dtype = 'float32'

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, dict, list, tuple)) and (not k.startswith('vqvae_single') or k=='vqvae_single_config') and k!='Out' and k!='first_stage_checkpoint' and k!='config']
config = {k: globals()[k] for k in config_keys}


# %% ------------- Arrange DDP ------------------------------

# various inits, derived attributes, I/O setup 
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# Check if all GPUs are being used - multi gpu run is not working TODO: fix it
if ddp:
    total_gpus = torch.cuda.device_count()
    print(f"Using Distributed Data Parallel (DDP) with {ddp_world_size} processes.")
    print(f"Total GPUs available: {total_gpus}")
    assert ddp_world_size <= total_gpus, "DDP world size exceeds the number of available GPUs."
    print(f"Each process is using GPU: {ddp_local_rank}")
else:
    print("Not using DDP. Running on a single GPU.")

if master_process:
    logging.basicConfig(format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, 'INFO'), datefmt='%H:%M:%S')
    logger = get_logger(save_logs_flag = save_logs_flag, print_logs = print_logs, experiment_dir = output_dir)


torch.manual_seed(base_seed + seed_offset)

torch.set_float32_matmul_precision('high') 
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
# NOTE: March, 6, 2025: Before using Float16, make sure that gradientscaler works.
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# %% ------------- Logging with wandb ------------------------------
if wandb_flag:
    if not master_process:
        wandb_flag = False # only the master process should log to wandb
    else:
        import wandb
        if init_from == 'resume':

            # Check if the run is already created or not
            # Specify the project details
            entity = "llm_analysis"  # Use the specified entity
            project = config['wandb_project_name']  # Use the project name from the config
            # Use the API to fetch all runs
            api = wandb.Api()
            runs = api.runs(f"{entity}/{project}")
            ckpt_path = os.path.join(output_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=device)
            config['wandb_run_id'] = checkpoint['config']['wandb_run_id']
            config['wandb_project_name'] = checkpoint['config']['wandb_project_name']
            config['wandb_run_name'] = checkpoint['config']['wandb_run_name']
            config['wandb_entity'] = checkpoint['config']['wandb_entity']
            if config['wandb_run_id'] in [run.id for run in runs]:
                logging.info('WandB run is previously created and found now. Resuming the run.')
                wandb.init(
                project=checkpoint['config']['wandb_project_name'],
                id=checkpoint['config']['wandb_run_id'],
                resume="allow",
                name=checkpoint['config']['wandb_run_name']
                )
            else:
                logging.info('WandB run is previously created and deleted. Creating a new one.')
                wandb.init(
                project=checkpoint['config']['wandb_project_name'],
                resume="never",
                name=checkpoint['config']['wandb_run_name']
                )
                config['wandb_run_id'] = wandb.run.id
            wandb.config.update(config, allow_val_change=True )

        elif init_from == 'scratch':
            wandb.init(project=wandb_project_name, name=wandb_run_name, group=wandb_group)
            if wandb_flag:
                config['wandb_run_id'] = wandb.run.id
                config['wandb_project_name'] = wandb_project_name
                config['wandb_run_name'] = wandb_run_name
                config['wandb_entity'] = wandb_entity
            wandb.config.update(config) # log all config parameters to wandb


config_save_path = os.path.join(output_dir, 'config.json')

# wait for the directory to be created before writing the config file, sometimes master process is slower

while not os.path.exists(output_dir):
    if master_process: os.makedirs(output_dir, exist_ok=True)
    time.sleep(0.0001)
if master_process:
    with open(config_save_path, 'w') as config_save_file:
        json.dump(config, config_save_file, indent=4)
        print(f'Saved config to {config_save_path}')


# %% ------------- Load the language model ------------------------------
if master_process:
    logging.info(f"Loading GPT model from {language_model_ckpt_folder_path}")

# Load checkpoint
ckpt_path = os.path.join(language_model_ckpt_folder_path, 'ckpt.pt')

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

if master_process:
    logging.info("Loading model checkpoint...")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

# Extract model configuration from checkpoint
model_args = checkpoint['model_args']
if master_process:
    logging.info(f"Model architecture: {model_args}")

# Create GPT config object
gpt_config = GPTConfig(**model_args)

# Initialize model
Lmodel = GPT(gpt_config)
Lmodel.eval()

# Load model state dict - handle compiled model checkpoint
state_dict = checkpoint['model']
if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
    # Strip _orig_mod. prefix from keys (from torch.compile)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
        new_state_dict[new_key] = value
    state_dict = new_state_dict

Lmodel.load_state_dict(state_dict, strict=True)

# Move to device
Lmodel.to(device)

if LM_compile:
    if master_process:
        logging.info("Compiling the language model...")
    Lmodel = torch.compile(Lmodel)  # requires PyTorch 2.0 

# %% --- initialize the VQ-VAE model ---

if init_from == 'scratch':
    if master_process:  
        logging.info("Initializing the model from scratch")

    vqvae_single_model = VQVAE_single(vqvae_single_d, vqvae_single_hidden_dim, vqvae_single_codebook_size, vqvae_single_beta, vqvae_single_codebook_reset_counter_multiplier, vqvae_single_config)
    vqvae_single_model.to(device)

    iter_num = 0
    iter_num_log = []
    recon_losses = []
    codebook_losses = []
    commitment_losses = []
    total_losses = []
    nrmses = []
    nrmses_per_vector = []
    nrmses_per_element = []


elif init_from == 'resume':
    if master_process:
        logging.info("Resuming the model from a checkpoint in {}".format(output_dir))
    ckpt_path = os.path.join(output_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    vqvae_single_model = VQVAE_single(vqvae_single_d, vqvae_single_hidden_dim, vqvae_single_codebook_size, vqvae_single_beta, vqvae_single_codebook_reset_counter_multiplier, vqvae_single_config)
    vqvae_single_model.load_state_dict(checkpoint['model_state_dict'])
    vqvae_single_model.to(device)

    iter_num = checkpoint['iter_num']
    iter_num_log = checkpoint['iter_num_log']
    recon_losses = checkpoint['recon_losses']
    codebook_losses = checkpoint['codebook_losses']
    commitment_losses = checkpoint['commitment_losses']
    total_losses = checkpoint['total_losses']
    nrmses = checkpoint['nrmses']
    nrmses_per_vector = checkpoint.get('nrmses_per_vector', [0.0] * len(recon_losses))
    nrmses_per_element = checkpoint.get('nrmses_per_element', [0.0] * len(recon_losses))
else:
    if master_process:
        logging.info(f"Invalid value for init_from. Must be 'scratch' or 'resume', not {init_from}")
vqvae_single_model.train()

 # %% Setting up the training 
optimizer = configure_optimizers(vqvae_single_model, weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # free up memory
    del checkpoint  # Delete the checkpoint
    # clear CUDA memory cache (if tensors were on GPU)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # force garbage collection to ensure memory is freed
    gc.collect()
    checkpoint = None 

# compile the model
if vq_compile:
    if master_process:
        logging.info("compiling the model...")
    vqvae_single_model.compile()  # requires PyTorch 2.0

# wrap model in DDP if needed
if ddp:
    vqvae_single_model = DDP(vqvae_single_model, device_ids=[ddp_local_rank])

# %% ------------- Load Data ------------------------------
# Path finding data loading - use JSONL file instead of binary CFG data
jsonl_filename = f"{split}.jsonl"
jsonl_path = os.path.join(dataset_dir, jsonl_filename)

if not os.path.exists(jsonl_path):
    raise FileNotFoundError(f"Path finding data file not found: {jsonl_path}")

if master_process:
    logging.info(f"Using path finding data file: {jsonl_filename}")

# Create PathFindingDataset
path_finding_dataset = PathFindingDataset(
    jsonl_file=jsonl_path,
    max_seq_length=max_seq_length,
    mask_edges=True,  # We'll focus on path generation, not edge part
    max_nodes=max_nodes,  # From path finding config
    number_of_hyphens=None  # Use default behavior from dataset
)

if master_process:
    logging.info(f"Path finding dataset loaded: {len(path_finding_dataset)} samples")
    logging.info(f"Vocabulary size: {path_finding_dataset.vocab_size}")

# Create DataLoader for path finding data
train_loader = DataLoader(
    path_finding_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=4 if device_type == 'cuda' else 0,
    drop_last=True  # Ensure consistent batch sizes
)

# Get the first batch and generate complete paths
goal_token_id = path_finding_dataset.vocab['2']  # Goal token ID

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process

# Create an iterator for the dataloader
train_iter = iter(train_loader)

batch = next(train_iter)

# Get prompts (edges ending with ":")
prompt_ids = batch['prompt_ids'].to(device)  # (B, seq_len)
prompt_attention_mask = batch['prompt_attention_mask'].to(device)  # (B, seq_len)

if master_process:
    logging.info(f"Prompt shape: {prompt_ids.shape}")
    logging.info(f"Goal token ID: {goal_token_id}")
    logging.info(f"Max new tokens: {max_new_tokens}")


# %% ------------- Training start ------------------------------

# Generate complete paths and get hidden states
with ctx:
    promptLen = prompt_ids.shape[-1]
    generated_ids, generated_mask, all_hidden_states, _, _ = path_finding_generate_with_hidden_states(Lmodel, prompt_ids, max_new_tokens, attention_mask=prompt_attention_mask, eos_token=goal_token_id)
    generated_hidden_states = all_hidden_states[:, :, promptLen-1:, :] # (B, L+1, T_new+1, D)
    generated_attention_masks = generated_mask[:, promptLen-1:] # (B, T_new+1)

# Extract hidden states for generated tokens (last max_new_tokens positions)
# Use index -2 for last transformer layer (not layer norm) 
if master_process:
    logging.info(f"Generated hidden states shape: {generated_hidden_states.shape}")
    logging.info(f"Generation attention masks shape: {generated_attention_masks.shape}")
    valid_tokens = generated_attention_masks.sum().item()
    total_tokens = generated_attention_masks.numel()
    logging.info(f"Valid generation tokens: {valid_tokens}/{total_tokens} ({100*valid_tokens/total_tokens:.1f}%)")


# %% ------------- Training loop ------------------------------
while True:
    lr = get_lr(iter_num, warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters, learning_rate=learning_rate, min_lr=min_lr) if decay_lr else learning_rate
    beta = get_beta(iter_num, warmup_iters=vqvae_single_warmup_iters_beta, beta_start=vqvae_single_betainit, beta_max=vqvae_single_betafinal) if vqvae_single_grow_beta else vqvae_single_beta

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Reset accumulators for the iteration
    total_loss_accum = 0.0
    recon_loss_accum = 0.0
    codebook_loss_accum = 0.0
    commitment_loss_accum = 0.0
    cosine_push_loss_accum = 0.0
    entropy_loss_accum = 0.0
    nrmse_accum = 0.0
    nrmse_per_vector_accum = 0.0
    nrmse_per_element_accum = 0.0
    unique_count_accum = 0.0
    if wandb_flag:
        nrmse_per_time_accum = 0.0

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        # Use generated hidden states directly (already extracted from -2 layer)
        H = generated_hidden_states[:, -2, :, :].to(dtype=ptdtype)  # (B, gen_T, d)
        # normalize H
        H = H / (torch.norm(H, dim=-1, keepdim=True) + 1e-8)

        if ddp:
            vqvae_single_model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            # Get both losses from the model - use generation attention masks
            # Ensure attention mask is contiguous
            attention_mask_contiguous = generated_attention_masks.contiguous()
            output, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count, cosine_push_loss, entropy_loss = vqvae_single_model(H, attention_mask_contiguous, beta=beta)
        
        total_loss /= gradient_accumulation_steps

        with torch.no_grad():
            # For NRMSE calculation, we need to denormalize the output to compare with original H
            if hasattr(vqvae_single_model, 'module'):
                # Handle DDP wrapped model
                output_denormalized = vqvae_single_model.module.denormalize(output)
            else:
                output_denormalized = vqvae_single_model.denormalize(output)
            
            nrmse_accum += torch.mean(compute_masked_nrmse(H, output_denormalized, attention_mask_contiguous)) / gradient_accumulation_steps
            nrmse_per_vector_accum += torch.mean(compute_masked_nrmse_per_vector(H, output_denormalized, attention_mask_contiguous)) / gradient_accumulation_steps
            nrmse_per_element_accum += torch.mean(compute_masked_nrmse_per_element(H, output_denormalized, attention_mask_contiguous)) / gradient_accumulation_steps

            if wandb_flag:
                # Calculate experimental time-wise NRMSE
                diff_norm = torch.norm(H - output_denormalized, dim=-1) # (B, gen_T)
                H_norm = torch.norm(H, dim=-1) # (B, gen_T)
                vector_nrmse = diff_norm / (H_norm + 1e-8)  # (B, gen_T)
                vector_nrmse = vector_nrmse.unsqueeze(-1)  # (B, gen_T, 1)
                # Note: attention_mask_contiguous is already right-padded, so no conversion needed !!!!!!! CHECK THIS IF AN ISSUE OCCURS...
                vector_nrmse_right_padded = vector_nrmse.squeeze(-1)
                masked_vector_nrmse_right_padded = vector_nrmse_right_padded * attention_mask_contiguous
                valid_count_per_time = attention_mask_contiguous.sum(dim=(0,))
                nrmse_per_time_accum += (masked_vector_nrmse_right_padded.sum(dim=(0,)) / (valid_count_per_time + 1e-8)) / gradient_accumulation_steps

        # Accumulate for logging
        recon_loss_accum += recon_loss.item() / gradient_accumulation_steps
        codebook_loss_accum += codebook_loss.item() / gradient_accumulation_steps
        commitment_loss_accum += commitment_loss.item() / gradient_accumulation_steps
        cosine_push_loss_accum += cosine_push_loss.item() / gradient_accumulation_steps
        entropy_loss_accum += entropy_loss.item() / gradient_accumulation_steps
        total_loss_accum += total_loss.item()
        unique_count_accum += unique_count / gradient_accumulation_steps

        total_loss.backward()

    if grad_clip > 0:
        grad_norm_value = torch.nn.utils.clip_grad_norm_(vqvae_single_model.parameters(), grad_clip)
    else:
        grad_norm_list = [p.grad.data.norm(2) for p in vqvae_single_model.parameters() if p.grad is not None]
        grad_norm_value = torch.norm(torch.stack(grad_norm_list)) if grad_norm_list else torch.tensor(0.0, device=device)

    param_norm_list = [p.data.norm(2) for p in vqvae_single_model.parameters() if p is not None]
    total_param_norm = torch.norm(torch.stack(param_norm_list)) if param_norm_list else torch.tensor(0.0, device=device)
    grad_norm_ratio = grad_norm_value / total_param_norm if total_param_norm.item() > 0 else torch.tensor(0.0, device=device)
    
    optimizer.step()
    optimizer.zero_grad()

    # Get the next batch and generate paths
    try:
        batch = next(train_iter)
    except StopIteration:
        # If we've gone through all batches, create a new iterator !!!!!!!!!!!!!!!
        train_iter = iter(train_loader)
        batch = next(train_iter)
    
    # Extract prompt data
    prompt_ids = batch['prompt_ids'].to(device)
    prompt_attention_mask = batch['prompt_attention_mask'].to(device)
    
    # Generate complete paths and get hidden states
    with ctx:
        promptLen = prompt_ids.shape[-1]
        generated_ids, generated_mask, all_hidden_states, _, _ = path_finding_generate_with_hidden_states(Lmodel, prompt_ids, max_new_tokens, attention_mask=prompt_attention_mask, eos_token=goal_token_id)
        generated_hidden_states = all_hidden_states[:, :, promptLen-1:, :] # (B, L+1, T_new+1, D)
        generated_attention_masks = generated_mask[:, promptLen-1:] # (B, T_new+1)
        
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0:
        iter_num_log.append(iter_num)
        if ddp:
            # Create tensors on the correct device
            recon_loss_accum_tensor = torch.tensor(recon_loss_accum, device=device)
            codebook_loss_accum_tensor = torch.tensor(codebook_loss_accum, device=device)
            commitment_loss_accum_tensor = torch.tensor(commitment_loss_accum, device=device)
            cosine_push_loss_accum_tensor = torch.tensor(cosine_push_loss_accum, device=device)
            entropy_loss_accum_tensor = torch.tensor(entropy_loss_accum, device=device)
            total_loss_accum_tensor = torch.tensor(total_loss_accum, device=device)
            nrmse_accum_tensor = torch.tensor(nrmse_accum, device=device)
            nrmse_per_vector_accum_tensor = torch.tensor(nrmse_per_vector_accum, device=device)
            nrmse_per_element_accum_tensor = torch.tensor(nrmse_per_element_accum, device=device)
            # All-reduce these tensors over all processes
            torch.distributed.all_reduce(recon_loss_accum_tensor)
            torch.distributed.all_reduce(codebook_loss_accum_tensor)
            torch.distributed.all_reduce(commitment_loss_accum_tensor)
            torch.distributed.all_reduce(cosine_push_loss_accum_tensor)
            torch.distributed.all_reduce(entropy_loss_accum_tensor)
            torch.distributed.all_reduce(total_loss_accum_tensor)
            torch.distributed.all_reduce(nrmse_accum_tensor)
            torch.distributed.all_reduce(nrmse_per_vector_accum_tensor)
            torch.distributed.all_reduce(nrmse_per_element_accum_tensor)
            world_size = torch.distributed.get_world_size()
            recon_loss_accum = recon_loss_accum_tensor.item() / world_size
            codebook_loss_accum = codebook_loss_accum_tensor.item() / world_size
            commitment_loss_accum = commitment_loss_accum_tensor.item() / world_size
            cosine_push_loss_accum = cosine_push_loss_accum_tensor.item() / world_size
            entropy_loss_accum = entropy_loss_accum_tensor.item() / world_size
            total_loss_accum = total_loss_accum_tensor.item() / world_size
            nrmse_accum = nrmse_accum_tensor.item() / world_size
            nrmse_per_vector_accum = nrmse_per_vector_accum_tensor.item() / world_size
            nrmse_per_element_accum = nrmse_per_element_accum_tensor.item() / world_size
        if master_process:
            logging.info(
                f'Iter: {iter_num}, Loss: {total_loss_accum:.4f}, '
                f'NRMSE: {nrmse_accum:.4f}, NRMSE_vec: {nrmse_per_vector_accum:.4f}, NRMSE_elem: {nrmse_per_element_accum:.4f}, '
                f'Recon Loss: {recon_loss_accum:.4f}, Codebook Loss: {codebook_loss_accum:.4f}, '
                f'Commitment Loss: {commitment_loss_accum:.4f}, Cosine Push Loss: {cosine_push_loss_accum:.4f}, '
                f'Entropy Loss: {entropy_loss_accum:.4f}, '
                f'Unique Vectors: {unique_count_accum:.2f}, '
                f'Grad Norm: {grad_norm_value.item():.4f}, Grad Norm/Param Norm: {grad_norm_ratio.item():.4f}, '
                f'Time: {dt:.2f}s'
            )
            if wandb_flag:
                log_data = {
                    'vqvae_single/iter_num': iter_num,
                    'vqvae_single/total_loss': total_loss_accum,
                    'vqvae_single/recon_loss': recon_loss_accum,
                    'vqvae_single/codebook_loss': codebook_loss_accum,
                    'vqvae_single/commitment_loss': commitment_loss_accum,
                    'vqvae_single/cosine_push_loss': cosine_push_loss_accum,
                    'vqvae_single/entropy_loss': entropy_loss_accum,
                    'vqvae_single/nrmse': nrmse_accum,
                    'vqvae_single/nrmse_per_vector': nrmse_per_vector_accum,
                    'vqvae_single/nrmse_per_element': nrmse_per_element_accum,
                    'vqvae_single/unique_vectors': unique_count_accum,
                    'vqvae_single/time': dt,
                    'vqvae_single/lr': lr,
                    'vqvae_single/grad_norm': grad_norm_value.item(),
                    'vqvae_single/grad_norm_ratio': grad_norm_ratio.item()
                }
                
                # Add experimental per-time NRMSE
                for t in range(nrmse_per_time_accum.shape[0]):
                    log_data[f'vqvae_single_experimental/nrmse_time_{t}'] = nrmse_per_time_accum[t].item()
                
                if torch.cuda.is_available():
                    log_data.update({
                        'vqvae_single/gpu_memory_allocated_GB': torch.cuda.memory_allocated(device) / 1024**3,
                        'vqvae_single/gpu_memory_reserved_GB': torch.cuda.memory_reserved(device) / 1024**3
                    })
                wandb.log(log_data)
                
                # Plot NRMSE analysis every 100 iterations
                if iter_num % 100 == 0 and master_process:
                    plt.figure(figsize=(8, 5))
                    
                    # Plot NRMSE per time step
                    plt.plot(nrmse_per_time_accum.cpu().numpy())
                    plt.title('NRMSE per Time Step')
                    plt.xlabel('Time Step')
                    plt.ylabel('NRMSE')
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'vqvae_single_nrmse_analysis.png'))
                    plt.close()
        total_losses.append(total_loss_accum)
        recon_losses.append(recon_loss_accum)
        codebook_losses.append(codebook_loss_accum)
        commitment_losses.append(commitment_loss_accum)
        nrmses.append(nrmse_accum)
        nrmses_per_vector.append(nrmse_per_vector_accum)
        nrmses_per_element.append(nrmse_per_element_accum)

        # Enhanced codebook analysis every 1000 iterations using internal metrics
        if iter_num % 250 == 0:
            if master_process:
                logging.info("Getting usage statistics from internal model metrics...")

                # Get model reference (handle DDP)
                model_ref = vqvae_single_model.module if hasattr(vqvae_single_model, 'module') else vqvae_single_model
                
                # Get usage statistics from the model
                usage_stats = model_ref.get_usage_statistics()
                similarity_stats = model_ref.compute_codebook_similarities()
                
                # Log usage statistics
                logging.info(f"VQ-VAE Single codebook analysis: Used {usage_stats['unique_vectors_used']} out of {model_ref.codebook_size} vectors, "
                           f"total vectors processed: {usage_stats['total_vectors_processed']}")
                
                # Plot usage histogram
                if usage_stats['unique_vectors_used'] > 0:
                    usage_counts = usage_stats['usage_counts'].cpu().numpy()
                    
                    plt.figure(figsize=(10,5))
                    plt.bar(range(len(usage_counts)), usage_counts)
                    plt.title(f'VQ-VAE Single Codebook Vector Usage (Internal Tracking)\n'
                             f'Used {usage_stats["unique_vectors_used"]} out of {model_ref.codebook_size} vectors, '
                             f'Total samples: {usage_stats["total_vectors_processed"]}')
                    plt.xlabel('Codebook Vector Index')
                    plt.ylabel('Usage Count')
                    plt.savefig(os.path.join(output_dir, 'vqvae_single_codebook_usage_internal.png'))
                    plt.close()
                    
                    # Plot similarity heatmap if we have similarities
                    if similarity_stats['similarities'] is not None:
                        similarities = similarity_stats['similarities'].cpu().numpy()
                        used_indices = similarity_stats['used_indices'].cpu().numpy()
                        
                        plt.figure(figsize=(10,10))
                        plt.imshow(similarities, cmap='viridis')
                        plt.colorbar()
                        plt.title(f'Cosine Similarities Between Used VQ-VAE Single Codebook Vectors (Internal Tracking)\n'
                                 f'Used {similarity_stats["num_used_vectors"]} out of {model_ref.codebook_size} vectors')
                        plt.xlabel('Used Codebook Vector Index')
                        plt.ylabel('Used Codebook Vector Index')
                        plt.savefig(os.path.join(output_dir, 'vqvae_single_codebook_similarities_internal.png'))
                        plt.close()
                    else:
                        logging.info("Not enough unique vectors for similarity analysis.")
                else:
                    logging.info("No vectors used yet, skipping usage plots.")

    if save_interval > 0 and iter_num % save_interval == 0 and master_process:
        tempTime = time.time()
        checkpoint = {
            'model_state_dict': vqvae_single_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter_num': iter_num,
            'iter_num_log': iter_num_log,
            'config': config,
            'recon_losses': recon_losses,
            'codebook_losses': codebook_losses,
            'commitment_losses': commitment_losses,
            'nrmses': nrmses,
            'nrmses_per_vector': nrmses_per_vector,
            'nrmses_per_element': nrmses_per_element,
            'total_losses': total_losses,
            'wandb_flag': wandb_flag
        }
        torch.save(checkpoint, os.path.join(output_dir, 'ckpt.pt'))
        logging.info(f'Saved checkpoint to {output_dir}/ckpt.pt. Time taken: {time.time()-tempTime:.2f}s')

    iter_num += 1
    # termination conditions
    if iter_num > max_iters:
        break

if master_process:
    checkpoint = {
        'model_state_dict': vqvae_single_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter_num': iter_num,
        'iter_num_log': iter_num_log,
        'config': config,
        'recon_losses': recon_losses,
        'codebook_losses': codebook_losses,
        'commitment_losses': commitment_losses,
        'nrmses': nrmses,
        'nrmses_per_vector': nrmses_per_vector,
        'nrmses_per_element': nrmses_per_element,
        'total_losses': total_losses,
        'wandb_flag': wandb_flag
    }
    torch.save(checkpoint, os.path.join(output_dir, 'ckpt.pt'))
    total_time = time.time() - t0
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info(f'Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s')
    if wandb_flag:
        wandb.finish()
    
if ddp:
    destroy_process_group()



# %%
