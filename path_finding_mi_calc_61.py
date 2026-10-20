# %% 
# %%

# %%

# %%
print('path_finding_mi_calc_61.py')
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

from minimal_VQVAEs import * 
from vqvae_utils import * 
import json
from tqdm import tqdm
from path_finding_llm_utilities import PathFindingDataset, custom_collate_fn, skip_time_steps
from torch.utils.data import DataLoader
from model import GPT, GPTConfig

# %%
# Load training configuration from a JSON file
try:
    parser = argparse.ArgumentParser(description="Calculate mutual information for VQ-VAE models")
    parser.add_argument('--dataset_dir', type=str, help="Path to dataset directory")
    parser.add_argument('--exp_dir', type=str, required=True, help="Path to experiment directory")
    parser.add_argument('--LLM_ckpt_folder_dir', type=str, help="Path to LLM checkpoint folder")
    parser.add_argument('--vqvae61_ckpt_folder_dir', type=str, help="Path to VQVAE1 checkpoint folder") 
    parser.add_argument('--vqvae_single_ckpt_folder_dir', type=str, help="Path to VQVAE single checkpoint folder")
    parser.add_argument('--output_dir', type=str, help="Path to output directory")
    parser.add_argument('--batch_size', type=int, help="Batch size")
    parser.add_argument('--sync_interval', type=int, help="How often to sync and report results (in batches)")
    parser.add_argument('--log_interval', type=int, help="How often to log progress (in batches)")
    parser.add_argument('--plot_interval', type=int, help="How often to generate plots (in batches, 0 to disable)")
    parser.add_argument('--store_samples_to_file', action='store_true', default=True, help="Store samples to HDF5 files for later analysis")
    parser.add_argument('--no_store_samples', dest='store_samples_to_file', action='store_false', help="Disable storing samples to files")
    parser.add_argument('--condition_on_correct_false', action='store_true', default=False, help="Create separate tracking for correct and false predictions")
    parser.add_argument('--temperature', type=float, default=0.0, help="Temperature for sampling")
    args = parser.parse_args()
except:
    # Create args namespace for interactive use
    args = argparse.Namespace()



args.LLM_ckpt_folder_dir = os.path.join(args.exp_dir, args.LLM_ckpt_folder_dir) if hasattr(args, 'LLM_ckpt_folder_dir') and args.LLM_ckpt_folder_dir is not None else None
args.vqvae_single_ckpt_folder_dir = args.vqvae_single_ckpt_folder_dir if hasattr(args, 'vqvae_single_ckpt_folder_dir') and args.vqvae_single_ckpt_folder_dir is not None else None
args.dataset_dir = args.dataset_dir if hasattr(args, 'dataset_dir') and args.dataset_dir is not None else args.exp_dir+'/samples'
args.output_dir = os.path.join(args.exp_dir, args.output_dir) if hasattr(args, 'output_dir') and args.output_dir is not None else os.path.join(args.exp_dir, 'mi_output')
args.batch_size = args.batch_size if hasattr(args, 'batch_size') and args.batch_size is not None else 64
args.sync_interval = args.sync_interval if hasattr(args, 'sync_interval') and args.sync_interval is not None else 100
args.log_interval = args.log_interval if hasattr(args, 'log_interval') and args.log_interval is not None else 100
args.plot_interval = args.plot_interval if hasattr(args, 'plot_interval') and args.plot_interval is not None else 10000
args.store_samples_to_file = args.store_samples_to_file if hasattr(args, 'store_samples_to_file') and args.store_samples_to_file is not None else True
args.condition_on_correct_false = args.condition_on_correct_false if hasattr(args, 'condition_on_correct_false') and args.condition_on_correct_false is not None else False
args.temperature = args.temperature if hasattr(args, 'temperature') and args.temperature is not None else 0.0
if args.exp_dir and args.LLM_ckpt_folder_dir is None:
    # Get list of folders in LLMout directory
    llm_out_dir = os.path.join(args.exp_dir, "LLMout") if args.LLM_ckpt_folder_dir is None else args.LLM_ckpt_folder_dir
    if os.path.exists(llm_out_dir):
        llm_folders = [f for f in os.listdir(llm_out_dir) if os.path.isdir(os.path.join(llm_out_dir, f))]
        if llm_folders:
            for folder in llm_folders:
                if "mtl_" not in folder and "ckpt.pt" in os.listdir(os.path.join(llm_out_dir, folder)):
                    print(f"Found LLM checkpoint in {os.path.join(llm_out_dir, folder)}")
                    args.LLM_ckpt_folder_dir = os.path.join(llm_out_dir, folder)
                    break
    
args.vqvae61_ckpt_folder_dir = os.path.join(args.exp_dir, "vqvae61_out") if args.vqvae61_ckpt_folder_dir is None else os.path.join(args.exp_dir, args.vqvae61_ckpt_folder_dir)
args.vqvae_single_ckpt_folder_dir = os.path.join(args.exp_dir, "vqvae_single_out") if args.vqvae_single_ckpt_folder_dir is None else os.path.join(args.exp_dir, args.vqvae_single_ckpt_folder_dir)

print("\nArguments:")
print("-" * 50)
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
print("-" * 50)
print()
# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Determine dtype based on device capabilities
# if 'cuda' in device and torch.cuda.is_bf16_supported():
#     dtype = 'bfloat16'

# First stage experiment path/directory
vqvae61_ckpt_path = args.vqvae61_ckpt_folder_dir
if vqvae61_ckpt_path is None:
    raise ValueError("First stage checkpoint model path must be provided")
elif os.path.isdir(vqvae61_ckpt_path):
    print(f'{vqvae61_ckpt_path} is a directory. Loading the latest ckpt.py file from this directory.')
    if not os.path.exists(os.path.join(vqvae61_ckpt_path, 'ckpt.pt')):
        raise ValueError(f"{os.path.join(vqvae61_ckpt_path, 'ckpt.pt')} not found")
    else:
        vqvae61_ckpt_path = os.path.join(vqvae61_ckpt_path, 'ckpt.pt')
elif not os.path.exists(vqvae61_ckpt_path):
    raise ValueError(f"{vqvae61_ckpt_path} not found")
# Load first stage checkpoint
vqvae61_checkpoint = torch.load(vqvae61_ckpt_path, map_location=device)


single_stage_ckpt_path = args.vqvae_single_ckpt_folder_dir
if single_stage_ckpt_path is None:
    raise ValueError("Single stage checkpoint model path must be provided")
elif os.path.isdir(single_stage_ckpt_path):
    print(f'{single_stage_ckpt_path} is a directory. Loading the latest ckpt.py file from this directory.')
    if not os.path.exists(os.path.join(single_stage_ckpt_path, 'ckpt.pt')):
        raise ValueError(f"{os.path.join(single_stage_ckpt_path, 'ckpt.pt')} not found")
    else:
        single_stage_ckpt_path = os.path.join(single_stage_ckpt_path, 'ckpt.pt')
elif not os.path.exists(single_stage_ckpt_path):
    raise ValueError(f"{single_stage_ckpt_path} not found")
# Load single stage checkpoint
single_stage_checkpoint = torch.load(single_stage_ckpt_path, map_location=device)

# %% ------------- Arrange DDP ------------------------------

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # Set random seed with offset for DDP
    torch.manual_seed(1337 + seed_offset)
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
    ddp_local_rank = 0

# various inits, derived attributes, I/O setup 
save_logs_flag = True
print_logs = master_process  # Only master process prints
output_dir = args.output_dir

# Create separate output directories for conditional tracking
if args.condition_on_correct_false:
    output_dir_correct = output_dir + "_correct"
    output_dir_false = output_dir + "_false"
else:
    output_dir_correct = None
    output_dir_false = None

if master_process:
    os.makedirs(output_dir, exist_ok=True)
    if args.condition_on_correct_false:
        os.makedirs(output_dir_correct, exist_ok=True)
        os.makedirs(output_dir_false, exist_ok=True)
    
    logger = get_logger(save_logs_flag = save_logs_flag, print_logs = print_logs, experiment_dir = output_dir)
    
    # Create separate loggers for correct and false conditions (no print, only save)
    if args.condition_on_correct_false:
        logger_correct = get_logger(save_logs_flag = save_logs_flag, print_logs = False, experiment_dir = output_dir_correct)
        logger_false = get_logger(save_logs_flag = save_logs_flag, print_logs = False, experiment_dir = output_dir_false)
    else:
        logger_correct = None
        logger_false = None
else:
    # Create a dummy logger for non-master processes that doesn't actually log
    logger = logging.getLogger('dummy')
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)  # Effectively disable logging
    logger_correct = None
    logger_false = None

if master_process:
    logger.info(f"DDP: rank {ddp_rank}/{ddp_world_size}, local_rank {ddp_local_rank}, master: {master_process}")
else:
    print(f"DDP: rank {ddp_rank}/{ddp_world_size}, local_rank {ddp_local_rank}, master: {master_process}")


dtype = vqvae61_checkpoint['config']['dtype']


torch.set_float32_matmul_precision('high') 
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
# NOTE: March, 6, 2025: Before using Float16, make sure that gradientscaler works.
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# %% ------------- Load the language model ------------------------------

# Load GPT model from checkpoint
checkpoint_path = os.path.join(args.LLM_ckpt_folder_dir, 'ckpt.pt')

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

if master_process:
    logging.info(f"Loading GPT model from checkpoint: {checkpoint_path}")

# Load checkpoint and extract model configuration and state
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
gpt_config = GPTConfig(**checkpoint['model_args'])
# gpt_config.block_size = max_seq_length #+ max_new_tokens + 8
Lmodel = GPT(gpt_config)
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

if master_process:
    logging.info(f"GPT model loaded with {sum(p.numel() for p in Lmodel.parameters()):,} parameters")

# Move model to device
Lmodel.to(device)
Lmodel.eval()  # Set to evaluation mode for hidden state extraction

# logging.info("compiling the language model...")
# Lmodel.compile()

# %%
# PF Configuration
split = 'mi'
max_seq_length = gpt_config.block_size
max_nodes = vqvae61_checkpoint['config']['max_nodes']
max_new_tokens = max_tau = 15
batch_size = vqvae61_checkpoint['config']['batch_size']
device_type = device_type = 'cuda' if 'cuda' in device else 'cpu'
dataset_dir = args.dataset_dir
eos_token_id = 2
json_file_path = os.path.join(dataset_dir, 'config_train.json')
with open(json_file_path, 'r') as f:
    config_train = json.load(f)
max_path_len = config_train['max_path_len']


# %% ------------- Load VQ-VAE 1 Encoder ------------------------------
# %%

# Create the encoder first.
encoder_model = Encoder61(
    L=vqvae61_checkpoint['config']['vqvae61_config']['L'],
    d=vqvae61_checkpoint['config']['vqvae61_config']['d'],
    d_aggregate=vqvae61_checkpoint['config']['vqvae61_config']['d_aggregate'],
    T_star=vqvae61_checkpoint['config']['vqvae61_config']['T_star'],
    num_layers_layerwise_stage=vqvae61_checkpoint['config']['vqvae61_config']['num_layers_layerwise_stage'],
    num_layers_aggregate_stage=vqvae61_checkpoint['config']['vqvae61_config']['num_layers_aggregate_stage'],
    config_layerwise_stage=vqvae61_checkpoint['config']['vqvae61_config']['layerwise_config'],
    config_aggregate_stage=vqvae61_checkpoint['config']['vqvae61_config']['aggregate_config']
)
encoder_model.to(device)
encoder_param_dict = {k.replace('module.encoder.',''):v for k,v in vqvae61_checkpoint['model_state_dict'].items() if 'module.encoder.' in k}
encoder_param_dict.update({k.replace('encoder.',''):v for k,v in vqvae61_checkpoint['model_state_dict'].items() if k.startswith('encoder.')})
encoder_model.load_state_dict(encoder_param_dict, strict=True)

# Create the decoder
decoder_model = Decoder61(
    L=vqvae61_checkpoint['config']['vqvae61_config']['L'],
    d=vqvae61_checkpoint['config']['vqvae61_config']['d'],
    d_aggregate=vqvae61_checkpoint['config']['vqvae61_config']['d_aggregate'],
    T_star=vqvae61_checkpoint['config']['vqvae61_config']['T_star'],
    T_max=max_seq_length,
    num_layers_aggregate_stage=vqvae61_checkpoint['config']['vqvae61_config']['num_layers_aggregate_stage'] ,
    num_layers_layerwise_stage=vqvae61_checkpoint['config']['vqvae61_config']['num_layers_layerwise_stage'],
    config_aggregate_stage=vqvae61_checkpoint['config']['vqvae61_config']['aggregate_config'],
    config_layerwise_stage=vqvae61_checkpoint['config']['vqvae61_config']['layerwise_config'],
    tied_encoder_proj=vqvae61_checkpoint['config']['vqvae61_config']['tied_encoder_proj']
)
decoder_model.to(device)
decoder_param_dict = {k.replace('module.decoder.',''):v for k,v in vqvae61_checkpoint['model_state_dict'].items() if 'module.decoder.' in k}
decoder_param_dict.update({k.replace('decoder.',''):v for k,v in vqvae61_checkpoint['model_state_dict'].items() if k.startswith('decoder.')})
decoder_model.load_state_dict(decoder_param_dict, strict=True)
vqvae61_model = VQVAE61(encoder_model, decoder_model, vqvae61_checkpoint['config']['vqvae61_config'])

vqvae61_codebook = nn.Embedding(vqvae61_checkpoint['config']['vqvae61_config']['codebook_size'], vqvae61_checkpoint['config']['vqvae61_config']['d_aggregate']*vqvae61_checkpoint['config']['vqvae61_config']['T_star'])
vqvae61_codebook.to(device)
codebook_param_dict = {k.replace('module.codebook.',''):v for k,v in vqvae61_checkpoint['model_state_dict'].items() if 'module.codebook.' in k}
codebook_param_dict.update({k.replace('codebook.',''):v for k,v in vqvae61_checkpoint['model_state_dict'].items() if k.startswith('codebook.')})
vqvae61_codebook.load_state_dict(codebook_param_dict, strict=True)
print('First stage encoder:')
checkModelLoadCorrect(encoder_model, encoder_param_dict)
print('First stage codebook:')
checkModelLoadCorrect(vqvae61_codebook, codebook_param_dict)
print('First stage decoder:')
checkModelLoadCorrect(decoder_model, decoder_param_dict)

encoder_model.eval()  # eval() only affects behavior during forward pass (e.g. dropout, batchnorm)
for param in encoder_model.parameters():
    param.requires_grad = False # requires_grad=False is still needed to prevent gradient computation
vqvae61_codebook.eval()
for param in vqvae61_codebook.parameters():
    param.requires_grad = False
decoder_model.eval()
for param in decoder_model.parameters():
    param.requires_grad = False

vqvae61_codebook_weights = vqvae61_codebook.weight

# %% Load VQ-VAE-Single

single_vqvae = VQVAE_single(
    model_dim = single_stage_checkpoint['config']['vqvae_single_config']['d'],
    hidden_dim = single_stage_checkpoint['config']['vqvae_single_config']['hidden_dim'],
    codebook_size = single_stage_checkpoint['config']['vqvae_single_config']['codebook_size'],
    beta = single_stage_checkpoint['config']['vqvae_single_config']['beta'],
    codebook_reset_counter_multiplier = single_stage_checkpoint['config']['vqvae_single_config']['codebook_reset_counter_multiplier'],
    config = single_stage_checkpoint['config']['vqvae_single_config']
)
single_vqvae.to(device)

single_vqvae_param_dict = {k.replace('module.',''):v for k,v in single_stage_checkpoint['model_state_dict'].items() if 'module.' in k}
single_vqvae_param_dict.update({k:v for k,v in single_stage_checkpoint['model_state_dict'].items() if 'module.' not in k})


# Load normalization values from first stage VQVAE1 model
single_normalization_value = None
for k, v in single_stage_checkpoint['model_state_dict'].items():
    if 'normalization_value' in k:
        single_normalization_value = v.to(device)
        print(f'Found normalization value in single stage checkpoint: {k}, {v}')
        break

assert single_normalization_value is not None, "No normalization values found in single checkpoint. This may affect model performance."

del single_vqvae_param_dict['normalization_value']

single_vqvae.load_state_dict(single_vqvae_param_dict, strict=True)

print('Single stage VQ-VAE:')
checkModelLoadCorrect(single_vqvae, single_vqvae_param_dict)

single_vqvae.eval()
for param in single_vqvae.parameters():
    param.requires_grad = False

single_vqvae_codebook_weights = single_vqvae.codebook.weight
single_vqvae_codebook_weights.to(device)


def normalize_single_stage(x):
    """Normalize input using first stage VQVAE1 normalization values."""
    return x / (single_normalization_value + 1e-8)

def denormalize_first_stage(x):
    """Denormalize input using first stage VQVAE1 normalization values."""
    return x * (single_normalization_value + 1e-8)

# %% ------------- Training loop ------------------------------


# Create PathFinding dataset and dataloader
skip_every_timestep = vqvae61_checkpoint['config']['skip_every_timestep']
skip_every_layer = vqvae61_checkpoint['config']['skip_every_layer']

# %%

path_finding_dataset = PathFindingDataset(
    jsonl_file=os.path.join(dataset_dir, f'{split}.jsonl'),
    max_seq_length=max_seq_length,
    mask_edges=True,
    max_nodes=max_nodes,
    number_of_hyphens= 0 # zero in our experiments 
)

train_loader = DataLoader(
    path_finding_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=4 if device_type == 'cuda' else 0
)

# For DDP, we need to ensure each GPU processes different parts of the dataset
if ddp:
    # Calculate the data range for this process
    total_batches = len(train_loader)
    batches_per_process = total_batches // ddp_world_size
    start_batch = ddp_rank * batches_per_process
    if ddp_rank == ddp_world_size - 1:  # Last process takes remaining batches
        end_batch = total_batches
    else:
        end_batch = start_batch + batches_per_process
    
    if master_process:
        logger.info(f"Total batches: {total_batches}")
        logger.info(f"Batches per process: {batches_per_process}")
        for rank in range(ddp_world_size):
            rank_start = rank * batches_per_process
            rank_end = total_batches if rank == ddp_world_size - 1 else rank_start + batches_per_process
            logger.info(f"Rank {rank}: batches {rank_start}-{rank_end-1}")
else:
    start_batch = 0
    end_batch = len(train_loader)

# %%

# Simple sample collection and periodic sync approach ########
from information_theory_utils import StreamingInfo
import torch.distributed as dist
from collections import defaultdict

def sync_and_update_streaming_info(local_samples, smi_dict, processed_samples_count, 
                                  local_samples_correct=None, smi_dict_correct=None,
                                  local_samples_false=None, smi_dict_false=None):
    """Sync samples from all GPUs and update StreamingInfo on master process"""
    if ddp:
        # Gather samples from all processes
        all_samples = [None] * ddp_world_size
        dist.all_gather_object(all_samples, local_samples)
        
        # Gather correct samples if conditional tracking is enabled
        if local_samples_correct is not None:
            all_samples_correct = [None] * ddp_world_size
            dist.all_gather_object(all_samples_correct, local_samples_correct)
        else:
            all_samples_correct = None
            
        # Gather false samples if conditional tracking is enabled
        if local_samples_false is not None:
            all_samples_false = [None] * ddp_world_size
            dist.all_gather_object(all_samples_false, local_samples_false)
        else:
            all_samples_false = None
        
        if master_process:
            # Combine samples from all processes
            combined_samples = defaultdict(list)
            for proc_samples in all_samples:
                for tau, samples in proc_samples.items():
                    combined_samples[tau].extend(samples)
            
            # Update StreamingInfo objects with new samples
            for tau, samples in combined_samples.items():
                if tau in smi_dict:
                    for sample in samples:
                        smi_dict[tau].update(sample)
            
            # Handle correct samples
            combined_samples_correct = defaultdict(list)
            if all_samples_correct is not None:
                for proc_samples in all_samples_correct:
                    for tau, samples in proc_samples.items():
                        combined_samples_correct[tau].extend(samples)
                
                for tau, samples in combined_samples_correct.items():
                    if tau in smi_dict_correct:
                        for sample in samples:
                            smi_dict_correct[tau].update(sample)
            
            # Handle false samples
            combined_samples_false = defaultdict(list)
            if all_samples_false is not None:
                for proc_samples in all_samples_false:
                    for tau, samples in proc_samples.items():
                        combined_samples_false[tau].extend(samples)
                
                for tau, samples in combined_samples_false.items():
                    if tau in smi_dict_false:
                        for sample in samples:
                            smi_dict_false[tau].update(sample)
            
            return dict(combined_samples), dict(combined_samples_correct), dict(combined_samples_false)
    else:
        # Single GPU - directly update StreamingInfo
        for tau, samples in local_samples.items():
            if tau in smi_dict:
                for sample in samples:
                    smi_dict[tau].update(sample)
        
        # Handle correct samples
        if local_samples_correct is not None:
            for tau, samples in local_samples_correct.items():
                if tau in smi_dict_correct:
                    for sample in samples:
                        smi_dict_correct[tau].update(sample)
        
        # Handle false samples
        if local_samples_false is not None:
            for tau, samples in local_samples_false.items():
                if tau in smi_dict_false:
                    for sample in samples:
                        smi_dict_false[tau].update(sample)
        
        return local_samples, local_samples_correct or {}, local_samples_false or {}
    
    return {}, {}, {}

def calculate_and_report_metrics(smi_dict, processed_samples_count, batch_idx, total_batches,
                               smi_dict_correct=None, smi_dict_false=None, 
                               logger_correct=None, logger_false=None):
    """Calculate and report current metrics"""
    if not master_process:
        return {}, {}, {}
    
    # Calculate results for all samples
    results = {}
    total_samples = 0
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PROGRESS REPORT - Batch {batch_idx+1}/{total_batches}")
    logger.info(f"Samples processed so far: {processed_samples_count}")
    logger.info(f"{'='*80}")
    
    for tau in range(0, max_tau + 1):
        if tau in smi_dict and smi_dict[tau].N > 0:
            results[tau] = {
                'mutual_information': smi_dict[tau].mutual_information("H", "h"),
                'joint_entropy': smi_dict[tau].joint_entropy(("H", "h")),
                'entropy_H': smi_dict[tau].entropy("H"),
                'entropy_h': smi_dict[tau].entropy("h"),
                'conditional_entropy_h_given_H': smi_dict[tau].conditional_entropy("h", "H"),
                'conditional_entropy_H_given_h': smi_dict[tau].conditional_entropy("H", "h"),
                'sample_count': smi_dict[tau].N
            }
            total_samples += smi_dict[tau].N
            
            logger.info(f'Tau {tau}: MI={results[tau]["mutual_information"]:.4f}, '
                       f'H(H,h)={results[tau]["joint_entropy"]:.4f}, '
                       f'H(H)={results[tau]["entropy_H"]:.4f}, '
                       f'H(h)={results[tau]["entropy_h"]:.4f}, '
                       f'Samples={results[tau]["sample_count"]}')
    
    logger.info(f"Total samples across all taus: {total_samples}")
    logger.info(f"{'='*80}\n")
    
    # Calculate results for correct samples (no print, only log to file)
    results_correct = {}
    if smi_dict_correct and logger_correct:
        total_samples_correct = 0
        logger_correct.info(f"\n{'='*80}")
        logger_correct.info(f"CORRECT SAMPLES PROGRESS REPORT - Batch {batch_idx+1}/{total_batches}")
        logger_correct.info(f"Samples processed so far: {processed_samples_count}")
        logger_correct.info(f"{'='*80}")
        
        for tau in range(0, max_tau + 1):
            if tau in smi_dict_correct and smi_dict_correct[tau].N > 0:
                results_correct[tau] = {
                    'mutual_information': smi_dict_correct[tau].mutual_information("H", "h"),
                    'joint_entropy': smi_dict_correct[tau].joint_entropy(("H", "h")),
                    'entropy_H': smi_dict_correct[tau].entropy("H"),
                    'entropy_h': smi_dict_correct[tau].entropy("h"),
                    'conditional_entropy_h_given_H': smi_dict_correct[tau].conditional_entropy("h", "H"),
                    'conditional_entropy_H_given_h': smi_dict_correct[tau].conditional_entropy("H", "h"),
                    'sample_count': smi_dict_correct[tau].N
                }
                total_samples_correct += smi_dict_correct[tau].N
                
                logger_correct.info(f'Tau {tau}: MI={results_correct[tau]["mutual_information"]:.4f}, '
                                  f'H(H,h)={results_correct[tau]["joint_entropy"]:.4f}, '
                                  f'H(H)={results_correct[tau]["entropy_H"]:.4f}, '
                                  f'H(h)={results_correct[tau]["entropy_h"]:.4f}, '
                                  f'Samples={results_correct[tau]["sample_count"]}')
        
        logger_correct.info(f"Total correct samples across all taus: {total_samples_correct}")
        logger_correct.info(f"{'='*80}\n")
    
    # Calculate results for false samples (no print, only log to file)
    results_false = {}
    if smi_dict_false and logger_false:
        total_samples_false = 0
        logger_false.info(f"\n{'='*80}")
        logger_false.info(f"FALSE SAMPLES PROGRESS REPORT - Batch {batch_idx+1}/{total_batches}")
        logger_false.info(f"Samples processed so far: {processed_samples_count}")
        logger_false.info(f"{'='*80}")
        
        for tau in range(0, max_tau + 1):
            if tau in smi_dict_false and smi_dict_false[tau].N > 0:
                results_false[tau] = {
                    'mutual_information': smi_dict_false[tau].mutual_information("H", "h"),
                    'joint_entropy': smi_dict_false[tau].joint_entropy(("H", "h")),
                    'entropy_H': smi_dict_false[tau].entropy("H"),
                    'entropy_h': smi_dict_false[tau].entropy("h"),
                    'conditional_entropy_h_given_H': smi_dict_false[tau].conditional_entropy("h", "H"),
                    'conditional_entropy_H_given_h': smi_dict_false[tau].conditional_entropy("H", "h"),
                    'sample_count': smi_dict_false[tau].N
                }
                total_samples_false += smi_dict_false[tau].N
                
                logger_false.info(f'Tau {tau}: MI={results_false[tau]["mutual_information"]:.4f}, '
                                f'H(H,h)={results_false[tau]["joint_entropy"]:.4f}, '
                                f'H(H)={results_false[tau]["entropy_H"]:.4f}, '
                                f'H(h)={results_false[tau]["entropy_h"]:.4f}, '
                                f'Samples={results_false[tau]["sample_count"]}')
        
        logger_false.info(f"Total false samples across all taus: {total_samples_false}")
        logger_false.info(f"{'='*80}\n")
    
    return results, results_correct, results_false

def create_plots_and_save(results, output_dir, processed_samples_count, master_process, logger, batch_idx=None, is_final=False):
    """Create and save plots for mutual information results"""
    if not results:
        print("No results to plot - no data was collected for any tau values.")
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Determine filename suffix based on whether this is final or intermediate
    if is_final:
        suffix = "final"
        title_suffix = "Final Results"
    else:
        suffix = f"batch_{batch_idx}"
        title_suffix = f"Batch {batch_idx}"
    
    # Extract data for plotting
    taus = sorted(results.keys())
    mutual_info = [results[tau]['mutual_information'] for tau in taus]
    joint_entropy = [results[tau]['joint_entropy'] for tau in taus]
    entropy_H = [results[tau]['entropy_H'] for tau in taus]
    entropy_h = [results[tau]['entropy_h'] for tau in taus]
    cond_entropy_h_given_H = [results[tau]['conditional_entropy_h_given_H'] for tau in taus]
    cond_entropy_H_given_h = [results[tau]['conditional_entropy_H_given_h'] for tau in taus]
    sample_counts = [results[tau]['sample_count'] for tau in taus]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    main_title = f'Information-Theoretic Metrics - {title_suffix}\nTotal Samples Processed: {processed_samples_count:,}'
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    # Plot 1: Mutual Information
    axes[0, 0].plot(taus, mutual_info, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Tau (τ)')
    axes[0, 0].set_ylabel('Mutual Information I(H;h)')
    axes[0, 0].set_title('Mutual Information vs Tau')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Joint Entropy
    axes[0, 1].plot(taus, joint_entropy, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Tau (τ)')
    axes[0, 1].set_ylabel('Joint Entropy H(H,h)')
    axes[0, 1].set_title('Joint Entropy vs Tau')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Individual Entropies
    axes[0, 2].plot(taus, entropy_H, 'r-', linewidth=2, label='H(H)')
    axes[0, 2].plot(taus, entropy_h, 'purple', linestyle='-', linewidth=2, label='H(h)')
    axes[0, 2].set_xlabel('Tau (τ)')
    axes[0, 2].set_ylabel('Entropy')
    axes[0, 2].set_title('Individual Entropies vs Tau')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Conditional Entropies
    axes[1, 0].plot(taus, cond_entropy_h_given_H, 'orange', linestyle='-', linewidth=2, label='H(h|H)')
    axes[1, 0].plot(taus, cond_entropy_H_given_h, 'brown', linestyle='-', linewidth=2, label='H(H|h)')
    axes[1, 0].set_xlabel('Tau (τ)')
    axes[1, 0].set_ylabel('Conditional Entropy')
    axes[1, 0].set_title('Conditional Entropies vs Tau')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Sample Counts
    axes[1, 1].bar(taus, sample_counts, alpha=0.7, color='skyblue', edgecolor='navy')
    axes[1, 1].set_xlabel('Tau (τ)')
    axes[1, 1].set_ylabel('Sample Count')
    axes[1, 1].set_title('Sample Counts vs Tau')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Information Decomposition
    axes[1, 2].plot(taus, mutual_info, 'b-', linewidth=2, label='I(H;h)')
    axes[1, 2].plot(taus, cond_entropy_h_given_H, 'orange', linestyle='-', linewidth=2, label='H(h|H)')
    axes[1, 2].set_xlabel('Tau (τ)')
    axes[1, 2].set_ylabel('Information (bits)')
    axes[1, 2].set_title('Information Decomposition: H(h) = I(H;h) + H(h|H)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file_path = os.path.join(output_dir, f'mutual_information_plots_{suffix}.png')
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
    if master_process:
        logger.info(f"Comprehensive plots saved to: {plot_file_path}")
    
    # # Also save as PDF for better quality
    # plot_pdf_path = os.path.join(output_dir, f'mutual_information_plots_{suffix}.pdf')
    # plt.savefig(plot_pdf_path, bbox_inches='tight')
    # if master_process:
    #     logger.info(f"PDF plots saved to: {plot_pdf_path}")
    
    # Create a separate detailed mutual information plot
    plt.figure(figsize=(10, 6))
    plt.plot(taus, mutual_info, 'b-', linewidth=3)
    plt.xlabel('Tau (τ)', fontsize=14)
    plt.ylabel('Mutual Information I(H;h) [bits]', fontsize=14)
    plt.title(f'Mutual Information - {title_suffix}\nTotal Samples: {processed_samples_count:,}', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value annotations on points
    for i, (tau, mi) in enumerate(zip(taus, mutual_info)):
        plt.annotate(f'{mi:.3f}', (tau, mi), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save the detailed MI plot
    mi_plot_path = os.path.join(output_dir, f'mutual_information_detailed_{suffix}.png')
    plt.savefig(mi_plot_path, dpi=300, bbox_inches='tight')
    # plt.savefig(os.path.join(output_dir, f'mutual_information_detailed_{suffix}.pdf'), bbox_inches='tight')
    if master_process:
        logger.info(f"Detailed mutual information plot saved to: {mi_plot_path}")
    
    # Close figures to save memory
    plt.close('all')


# %% 
# Only master process maintains StreamingInfo objects
if master_process:
    # Create samples directory for file storage
    samples_dir = os.path.join(output_dir, 'samples') if args.store_samples_to_file else None
    if samples_dir:
        os.makedirs(samples_dir, exist_ok=True)
        logger.info(f"Sample storage directory created: {samples_dir}")
    
    # Create additional samples directories for correct and false conditions
    if args.condition_on_correct_false:
        samples_dir_correct = os.path.join(output_dir_correct, 'samples') if args.store_samples_to_file else None
        samples_dir_false = os.path.join(output_dir_false, 'samples') if args.store_samples_to_file else None
        if samples_dir_correct:
            os.makedirs(samples_dir_correct, exist_ok=True)
            logger.info(f"Correct samples storage directory created: {samples_dir_correct}")
        if samples_dir_false:
            os.makedirs(samples_dir_false, exist_ok=True)
            logger.info(f"False samples storage directory created: {samples_dir_false}")
    else:
        samples_dir_correct = None
        samples_dir_false = None
    
    variables = {i:["H", "h"] for i in range(0,max_tau+1)}
    combos_to_track_dict = {i:[("H",), ("h",), ("H", "h")] for i in range(0,max_tau+1)}
    
    # Create StreamingInfo objects for all samples
    smi_dict = {}
    for i in range(0, max_tau+1):
        sample_file_path = os.path.join(samples_dir, f'samples_tau_{i}.h5') if args.store_samples_to_file else None
        smi_dict[i] = StreamingInfo(
            variables=variables[i], 
            combos_to_track=combos_to_track_dict[i], 
            base=2, 
            store_samples=False,  # Keep memory storage disabled for efficiency
            store_samples_to_file=args.store_samples_to_file,
            sample_file_path=sample_file_path,
            compression='gzip',
            chunk_size=10000
        )
    
    # Create StreamingInfo objects for correct samples
    if args.condition_on_correct_false:
        smi_dict_correct = {}
        for i in range(0, max_tau+1):
            sample_file_path_correct = os.path.join(samples_dir_correct, f'samples_tau_{i}.h5') if args.store_samples_to_file else None
            smi_dict_correct[i] = StreamingInfo(
                variables=variables[i], 
                combos_to_track=combos_to_track_dict[i], 
                base=2, 
                store_samples=False,
                store_samples_to_file=args.store_samples_to_file,
                sample_file_path=sample_file_path_correct,
                compression='gzip',
                chunk_size=10000
            )
        
        # Create StreamingInfo objects for false samples
        smi_dict_false = {}
        for i in range(0, max_tau+1):
            sample_file_path_false = os.path.join(samples_dir_false, f'samples_tau_{i}.h5') if args.store_samples_to_file else None
            smi_dict_false[i] = StreamingInfo(
                variables=variables[i], 
                combos_to_track=combos_to_track_dict[i], 
                base=2, 
                store_samples=False,
                store_samples_to_file=args.store_samples_to_file,
                sample_file_path=sample_file_path_false,
                compression='gzip',
                chunk_size=10000
            )
    else:
        smi_dict_correct = {}
        smi_dict_false = {}
else:
    smi_dict = {}
    smi_dict_correct = {}
    smi_dict_false = {}



# Sample collection dictionaries for each process
local_sample_buffer = defaultdict(list)  # tau -> list of samples
local_sample_buffer_correct = defaultdict(list) if args.condition_on_correct_false else {}  # tau -> list of correct samples
local_sample_buffer_false = defaultdict(list) if args.condition_on_correct_false else {}  # tau -> list of false samples
processed_samples_count = 0

if master_process:
    logger.info(f"Processing batches {start_batch}-{end_batch-1} on {ddp_world_size} GPU(s)")
    logger.info(f"Sync interval: {args.sync_interval} batches")
    logger.info(f"Log interval: {args.log_interval} batches")
    logger.info(f"Plot interval: {args.plot_interval} batches {'(disabled)' if args.plot_interval <= 0 else ''}")
    logger.info(f"Sample file storage: {'enabled' if args.store_samples_to_file else 'disabled'}")
    if args.store_samples_to_file:
        logger.info(f"Samples will be stored in: {samples_dir}")

# %% ------------- batch loop ------------------------------


# Process through every batch in the dataset

for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Processing batches"):
    # Skip batches not assigned to this process in DDP mode
    if ddp and (batch_idx < start_batch or batch_idx >= end_batch):
        continue

    prompt_ids = batch["prompt_ids"].to(device)
    prompt_attention_mask = batch["prompt_attention_mask"].to(device)
    # correct_path_ids = batch["correct_path_ids"]
    # original_prompt_lens = batch["original_prompt_len"]
            
    batch_size = prompt_ids.size(0)
    prompt_length = prompt_ids.size(1)

    with ctx and torch.no_grad():

        # Generate paths from prompts
        generated_sequences ,generated_masks, all_hidden_states, gen_tokens_hidden_states, generated_mask_tokens = path_finding_generate_with_hidden_states(Lmodel, prompt_ids, max_new_tokens, attention_mask=prompt_attention_mask, eos_token=eos_token_id, temperature=args.temperature)

        # Update attention masks for the full generated sequences
        current_batch_size, full_seq_len = generated_sequences.shape
        # full_seq_attn_mask = torch.zeros_like(generated_sequences, dtype=torch.bool)
        full_seq_attn_mask = generated_masks

        # Mark original valid tokens for all samples at once
        orig_valid_tokens = prompt_attention_mask.sum(dim=1)  # Shape: (current_batch_size,)
        orig_seq_len = prompt_attention_mask.shape[1]
        

        hidden_states_tensor = all_hidden_states

        hidden_states_tensor = hidden_states_tensor / (torch.norm(hidden_states_tensor, dim=-1, keepdim=True) + 1e-8)
        
        prefix_H = hidden_states_tensor[:, :-2, :orig_seq_len, :] #.to(dtype = ptdtype)

        if skip_every_timestep > 1:
            prompt_ids_skipped, prompt_attention_mask_skipped, prefix_H = skip_time_steps(prompt_ids, prompt_attention_mask, prefix_H, max_nodes)
        else:
            prompt_attention_mask_skipped = prompt_attention_mask
            prompt_ids_skipped = prompt_ids
        if skip_every_layer > 1:
            prefix_H = prefix_H[:, ::skip_every_layer, :, :]

        z_e = encoder_model(prefix_H, prompt_attention_mask_skipped)
        distances = torch.sum(z_e**2, dim=1, keepdim=True) + torch.sum(vqvae61_codebook_weights**2, dim=1) - 2 * torch.matmul(z_e, vqvae61_codebook_weights.t())
        prefix_H_encodings = torch.argmin(distances, dim=1)

        generated_hs = hidden_states_tensor[:, -2, orig_seq_len-1:, :].contiguous() #.to(dtype = ptdtype).contiguous()
        generated_hs = normalize_single_stage(generated_hs)

        generated_hs_flat = generated_hs.view(-1, generated_hs.shape[-1])
        projected_hs_flat = single_vqvae.forward_proj(generated_hs_flat)
        distances_flat = torch.sum(projected_hs_flat**2, dim=1, keepdim=True) + torch.sum(single_vqvae_codebook_weights**2, dim=1) - 2 * torch.matmul(projected_hs_flat, single_vqvae_codebook_weights.t())
        prefix_h_encodings_flat = torch.argmin(distances_flat, dim=1)
        prefix_h_encodings = prefix_h_encodings_flat.view(generated_hs.shape[0], generated_hs.shape[1])

        # Get indicator vectors for conditional tracking if enabled
        if args.condition_on_correct_false:
            correct1_path = torch.tensor([temp[0] for temp in batch["correct_path_ids"]])[:,1:-1].to(device)
            correct2_path = torch.tensor([temp[1] for temp in batch["correct_path_ids"]])[:,1:-1].to(device)
            model_out_path = generated_sequences[:,prompt_length+1:prompt_length+1+max_path_len]
            # Check which entries match correct1_path or correct2_path
            matches_correct1 = torch.all(model_out_path == correct1_path, dim=1)
            matches_correct2 = torch.all(model_out_path == correct2_path, dim=1)
            correct_indicator = matches_correct1 | matches_correct2
            false_indicator = ~correct_indicator
        else:
            correct_indicator = None
            false_indicator = None

        # Collect samples in local buffer instead of directly updating StreamingInfo
        collect_start_idx = orig_seq_len-1
        active_samples = torch.ones(current_batch_size, dtype=torch.bool, device=device)
        
        for j in range(collect_start_idx, full_seq_len):
            if not active_samples.any():  # All samples have hit EOS, stop
                break

            tau = j - collect_start_idx
            if tau <= max_tau:
                for sample_idx in range(current_batch_size):    
                    if active_samples[sample_idx]:
                        sample = {
                            "H": prefix_H_encodings[sample_idx].item(), 
                            "h": prefix_h_encodings[sample_idx, j-collect_start_idx].item()
                        }
                        # Always add to main buffer
                        local_sample_buffer[tau].append(sample)
                        
                        # Add to conditional buffers if enabled
                        if args.condition_on_correct_false:
                            if correct_indicator[sample_idx]:
                                local_sample_buffer_correct[tau].append(sample)
                            if false_indicator[sample_idx]:
                                local_sample_buffer_false[tau].append(sample)

            # Check which active samples have EOS at position j
            eos_at_j = (generated_sequences[:, j] == eos_token_id) & active_samples
            active_samples[eos_at_j] = False

    # Update processed samples count
    processed_samples_count += current_batch_size
    
    # Log progress
    if batch_idx % args.log_interval == 0:
        samples_in_buffer = sum(len(samples) for samples in local_sample_buffer.values())
        if args.condition_on_correct_false:
            samples_in_buffer_correct = sum(len(samples) for samples in local_sample_buffer_correct.values())
            samples_in_buffer_false = sum(len(samples) for samples in local_sample_buffer_false.values())
            if ddp:
                print(f"Rank {ddp_rank}: Batch {batch_idx}, Processed: {processed_samples_count}, "
                      f"Buffer: {samples_in_buffer} (Correct: {samples_in_buffer_correct}, False: {samples_in_buffer_false})")
            else:
                print(f"Batch {batch_idx}, Processed: {processed_samples_count}, "
                      f"Buffer: {samples_in_buffer} (Correct: {samples_in_buffer_correct}, False: {samples_in_buffer_false})")
        else:
            if ddp:
                print(f"Rank {ddp_rank}: Batch {batch_idx}, Processed: {processed_samples_count}, Buffer: {samples_in_buffer}")
            else:
                print(f"Batch {batch_idx}, Processed: {processed_samples_count}, Buffer: {samples_in_buffer}")
    
    # Periodic sync and reporting
    if batch_idx % args.sync_interval == 0 and batch_idx > 0:
        if ddp:
            dist.barrier()  # Synchronize all processes
        
        # Sync samples and update StreamingInfo
        if args.condition_on_correct_false:
            synced_samples, synced_samples_correct, synced_samples_false = sync_and_update_streaming_info(
                local_sample_buffer, smi_dict, processed_samples_count,
                local_sample_buffer_correct, smi_dict_correct,
                local_sample_buffer_false, smi_dict_false
            )
        else:
            synced_samples, _, _ = sync_and_update_streaming_info(local_sample_buffer, smi_dict, processed_samples_count)
        
        # Flush samples to file if file storage is enabled
        if master_process and args.store_samples_to_file:
            for tau in smi_dict:
                smi_dict[tau].flush_to_file()
            
            if args.condition_on_correct_false:
                for tau in smi_dict_correct:
                    smi_dict_correct[tau].flush_to_file()
                for tau in smi_dict_false:
                    smi_dict_false[tau].flush_to_file()
        
        # Calculate and report metrics
        if args.condition_on_correct_false:
            results, results_correct, results_false = calculate_and_report_metrics(
                smi_dict, processed_samples_count, batch_idx, end_batch,
                smi_dict_correct, smi_dict_false, logger_correct, logger_false
            )
        else:
            results, _, _ = calculate_and_report_metrics(smi_dict, processed_samples_count, batch_idx, end_batch)
            results_correct = {}
            results_false = {}
        
        # Generate plots if plot_interval is enabled
        if args.plot_interval > 0 and batch_idx % args.plot_interval == 0:
            if results:
                create_plots_and_save(results, output_dir, processed_samples_count, master_process, logger, batch_idx=batch_idx, is_final=False)
            if args.condition_on_correct_false:
                if results_correct:
                    create_plots_and_save(results_correct, output_dir_correct, processed_samples_count, master_process, logger_correct, batch_idx=batch_idx, is_final=False)
                if results_false:
                    create_plots_and_save(results_false, output_dir_false, processed_samples_count, master_process, logger_false, batch_idx=batch_idx, is_final=False)
        
        # Clear local buffers after sync
        local_sample_buffer.clear()
        if args.condition_on_correct_false:
            local_sample_buffer_correct.clear()
            local_sample_buffer_false.clear()

# Final sync and reporting
if ddp:
    dist.barrier()

# Final sync of remaining samples
if local_sample_buffer or (args.condition_on_correct_false and (local_sample_buffer_correct or local_sample_buffer_false)):
    if args.condition_on_correct_false:
        synced_samples, synced_samples_correct, synced_samples_false = sync_and_update_streaming_info(
            local_sample_buffer, smi_dict, processed_samples_count,
            local_sample_buffer_correct, smi_dict_correct,
            local_sample_buffer_false, smi_dict_false
        )
    else:
        synced_samples, _, _ = sync_and_update_streaming_info(local_sample_buffer, smi_dict, processed_samples_count)
    
    local_sample_buffer.clear()
    if args.condition_on_correct_false:
        local_sample_buffer_correct.clear()
        local_sample_buffer_false.clear()

# Final flush and cleanup of file storage
if master_process and args.store_samples_to_file:
    for tau in smi_dict:
        smi_dict[tau].flush_to_file()
        # Get file statistics before closing
        if smi_dict[tau].get_file_sample_count() > 0:
            logger.info(f"Tau {tau}: {smi_dict[tau].get_file_sample_count()} samples saved to file")
    
    if args.condition_on_correct_false:
        for tau in smi_dict_correct:
            smi_dict_correct[tau].flush_to_file()
            if smi_dict_correct[tau].get_file_sample_count() > 0:
                logger_correct.info(f"Tau {tau}: {smi_dict_correct[tau].get_file_sample_count()} correct samples saved to file")
        
        for tau in smi_dict_false:
            smi_dict_false[tau].flush_to_file()
            if smi_dict_false[tau].get_file_sample_count() > 0:
                logger_false.info(f"Tau {tau}: {smi_dict_false[tau].get_file_sample_count()} false samples saved to file")
    
    logger.info("Closing sample storage files...")
    for tau in smi_dict:
        smi_dict[tau].close_file()
    
    if args.condition_on_correct_false:
        logger_correct.info("Closing correct sample storage files...")
        for tau in smi_dict_correct:
            smi_dict_correct[tau].close_file()
        
        logger_false.info("Closing false sample storage files...")
        for tau in smi_dict_false:
            smi_dict_false[tau].close_file()

# %%

# %% ------------- Final Results Calculation ------------------------------

# Final results calculation and reporting (only on master process)
if master_process:
    logger.info("\n" + "="*100)
    logger.info("FINAL RESULTS - PROCESSING COMPLETE")
    logger.info("="*100)
    
    total_samples_processed = sum(smi_dict[tau].N for tau in smi_dict if smi_dict[tau].N > 0)
    expected_samples = processed_samples_count * ddp_world_size if ddp else processed_samples_count
    
    logger.info(f"Total samples processed across all GPUs: {total_samples_processed}")
    logger.info(f"Expected samples based on batch count: {expected_samples}")
    logger.info("="*100)

    # Calculate results for all samples
    results = {}
    for tau in range(0, max_tau + 1):
        if tau in smi_dict and smi_dict[tau].N > 0:  # Only process if we have data
            results[tau] = {
                'mutual_information': smi_dict[tau].mutual_information("H", "h"),
                'joint_entropy': smi_dict[tau].joint_entropy(("H", "h")),
                'entropy_H': smi_dict[tau].entropy("H"),
                'entropy_h': smi_dict[tau].entropy("h"),
                'conditional_entropy_h_given_H': smi_dict[tau].conditional_entropy("h", "H"),
                'conditional_entropy_H_given_h': smi_dict[tau].conditional_entropy("H", "h"),
                'sample_count': smi_dict[tau].N
            }
            
            logger.info(f'Tau {tau}:')
            logger.info(f'  Mutual information I(H;h): {results[tau]["mutual_information"]:.6f}')
            logger.info(f'  Joint entropy H(H,h): {results[tau]["joint_entropy"]:.6f}')
            logger.info(f'  Entropy H(H): {results[tau]["entropy_H"]:.6f}')
            logger.info(f'  Entropy H(h): {results[tau]["entropy_h"]:.6f}')
            logger.info(f'  Conditional entropy H(h|H): {results[tau]["conditional_entropy_h_given_H"]:.6f}')
            logger.info(f'  Conditional entropy H(H|h): {results[tau]["conditional_entropy_H_given_h"]:.6f}')
            logger.info(f'  Sample count: {results[tau]["sample_count"]}')
            logger.info('')

    # Calculate results for correct samples (no print, only log to file)
    results_correct = {}
    if args.condition_on_correct_false:
        total_correct_samples = sum(smi_dict_correct[tau].N for tau in smi_dict_correct if smi_dict_correct[tau].N > 0)
        logger_correct.info("\n" + "="*100)
        logger_correct.info("FINAL CORRECT RESULTS - PROCESSING COMPLETE")
        logger_correct.info("="*100)
        logger_correct.info(f"Total correct samples processed: {total_correct_samples}")
        logger_correct.info("="*100)
        
        for tau in range(0, max_tau + 1):
            if tau in smi_dict_correct and smi_dict_correct[tau].N > 0:
                results_correct[tau] = {
                    'mutual_information': smi_dict_correct[tau].mutual_information("H", "h"),
                    'joint_entropy': smi_dict_correct[tau].joint_entropy(("H", "h")),
                    'entropy_H': smi_dict_correct[tau].entropy("H"),
                    'entropy_h': smi_dict_correct[tau].entropy("h"),
                    'conditional_entropy_h_given_H': smi_dict_correct[tau].conditional_entropy("h", "H"),
                    'conditional_entropy_H_given_h': smi_dict_correct[tau].conditional_entropy("H", "h"),
                    'sample_count': smi_dict_correct[tau].N
                }
                
                logger_correct.info(f'Tau {tau}:')
                logger_correct.info(f'  Mutual information I(H;h): {results_correct[tau]["mutual_information"]:.6f}')
                logger_correct.info(f'  Joint entropy H(H,h): {results_correct[tau]["joint_entropy"]:.6f}')
                logger_correct.info(f'  Entropy H(H): {results_correct[tau]["entropy_H"]:.6f}')
                logger_correct.info(f'  Entropy H(h): {results_correct[tau]["entropy_h"]:.6f}')
                logger_correct.info(f'  Conditional entropy H(h|H): {results_correct[tau]["conditional_entropy_h_given_H"]:.6f}')
                logger_correct.info(f'  Conditional entropy H(H|h): {results_correct[tau]["conditional_entropy_H_given_h"]:.6f}')
                logger_correct.info(f'  Sample count: {results_correct[tau]["sample_count"]}')
                logger_correct.info('')

    # Calculate results for false samples (no print, only log to file)
    results_false = {}
    if args.condition_on_correct_false:
        total_false_samples = sum(smi_dict_false[tau].N for tau in smi_dict_false if smi_dict_false[tau].N > 0)
        logger_false.info("\n" + "="*100)
        logger_false.info("FINAL FALSE RESULTS - PROCESSING COMPLETE")
        logger_false.info("="*100)
        logger_false.info(f"Total false samples processed: {total_false_samples}")
        logger_false.info("="*100)
        
        for tau in range(0, max_tau + 1):
            if tau in smi_dict_false and smi_dict_false[tau].N > 0:
                results_false[tau] = {
                    'mutual_information': smi_dict_false[tau].mutual_information("H", "h"),
                    'joint_entropy': smi_dict_false[tau].joint_entropy(("H", "h")),
                    'entropy_H': smi_dict_false[tau].entropy("H"),
                    'entropy_h': smi_dict_false[tau].entropy("h"),
                    'conditional_entropy_h_given_H': smi_dict_false[tau].conditional_entropy("h", "H"),
                    'conditional_entropy_H_given_h': smi_dict_false[tau].conditional_entropy("H", "h"),
                    'sample_count': smi_dict_false[tau].N
                }
                
                logger_false.info(f'Tau {tau}:')
                logger_false.info(f'  Mutual information I(H;h): {results_false[tau]["mutual_information"]:.6f}')
                logger_false.info(f'  Joint entropy H(H,h): {results_false[tau]["joint_entropy"]:.6f}')
                logger_false.info(f'  Entropy H(H): {results_false[tau]["entropy_H"]:.6f}')
                logger_false.info(f'  Entropy H(h): {results_false[tau]["entropy_h"]:.6f}')
                logger_false.info(f'  Conditional entropy H(h|H): {results_false[tau]["conditional_entropy_h_given_H"]:.6f}')
                logger_false.info(f'  Conditional entropy H(H|h): {results_false[tau]["conditional_entropy_H_given_h"]:.6f}')
                logger_false.info(f'  Sample count: {results_false[tau]["sample_count"]}')
                logger_false.info('')
else:
    results = {}
    results_correct = {}
    results_false = {}

# Save comprehensive results to text file (only on master process)
if master_process:
    results_file_path = os.path.join(output_dir, 'mutual_information_results_all_taus.txt')
    with open(results_file_path, 'w') as f:
        f.write("Information Theoretical Calculation Results Across All Tau Values\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset directory: {dataset_dir}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Max tau: {max_tau}\n")
        f.write(f"Max new tokens: {max_new_tokens}\n")
        f.write(f"Max seq length: {max_seq_length}\n")
        f.write(f"Max nodes: {max_nodes}\n")
        f.write(f"EOS token ID: {eos_token_id}\n")
        f.write(f"World size (GPUs used): {ddp_world_size}\n")
        f.write("\n" + "=" * 70 + "\n\n")
        
        # Write detailed results for each tau
        for tau in sorted(results.keys()):
            f.write(f"TAU = {tau}\n")
            f.write("-" * 20 + "\n")
            f.write(f"Mutual information I(H;h):     {results[tau]['mutual_information']:.6f}\n")
            f.write(f"Joint entropy H(H,h):          {results[tau]['joint_entropy']:.6f}\n")
            f.write(f"Entropy H(H):                  {results[tau]['entropy_H']:.6f}\n")
            f.write(f"Entropy H(h):                  {results[tau]['entropy_h']:.6f}\n")
            f.write(f"Conditional entropy H(h|H):    {results[tau]['conditional_entropy_h_given_H']:.6f}\n")
            f.write(f"Conditional entropy H(H|h):    {results[tau]['conditional_entropy_H_given_h']:.6f}\n")
            f.write(f"Sample count:                  {results[tau]['sample_count']}\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write(f"Calculation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_dir}\n")
    
    logger.info(f"Comprehensive results saved to: {results_file_path}")
    
    # Save conditional results to text files if enabled
    if args.condition_on_correct_false:
        # Save correct results text file
        if results_correct:
            results_file_path_correct = os.path.join(output_dir_correct, 'mutual_information_results_all_taus.txt')
            with open(results_file_path_correct, 'w') as f:
                f.write("Information Theoretical Calculation Results - CORRECT SAMPLES\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Dataset directory: {dataset_dir}\n")
                f.write(f"Batch size: {batch_size}\n")
                f.write(f"Max tau: {max_tau}\n")
                f.write(f"Max new tokens: {max_new_tokens}\n")
                f.write(f"Max seq length: {max_seq_length}\n")
                f.write(f"Max nodes: {max_nodes}\n")
                f.write(f"EOS token ID: {eos_token_id}\n")
                f.write(f"World size (GPUs used): {ddp_world_size}\n")
                f.write(f"Condition type: CORRECT samples only\n")
                f.write("\n" + "=" * 70 + "\n\n")
                
                for tau in sorted(results_correct.keys()):
                    f.write(f"TAU = {tau}\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Mutual information I(H;h):     {results_correct[tau]['mutual_information']:.6f}\n")
                    f.write(f"Joint entropy H(H,h):          {results_correct[tau]['joint_entropy']:.6f}\n")
                    f.write(f"Entropy H(H):                  {results_correct[tau]['entropy_H']:.6f}\n")
                    f.write(f"Entropy H(h):                  {results_correct[tau]['entropy_h']:.6f}\n")
                    f.write(f"Conditional entropy H(h|H):    {results_correct[tau]['conditional_entropy_h_given_H']:.6f}\n")
                    f.write(f"Conditional entropy H(H|h):    {results_correct[tau]['conditional_entropy_H_given_h']:.6f}\n")
                    f.write(f"Sample count:                  {results_correct[tau]['sample_count']}\n")
                    f.write("\n")
                
                f.write("=" * 70 + "\n")
                f.write(f"Calculation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output directory: {output_dir_correct}\n")
            
            logger_correct.info(f"Correct comprehensive results saved to: {results_file_path_correct}")
        
        # Save false results text file
        if results_false:
            results_file_path_false = os.path.join(output_dir_false, 'mutual_information_results_all_taus.txt')
            with open(results_file_path_false, 'w') as f:
                f.write("Information Theoretical Calculation Results - FALSE SAMPLES\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Dataset directory: {dataset_dir}\n")
                f.write(f"Batch size: {batch_size}\n")
                f.write(f"Max tau: {max_tau}\n")
                f.write(f"Max new tokens: {max_new_tokens}\n")
                f.write(f"Max seq length: {max_seq_length}\n")
                f.write(f"Max nodes: {max_nodes}\n")
                f.write(f"EOS token ID: {eos_token_id}\n")
                f.write(f"World size (GPUs used): {ddp_world_size}\n")
                f.write(f"Condition type: FALSE samples only\n")
                f.write("\n" + "=" * 70 + "\n\n")
                
                for tau in sorted(results_false.keys()):
                    f.write(f"TAU = {tau}\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Mutual information I(H;h):     {results_false[tau]['mutual_information']:.6f}\n")
                    f.write(f"Joint entropy H(H,h):          {results_false[tau]['joint_entropy']:.6f}\n")
                    f.write(f"Entropy H(H):                  {results_false[tau]['entropy_H']:.6f}\n")
                    f.write(f"Entropy H(h):                  {results_false[tau]['entropy_h']:.6f}\n")
                    f.write(f"Conditional entropy H(h|H):    {results_false[tau]['conditional_entropy_h_given_H']:.6f}\n")
                    f.write(f"Conditional entropy H(H|h):    {results_false[tau]['conditional_entropy_H_given_h']:.6f}\n")
                    f.write(f"Sample count:                  {results_false[tau]['sample_count']}\n")
                    f.write("\n")
                
                f.write("=" * 70 + "\n")
                f.write(f"Calculation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output directory: {output_dir_false}\n")
            
            logger_false.info(f"False comprehensive results saved to: {results_file_path_false}")

    # Create final plots for all samples
    create_plots_and_save(results, output_dir, processed_samples_count, master_process, logger, batch_idx=None, is_final=True)
    
    # Create final plots for conditional samples if enabled
    if args.condition_on_correct_false:
        if results_correct:
            create_plots_and_save(results_correct, output_dir_correct, processed_samples_count, master_process, logger_correct, batch_idx=None, is_final=True)
        if results_false:
            create_plots_and_save(results_false, output_dir_false, processed_samples_count, master_process, logger_false, batch_idx=None, is_final=True)
    
    # Save results as JSON for further analysis - all samples
    json_results_path = os.path.join(output_dir, 'mutual_information_results.json')
    with open(json_results_path, 'w') as f:
        json_results = {
            'metadata': {
                'dataset_dir': dataset_dir,
                'batch_size': batch_size,
                'max_tau': max_tau,
                'max_new_tokens': max_new_tokens,
                'max_seq_length': max_seq_length,
                'max_nodes': max_nodes,
                'eos_token_id': eos_token_id,
                'world_size': ddp_world_size,
                'total_samples_processed': processed_samples_count,
                'condition_on_correct_false': args.condition_on_correct_false,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': results
        }
        json.dump(json_results, f, indent=2)
    logger.info(f"JSON results saved to: {json_results_path}")
    
    # Save conditional results as JSON if enabled
    if args.condition_on_correct_false:
        # Save correct results JSON
        if results_correct:
            json_results_path_correct = os.path.join(output_dir_correct, 'mutual_information_results.json')
            with open(json_results_path_correct, 'w') as f:
                json_results_correct = {
                    'metadata': {
                        'dataset_dir': dataset_dir,
                        'batch_size': batch_size,
                        'max_tau': max_tau,
                        'max_new_tokens': max_new_tokens,
                        'max_seq_length': max_seq_length,
                        'max_nodes': max_nodes,
                        'eos_token_id': eos_token_id,
                        'world_size': ddp_world_size,
                        'total_samples_processed': processed_samples_count,
                        'condition_type': 'correct',
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'results': results_correct
                }
                json.dump(json_results_correct, f, indent=2)
            logger_correct.info(f"Correct JSON results saved to: {json_results_path_correct}")
        
        # Save false results JSON
        if results_false:
            json_results_path_false = os.path.join(output_dir_false, 'mutual_information_results.json')
            with open(json_results_path_false, 'w') as f:
                json_results_false = {
                    'metadata': {
                        'dataset_dir': dataset_dir,
                        'batch_size': batch_size,
                        'max_tau': max_tau,
                        'max_new_tokens': max_new_tokens,
                        'max_seq_length': max_seq_length,
                        'max_nodes': max_nodes,
                        'eos_token_id': eos_token_id,
                        'world_size': ddp_world_size,
                        'total_samples_processed': processed_samples_count,
                        'condition_type': 'false',
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'results': results_false
                }
                json.dump(json_results_false, f, indent=2)
            logger_false.info(f"False JSON results saved to: {json_results_path_false}")

# Clean up DDP
if ddp:
    destroy_process_group()

# %%