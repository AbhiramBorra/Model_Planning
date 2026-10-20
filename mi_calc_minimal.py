# %% 
"""
Mutual information calculation for the minimal VQ-VAE-like model.
"""
# %%

# %%

# %%
print('mi_calc_minimal.py')
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

# %%
# Load training configuration from a JSON file
try:
    parser = argparse.ArgumentParser(description="Calculate mutual information for VQ-VAE models")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to dataset directory")
    parser.add_argument('--exp_dir', type=str, required=True, help="Path to experiment directory")
    parser.add_argument('--LLM_ckpt_folder_dir', type=str, help="Path to LLM checkpoint folder")
    parser.add_argument('--vqvae1_ckpt_folder_dir', type=str, help="Path to VQVAE1 checkpoint folder") 
    parser.add_argument('--vqvae2_ckpt_folder_dir', type=str, help="Path to VQVAE2 checkpoint folder")
    parser.add_argument('--vqvae_single_ckpt_folder_dir', type=str, help="Path to VQVAE single checkpoint folder")
    parser.add_argument('--output_dir', type=str, help="Path to output directory")
    parser.add_argument('--batch_size', type=int, help="Batch size")
    parser.add_argument('--sync_interval', type=int, help="How often to sync and report results (in batches)")
    parser.add_argument('--log_interval', type=int, help="How often to log progress (in batches)")
    parser.add_argument('--plot_interval', type=int, help="How often to generate plots (in batches, 0 to disable)")
    parser.add_argument('--store_samples_to_file', action='store_true', default=True, help="Store samples to HDF5 files for later analysis")
    parser.add_argument('--no_store_samples', dest='store_samples_to_file', action='store_false', help="Disable storing samples to files")
    args = parser.parse_args()
except:
    # Create args namespace for interactive use
    args = argparse.Namespace()
    args.dataset_dir = "data/context_free_grammar/cfg_s13333-61-_rd345_rl23_4000k"
    args.exp_dir = './exp_cfg_s13333-61-_rd345_rl23_4000k'
    args.store_samples_to_file = True


args.LLM_ckpt_folder_dir = args.LLM_ckpt_folder_dir if hasattr(args, 'LLM_ckpt_folder_dir') and args.LLM_ckpt_folder_dir is not None else None
args.vqvae1_ckpt_folder_dir = args.vqvae1_ckpt_folder_dir if hasattr(args, 'vqvae1_ckpt_folder_dir') and args.vqvae1_ckpt_folder_dir is not None else None
args.vqvae2_ckpt_folder_dir = args.vqvae2_ckpt_folder_dir if hasattr(args, 'vqvae2_ckpt_folder_dir') and args.vqvae2_ckpt_folder_dir is not None else None 
args.vqvae_single_ckpt_folder_dir = args.vqvae_single_ckpt_folder_dir if hasattr(args, 'vqvae_single_ckpt_folder_dir') and args.vqvae_single_ckpt_folder_dir is not None else None
args.output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir is not None else args.exp_dir+'/mi_output'
args.batch_size = args.batch_size if hasattr(args, 'batch_size') and args.batch_size is not None else 64
args.sync_interval = args.sync_interval if hasattr(args, 'sync_interval') and args.sync_interval is not None else 100
args.log_interval = args.log_interval if hasattr(args, 'log_interval') and args.log_interval is not None else 100
args.plot_interval = args.plot_interval if hasattr(args, 'plot_interval') and args.plot_interval is not None else 10000
args.store_samples_to_file = args.store_samples_to_file if hasattr(args, 'store_samples_to_file') and args.store_samples_to_file is not None else True
if args.exp_dir:
    # Get list of folders in LLMout directory
    llm_out_dir = os.path.join(args.exp_dir, "LLMout") if args.LLM_ckpt_folder_dir is None else os.path.join(args.exp_dir, args.LLM_ckpt_folder_dir)
    if os.path.exists(llm_out_dir):
        llm_folders = [f for f in os.listdir(llm_out_dir) if os.path.isdir(os.path.join(llm_out_dir, f))]
        if llm_folders:
            args.LLM_ckpt_folder_dir = os.path.join(llm_out_dir, llm_folders[0])
    
    args.vqvae1_ckpt_folder_dir = os.path.join(args.exp_dir, "vqvae1_out") if args.vqvae1_ckpt_folder_dir is None else os.path.join(args.exp_dir, args.vqvae1_ckpt_folder_dir)
    args.vqvae2_ckpt_folder_dir = os.path.join(args.exp_dir, "vqvae2_out") if args.vqvae2_ckpt_folder_dir is None else os.path.join(args.exp_dir, args.vqvae2_ckpt_folder_dir)
    args.vqvae_single_ckpt_folder_dir = os.path.join(args.exp_dir, "vqvae_single_out") if args.vqvae_single_ckpt_folder_dir is None else os.path.join(args.exp_dir, args.vqvae_single_ckpt_folder_dir)


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda' and not torch.cuda.is_available():
    device = 'cpu'
    if master_process:
        logger.info("Warning: CUDA is not available, using CPU instead.")

# Determine dtype based on device capabilities
if 'cuda' in device and torch.cuda.is_bf16_supported():
    dtype = 'bfloat16'
else:
    dtype = 'float32'
    if master_process and 'cuda' in device:
        logger.info("Warning: bfloat16 not supported on this device, using float32 instead.")


# First stage experiment path/directory
first_stage_ckpt_path = args.vqvae1_ckpt_folder_dir
if first_stage_ckpt_path is None:
    raise ValueError("First stage checkpoint model path must be provided")
elif os.path.isdir(first_stage_ckpt_path):
    print(f'{first_stage_ckpt_path} is a directory. Loading the latest ckpt.py file from this directory.')
    if not os.path.exists(os.path.join(first_stage_ckpt_path, 'ckpt.pt')):
        raise ValueError(f"{os.path.join(first_stage_ckpt_path, 'ckpt.pt')} not found")
    else:
        first_stage_ckpt_path = os.path.join(first_stage_ckpt_path, 'ckpt.pt')
elif not os.path.exists(first_stage_ckpt_path):
    raise ValueError(f"{first_stage_ckpt_path} not found")
# Load first stage checkpoint
first_stage_checkpoint = torch.load(first_stage_ckpt_path, map_location=device)

second_stage_ckpt_path = args.vqvae2_ckpt_folder_dir
if second_stage_ckpt_path is None:
    raise ValueError("Second stage checkpoint model path must be provided")
elif os.path.isdir(second_stage_ckpt_path):
    print(f'{second_stage_ckpt_path} is a directory. Loading the latest ckpt.py file from this directory.')
    if not os.path.exists(os.path.join(second_stage_ckpt_path, 'ckpt.pt')):
        raise ValueError(f"{os.path.join(second_stage_ckpt_path, 'ckpt.pt')} not found")
    else:
        second_stage_ckpt_path = os.path.join(second_stage_ckpt_path, 'ckpt.pt')
elif not os.path.exists(second_stage_ckpt_path):
    raise ValueError(f"{second_stage_ckpt_path} not found")
# Load first stage checkpoint
second_stage_checkpoint = torch.load(second_stage_ckpt_path, map_location=device)

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

if master_process:
    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger(save_logs_flag = save_logs_flag, print_logs = print_logs, experiment_dir = output_dir)
else:
    # Create a dummy logger for non-master processes that doesn't actually log
    logger = logging.getLogger('dummy')
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)  # Effectively disable logging

if master_process:
    logger.info(f"DDP: rank {ddp_rank}/{ddp_world_size}, local_rank {ddp_local_rank}, master: {master_process}")
else:
    print(f"DDP: rank {ddp_rank}/{ddp_world_size}, local_rank {ddp_local_rank}, master: {master_process}")




torch.set_float32_matmul_precision('high') 
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
# NOTE: March, 6, 2025: Before using Float16, make sure that gradientscaler works.
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# %% ------------- Load the language model ------------------------------
Lmodel = nanoLLM(model_name = args.LLM_ckpt_folder_dir, base_dir = None)
Lmodel.eval()
Lmodel.to(device)


# %% ------------- Load VQ-VAE 1 Encoder ------------------------------
first_stage_encoder = Encoder1(
    L=first_stage_checkpoint['config']['vqvae1_config']['L'], 
    d=first_stage_checkpoint['config']['vqvae1_config']['d'], 
    d2=first_stage_checkpoint['config']['vqvae1_config']['d2'], 
    num_layers_layerwise_stage=first_stage_checkpoint['config']['vqvae1_config']['num_layers_layerwise_stage'],
    num_layers_aggregate_stage=first_stage_checkpoint['config']['vqvae1_config']['num_layers_aggregate_stage'],
    config_layerwise_stage=first_stage_checkpoint['config']['vqvae1_config']['config_layerwise_stage'],
    config_aggregate_stage=first_stage_checkpoint['config']['vqvae1_config']['config_aggregate_stage']
)

first_stage_encoder.to(device)

encoder_param_dict = {k.replace('module.encoder.',''):v for k,v in first_stage_checkpoint['model_state_dict'].items() if 'module.encoder.' in k}
encoder_param_dict.update({k.replace('encoder.',''):v for k,v in first_stage_checkpoint['model_state_dict'].items() if k.startswith('encoder.')})
first_stage_encoder.load_state_dict(encoder_param_dict, strict=True)

tied_encoder_proj = None if first_stage_checkpoint['config']['vqvae1_config']['config_layerwise_stage']['tied_encoder_proj'] == False else first_stage_encoder.proj
first_stage_decoder = Decoder1(
    L=first_stage_checkpoint['config']['vqvae1_config']['L'], 
    d=first_stage_checkpoint['config']['vqvae1_config']['d'], 
    d2=first_stage_checkpoint['config']['vqvae1_config']['d2'], 
    num_layers_aggregate_stage=first_stage_checkpoint['config']['vqvae1_config']['num_layers_aggregate_stage'],
    num_layers_layerwise_stage=first_stage_checkpoint['config']['vqvae1_config']['num_layers_layerwise_stage'],
    config_layerwise_stage=first_stage_checkpoint['config']['vqvae1_config']['config_layerwise_stage'],
    config_aggregate_stage=first_stage_checkpoint['config']['vqvae1_config']['config_aggregate_stage'],
    tied_encoder_proj=tied_encoder_proj
)

first_stage_decoder.to(device)

decoder_param_dict = {k.replace('module.decoder.',''):v for k,v in first_stage_checkpoint['model_state_dict'].items() if 'module.decoder.' in k}
decoder_param_dict.update({k.replace('decoder.',''):v for k,v in first_stage_checkpoint['model_state_dict'].items() if k.startswith('decoder.')})
first_stage_decoder.load_state_dict(decoder_param_dict, strict=True)

first_stage_codebook = nn.Embedding(first_stage_checkpoint['config']['vqvae1_config']['codebook_size'], first_stage_checkpoint['config']['vqvae1_config']['d2'])
first_stage_codebook.to(device)
codebook_param_dict = {k.replace('module.codebook.',''):v for k,v in first_stage_checkpoint['model_state_dict'].items() if 'module.codebook.' in k}
codebook_param_dict.update({k.replace('codebook.',''):v for k,v in first_stage_checkpoint['model_state_dict'].items() if k.startswith('codebook.')})
first_stage_codebook.load_state_dict(codebook_param_dict, strict=True)
print('First stage encoder:')
checkModelLoadCorrect(first_stage_encoder, encoder_param_dict)
print('First stage codebook:')
checkModelLoadCorrect(first_stage_codebook, codebook_param_dict)
print('First stage decoder:')
checkModelLoadCorrect(first_stage_decoder, decoder_param_dict)

first_stage_encoder.eval()  # eval() only affects behavior during forward pass (e.g. dropout, batchnorm)
for param in first_stage_encoder.parameters():
    param.requires_grad = False # requires_grad=False is still needed to prevent gradient computation
first_stage_codebook.eval()
for param in first_stage_codebook.parameters():
    param.requires_grad = False
first_stage_decoder.eval()
for param in first_stage_decoder.parameters():
    param.requires_grad = False

first_stage_codebook_weights = first_stage_codebook.weight

# Load normalization values from first stage VQVAE1 model
first_stage_normalization_values = None
for k, v in first_stage_checkpoint['model_state_dict'].items():
    if 'normalization_values' in k:
        first_stage_normalization_values = v.to(device)
        print(f'Found normalization values in first stage checkpoint: {k}, {v}')
        break

assert first_stage_normalization_values is not None, "No normalization values found in first stage checkpoint. This may affect model performance."

def normalize_first_stage(x):
    """Normalize input using first stage VQVAE1 normalization values."""
    if first_stage_normalization_values is not None:
        # Expand normalization values to match x shape: (1, L, 1, 1)
        norm_values_expanded = first_stage_normalization_values.view(1, -1, 1, 1)
        return x / (norm_values_expanded + 1e-8)
    else:
        return x

def denormalize_first_stage(x):
    """Denormalize input using first stage VQVAE1 normalization values."""
    if first_stage_normalization_values is not None:
        # Expand normalization values to match x shape: (1, L, 1, 1)
        norm_values_expanded = first_stage_normalization_values.view(1, -1, 1, 1)
        return x * (norm_values_expanded + 1e-8)
    else:
        return x


# %%

# %% ------------- Load VQ-VAE 2 Encoder ------------------------------

second_stage_encoder = Encoder2(
    d2 = first_stage_checkpoint['config']['vqvae1_config']['d2'],
    min_seq_len = second_stage_checkpoint['config']['min_max_prefix_len'][0],
    D = second_stage_checkpoint['config']['vqvae2_config']['D'],
    num_layers = second_stage_checkpoint['config']['vqvae2_config']['encoder_num_layers'],
    config = second_stage_checkpoint['config']['vqvae2_config']['encoder_config']
)

second_stage_encoder.to(device)

encoder_param_dict = {k.replace('module.encoder.',''):v for k,v in second_stage_checkpoint['model_state_dict'].items() if 'module.encoder.' in k}
encoder_param_dict.update({k.replace('encoder.',''):v for k,v in second_stage_checkpoint['model_state_dict'].items() if k.startswith('encoder.')})
second_stage_encoder.load_state_dict(encoder_param_dict, strict=True)

second_stage_decoder = Decoder2(
    d2 = first_stage_checkpoint['config']['vqvae1_config']['d2'],
    max_prefix_len = second_stage_checkpoint['config']['min_max_prefix_len'][1],
    D = second_stage_checkpoint['config']['vqvae2_config']['D'],
    num_layers = second_stage_checkpoint['config']['vqvae2_config']['decoder_num_layers'],
    config = second_stage_checkpoint['config']['vqvae2_config']['decoder_config']
)

second_stage_decoder.to(device)
decoder_param_dict = {k.replace('module.decoder.',''):v for k,v in second_stage_checkpoint['model_state_dict'].items() if 'module.decoder.' in k}
decoder_param_dict.update({k.replace('decoder.',''):v for k,v in second_stage_checkpoint['model_state_dict'].items() if k.startswith('decoder.')})
second_stage_decoder.load_state_dict(decoder_param_dict, strict=True)

second_stage_codebook = nn.Embedding(second_stage_checkpoint['config']['vqvae2_config']['codebook_size'], second_stage_checkpoint['config']['vqvae2_config']['D'])
second_stage_codebook.to(device)
codebook_param_dict = {k.replace('module.codebook.',''):v for k,v in second_stage_checkpoint['model_state_dict'].items() if 'module.codebook.' in k}
codebook_param_dict.update({k.replace('codebook.',''):v for k,v in second_stage_checkpoint['model_state_dict'].items() if k.startswith('codebook.')})
second_stage_codebook.load_state_dict(codebook_param_dict, strict=True)

print('Second stage encoder:')
checkModelLoadCorrect(second_stage_encoder, encoder_param_dict)
print('Second stage decoder:')
checkModelLoadCorrect(second_stage_decoder, decoder_param_dict)
print('Second stage codebook:')
checkModelLoadCorrect(second_stage_codebook, codebook_param_dict)

second_stage_encoder.eval()
for param in second_stage_encoder.parameters():
    param.requires_grad = False
second_stage_decoder.eval()
for param in second_stage_decoder.parameters():
    param.requires_grad = False
second_stage_codebook.eval()
for param in second_stage_codebook.parameters():
    param.requires_grad = False

second_stage_codebook_weights = second_stage_codebook.weight

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
split = 'train'
dataset_dir = args.dataset_dir
batch_size = args.batch_size
min_max_prefix_len = second_stage_checkpoint['config']['min_max_prefix_len']

prefix_meta_path = os.path.join(dataset_dir, f'prefix_meta_{min_max_prefix_len[0]}_{min_max_prefix_len[1]}.pkl')
with open(prefix_meta_path, 'rb') as f:
    prefix_meta = pickle.load(f)
    pad_token_id = prefix_meta['prefix_padding']
    min_prefix_length = prefix_meta['min_prefix_length']
    max_prefix_length = prefix_meta['max_prefix_length']
    # eos_token_id = int([temp for temp in dataset_dir.split('_') if temp.startswith('s')][0][-1]) + 1# prefix_meta['eos_token_id']
    temp =[temp for temp in dataset_dir.split('_') if temp.startswith('s')][0]
    last = temp[-1]
    if last == '-':
        last = temp[:-1].split('-')[-1]
    eos_token_id = int(last) + 1
    bos_token_id = 0
    pad_token_id = eos_token_id + 1
    longest_length = prefix_meta['longest_length']
    shortest_length = prefix_meta['shortest_length']
    if master_process:
        logger.info(f'pad_token_id: {pad_token_id}')
        logger.info(f'min_prefix_length: {min_prefix_length}')
        logger.info(f'max_prefix_length: {max_prefix_length}')
        logger.info(f'eos_token_id: {eos_token_id}')
        logger.info(f'bos_token_id: {bos_token_id}')
        logger.info(f'longest_length: {longest_length}')
        logger.info(f'shortest_length: {shortest_length}')

# Create the dataloader for CFG data
train_loader = create_cfg_dataloader(
    data_path=os.path.join(dataset_dir, f'{split}_prefixes{min_max_prefix_len[0]}_{min_max_prefix_len[1]}.bin'),
    batch_size=batch_size,
    seq_length=min_max_prefix_len[1],
    pad_token_id=pad_token_id,
    shuffle=False,  # Set to False for processing all data systematically
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

# Simple sample collection and periodic sync approach ########
from information_theory_utils import StreamingInfo
import torch.distributed as dist
from collections import defaultdict

def sync_and_update_streaming_info(local_samples, smi_dict, processed_samples_count):
    """Sync samples from all GPUs and update StreamingInfo on master process"""
    if ddp:
        # Gather samples from all processes
        all_samples = [None] * ddp_world_size
        dist.all_gather_object(all_samples, local_samples)
        
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
            
            return dict(combined_samples)
    else:
        # Single GPU - directly update StreamingInfo
        for tau, samples in local_samples.items():
            if tau in smi_dict:
                for sample in samples:
                    smi_dict[tau].update(sample)
        return local_samples
    
    return {}

def calculate_and_report_metrics(smi_dict, processed_samples_count, batch_idx, total_batches):
    """Calculate and report current metrics"""
    if not master_process:
        return {}
    
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
    
    return results

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

max_tau = longest_length - min_prefix_length

# Only master process maintains StreamingInfo objects
if master_process:
    # Create samples directory for file storage
    samples_dir = os.path.join(output_dir, 'samples') if args.store_samples_to_file else None
    if samples_dir:
        os.makedirs(samples_dir, exist_ok=True)
        logger.info(f"Sample storage directory created: {samples_dir}")
    
    variables = {i:["H", "h"] for i in range(0,max_tau+1)}
    combos_to_track_dict = {i:[("H",), ("h",), ("H", "h")] for i in range(0,max_tau+1)}
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
else:
    smi_dict = {}

max_new_tokens = longest_length - shortest_length

# Sample collection dictionaries for each process
local_sample_buffer = defaultdict(list)  # tau -> list of samples
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
for batch_idx, (sequences, attention_masks) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Processing batches"):
    # Skip batches not assigned to this process in DDP mode
    if ddp and (batch_idx < start_batch or batch_idx >= end_batch):
        continue
        
    sequences = sequences.to(device)
    sequences[sequences == pad_token_id] = 0 # replace padding tokens with 0
    attention_masks = attention_masks.to(device)

    with ctx and torch.no_grad():
        # Generate the rest of the sequence using the trained LLM
        max_new_tokens = longest_length - shortest_length
        
        # Generate sequences - this will append new tokens to the right
        generated_sequences = Lmodel.generate(
            input_ids=sequences,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            eos_token=eos_token_id,
            attention_mask=attention_masks
        )
        
        # Update attention masks for the full generated sequences
        current_batch_size, full_seq_len = generated_sequences.shape
        full_seq_attn_mask = torch.zeros_like(generated_sequences, dtype=torch.bool)
        
        # Mark original valid tokens for all samples at once
        orig_valid_tokens = attention_masks.sum(dim=1)  # Shape: (current_batch_size,)
        orig_seq_len = attention_masks.shape[1]
        
        full_seq_attn_mask[:,:orig_seq_len] = attention_masks
        
        # Process generated tokens efficiently
        gen_start_idx = orig_seq_len
        active_samples = torch.ones(current_batch_size, dtype=torch.bool, device=device)  # Track which samples are still active
        
        for j in range(gen_start_idx, full_seq_len):
            if not active_samples.any():  # All samples have hit EOS, stop
                break
                
            # Check which active samples have EOS at position j
            eos_at_j = (generated_sequences[:, j] == eos_token_id) & active_samples
            
            # For active samples that don't have EOS at j, mark position as 1
            no_eos_at_j = active_samples & ~eos_at_j
            full_seq_attn_mask[no_eos_at_j, j] = 1
            
            # For active samples that have EOS at j, mark position as 1 and deactivate them
            full_seq_attn_mask[eos_at_j, j] = 1
            active_samples[eos_at_j] = False
        
        # Generate hidden states H using the full generated sequences
        hidden_states_tensor = Lmodel.generate_prefix_hidden_states(
            input_ids=generated_sequences,
            attention_mask=full_seq_attn_mask
        )

        hidden_states_tensor = hidden_states_tensor / (torch.norm(hidden_states_tensor, dim=-1, keepdim=True) + 1e-8)
        
        prefix_H = hidden_states_tensor[:, :-2, :orig_seq_len, :] #.to(dtype = ptdtype)
        prefix_H = normalize_first_stage(prefix_H)
        out_first_stage = first_stage_encoder(prefix_H, attention_masks)
        out_first_stage_right_padded, mask_right_padded = convert_left_to_right_padding(out_first_stage, attention_masks)
        z_e = second_stage_encoder(out_first_stage_right_padded, padding_mask=mask_right_padded)
        distances = torch.sum(z_e**2, dim=1, keepdim=True) + torch.sum(second_stage_codebook_weights**2, dim=1) - 2 * torch.matmul(z_e, second_stage_codebook_weights.t())
        prefix_H_encodings = torch.argmin(distances, dim=1)
        

        generated_hs = hidden_states_tensor[:, -2, orig_seq_len-1:, :].contiguous() #.to(dtype = ptdtype).contiguous()
        generated_hs = normalize_single_stage(generated_hs)

        generated_hs_flat = generated_hs.view(-1, generated_hs.shape[-1])
        projected_hs_flat = single_vqvae.forward_proj(generated_hs_flat)
        distances_flat = torch.sum(projected_hs_flat**2, dim=1, keepdim=True) + torch.sum(single_vqvae_codebook_weights**2, dim=1) - 2 * torch.matmul(projected_hs_flat, single_vqvae_codebook_weights.t())
        prefix_h_encodings_flat = torch.argmin(distances_flat, dim=1)
        prefix_h_encodings = prefix_h_encodings_flat.view(generated_hs.shape[0], generated_hs.shape[1])

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
                        local_sample_buffer[tau].append(sample)

            # Check which active samples have EOS at position j
            eos_at_j = (generated_sequences[:, j] == eos_token_id) & active_samples
            active_samples[eos_at_j] = False

    # Update processed samples count
    processed_samples_count += current_batch_size
    
    # Log progress
    if batch_idx % args.log_interval == 0:
        samples_in_buffer = sum(len(samples) for samples in local_sample_buffer.values())
        if ddp:
            print(f"Rank {ddp_rank}: Batch {batch_idx}, Processed: {processed_samples_count}, Buffer: {samples_in_buffer}")
        else:
            print(f"Batch {batch_idx}, Processed: {processed_samples_count}, Buffer: {samples_in_buffer}")
    
    # Periodic sync and reporting
    if batch_idx % args.sync_interval == 0 and batch_idx > 0:
        if ddp:
            dist.barrier()  # Synchronize all processes
        
        # Sync samples and update StreamingInfo
        synced_samples = sync_and_update_streaming_info(local_sample_buffer, smi_dict, processed_samples_count)
        
        # Flush samples to file if file storage is enabled
        if master_process and args.store_samples_to_file:
            for tau in smi_dict:
                smi_dict[tau].flush_to_file()
        
        # Calculate and report metrics
        results = calculate_and_report_metrics(smi_dict, processed_samples_count, batch_idx, end_batch)
        
        # Generate plots if plot_interval is enabled
        if args.plot_interval > 0 and batch_idx % args.plot_interval == 0 and results:
            create_plots_and_save(results, output_dir, processed_samples_count, master_process, logger, batch_idx=batch_idx, is_final=False)
        
        # Clear local buffer after sync
        local_sample_buffer.clear()

# Final sync and reporting
if ddp:
    dist.barrier()

# Final sync of remaining samples
if local_sample_buffer:
    synced_samples = sync_and_update_streaming_info(local_sample_buffer, smi_dict, processed_samples_count)
    local_sample_buffer.clear()

# Final flush and cleanup of file storage
if master_process and args.store_samples_to_file:
    for tau in smi_dict:
        smi_dict[tau].flush_to_file()
        # Get file statistics before closing
        if smi_dict[tau].get_file_sample_count() > 0:
            logger.info(f"Tau {tau}: {smi_dict[tau].get_file_sample_count()} samples saved to file")
    
    logger.info("Closing sample storage files...")
    for tau in smi_dict:
        smi_dict[tau].close_file()

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
else:
    results = {}

# Save comprehensive results to text file (only on master process)
if master_process:
    results_file_path = os.path.join(output_dir, 'mutual_information_results_all_taus.txt')
    with open(results_file_path, 'w') as f:
        f.write("Information Theoretical Calculation Results Across All Tau Values\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset directory: {dataset_dir}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Max tau: {max_tau}\n")
        f.write(f"Min prefix length: {min_prefix_length}\n")
        f.write(f"Max prefix length: {max_prefix_length}\n")
        f.write(f"Longest sequence length: {longest_length}\n")
        f.write(f"Shortest sequence length: {shortest_length}\n")
        f.write(f"EOS token ID: {eos_token_id}\n")
        f.write(f"Pad token ID: {pad_token_id}\n")
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

    # Create final plots
    create_plots_and_save(results, output_dir, processed_samples_count, master_process, logger, batch_idx=None, is_final=True)
    
    # Save results as JSON for further analysis
    json_results_path = os.path.join(output_dir, 'mutual_information_results.json')
    with open(json_results_path, 'w') as f:
        json_results = {
            'metadata': {
                'dataset_dir': dataset_dir,
                'batch_size': batch_size,
                'max_tau': max_tau,
                'min_prefix_length': min_prefix_length,
                'max_prefix_length': max_prefix_length,
                'longest_length': longest_length,
                'shortest_length': shortest_length,
                'eos_token_id': eos_token_id,
                'pad_token_id': pad_token_id,
                'world_size': ddp_world_size,
                'total_samples_processed': processed_samples_count,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': results
        }
        json.dump(json_results, f, indent=2)
    logger.info(f"JSON results saved to: {json_results_path}")

# Clean up DDP
if ddp:
    destroy_process_group()

# %%