
# %%

# %%

# %%
print('mi_calc_full_path_long2_61.py')
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
    parser.add_argument('--vqvae61_ckpt_folder_dir', type=str, help="Path to VQVAE61 checkpoint folder") 
    parser.add_argument('--output_dir', type=str, help="Path to output directory")
    parser.add_argument('--batch_size', type=int, help="Batch size")
    parser.add_argument('--sync_interval', type=int, help="How often to sync and report results (in batches)")
    parser.add_argument('--log_interval', type=int, help="How often to log progress (in batches)")
    parser.add_argument('--plot_interval', type=int, help="How often to generate plots (in batches, 0 to disable)")
    parser.add_argument('--store_samples_to_file', action='store_true', default=True, help="Store samples to HDF5 files for later analysis")
    parser.add_argument('--no_store_samples', dest='store_samples_to_file', action='store_false', help="Disable storing samples to files")
    parser.add_argument('--temperature', type=float,default=0.0, help="Temperature for sampling")
    args = parser.parse_args()
except:
    # Create args namespace for interactive use
    args = argparse.Namespace()



args.LLM_ckpt_folder_dir = os.path.join(args.exp_dir, args.LLM_ckpt_folder_dir) if hasattr(args, 'LLM_ckpt_folder_dir') and args.LLM_ckpt_folder_dir is not None else None
args.vqvae61_ckpt_folder_dir = args.vqvae61_ckpt_folder_dir if hasattr(args, 'vqvae61_ckpt_folder_dir') and args.vqvae61_ckpt_folder_dir is not None else None
args.dataset_dir = args.dataset_dir if hasattr(args, 'dataset_dir') and args.dataset_dir is not None else args.exp_dir+'/samples'
args.output_dir = os.path.join(args.exp_dir, args.output_dir) if hasattr(args, 'output_dir') and args.output_dir is not None else os.path.join(args.exp_dir, 'path_mi_output')
args.batch_size = args.batch_size if hasattr(args, 'batch_size') and args.batch_size is not None else 64
args.sync_interval = args.sync_interval if hasattr(args, 'sync_interval') and args.sync_interval is not None else 100
args.log_interval = args.log_interval if hasattr(args, 'log_interval') and args.log_interval is not None else 100
args.plot_interval = args.plot_interval if hasattr(args, 'plot_interval') and args.plot_interval is not None else 10000
args.store_samples_to_file = args.store_samples_to_file if hasattr(args, 'store_samples_to_file') and args.store_samples_to_file is not None else True
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


# %% Load some data attributes - for path vqvae
json_file_path = os.path.join(dataset_dir, 'config_train.json')
with open(json_file_path, 'r') as f:
    config_train = json.load(f)
max_nodes = config_train['max_nodes']
max_path_len = config_train['max_path_len']
vocab_size = max_nodes + 6 # To guarantee the coverage of the vocabulary + 6
number_of_correct_paths = config_train['n_paths']
number_of_decoy_paths = config_train['n_decoy']





# %%
skip_every_timestep = vqvae61_checkpoint['config']['skip_every_timestep']
skip_every_layer = vqvae61_checkpoint['config']['skip_every_layer']

# %%

# %% Create PathFinding dataset and dataloader
path_finding_dataset = PathFindingDataset(
    jsonl_file=os.path.join(dataset_dir, f'{split}.jsonl'),
    max_seq_length=max_seq_length,
    mask_edges=True,
    max_nodes=max_nodes,
    number_of_hyphens=0 # zero in our experiments 
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

# %%
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
                for variable_name, samples in proc_samples.items():
                    combined_samples[variable_name].extend(samples)
            
            # Update StreamingInfo objects with new samples
            for variable_name, samples in combined_samples.items():
                if variable_name in smi_dict:
                    for sample in samples:
                        smi_dict[variable_name].update(sample)
            
            return dict(combined_samples)
    else:
        # Single GPU - directly update StreamingInfo
        for variable_name, samples in local_samples.items():
            if variable_name in smi_dict:
                for sample in samples:
                    smi_dict[variable_name].update(sample)
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
    
    for variable_name in variable_names_to_track:
        if variable_name in smi_dict and smi_dict[variable_name].N > 0:
            results[variable_name] = {
                'mutual_information': smi_dict[variable_name].mutual_information("H", "h"),
                'joint_entropy': smi_dict[variable_name].joint_entropy(("H", "h")),
                'entropy_H': smi_dict[variable_name].entropy("H"),
                'entropy_h': smi_dict[variable_name].entropy("h"),
                'conditional_entropy_h_given_H': smi_dict[variable_name].conditional_entropy("h", "H"),
                'conditional_entropy_H_given_h': smi_dict[variable_name].conditional_entropy("H", "h"),
                'sample_count': smi_dict[variable_name].N
            }
            total_samples += smi_dict[variable_name].N
            
            logger.info(f'{variable_name}: MI={results[variable_name]["mutual_information"]:.4f}, '
                       f'H(H,h)={results[variable_name]["joint_entropy"]:.4f}, '
                       f'H(H)={results[variable_name]["entropy_H"]:.4f}, '
                       f'H(h)={results[variable_name]["entropy_h"]:.4f}, '
                       f'Samples={results[variable_name]["sample_count"]}')
    
    logger.info(f"Total samples across all variables: {total_samples}")
    logger.info(f"{'='*80}\n")
    
    return results

def create_plots_and_save(results, output_dir, processed_samples_count, master_process, logger, batch_idx=None, is_final=False):
    """Create and save plots for mutual information results"""
    if not results:
        print("No results to plot - no data was collected for any variables.")
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
    variable_names = sorted(results.keys())
    mutual_info = [results[var]['mutual_information'] for var in variable_names]
    joint_entropy = [results[var]['joint_entropy'] for var in variable_names]
    entropy_H = [results[var]['entropy_H'] for var in variable_names]
    entropy_h = [results[var]['entropy_h'] for var in variable_names]
    cond_entropy_h_given_H = [results[var]['conditional_entropy_h_given_H'] for var in variable_names]
    cond_entropy_H_given_h = [results[var]['conditional_entropy_H_given_h'] for var in variable_names]
    sample_counts = [results[var]['sample_count'] for var in variable_names]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    main_title = f'Information-Theoretic Metrics - {title_suffix}\nTotal Samples Processed: {processed_samples_count:,}'
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    # Create x positions for bars
    x_pos = np.arange(len(variable_names))
    
    # Plot 1: Mutual Information
    axes[0, 0].bar(x_pos, mutual_info, alpha=0.7, color='blue', edgecolor='navy')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(variable_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Mutual Information I(H;h)')
    axes[0, 0].set_title('Mutual Information by Variable')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Joint Entropy
    axes[0, 1].bar(x_pos, joint_entropy, alpha=0.7, color='green', edgecolor='darkgreen')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(variable_names, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Joint Entropy H(H,h)')
    axes[0, 1].set_title('Joint Entropy by Variable')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Individual Entropies
    width = 0.35
    axes[0, 2].bar(x_pos - width/2, entropy_H, width, alpha=0.7, color='red', edgecolor='darkred', label='H(H)')
    axes[0, 2].bar(x_pos + width/2, entropy_h, width, alpha=0.7, color='purple', edgecolor='darkviolet', label='H(h)')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(variable_names, rotation=45, ha='right')
    axes[0, 2].set_ylabel('Entropy')
    axes[0, 2].set_title('Individual Entropies by Variable')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Conditional Entropies
    axes[1, 0].bar(x_pos - width/2, cond_entropy_h_given_H, width, alpha=0.7, color='orange', edgecolor='darkorange', label='H(h|H)')
    axes[1, 0].bar(x_pos + width/2, cond_entropy_H_given_h, width, alpha=0.7, color='brown', edgecolor='darkred', label='H(H|h)')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(variable_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Conditional Entropy')
    axes[1, 0].set_title('Conditional Entropies by Variable')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Sample Counts
    axes[1, 1].bar(x_pos, sample_counts, alpha=0.7, color='skyblue', edgecolor='navy')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(variable_names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Sample Count')
    axes[1, 1].set_title('Sample Counts by Variable')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Information Decomposition
    axes[1, 2].bar(x_pos - width/2, mutual_info, width, alpha=0.7, color='blue', edgecolor='navy', label='I(H;h)')
    axes[1, 2].bar(x_pos + width/2, cond_entropy_h_given_H, width, alpha=0.7, color='orange', edgecolor='darkorange', label='H(h|H)')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(variable_names, rotation=45, ha='right')
    axes[1, 2].set_ylabel('Information (bits)')
    axes[1, 2].set_title('Information Decomposition: H(h) = I(H;h) + H(h|H)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file_path = os.path.join(output_dir, f'mutual_information_plots_{suffix}.png')
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
    if master_process:
        logger.info(f"Comprehensive plots saved to: {plot_file_path}")
    
    # Create a separate detailed mutual information plot
    plt.figure(figsize=(12, 8))
    plt.bar(x_pos, mutual_info, alpha=0.7, color='blue', edgecolor='navy')
    plt.xticks(x_pos, variable_names, rotation=45, ha='right')
    plt.xlabel('Variables', fontsize=14)
    plt.ylabel('Mutual Information I(H;h) [bits]', fontsize=14)
    plt.title(f'Mutual Information - {title_suffix}\nTotal Samples: {processed_samples_count:,}', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations on bars
    for i, (var, mi) in enumerate(zip(variable_names, mutual_info)):
        plt.annotate(f'{mi:.3f}', (i, mi), textcoords="offset points", xytext=(0,5), ha='center', fontsize=10)
    
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

variable_names_to_track = ["correct1_t1", "correct1_t2", "correct1_t3", "correct1_t4", "correct2_t1", "correct2_t2", "correct2_t3", "correct2_t4", "decoy1_t1", "decoy1_t2", "decoy1_t3", "decoy1_t4", "model_out_t1", "model_out_t2", "model_out_t3", "model_out_t4", "model_out_when_correct_t1", "model_out_when_correct_t2", "model_out_when_correct_t3", "model_out_when_correct_t4",\
                    "model_out_when_incorrect_t1", "model_out_when_incorrect_t2", "model_out_when_incorrect_t3", "model_out_when_incorrect_t4", "same_correct_when_correct_t1", "same_correct_when_correct_t2", "same_correct_when_correct_t3", "same_correct_when_correct_t4", "other_correct_when_correct_t1", "other_correct_when_correct_t2", "other_correct_when_correct_t3", "other_correct_when_correct_t4", "decoy_when_correct_t1", "decoy_when_correct_t2", "decoy_when_correct_t3", "decoy_when_correct_t4", \
                    "correct1_when_incorrect_t1", "correct1_when_incorrect_t2", "correct1_when_incorrect_t3", "correct1_when_incorrect_t4", "correct2_when_incorrect_t1", "correct2_when_incorrect_t2", "correct2_when_incorrect_t3", "correct2_when_incorrect_t4", "decoy_when_incorrect_t1", "decoy_when_incorrect_t2", "decoy_when_incorrect_t3", "decoy_when_incorrect_t4"]

# Only master process maintains StreamingInfo objects
if master_process:
    # Create samples directory for file storage
    samples_dir = os.path.join(output_dir, 'samples') if args.store_samples_to_file else None
    if samples_dir:
        os.makedirs(samples_dir, exist_ok=True)
        logger.info(f"Sample storage directory created: {samples_dir}")
    
    variables = {name:["H", "h"] for name in variable_names_to_track}
    combos_to_track_dict = {name:[("H",), ("h",), ("H", "h")] for name in variable_names_to_track}
    smi_dict = {}
    
    for name in variable_names_to_track:
        sample_file_path = os.path.join(samples_dir, f'samples_{name}.h5') if args.store_samples_to_file else None
        smi_dict[name] = StreamingInfo(
            variables=variables[name], 
            combos_to_track=combos_to_track_dict[name], 
            base=2, 
            store_samples=False,  # Keep memory storage disabled for efficiency
            store_samples_to_file=args.store_samples_to_file,
            sample_file_path=sample_file_path,
            compression='gzip',
            chunk_size=10000
        )
else:
    smi_dict = {}


# %%

# Sample collection dictionaries for each process
local_sample_buffer = defaultdict(list)  # variable name -> list of samples
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
            
    batch_size = prompt_ids.size(0)
    prompt_length = prompt_ids.size(1)  

    with ctx and torch.no_grad():

        # Generate paths from prompts
        generated_sequences ,generated_masks, all_hidden_states, gen_tokens_hidden_states, generated_mask_tokens = path_finding_generate_with_hidden_states(Lmodel, prompt_ids, max_new_tokens, attention_mask=prompt_attention_mask, eos_token=eos_token_id, temperature=args.temperature)
        if generated_sequences.shape[1] - 2 - prompt_length < max_path_len:
            logger.info(f"Batch {batch_idx}: Generated sequence length {generated_sequences.shape[1] - 2 - prompt_length} is less than max_path_len {max_path_len}, skipping this batch")
            continue

        prefix_H = all_hidden_states[:,:-2,:prompt_length,:]


        if skip_every_timestep > 1:
            prompt_ids_skipped, prompt_attention_mask_skipped, prefix_H = skip_time_steps(prompt_ids, prompt_attention_mask, prefix_H, max_nodes)
        else:
            prompt_attention_mask_skipped = prompt_attention_mask
            prompt_ids_skipped = prompt_ids
        if skip_every_layer > 1:
            prefix_H = prefix_H[:, ::skip_every_layer, :, :]
            



        prefix_H = prefix_H / (torch.norm(prefix_H, dim=-1, keepdim=True) + 1e-8)
        z_e = encoder_model(prefix_H, prompt_attention_mask_skipped)
        distances = torch.sum(z_e**2, dim=1, keepdim=True) + torch.sum(vqvae61_codebook_weights**2, dim=1) - 2 * torch.matmul(z_e, vqvae61_codebook_weights.t())
        prefix_H_encodings = torch.argmin(distances, dim=1)

        model_out_path = generated_sequences[:,prompt_length+1:prompt_length+1+max_path_len]
        model_out_path_t1 = model_out_path[:,0]
        model_out_path_t2 = model_out_path[:,1]
        model_out_path_t3 = model_out_path[:,2]
        model_out_path_t4 = model_out_path[:,3]
        model_out_encodings_t1 = model_out_path_t1
        model_out_encodings_t2 = model_out_path_t2
        model_out_encodings_t3 = model_out_path_t3
        model_out_encodings_t4 = model_out_path_t4

        correct1_path = torch.tensor([temp[0] for temp in batch["correct_path_ids"]])[:,1:-1].to(device)
        correct1_path_t1 = correct1_path[:,0]
        correct1_path_t2 = correct1_path[:,1]
        correct1_path_t3 = correct1_path[:,2]
        correct1_path_t4 = correct1_path[:,3]
        correct1_path_encodings_t1 = correct1_path_t1
        correct1_path_encodings_t2 = correct1_path_t2
        correct1_path_encodings_t3 = correct1_path_t3
        correct1_path_encodings_t4 = correct1_path_t4
            
        correct2_path = torch.tensor([temp[1] for temp in batch["correct_path_ids"]])[:,1:-1].to(device)
        correct2_path_t1 = correct2_path[:,0]
        correct2_path_t2 = correct2_path[:,1]
        correct2_path_t3 = correct2_path[:,2]
        correct2_path_t4 = correct2_path[:,3]
        correct2_path_encodings_t1 = correct2_path_t1
        correct2_path_encodings_t2 = correct2_path_t2
        correct2_path_encodings_t3 = correct2_path_t3
        correct2_path_encodings_t4 = correct2_path_t4

        decoy_path = torch.tensor([temp[0] for temp in batch["decoy_path_ids"]])[:,1:-1].to(device)
        decoy_path_t1 = decoy_path[:,0]
        decoy_path_t2 = decoy_path[:,1]
        decoy_path_t3 = decoy_path[:,2]
        decoy_path_t4 = decoy_path[:,3]
        decoy_path_encodings_t1 = decoy_path_t1
        decoy_path_encodings_t2 = decoy_path_t2
        decoy_path_encodings_t3 = decoy_path_t3
        decoy_path_encodings_t4 = decoy_path_t4
        
        # Check which entries match correct1_path or correct2_path
        matches_correct1 = torch.all(model_out_path == correct1_path, dim=1)
        matches_correct2 = torch.all(model_out_path == correct2_path, dim=1)
        corrects = matches_correct1 | matches_correct2
        # For entries that are correct, determine if they match correct1 or correct2
        correct_path_indicators = torch.zeros(batch_size, dtype=torch.long, device=device)  # 0: incorrect, 1: correct1, 2: correct2
        correct_path_indicators[matches_correct1] = 1
        correct_path_indicators[matches_correct2] = 2

        for sample_idx in range(batch_size):   

            name = "correct1_t1" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": correct1_path_encodings_t1[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)

            name = "correct1_t2" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": correct1_path_encodings_t2[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)

            name = "correct1_t3" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": correct1_path_encodings_t3[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)

            name = "correct1_t4" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": correct1_path_encodings_t4[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)

            name = "correct2_t1" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": correct2_path_encodings_t1[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)

            name = "correct2_t2" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": correct2_path_encodings_t2[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)

            name = "correct2_t3" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": correct2_path_encodings_t3[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)

            name = "correct2_t4" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": correct2_path_encodings_t4[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)

            name = "decoy1_t1" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": decoy_path_encodings_t1[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)

            name = "decoy1_t2" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": decoy_path_encodings_t2[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)
            
            name = "decoy1_t3" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": decoy_path_encodings_t3[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)

            name = "decoy1_t4" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": decoy_path_encodings_t4[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)


            name = "model_out_t1" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": model_out_encodings_t1[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)

            name = "model_out_t2" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": model_out_encodings_t2[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)
            
            name = "model_out_t3" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": model_out_encodings_t3[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)

            name = "model_out_t4" 
            sample = {
                "H": prefix_H_encodings[sample_idx].item(), 
                "h": model_out_encodings_t4[sample_idx].item()
            }
            local_sample_buffer[name].append(sample)


            if corrects[sample_idx]:
                name = "model_out_when_correct_t1" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": model_out_encodings_t1[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "model_out_when_correct_t2" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": model_out_encodings_t2[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)
                
                name = "model_out_when_correct_t3" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": model_out_encodings_t3[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "model_out_when_correct_t4" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": model_out_encodings_t4[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "other_correct_when_correct_t1" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct2_path_encodings_t1[sample_idx].item() if correct_path_indicators[sample_idx] == 1 else correct1_path_encodings_t1[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "other_correct_when_correct_t2" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct2_path_encodings_t2[sample_idx].item() if correct_path_indicators[sample_idx] == 1 else correct1_path_encodings_t2[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)
                
                name = "other_correct_when_correct_t3" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct2_path_encodings_t3[sample_idx].item() if correct_path_indicators[sample_idx] == 1 else correct1_path_encodings_t3[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "other_correct_when_correct_t4" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct2_path_encodings_t4[sample_idx].item() if correct_path_indicators[sample_idx] == 1 else correct1_path_encodings_t4[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "same_correct_when_correct_t1" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct1_path_encodings_t1[sample_idx].item() if correct_path_indicators[sample_idx] == 1 else correct2_path_encodings_t1[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "same_correct_when_correct_t2" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct1_path_encodings_t2[sample_idx].item() if correct_path_indicators[sample_idx] == 1 else correct2_path_encodings_t2[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)
                
                name = "same_correct_when_correct_t3" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct1_path_encodings_t3[sample_idx].item() if correct_path_indicators[sample_idx] == 1 else correct2_path_encodings_t3[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "same_correct_when_correct_t4" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct1_path_encodings_t4[sample_idx].item() if correct_path_indicators[sample_idx] == 1 else correct2_path_encodings_t4[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "decoy_when_correct_t1" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": decoy_path_encodings_t1[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "decoy_when_correct_t2" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": decoy_path_encodings_t2[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)
                
                name = "decoy_when_correct_t3" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": decoy_path_encodings_t3[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "decoy_when_correct_t4" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": decoy_path_encodings_t4[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

            else:
                name = "model_out_when_incorrect_t1" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": model_out_encodings_t1[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "model_out_when_incorrect_t2" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": model_out_encodings_t2[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)
                
                name = "model_out_when_incorrect_t3" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": model_out_encodings_t3[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)
                
                name = "model_out_when_incorrect_t4" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": model_out_encodings_t4[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "correct1_when_incorrect_t1" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct1_path_encodings_t1[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "correct1_when_incorrect_t2" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct1_path_encodings_t2[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "correct1_when_incorrect_t3" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct1_path_encodings_t3[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "correct1_when_incorrect_t4" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct1_path_encodings_t4[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "correct2_when_incorrect_t1" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct2_path_encodings_t1[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "correct2_when_incorrect_t2" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct2_path_encodings_t2[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)
                
                name = "correct2_when_incorrect_t3" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct2_path_encodings_t3[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "correct2_when_incorrect_t4" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": correct2_path_encodings_t4[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "decoy_when_incorrect_t1" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": decoy_path_encodings_t1[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "decoy_when_incorrect_t2" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": decoy_path_encodings_t2[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)
                
                name = "decoy_when_incorrect_t3" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": decoy_path_encodings_t3[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

                name = "decoy_when_incorrect_t4" 
                sample = {
                    "H": prefix_H_encodings[sample_idx].item(), 
                    "h": decoy_path_encodings_t4[sample_idx].item()
                }
                local_sample_buffer[name].append(sample)

    # Update processed samples count
    processed_samples_count += batch_size
    
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
            for name in smi_dict:
                smi_dict[name].flush_to_file()
        
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
    for name in smi_dict:
        smi_dict[name].flush_to_file()
        # Get file statistics before closing
        if smi_dict[name].get_file_sample_count() > 0:
            logger.info(f"Variable {name}: {smi_dict[name].get_file_sample_count()} samples saved to file")
    
    logger.info("Closing sample storage files...")
    for name in smi_dict:
        smi_dict[name].close_file()

# %%

# %% ------------- Final Results Calculation ------------------------------

# Final results calculation and reporting (only on master process)
if master_process:
    logger.info("\n" + "="*100)
    logger.info("FINAL RESULTS - PROCESSING COMPLETE")
    logger.info("="*100)
    
    total_samples_processed = sum(smi_dict[var].N for var in smi_dict if smi_dict[var].N > 0)
    expected_samples = processed_samples_count * ddp_world_size if ddp else processed_samples_count
    
    logger.info(f"Total samples processed across all GPUs: {total_samples_processed}")
    logger.info(f"Expected samples based on batch count: {expected_samples}")
    logger.info("="*100)

    results = {}
    for variable_name in variable_names_to_track:
        if variable_name in smi_dict and smi_dict[variable_name].N > 0:  # Only process if we have data
            results[variable_name] = {
                'mutual_information': smi_dict[variable_name].mutual_information("H", "h"),
                'joint_entropy': smi_dict[variable_name].joint_entropy(("H", "h")),
                'entropy_H': smi_dict[variable_name].entropy("H"),
                'entropy_h': smi_dict[variable_name].entropy("h"),
                'conditional_entropy_h_given_H': smi_dict[variable_name].conditional_entropy("h", "H"),
                'conditional_entropy_H_given_h': smi_dict[variable_name].conditional_entropy("H", "h"),
                'sample_count': smi_dict[variable_name].N
            }
            
            logger.info(f'Variable {variable_name}:')
            logger.info(f'  Mutual information I(H;h): {results[variable_name]["mutual_information"]:.6f}')
            logger.info(f'  Joint entropy H(H,h): {results[variable_name]["joint_entropy"]:.6f}')
            logger.info(f'  Entropy H(H): {results[variable_name]["entropy_H"]:.6f}')
            logger.info(f'  Entropy H(h): {results[variable_name]["entropy_h"]:.6f}')
            logger.info(f'  Conditional entropy H(h|H): {results[variable_name]["conditional_entropy_h_given_H"]:.6f}')
            logger.info(f'  Conditional entropy H(H|h): {results[variable_name]["conditional_entropy_H_given_h"]:.6f}')
            logger.info(f'  Sample count: {results[variable_name]["sample_count"]}')
            logger.info('')
else:
    results = {}

# Save comprehensive results to text file (only on master process)
if master_process:
    results_file_path = os.path.join(output_dir, 'mutual_information_results_all_variables.txt')
    with open(results_file_path, 'w') as f:
        f.write("Information Theoretical Calculation Results Across All Variables\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset directory: {dataset_dir}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Variables tracked: {', '.join(variable_names_to_track)}\n")
        f.write(f"Max new tokens: {max_new_tokens}\n")
        f.write(f"Max seq length: {max_seq_length}\n")
        f.write(f"Max nodes: {max_nodes}\n")
        f.write(f"EOS token ID: {eos_token_id}\n")
        f.write(f"World size (GPUs used): {ddp_world_size}\n")
        f.write("\n" + "=" * 70 + "\n\n")
        
        # Write detailed results for each variable
        for variable_name in sorted(results.keys()):
            f.write(f"VARIABLE = {variable_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mutual information I(H;h):     {results[variable_name]['mutual_information']:.6f}\n")
            f.write(f"Joint entropy H(H,h):          {results[variable_name]['joint_entropy']:.6f}\n")
            f.write(f"Entropy H(H):                  {results[variable_name]['entropy_H']:.6f}\n")
            f.write(f"Entropy H(h):                  {results[variable_name]['entropy_h']:.6f}\n")
            f.write(f"Conditional entropy H(h|H):    {results[variable_name]['conditional_entropy_h_given_H']:.6f}\n")
            f.write(f"Conditional entropy H(H|h):    {results[variable_name]['conditional_entropy_H_given_h']:.6f}\n")
            f.write(f"Sample count:                  {results[variable_name]['sample_count']}\n")
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
                'variables_tracked': variable_names_to_track,
                'max_new_tokens': max_new_tokens,
                'max_seq_length': max_seq_length,
                'max_nodes': max_nodes,
                'eos_token_id': eos_token_id,
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