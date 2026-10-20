
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
from path_finding_llm_utilities import PathFindingDataset, custom_collate_fn
from torch.utils.data import DataLoader
from model import GPT, GPTConfig


# %%
# Load training configuration from a JSON file
try:
    parser = argparse.ArgumentParser(description="Train VQ-VAE-like model")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    config_path = args.config
except:
    raise RuntimeError("Failed to parse command line arguments for config file. Please provide --config argument.")
    # config_path = 'PF_15XL/vqvae_path_config.json'

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
# language_model_ckpt_folder_path = config.get('language_model_ckpt_folder_path', None)
# LM_compile = config.get('LM_compile', True)  # use PyTorch 2.0 to compile the model to be faster

# Data for training the model
gradient_accumulation_steps = config.get('gradient_accumulation_steps')  # used to simulate larger batch sizes
batch_size = config.get('batch_size')  # if gradient_accumulation_steps > 1, this is the micro-batch size
# max_new_tokens = config.get('max_new_tokens', 9)  # number of tokens generated in each sample

dataset_dir = config.get('dataset_dir', None)
# For path finding, we use the split from config (e.g., 'vqvae') to determine which JSONL file to use
# split = config.get('split', 'vqvae')
max_seq_length = config.get('max_seq_length')  # Maximum sequence length for path finding
# max_nodes = config.get('max_nodes')  # Maximum number of nodes for path finding
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
vqvae_path_grow_beta = config['vqvae_path_config']['grow_beta']  # whether to grow the beta parameter
vqvae_path_betainit = config['vqvae_path_config']['betainit']  # initial value of beta
vqvae_path_betafinal = config['vqvae_path_config']['betafinal']  # final value of beta
vqvae_path_warmup_iters_beta = config['vqvae_path_config']['warmup_iters_beta'] #let's try this once  # how many steps to warm up beta for IN THE FUTURE WE MAY WANT TO CHANGE THIS TO BE DIFFERENT FROM warmup_iters

#VQ-VAE_path model parameters
vqvae_path_config = config['vqvae_path_config']
vqvae_path_codebook_size = vqvae_path_config['codebook_size']
vqvae_path_codebook_reset_counter_multiplier = vqvae_path_config['codebook_reset_counter_multiplier']
vqvae_path_beta = vqvae_path_config['beta']
vqvae_path_d_model = vqvae_path_config['d_model']
vqvae_path_num_layers = vqvae_path_config['num_layers']
vqvae_path_cosine_push_weight = vqvae_path_config['cosine_push_weight']
vqvae_path_entropy_loss_weight = vqvae_path_config['entropy_loss_weight']
vqvae_path_entropy_temperature = vqvae_path_config['entropy_temperature']
vqvae_path_mask_prob = vqvae_path_config['mask_prob']
vqvae_path_usage_tracking_window = vqvae_path_config['usage_tracking_window']
vqvae_path_config_transformer = vqvae_path_config['config_transformer']
vqvae_path_tied_token_embedding_weight = vqvae_path_config['tied_token_embedding_weight']

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

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, dict, list, tuple)) and (not k.startswith('vqvae_path') or k=='vqvae_path_config') and k!='Out' and k!='first_stage_checkpoint' and k!='config']
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


# %% Pull the dataset related information
json_file_path = os.path.join(dataset_dir, 'config_train.json')
with open(json_file_path, 'r') as f:
    config_train = json.load(f)
max_nodes = config_train['max_nodes']
max_seq_length = config_train['max_path_len']
vocab_size = max_nodes + 6 # To guarantee the coverage of the vocabulary + 6

# %% --- initialize the VQ-VAE model ---

if init_from == 'scratch':
    if master_process:  
        logging.info("Initializing the model from scratch")

    encoder_model = Path_Encoder(
        vocab_size=vocab_size,
        d_model=vqvae_path_d_model,
        T=max_seq_length,
        num_layers=vqvae_path_num_layers,
        config=vqvae_path_config_transformer
    )

    if vqvae_path_tied_token_embedding_weight:
        tied_token_embedding_weight = encoder_model.token_embed.weight
    else:
        tied_token_embedding_weight = None

    decoder_model = Path_Decoder(
        vocab_size=vocab_size,
        d_model=vqvae_path_d_model,
        T=max_seq_length,
        num_layers=vqvae_path_num_layers,
        config=vqvae_path_config_transformer,
        tied_token_embedding_weight=tied_token_embedding_weight
    )
    vqvae_path_model = Path_VQVAE(encoder_model, decoder_model, vqvae_path_config)
    
    vqvae_path_model.to(device)

    iter_num = 0
    iter_num_log = []
    recon_losses = []
    codebook_losses = []
    commitment_losses = []
    total_losses = []
    token_accuracies = []
    sequence_accuracies = []
    position_accuracies = []

elif init_from == 'resume':
    if master_process:
        logging.info("Resuming the model from a checkpoint in {}".format(output_dir))
    ckpt_path = os.path.join(output_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    encoder_model = Path_Encoder(
        vocab_size=vocab_size,
        d_model=vqvae_path_d_model,
        T=max_seq_length,
        num_layers=vqvae_path_num_layers,
        config=vqvae_path_config_transformer
    )

    if vqvae_path_tied_token_embedding_weight:
        tied_token_embedding_weight = encoder_model.token_embed.weight
    else:
        tied_token_embedding_weight = None

    decoder_model = Path_Decoder(
        vocab_size=vocab_size,
        d_model=vqvae_path_d_model,
        T=max_seq_length,
        num_layers=vqvae_path_num_layers,
        config=vqvae_path_config_transformer,
        tied_token_embedding_weight=tied_token_embedding_weight
    )
    vqvae_path_model = Path_VQVAE(encoder_model, decoder_model, vqvae_path_config)
    
    vqvae_path_model.load_state_dict(checkpoint['model_state_dict'])
    vqvae_path_model.to(device)

    iter_num = checkpoint['iter_num']
    iter_num_log = checkpoint['iter_num_log']
    recon_losses = checkpoint['recon_losses']
    codebook_losses = checkpoint['codebook_losses']
    commitment_losses = checkpoint['commitment_losses']
    total_losses = checkpoint['total_losses']
    token_accuracies = checkpoint['token_accuracies']
    sequence_accuracies = checkpoint.get('sequence_accuracies', [0.0] * len(recon_losses))
    position_accuracies = checkpoint.get('position_accuracies', [0.0] * len(recon_losses))
else:
    if master_process:
        logging.info(f"Invalid value for init_from. Must be 'scratch' or 'resume', not {init_from}")
    raise ValueError(f"Invalid value for init_from: {init_from}")

vqvae_path_model.train()

 # %% Setting up the training 
optimizer = configure_optimizers(vqvae_path_model, weight_decay, learning_rate, (beta1, beta2), device_type)
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
    vqvae_path_model.compile()  # requires PyTorch 2.0

# wrap model in DDP if needed
if ddp:
    vqvae_path_model = DDP(vqvae_path_model, device_ids=[ddp_local_rank])

# %% ------------- Training loop ------------------------------

while True:
    t0 = time.time()
    lr = get_lr(iter_num, warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters, learning_rate=learning_rate, min_lr=min_lr) if decay_lr else learning_rate
    beta = get_beta(iter_num, warmup_iters=vqvae_path_warmup_iters_beta, beta_start=vqvae_path_betainit, beta_max=vqvae_path_betafinal) if vqvae_path_grow_beta else vqvae_path_beta

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Reset accumulators for the iteration
    total_loss_accum = 0.0
    recon_loss_accum = 0.0
    codebook_loss_accum = 0.0
    commitment_loss_accum = 0.0
    cosine_push_loss_accum = 0.0
    entropy_loss_accum = 0.0
    token_accuracy_accum = 0.0
    sequence_accuracy_accum = 0.0
    position_accuracy_accum = 0.0
    unique_count_accum = 0.0

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):

        # generate data here, B by T random node ids with replacement
        X = torch.randint(0, vocab_size, (batch_size, max_seq_length), device=device)

        if ddp:
            vqvae_path_model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            out_logits, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count, cosine_push_loss, entropy_loss = vqvae_path_model(X)
            # out_logits is B, T, vocab_size
        total_loss /= gradient_accumulation_steps

        with torch.no_grad():
            # For accuracy calculation, we compare predicted tokens with original input tokens
            # out_logits is (B, T, vocab_size), X is (B, T)
            predicted_tokens = torch.argmax(out_logits, dim=-1)  # (B, T)
            correct_predictions = (predicted_tokens == X).float()  # (B, T)
            
            # Calculate token-level accuracy
            token_accuracy = torch.mean(correct_predictions)
            token_accuracy_accum += token_accuracy / gradient_accumulation_steps
            
            # Calculate sequence-level accuracy (all tokens in sequence must be correct)
            sequence_accuracy = torch.mean(torch.all(correct_predictions == 1, dim=-1).float())
            sequence_accuracy_accum += sequence_accuracy / gradient_accumulation_steps
            
            # Calculate position-wise accuracy (accuracy at each position)
            position_accuracy = torch.mean(correct_predictions, dim=0)  # (T,)
            position_accuracy_accum += position_accuracy / gradient_accumulation_steps

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
        grad_norm_value = torch.nn.utils.clip_grad_norm_(vqvae_path_model.parameters(), grad_clip)
    else:
        grad_norm_list = [p.grad.data.norm(2) for p in vqvae_path_model.parameters() if p.grad is not None]
        grad_norm_value = torch.norm(torch.stack(grad_norm_list)) if grad_norm_list else torch.tensor(0.0, device=device)

    param_norm_list = [p.data.norm(2) for p in vqvae_path_model.parameters() if p is not None]
    total_param_norm = torch.norm(torch.stack(param_norm_list)) if param_norm_list else torch.tensor(0.0, device=device)
    grad_norm_ratio = grad_norm_value / total_param_norm if total_param_norm.item() > 0 else torch.tensor(0.0, device=device)
    
    optimizer.step()
    optimizer.zero_grad()
    
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
            token_accuracy_accum_tensor = torch.tensor(token_accuracy_accum, device=device)
            sequence_accuracy_accum_tensor = torch.tensor(sequence_accuracy_accum, device=device)
            position_accuracy_accum_tensor = torch.tensor(position_accuracy_accum, device=device)
            # All-reduce these tensors over all processes
            torch.distributed.all_reduce(recon_loss_accum_tensor)
            torch.distributed.all_reduce(codebook_loss_accum_tensor)
            torch.distributed.all_reduce(commitment_loss_accum_tensor)
            torch.distributed.all_reduce(cosine_push_loss_accum_tensor)
            torch.distributed.all_reduce(entropy_loss_accum_tensor)
            torch.distributed.all_reduce(total_loss_accum_tensor)
            torch.distributed.all_reduce(token_accuracy_accum)
            torch.distributed.all_reduce(sequence_accuracy_accum_tensor)
            torch.distributed.all_reduce(position_accuracy_accum_tensor)
            world_size = torch.distributed.get_world_size()
            recon_loss_accum = recon_loss_accum_tensor.item() / world_size
            codebook_loss_accum = codebook_loss_accum_tensor.item() / world_size
            commitment_loss_accum = commitment_loss_accum_tensor.item() / world_size
            cosine_push_loss_accum = cosine_push_loss_accum_tensor.item() / world_size
            entropy_loss_accum = entropy_loss_accum_tensor.item() / world_size
            total_loss_accum = total_loss_accum_tensor.item() / world_size
            token_accuracy_accum = token_accuracy_accum_tensor.item() / world_size
            sequence_accuracy_accum = sequence_accuracy_accum_tensor.item() / world_size
            position_accuracy_accum = position_accuracy_accum_tensor / world_size
        if master_process:
            logging.info(
                f'Iter: {iter_num}, Loss: {total_loss_accum:.4f}, '
                f'Token Accuracy: {token_accuracy_accum:.4f}, Sequence Acc: {sequence_accuracy_accum:.4f}, Position Acc: {position_accuracy_accum.cpu().numpy()}, '
                f'Recon Loss: {recon_loss_accum:.4f}, Codebook Loss: {codebook_loss_accum:.4f}, '
                f'Commitment Loss: {commitment_loss_accum:.4f}, Cosine Push Loss: {cosine_push_loss_accum:.4f}, '
                f'Entropy Loss: {entropy_loss_accum:.4f}, '
                f'Unique Vectors: {unique_count_accum:.2f}, '
                f'Grad Norm: {grad_norm_value.item():.4f}, Grad Norm/Param Norm: {grad_norm_ratio.item():.4f}, '
                f'Time: {dt:.2f}s'
            )
            if wandb_flag:
                log_data = {
                    'vqvae_path/iter_num': iter_num,
                    'vqvae_path/total_loss': total_loss_accum,
                    'vqvae_path/recon_loss': recon_loss_accum,
                    'vqvae_path/codebook_loss': codebook_loss_accum,
                    'vqvae_path/commitment_loss': commitment_loss_accum,
                    'vqvae_path/cosine_push_loss': cosine_push_loss_accum,
                    'vqvae_path/entropy_loss': entropy_loss_accum,
                    'vqvae_path/token_accuracy': token_accuracy_accum,
                    'vqvae_path/sequence_accuracy': sequence_accuracy_accum,
                    'vqvae_path/position_accuracy': position_accuracy_accum,
                    'vqvae_path/unique_vectors': unique_count_accum,
                    'vqvae_path/time': dt,
                    'vqvae_path/lr': lr,
                    'vqvae_path/grad_norm': grad_norm_value.item(),
                    'vqvae_path/grad_norm_ratio': grad_norm_ratio.item()
                }
                
                for t in range(position_accuracy_accum.shape[0]):
                    log_data[f'vqvae_path_experimental/position_accuracy_time_{t}'] = position_accuracy_accum[t].item()
                
                if torch.cuda.is_available():
                    log_data.update({
                        'vqvae_path/gpu_memory_allocated_GB': torch.cuda.memory_allocated(device) / 1024**3,
                        'vqvae_path/gpu_memory_reserved_GB': torch.cuda.memory_reserved(device) / 1024**3
                    })
                wandb.log(log_data)
                
                if iter_num % 100 == 0 and master_process:
                    plt.figure(figsize=(8, 5))
                    
                    # Plot Position Accuracy per time step
                    plt.plot(position_accuracy_accum.cpu().numpy())
                    plt.title('Position Accuracy per Time Step')
                    plt.xlabel('Time Step')
                    plt.ylabel('Position Accuracy')
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'vqvae_path_position_accuracy_analysis.png'))
                    plt.close()

                    # Log and save cosine similarities of path encoder embedding vector 
                    with torch.no_grad():
                        # Get model reference (handle DDP)
                        model_ref = vqvae_path_model.module if hasattr(vqvae_path_model, 'module') else vqvae_path_model
                        
                        # Get encoder embedding weights
                        encoder_embeddings = model_ref.encoder.token_embed.weight.data  # (vocab_size, d_model)
                        
                        # Compute cosine similarities between embedding vectors
                        encoder_embeddings_normalized = F.normalize(encoder_embeddings, p=2, dim=1)
                        encoder_similarity_matrix = torch.mm(encoder_embeddings_normalized, encoder_embeddings_normalized.t())
                        
                    # Plot encoder embedding similarity heatmap
                    plt.figure(figsize=(12, 10))
                    plt.imshow(encoder_similarity_matrix.cpu().numpy(), cmap='viridis', vmin=-1, vmax=1)
                    plt.colorbar(label='Cosine Similarity')
                    plt.title(f'Path_VQVAE Encoder Token Embedding Cosine Similarities\n'
                            f'Vocab Size: {model_ref.encoder.vocab_size}, Embedding Dim: {model_ref.encoder.d_model}')
                    plt.xlabel('Token Index')
                    plt.ylabel('Token Index')
                    plt.savefig(os.path.join(output_dir, 'vqvae_path_encoder_embedding_similarities.png'), dpi=150, bbox_inches='tight')
                    plt.close()


        total_losses.append(total_loss_accum)
        recon_losses.append(recon_loss_accum)
        codebook_losses.append(codebook_loss_accum)
        commitment_losses.append(commitment_loss_accum)
        token_accuracies.append(token_accuracy_accum)
        sequence_accuracies.append(sequence_accuracy_accum)
        position_accuracies.append(position_accuracy_accum)

        # Enhanced codebook analysis every 1000 iterations using internal metrics
        if iter_num % 250 == 0:
            if master_process:
                logging.info("Getting usage statistics from internal model metrics...")

                # Get model reference (handle DDP)
                model_ref = vqvae_path_model.module if hasattr(vqvae_path_model, 'module') else vqvae_path_model
                
                # Get usage statistics from the model
                usage_stats = model_ref.get_usage_statistics()
                similarity_stats = model_ref.compute_codebook_similarities()
                
                # Log usage statistics
                logging.info(f"Path_VQVAE codebook analysis: Used {usage_stats['unique_vectors_used']} out of {model_ref.codebook_size} vectors, "
                           f"total vectors processed: {usage_stats['total_vectors_processed']}")
                
                # Plot usage histogram
                if usage_stats['unique_vectors_used'] > 0:
                    usage_counts = usage_stats['usage_counts'].cpu().numpy()
                    
                    plt.figure(figsize=(10,5))
                    plt.bar(range(len(usage_counts)), usage_counts)
                    plt.title(f'Path_VQVAE Codebook Vector Usage (Internal Tracking)\n'
                             f'Used {usage_stats["unique_vectors_used"]} out of {model_ref.codebook_size} vectors, '
                             f'Total samples: {usage_stats["total_vectors_processed"]}')
                    plt.xlabel('Codebook Vector Index')
                    plt.ylabel('Usage Count')
                    plt.savefig(os.path.join(output_dir, 'vqvae_path_codebook_usage_internal.png'))
                    plt.close()
                    
                    # Plot similarity heatmap if we have similarities
                    if similarity_stats['similarities'] is not None:
                        similarities = similarity_stats['similarities'].cpu().numpy()
                        used_indices = similarity_stats['used_indices'].cpu().numpy()
                        
                        plt.figure(figsize=(10,10))
                        plt.imshow(similarities, cmap='viridis')
                        plt.colorbar()
                        plt.title(f'Cosine Similarities Between Used Path_VQVAE Codebook Vectors (Internal Tracking)\n'
                                 f'Used {similarity_stats["num_used_vectors"]} out of {model_ref.codebook_size} vectors')
                        plt.xlabel('Used Codebook Vector Index')
                        plt.ylabel('Used Codebook Vector Index')
                        plt.savefig(os.path.join(output_dir, 'vqvae_path_codebook_similarities_internal.png'))
                        plt.close()
                    else:
                        logging.info("Not enough unique vectors for similarity analysis.")
                else:
                    logging.info("No vectors used yet, skipping usage plots.")

    if save_interval > 0 and iter_num % save_interval == 0 and master_process:
        tempTime = time.time()
        checkpoint = {
            'model_state_dict': vqvae_path_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter_num': iter_num,
            'iter_num_log': iter_num_log,
            'config': config,
            'recon_losses': recon_losses,
            'codebook_losses': codebook_losses,
            'commitment_losses': commitment_losses,
            'token_accuracies': token_accuracies,
            'sequence_accuracies': sequence_accuracies,
            'position_accuracies': position_accuracies,
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
        'model_state_dict': vqvae_path_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter_num': iter_num,
        'iter_num_log': iter_num_log,
        'config': config,
        'recon_losses': recon_losses,
        'codebook_losses': codebook_losses,
        'commitment_losses': commitment_losses,
        'token_accuracies': token_accuracies,
        'sequence_accuracies': sequence_accuracies,
        'position_accuracies': position_accuracies,
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
