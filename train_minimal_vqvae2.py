
# %%

# %%

# %%
print('train_minimal_vqvae2.py')
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

# %%
# Load training configuration from a JSON file
try:
    parser = argparse.ArgumentParser(description="Train VQ-VAE-like Stage 2 model")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()

    config_path = args.config
except:
    # raise ValueError("No config file provided")
    config_path = './exp_cfg_s14448_rd3456_rl234_4000k/vqvae2_config.json'
# %%
# %%
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
    print('Loaded config from', config_path)

# %%

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
gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)  # used to simulate larger batch sizes
batch_size = config.get('batch_size', 3)  # if gradient_accumulation_steps > 1, this is the micro-batch size
# max_new_tokens = config.get('max_new_tokens', 9)  # number of tokens generated in each sample

dataset_dir = config.get('dataset_dir', None)
min_max_prefix_len = config.get('min_max_prefix_len', None) # 2, 7
split = config.get('split', 'train')
                   
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

#VQ-VAE2 model parameters
vqvae2_config = config.get('vqvae2_config', {})
vqvae2_codebook_size = vqvae2_config.get('codebook_size', None)
vqvae2_codebook_reset_counter_multiplier = vqvae2_config.get('codebook_reset_counter_multiplier', 10) # 0 for no reset
vqvae2_beta = vqvae2_config.get('beta', 0.25)
vqvae2_D = vqvae2_config.get('D', 512)
vqvae2_T_star = vqvae2_config.get('T_star', 20)

# Commitment loss beta growth settings
vqvae2_grow_beta = vqvae2_config.get('vqvae2_grow_beta', True)  # whether to grow the beta parameter
vqvae2_betainit = vqvae2_config.get('vqvae2_betainit', 0.1)  # initial value of beta
vqvae2_betafinal = vqvae2_config.get('vqvae2_betafinal', 0.4)  # final value of beta
vqvae2_warmup_iters_beta = warmup_iters  # how many steps to warm up beta for (may change in future)

# Encoder configuration
vqvae2_encoder_config = vqvae2_config.get('encoder_config', {})
vqvae2_encoder_num_layers = vqvae2_config.get('encoder_num_of_transf_layers', 12)

vqvae2_cosine_push_weight = vqvae2_config['cosine_push_weight']
vqvae2_entropy_loss_weight = vqvae2_config['entropy_loss_weight']
vqvae2_entropy_temperature = vqvae2_config['entropy_temperature']
vqvae2_mask_prob = vqvae2_config['mask_prob']

# Decoder configuration
vqvae2_decoder_config = vqvae2_config.get('decoder_config', {})
vqvae2_decoder_num_layers = vqvae2_config.get('decoder_num_of_transf_layers', 12)
vqvae2_config = {k.replace('vqvae2_',''): globals()[k] for k in [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, dict)) and k.startswith('vqvae2') and k != 'vqvae2_config']}

# Experimental features are now read from config file

# VQ-model compile
vqvae2_compile = config.get('vqvae2_compile', True)  # use PyTorch 2.0 to compile the model to be faster

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
# First stage experiment path/directory
first_stage_ckpt_path = config.get('first_stage_ckpt_path', None)
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
config['first_stage_ckpt_path'] = first_stage_ckpt_path
first_stage_model_compile = config.get('first_stage_model_compile', True)  # use PyTorch 2.0 to compile the model to be faster
# Load first stage checkpoint
first_stage_checkpoint = torch.load(first_stage_ckpt_path, map_location=device)

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, dict, list, tuple)) and (not k.startswith('vqvae2') or k=='vqvae2_config') and k!='Out' and k!='first_stage_checkpoint' and k!='config']
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
            # Resume may not work properly. TODO: check and fix if needed.
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
            wandb.init(project=wandb_project_name, name=wandb_run_name, group=wandb_group, entity=wandb_entity)
            if wandb_flag:
                config['wandb_run_id'] = wandb.run.id
                config['wandb_project_name'] = wandb_project_name
                config['wandb_run_name'] = wandb_run_name
            wandb.config.update(config) # log all config parameters to wandb



config_save_path = os.path.join(output_dir, 'exp_config.json')

# wait for the directory to be created before writing the config file, sometimes master process is slower

while not os.path.exists(output_dir):
    if master_process: os.makedirs(output_dir, exist_ok=True)
    time.sleep(0.0001)
if master_process:
    with open(config_save_path, 'w') as config_save_file:
        json.dump(config, config_save_file, indent=4)
        print(f'Saved config to {config_save_path}')


# %% ------------- Load the language model ------------------------------
Lmodel = nanoLLM(model_name = language_model_ckpt_folder_path, base_dir = None)
Lmodel.eval()
Lmodel.to(device)
if LM_compile:
    if master_process:
        logging.info("compiling the language model...")
    Lmodel.compile()  # requires PyTorch 2.0 

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

# Load normalization values from first stage VQVAE1 model
first_stage_normalization_values = None
for k, v in first_stage_checkpoint['model_state_dict'].items():
    if 'normalization_values' in k:
        first_stage_normalization_values = v.to(device)
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

print('First stage encoder:')
checkModelLoadCorrect(first_stage_encoder, encoder_param_dict)
print('First stage codebook:')
checkModelLoadCorrect(first_stage_codebook, codebook_param_dict)

first_stage_encoder.eval()  # eval() only affects behavior during forward pass (e.g. dropout, batchnorm)
for param in first_stage_encoder.parameters():
    param.requires_grad = False # requires_grad=False is still needed to prevent gradient computation
first_stage_codebook.eval()
for param in first_stage_codebook.parameters():
    param.requires_grad = False
if first_stage_model_compile:
    if master_process:
        logging.info("compiling the first stage encoder...")
    first_stage_encoder.compile()  # requires PyTorch 2.

first_stage_codebook_weights = first_stage_codebook.weight


# %% --- initialize the VQ-VAE 2 model ---

# Create the encoder first.
encoder_model = Encoder2(
    d2 = first_stage_checkpoint['config']['vqvae1_config']['d2'], 
    min_seq_len = min_max_prefix_len[0],
    D = vqvae2_D,
    num_layers = vqvae2_encoder_num_layers,
    config = vqvae2_encoder_config
)

# Create the decoder
decoder_model = Decoder2(
    d2=first_stage_checkpoint['config']['vqvae1_config']['d2'],
    max_prefix_len = min_max_prefix_len[1],
    D = vqvae2_D,
    num_layers = vqvae2_decoder_num_layers,
    config = vqvae2_decoder_config
)
vqvae2_model = VQVAE2(encoder_model, decoder_model, vqvae2_config)

if init_from == 'scratch':
    if master_process:  
        logging.info("Initializing the model from scratch")

    iter_num = 0
    iter_num_log = []
    recon_losses = []
    codebook_losses = []
    commitment_losses = []
    total_losses = []
    nrmses = []
    nrmses_per_vector = []
    nrmses_per_element = []
    overall_nrmses = []
    overall_nrmses_per_vector = []
    overall_nrmses_per_element = []
    # NEW for codebook collape - Add tracking for new regularizers
    cosine_push_losses = []
    entropy_losses = []
elif init_from == 'resume':
    # TODO: check if this works.
    if master_process:
        logging.info("Resuming the model from a checkpoint in {}".format(output_dir))
    ckpt_path = os.path.join(output_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    vqvae2_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    iter_num = checkpoint['iter_num']
    iter_num_log = checkpoint['iter_num_log']
    recon_losses = checkpoint['recon_losses']
    codebook_losses = checkpoint['codebook_losses']
    commitment_losses = checkpoint['commitment_losses']
    total_losses = checkpoint['total_losses']
    nrmses = checkpoint['nrmses']
    nrmses_per_vector = checkpoint.get('nrmses_per_vector', [0.0] * len(recon_losses))
    nrmses_per_element = checkpoint.get('nrmses_per_element', [0.0] * len(recon_losses))
    overall_nrmses = checkpoint.get('overall_nrmses', [0.0] * len(recon_losses))
    overall_nrmses_per_vector = checkpoint.get('overall_nrmses_per_vector', [0.0] * len(recon_losses))
    overall_nrmses_per_element = checkpoint.get('overall_nrmses_per_element', [0.0] * len(recon_losses))
    # NEW for codebook collape - Handle backwards compatibility for new loss types
    cosine_push_losses = checkpoint.get('cosine_push_losses', [0.0] * len(recon_losses))
    entropy_losses = checkpoint.get('entropy_losses', [0.0] * len(recon_losses))
else:
    if master_process:
        logging.info(f"Invalid value for init_from. Must be 'scratch' or 'resume', not {init_from}")

vqvae2_model.to(device)
vqvae2_model.train()
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

logging.info(f"Encoder trainable params: {count_params(encoder_model)}")
logging.info(f"Decoder trainable params: {count_params(decoder_model)}")
logging.info(f"Total trainable params: {count_params(vqvae2_model)}")

 # %% Setting up the training 
optimizer = configure_optimizers(vqvae2_model, weight_decay, learning_rate, (beta1, beta2), device_type)
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
if vqvae2_compile:
    if master_process:
        logging.info("compiling the model...")
    vqvae2_model.compile()  # requires PyTorch 2.0

# wrap model in DDP if needed
if ddp:
    vqvae2_model = DDP(vqvae2_model, device_ids=[ddp_local_rank])

# %% ------------- Training loop ------------------------------
split = 'train'

prefix_meta_path = os.path.join(dataset_dir, f'prefix_meta_{min_max_prefix_len[0]}_{min_max_prefix_len[1]}.pkl')
with open(prefix_meta_path, 'rb') as f:
    prefix_meta = pickle.load(f)
    pad_token_id = prefix_meta['prefix_padding']

# Create the dataloader for CFG data
train_loader = create_cfg_dataloader(
    data_path=os.path.join(dataset_dir, f'{split}_prefixes{min_max_prefix_len[0]}_{min_max_prefix_len[1]}.bin'),
    batch_size=batch_size,
    seq_length=min_max_prefix_len[1],
    pad_token_id=pad_token_id,
    shuffle=True,
    num_workers=4 if device_type == 'cuda' else 0
)



# Initialize hidden states with first batch
sequences, attention_masks = next(iter(train_loader))
sequences = sequences.to(device)
# TODO: turn off/on this after debugging, for now, it is a quick fix
# sequences[sequences == pad_token_id] = 0
attention_masks = attention_masks.to(device) # (B, T)
with ctx:
    hidden_states_tensor = Lmodel.generate_prefix_hidden_states(sequences, attention_masks)


t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process

# %%
while True:
    lr = get_lr(iter_num, warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters, learning_rate=learning_rate, min_lr=min_lr) if decay_lr else learning_rate
    beta = get_beta(iter_num, warmup_iters=vqvae2_warmup_iters_beta, beta_start=vqvae2_betainit, beta_max=vqvae2_betafinal) if vqvae2_grow_beta else vqvae2_beta

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Reset accumulators for the iteration
    total_loss_accum = 0.0
    recon_loss_accum = 0.0
    codebook_loss_accum = 0.0
    commitment_loss_accum = 0.0
    nrmse_accum = 0.0
    nrmse_per_vector_accum = 0.0
    nrmse_per_element_accum = 0.0
    overall_nrmse_accum = 0.0
    overall_nrmse_per_vector_accum = 0.0
    overall_nrmse_per_element_accum = 0.0
    # NEW for codebook collape - Add accumulators for new regularizers
    cosine_push_loss_accum = 0.0
    entropy_loss_accum = 0.0
    unique_count_accum = 0.0
    if wandb_flag:
        nrmse_per_time_accum = 0.0
        nrmse_per_layer_accum = 0.0

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        H = hidden_states_tensor[:, :-2, :, :].to(dtype = ptdtype) # (B, L, T, d)

        H = H / (torch.norm(H, dim=-1, keepdim=True) + 1e-8)
        with ctx:
            with torch.no_grad():
                # Normalize H before passing to first stage encoder
                H_normalized = normalize_first_stage(H)
                out_first_stage = first_stage_encoder(H_normalized, attention_masks) 
                out_first_stage_right_padded, mask_right_padded = convert_left_to_right_padding(out_first_stage, attention_masks)
                # tokenize the output of the first stage
                B, T, d2 = out_first_stage_right_padded.shape
        # print( (out_first_stage_right_padded.norm(dim=-1)*mask_right_padded).sum()/ (mask_right_padded.sum()))
        if ddp:
            vqvae2_model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            # NEW for codebook collape - Get losses from the model (now includes cosine-push and entropy losses)
            # output, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count = vqvae2_model(out_first_stage_right_padded, mask_right_padded, beta=beta)
            output, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count, cosine_push_loss, entropy_loss = vqvae2_model(out_first_stage_right_padded, mask_right_padded, beta=beta)
        total_loss /= gradient_accumulation_steps

        output = output


        # Calculate the overall NRMSE
        with ctx:
            with torch.no_grad():
                output_left_padded, _ = convert_right_to_left_padding(output, mask_right_padded) 
                B, T, d2 = output_left_padded.shape
                output_for_overall_nrmse_flat = output_left_padded.view(-1, d2)
                distances = torch.sum(output_for_overall_nrmse_flat**2, dim=1, keepdim=True) + torch.sum(first_stage_codebook_weights**2, dim=1) - 2 * torch.matmul(output_for_overall_nrmse_flat, first_stage_codebook_weights.t())
                encoding_indices = torch.argmin(distances, dim=1)  # shape: (B*T,)
                z_q_flat = first_stage_codebook.weight[encoding_indices]  # shape: (B*T, d2)
                z_q = z_q_flat.view(B, T, d2)  # shape: (B, T, d2)
                x_overall_recon = first_stage_decoder(z_q, padding_mask=attention_masks)
                # Denormalize the reconstruction before comparing with original H
                x_overall_recon_denormalized = denormalize_first_stage(x_overall_recon) # (B, L, T, d)

                # Store variables for experimental NRMSE calculation (outside gradient accumulation loop)
                if wandb_flag:
                    H_final = H.clone()
                    x_overall_recon_denormalized_final = x_overall_recon_denormalized.clone()
                    attention_masks_final = attention_masks.clone()
            
                    # Calculate experimental layer-wise and time-wise NRMSE for wandb logging
                    diff_norm = torch.norm(H_final - x_overall_recon_denormalized_final, dim=-1) # (B, L, T)
                    H_norm = torch.norm(H_final, dim=-1) # (B, L, T)
                    vector_nrmse = diff_norm / (H_norm + 1e-8) # (B, L, T)
                    
                    # make them right padded
                    vector_nrmse_right_padded, attention_masks_right_padded = convert_left_to_right_padding(vector_nrmse.permute(0,2,1), attention_masks_final)
                    vector_nrmse_right_padded = vector_nrmse_right_padded.permute(0,2,1)

                    # Expand attention mask to match shape: (B, L, seq_len)
                    mask_exp_right_padded = attention_masks_right_padded.unsqueeze(1).expand(-1, H_final.shape[1], -1).to(H_final.dtype)
                    
                    # Apply mask
                    masked_vector_nrmse_right_padded = vector_nrmse_right_padded * mask_exp_right_padded  # (B, L, seq_len)
                    
                    # Per-time-index NRMSE: mean across batch and layer dimensions
                    # Result shape: (seq_len,)
                    valid_count_per_time = mask_exp_right_padded.sum(dim=(0, 1))  # (seq_len,) - count valid entries per time index
                    nrmse_per_time_accum += (masked_vector_nrmse_right_padded.sum(dim=(0, 1)) / (valid_count_per_time + 1e-8)) / gradient_accumulation_steps  # (seq_len,)
                    
                    # Per-layer NRMSE: mean across batch and seq_len dimensions  
                    # Result shape: (L,)
                    valid_count_per_layer = mask_exp_right_padded.sum(dim=(0, 2))  # (L,) - count valid entries per layer
                    nrmse_per_layer_accum += (masked_vector_nrmse_right_padded.sum(dim=(0, 2)) / (valid_count_per_layer + 1e-8)) / gradient_accumulation_steps  # (L,)

        # Accumulate for logging
        recon_loss_accum += recon_loss.item() / gradient_accumulation_steps
        codebook_loss_accum += codebook_loss.item() / gradient_accumulation_steps
        commitment_loss_accum += commitment_loss.item() / gradient_accumulation_steps
        # NEW for codebook collape - Accumulate new regularizer losses
        cosine_push_loss_accum += cosine_push_loss.item() / gradient_accumulation_steps
        entropy_loss_accum += entropy_loss.item() / gradient_accumulation_steps
        unique_count_accum += unique_count / gradient_accumulation_steps
        total_loss_accum += total_loss.item()
        
        with torch.no_grad():
            nrmse_accum += torch.mean(compute_masked_nrmse(out_first_stage_right_padded, output, mask_right_padded)) / gradient_accumulation_steps
            nrmse_per_vector_accum += torch.mean(compute_masked_nrmse_per_vector(out_first_stage_right_padded, output, mask_right_padded)) / gradient_accumulation_steps
            nrmse_per_element_accum += torch.mean(compute_masked_nrmse_per_element(out_first_stage_right_padded, output, mask_right_padded)) / gradient_accumulation_steps
            overall_nrmse_accum += torch.mean(compute_masked_nrmse(H, x_overall_recon_denormalized, attention_masks)) / gradient_accumulation_steps
            overall_nrmse_per_vector_accum += torch.mean(compute_masked_nrmse_per_vector(H, x_overall_recon_denormalized, attention_masks)) / gradient_accumulation_steps
            overall_nrmse_per_element_accum += torch.mean(compute_masked_nrmse_per_element(H, x_overall_recon_denormalized, attention_masks)) / gradient_accumulation_steps
        
        total_loss.backward()

    if grad_clip > 0:
        grad_norm_value = torch.nn.utils.clip_grad_norm_(vqvae2_model.parameters(), grad_clip)
    else:
        grad_norm_list = [p.grad.data.norm(2) for p in vqvae2_model.parameters() if p.grad is not None]
        grad_norm_value = torch.norm(torch.stack(grad_norm_list)) if grad_norm_list else torch.tensor(0.0, device=device)

    param_norm_list = [p.data.norm(2) for p in vqvae2_model.parameters() if p is not None]
    total_param_norm = torch.norm(torch.stack(param_norm_list)) if param_norm_list else torch.tensor(0.0, device=device)
    grad_norm_ratio = grad_norm_value / total_param_norm if total_param_norm.item() > 0 else torch.tensor(0.0, device=device)
    
    optimizer.step()
    optimizer.zero_grad()
    
    # NEW for codebook collape - Normalize codebook vectors after optimizer step (for cosine-push regularization)
    if hasattr(vqvae2_model, 'module'):
        # Handle DDP wrapped model
        vqvae2_model.module.normalize_codebook_vectors()
    else:
        vqvae2_model.normalize_codebook_vectors()

    sequences, attention_masks = next(iter(train_loader))
    sequences = sequences.to(device)
    # TODO: turn off/on this after debugging, for now, it is a quick fix
    # sequences[sequences == pad_token_id] = 0
    attention_masks = attention_masks.to(device)
    with ctx:
        hidden_states_tensor = Lmodel.generate_prefix_hidden_states(sequences, attention_masks)


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
            # NEW for codebook collape - Add new regularizer losses to DDP reduction
            cosine_push_loss_accum_tensor = torch.tensor(cosine_push_loss_accum, device=device)
            entropy_loss_accum_tensor = torch.tensor(entropy_loss_accum, device=device)
            total_loss_accum_tensor = torch.tensor(total_loss_accum, device=device)
            nrmse_accum_tensor = torch.tensor(nrmse_accum, device=device)
            nrmse_per_vector_accum_tensor = torch.tensor(nrmse_per_vector_accum, device=device)
            nrmse_per_element_accum_tensor = torch.tensor(nrmse_per_element_accum, device=device)
            overall_nrmse_accum_tensor = torch.tensor(overall_nrmse_accum, device=device)
            overall_nrmse_per_vector_accum_tensor = torch.tensor(overall_nrmse_per_vector_accum, device=device)
            overall_nrmse_per_element_accum_tensor = torch.tensor(overall_nrmse_per_element_accum, device=device)
            # All-reduce these tensors over all processes
            torch.distributed.all_reduce(recon_loss_accum_tensor)
            torch.distributed.all_reduce(codebook_loss_accum_tensor)
            torch.distributed.all_reduce(commitment_loss_accum_tensor)
            # NEW for codebook collape - All-reduce new regularizer losses
            torch.distributed.all_reduce(cosine_push_loss_accum_tensor)
            torch.distributed.all_reduce(entropy_loss_accum_tensor)
            torch.distributed.all_reduce(total_loss_accum_tensor)
            torch.distributed.all_reduce(nrmse_accum_tensor)
            torch.distributed.all_reduce(nrmse_per_vector_accum_tensor)
            torch.distributed.all_reduce(nrmse_per_element_accum_tensor)
            torch.distributed.all_reduce(overall_nrmse_accum_tensor)
            torch.distributed.all_reduce(overall_nrmse_per_vector_accum_tensor)
            torch.distributed.all_reduce(overall_nrmse_per_element_accum_tensor)
            if wandb_flag:
                torch.distributed.all_reduce(nrmse_per_time_accum)
                torch.distributed.all_reduce(nrmse_per_layer_accum)
            world_size = torch.distributed.get_world_size()
            recon_loss_accum = recon_loss_accum_tensor.item() / world_size
            codebook_loss_accum = codebook_loss_accum_tensor.item() / world_size
            commitment_loss_accum = commitment_loss_accum_tensor.item() / world_size
            # NEW for codebook collape - Average new regularizer losses across processes
            cosine_push_loss_accum = cosine_push_loss_accum_tensor.item() / world_size
            entropy_loss_accum = entropy_loss_accum_tensor.item() / world_size
            total_loss_accum = total_loss_accum_tensor.item() / world_size
            nrmse_accum = nrmse_accum_tensor.item() / world_size
            nrmse_per_vector_accum = nrmse_per_vector_accum_tensor.item() / world_size
            nrmse_per_element_accum = nrmse_per_element_accum_tensor.item() / world_size
            overall_nrmse_accum = overall_nrmse_accum_tensor.item() / world_size
            overall_nrmse_per_vector_accum = overall_nrmse_per_vector_accum_tensor.item() / world_size
            overall_nrmse_per_element_accum = overall_nrmse_per_element_accum_tensor.item() / world_size
            if wandb_flag:
                nrmse_per_time_accum = nrmse_per_time_accum / world_size
                nrmse_per_layer_accum = nrmse_per_layer_accum / world_size
        if master_process:
            # NEW for codebook collape - Build logging message with conditional new regularizer losses
            log_msg = (
                f'Iter: {iter_num}, Loss: {total_loss_accum:.4f}, '
                f'NRMSE: {nrmse_accum:.4f}, NRMSE_vec: {nrmse_per_vector_accum:.4f}, NRMSE_elem: {nrmse_per_element_accum:.4f}, '
                f'Overall NRMSE: {overall_nrmse_accum:.4f}, Overall_vec: {overall_nrmse_per_vector_accum:.4f}, Overall_elem: {overall_nrmse_per_element_accum:.4f}, '
                f'Recon Loss: {recon_loss_accum:.4f}, Codebook Loss: {codebook_loss_accum:.4f}, '
                f'Commitment Loss: {commitment_loss_accum:.4f}, '
            )
            if cosine_push_loss_accum > 0:
                log_msg += f'Cosine Push Loss: {cosine_push_loss_accum:.4f}, '
            if entropy_loss_accum > 0:
                log_msg += f'Entropy Loss: {entropy_loss_accum:.4f}, '
            log_msg += (
                f'Unique Vectors: {unique_count_accum:.2f}, '
                f'Grad Norm: {grad_norm_value.item():.4f}, Grad Norm/Param Norm: {grad_norm_ratio.item():.4f}, '
                f'Time: {dt:.2f}s'
            )
            logging.info(log_msg)
            if wandb_flag:
                log_data = {
                    'vqvae2/iter_num': iter_num,
                    'vqvae2/total_loss': total_loss_accum,
                    'vqvae2/recon_loss': recon_loss_accum,
                    'vqvae2/codebook_loss': codebook_loss_accum,
                    'vqvae2/commitment_loss': commitment_loss_accum,
                    # NEW for codebook collape - Add new regularizer losses to wandb logging
                    'vqvae2/cosine_push_loss': cosine_push_loss_accum,
                    'vqvae2/entropy_loss': entropy_loss_accum,
                    'vqvae2/nrmse': nrmse_accum,
                    'vqvae2/nrmse_per_vector': nrmse_per_vector_accum,
                    'vqvae2/nrmse_per_element': nrmse_per_element_accum,
                    'vqvae2/overall_nrmse': overall_nrmse_accum,
                    'vqvae2/overall_nrmse_per_vector': overall_nrmse_per_vector_accum,
                    'vqvae2/overall_nrmse_per_element': overall_nrmse_per_element_accum,
                    'vqvae2/time': dt,
                    'vqvae2/lr': lr,
                    'vqvae2/unique_vectors': unique_count_accum,
                    'vqvae2/grad_norm': grad_norm_value.item(),
                    'vqvae2/grad_norm_ratio': grad_norm_ratio.item()
                }
                
                # Add experimental per-time and per-layer NRMSE
                for t in range(nrmse_per_time_accum.shape[0]):
                    log_data[f'vqvae2_experimental/nrmse_time_{t}'] = nrmse_per_time_accum[t].item()
                
                for l in range(nrmse_per_layer_accum.shape[0]):
                    log_data[f'vqvae2_experimental/nrmse_layer_{l}'] = nrmse_per_layer_accum[l].item()
                
                if torch.cuda.is_available():
                    log_data.update({
                        'vqvae2/gpu_memory_allocated_GB': torch.cuda.memory_allocated(device) / 1024**3,
                        'vqvae2/gpu_memory_reserved_GB': torch.cuda.memory_reserved(device) / 1024**3
                    })
                wandb.log(log_data)
                
                # Plot NRMSE analysis every 100 iterations
                if iter_num % 100 == 0 and master_process:
                    plt.figure(figsize=(12, 5))
                    
                    # Plot NRMSE per time step
                    plt.subplot(1, 2, 1)
                    plt.plot(nrmse_per_time_accum.cpu().numpy())
                    plt.title('NRMSE per Time Step')
                    plt.xlabel('Time Step')
                    plt.ylabel('NRMSE')
                    plt.grid(True)
                    
                    # Plot NRMSE per layer
                    plt.subplot(1, 2, 2)
                    plt.plot(nrmse_per_layer_accum.cpu().numpy())
                    plt.title('NRMSE per Layer')
                    plt.xlabel('Layer')
                    plt.ylabel('NRMSE') 
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'vqvae2_nrmse_analysis.png'))
                    plt.close()
        total_losses.append(total_loss_accum)
        recon_losses.append(recon_loss_accum)
        codebook_losses.append(codebook_loss_accum)
        commitment_losses.append(commitment_loss_accum)
        # NEW for codebook collape - Store new regularizer losses
        cosine_push_losses.append(cosine_push_loss_accum)
        entropy_losses.append(entropy_loss_accum)
        nrmses.append(nrmse_accum)
        nrmses_per_vector.append(nrmse_per_vector_accum)
        nrmses_per_element.append(nrmse_per_element_accum)
        overall_nrmses.append(overall_nrmse_accum)
        overall_nrmses_per_vector.append(overall_nrmse_per_vector_accum)
        overall_nrmses_per_element.append(overall_nrmse_per_element_accum)

        # Codebook analysis using model's internal tracking every 1000 iterations
        if iter_num % 1000 == 0 and master_process:
            # Get usage statistics from model's internal tracking
            if hasattr(vqvae2_model, 'module'):
                # Handle DDP wrapped model
                usage_stats = vqvae2_model.module.get_usage_statistics()
                similarity_stats = vqvae2_model.module.compute_codebook_similarities()
            else:
                usage_stats = vqvae2_model.get_usage_statistics()
                similarity_stats = vqvae2_model.compute_codebook_similarities()
            
            # Log usage statistics
            logging.info(f"VQ-VAE2 codebook usage: {usage_stats['unique_vectors_used']}/{len(usage_stats['usage_counts'])} vectors used, "
                        f"Total vectors processed: {usage_stats['total_vectors_processed']}")
            
            # Plot usage histogram
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(usage_stats['usage_counts'])), usage_stats['usage_counts'].cpu().numpy())
            plt.title(f'VQ-VAE2 Codebook Vector Usage\n'
                     f'Total vectors processed: {usage_stats["total_vectors_processed"]}, '
                     f'Unique vectors used: {usage_stats["unique_vectors_used"]}/{len(usage_stats["usage_counts"])}')
            plt.xlabel('Codebook Vector Index')
            plt.ylabel('Usage Count')
            plt.savefig(os.path.join(output_dir, 'vqvae2_codebook_usage.png'))
            plt.close()
            
            # Plot similarity heatmap if we have used vectors
            if similarity_stats['similarities'] is not None:
                plt.figure(figsize=(10, 10))
                plt.imshow(similarity_stats['similarities'].cpu().numpy(), cmap='viridis')
                plt.colorbar()
                plt.title(f'Cosine Similarities Between Used VQ-VAE2 Codebook Vectors\n'
                         f'Used vectors: {similarity_stats["num_used_vectors"]}')
                plt.xlabel('Used Vector Index')
                plt.ylabel('Used Vector Index')
                plt.savefig(os.path.join(output_dir, 'vqvae2_codebook_similarities.png'))
                plt.close()
                
            # Log to wandb if enabled
            if wandb_flag:
                wandb.log({
                    'vqvae2_analysis/unique_vectors_used': usage_stats['unique_vectors_used'],
                    'vqvae2_analysis/total_vectors_processed': usage_stats['total_vectors_processed'],
                    'vqvae2_analysis/usage_percentage': usage_stats['unique_vectors_used'] / len(usage_stats['usage_counts']) * 100
                })

    if save_interval > 0 and iter_num % save_interval == 0 and master_process:
        tempTime = time.time()
        checkpoint = {
            'model_state_dict': vqvae2_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter_num': iter_num,
            'iter_num_log': iter_num_log,
            'config': config,
            'recon_losses': recon_losses,
            'codebook_losses': codebook_losses,
            'commitment_losses': commitment_losses,
            # NEW for codebook collape - Save new regularizer losses
            'cosine_push_losses': cosine_push_losses,
            'entropy_losses': entropy_losses,
            'nrmses': nrmses,
            'nrmses_per_vector': nrmses_per_vector,
            'nrmses_per_element': nrmses_per_element,
            'overall_nrmses': overall_nrmses,
            'overall_nrmses_per_vector': overall_nrmses_per_vector,
            'overall_nrmses_per_element': overall_nrmses_per_element,
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
        'model_state_dict': vqvae2_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter_num': iter_num,
        'iter_num_log': iter_num_log,
        'config': config,
        'recon_losses': recon_losses,
        'codebook_losses': codebook_losses,
        'commitment_losses': commitment_losses,
        # NEW for codebook collape - Save new regularizer losses
        'cosine_push_losses': cosine_push_losses,
        'entropy_losses': entropy_losses,
        'nrmses': nrmses,
        'nrmses_per_vector': nrmses_per_vector,
        'nrmses_per_element': nrmses_per_element,
        'overall_nrmses': overall_nrmses,
        'overall_nrmses_per_vector': overall_nrmses_per_vector,
        'overall_nrmses_per_element': overall_nrmses_per_element,
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

# %%
    
if ddp:
    destroy_process_group()
# %%

# %%
