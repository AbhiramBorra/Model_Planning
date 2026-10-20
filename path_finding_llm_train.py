# %%

"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from model import GPTConfig, GPT, Block, LayerNorm

from llm_train_utils import get_lr, save_training_metadata #, save_cfg_accuracy_log, save_training_metadata, evaluate_cfg_accuracy, load_cfg_from_dataset
# from data.context_free_grammar.CFG_data_generation import CFG

from path_finding_llm_utilities import PathFindingDataset, custom_collate_fn
from path_finding_llm_evaluation import evaluate_path_accuracy
# %%

run_name = "pathfinding" # name of the run, used for wandb and output directory
# -----------------------------------------------------------------------------
save_interval = 20000
eval_interval = 500  # Used for BOTH loss and accuracy evaluation
log_interval = 1
eval_iters = 200
base_seed = 1337
# Path finding accuracy evaluation
accuracy_eval_samples = 1000  # Increased from 250 to reduce memorization
debug_loss_comparison = True  # Set to True to compare old vs new loss estimation

eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'MinimalLLM_Path_finding'
wandb_run_name = run_name # 'run' + str(time.time())
wandb_entity = 'llm_analysis' # 'HyperAttention' # the team name in wandb (the team may be in different organization)

# Multi-token loss (MTL) settings
multi_token_loss = None # None for standard training, "shared_heads" for MTL
mtl_length = 1 # how many tokens to predict (>=1)
mtl_shared_head_nb_blocks = 1 # how many blocks are shared (>=1)

out_dir = 'out/'+run_name
#dataset = 'openwebtext' 
dataset_dir = 'path_finding/samples'
gradient_accumulation_steps = 2  # used to simulate larger batch sizes
batch_size = 12*8 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 512
num_hyphens = 0 # Let's make default zero in our experiments 
# model
n_layer = 4
n_head = 4
n_embd = 16*4
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 20000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 100 # how many steps to warm up for
lr_decay_iters = 20000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# try:
exec(open('configurator.py').read()) # overrides from command line or config file
# except:
#     print("Using MTL DNM config")
#     exec(open('PF_15XL/mtl_dnm.py').read()) # 
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
# %%

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
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
# %%


if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(base_seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# %%


# Load dataset configuration
train_config = open(os.path.join(dataset_dir, 'config_train.json'), 'r').read()
train_config = json.loads(train_config)
max_nodes = train_config['max_nodes']

# Create datasets from JSONL files
train_jsonl_path = os.path.join(dataset_dir, 'train.jsonl')
val_jsonl_path = os.path.join(dataset_dir, 'val.jsonl')

# Initialize train dataset
train_dataset = PathFindingDataset(
    jsonl_file=train_jsonl_path, 
    max_seq_length=block_size, 
    mask_edges=True, 
    max_nodes=max_nodes,
    number_of_hyphens = num_hyphens
)

# Initialize validation dataset with same vocabulary
val_dataset = PathFindingDataset(
    jsonl_file=val_jsonl_path,
    tokenizer_vocab=train_dataset.vocab,  # Use same vocab as train
    max_seq_length=block_size,
    mask_edges=True,
    number_of_hyphens = num_hyphens,
    max_nodes=max_nodes
)

# %%


vocab_size = train_dataset.vocab_size

# Create DataLoaders with proper DDP support
# Initialize samplers globally
train_sampler = None
val_sampler = None

if ddp:
    train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=(train_sampler is None),  # Only shuffle if not using DistributedSampler
    sampler=train_sampler,
    num_workers=0,  # Set to 0 for simplicity, can increase if needed
    collate_fn=custom_collate_fn,
    pin_memory=(device_type == 'cuda')
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    sampler=val_sampler,
    num_workers=0,
    collate_fn=custom_collate_fn,
    pin_memory=(device_type == 'cuda')
)

train_iterator = iter(train_loader)
val_iterator = iter(val_loader)

# %%

# New data loader function using PathFindingDataset
def get_batch(split):
    global train_iterator, val_iterator, current_epoch, train_sampler, val_sampler
    try:
        if split == 'train':
            batch = next(train_iterator)
        else:
            batch = next(val_iterator)
    except StopIteration:
        # If we've exhausted the iterator, create a new one
        if split == 'train':
            current_epoch += 1
            # Update distributed sampler epoch for proper shuffling
            if ddp and train_sampler is not None:
                train_sampler.set_epoch(current_epoch)
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        else:
            # For validation, also update epoch but typically val doesn't need shuffling
            if ddp and val_sampler is not None:
                val_sampler.set_epoch(current_epoch)
            val_iterator = iter(val_loader)
            batch = next(val_iterator)
    
    # Extract input_ids and labels from the batch
    x = batch['input_ids']
    y = batch['labels']
    
    # For MTL, generate multiple y targets by shifting left
    if multi_token_loss == "shared_heads" and mtl_length > 1:
        y_list = [y]  # First element is regular y
        for shift_amount in range(1, mtl_length):
            # Shift y to left by shift_amount by dropping first entries and adding -1 to the end
            y_shifted = torch.cat([y[:, shift_amount:], torch.full((y.size(0), shift_amount), -1, dtype=y.dtype)], dim=1)
            y_list.append(y_shifted)
        y = y_list  # Replace y with list of targets
    
    # Move to device
    # old code: x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    if device_type == 'cuda':
        x = x.to(device, non_blocking=True)
        if isinstance(y, list):
            y = [y_i.to(device, non_blocking=True) for y_i in y]
        else:
            y = y.to(device, non_blocking=True)
    else:
        x = x.to(device)
        if isinstance(y, list):
            y = [y_i.to(device) for y_i in y]
        else:
            y = y.to(device)
        
    return x, y, batch['correct_path_ids'], batch['decoy_path_ids']

# %%

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
accuracy_history = []  # Track accuracy over time

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # Use the vocabulary size from our PathFindingDataset
    print(f"Using vocab size from PathFindingDataset: {vocab_size}")
    model_args['vocab_size'] = vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    # Load accuracy history if available
    if 'accuracy_history' in checkpoint:
        accuracy_history = checkpoint['accuracy_history']
        print(f"Loaded accuracy history with {len(accuracy_history)} entries")
    
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    print(f"Cropping model block size from {model.config.block_size} to {block_size}")
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
else:
    print(f"Model block size is already {model.config.block_size}, no cropping needed")
model.to(device)
# %%

# Create additional MTL modules if needed
mtl_modules = None
if multi_token_loss == "shared_heads" and mtl_length > 1:
    print(f"Creating MTL modules: {mtl_length - 1} modules with {mtl_shared_head_nb_blocks} blocks each")
    
    # Create mtl_length - 1 independent modules
    mtl_modules = []
    for i in range(mtl_length - 1):
        # Each module has mtl_shared_head_nb_blocks blocks only
        # ln_f and lm_head are shared from the original model
        module_blocks = nn.ModuleList([Block(model.config) for _ in range(mtl_shared_head_nb_blocks)])
        
        # Initialize weights similar to main model (will be overridden if loading from checkpoint)
        for block in module_blocks:
            block.apply(model._init_weights)
        
        mtl_module = {
            'blocks': module_blocks
        }
        mtl_modules.append(mtl_module)
    
    # Load MTL modules from checkpoint if resuming
    if init_from == 'resume' and 'mtl_modules' in checkpoint and checkpoint['mtl_modules'] is not None:
        saved_mtl_modules = checkpoint['mtl_modules']
        if len(saved_mtl_modules) == len(mtl_modules):
            print("Loading MTL modules from checkpoint...")
            for i, (mtl_module, saved_state) in enumerate(zip(mtl_modules, saved_mtl_modules)):
                # Handle module. prefixes in state dict
                state_dict = saved_state['blocks']
                unwanted_prefix = 'module.'
                for k, v in list(state_dict.items()):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                
                mtl_module['blocks'].load_state_dict(state_dict)
                # ln_f and lm_head are shared, so no need to load them separately
            print("MTL modules loaded successfully")
        else:
            print(f"Warning: MTL module count mismatch. Expected {len(mtl_modules)}, found {len(saved_mtl_modules)}. Using fresh initialization.")
    # Move MTL modules to device
    for mtl_module in mtl_modules:
        mtl_module['blocks'].to(device)
        # ln_f and lm_head are part of the main model, so they're already on device
# %%

# Save training metadata
save_training_metadata(config, model_args, out_dir)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
# Use model's configure_optimizers but add MTL parameters if needed
if mtl_modules is not None:
    # Create optimizer with both model and MTL parameters
    param_dict = {}
    # Add model parameters
    for pn, p in model.named_parameters():
        if p.requires_grad:
            param_dict[f"model.{pn}"] = p
    
    # Add MTL parameters (only blocks, since ln_f and lm_head are shared)
    for i, mtl_module in enumerate(mtl_modules):
        for pn, p in mtl_module['blocks'].named_parameters():
            if p.requires_grad:
                param_dict[f"mtl_{i}.blocks.{pn}"] = p
    
    # Create optim groups similar to model.configure_optimizers
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    # Create optimizer
    import inspect
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), **extra_args)
    
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors (incl. MTL): {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors (incl. MTL): {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
else:
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# %%

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    
    # Also wrap MTL modules in DDP if they exist
    if mtl_modules is not None:
        for mtl_module in mtl_modules:
            mtl_module['blocks'] = DDP(mtl_module['blocks'], device_ids=[ddp_local_rank])
            # ln_f and lm_head are part of the main model which is already wrapped in DDP

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y,_, _= get_batch(split)
            with ctx:
                if multi_token_loss == "shared_heads" and mtl_length > 1:
                    logits, loss, individual_losses = model(X, Y, mtl_modules=mtl_modules, mtl_shared_head_nb_blocks=mtl_shared_head_nb_blocks)
                else:
                    logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Alternative loss estimation using consistent subset
@torch.no_grad()
def estimate_loss_consistent():
    """Estimate loss using the same subset as accuracy evaluation for consistency"""
    out = {}
    model.eval()
    
    # Train loss - use same method as before
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y, _, _ = get_batch('train')
        with ctx:
            if multi_token_loss == "shared_heads" and mtl_length > 1:
                logits, loss, individual_losses = model(X, Y, mtl_modules=mtl_modules, mtl_shared_head_nb_blocks=mtl_shared_head_nb_blocks)
            else:
                logits, loss = model(X, Y)
        losses[k] = loss.item()
    out['train'] = losses.mean()
    
    # Validation loss - use consistent subset
    val_losses = []
    val_iter = iter(loss_eval_loader)
    try:
        for k in range(min(eval_iters, len(loss_eval_loader))):
            batch = next(val_iter)
            X = batch['input_ids'].to(device, non_blocking=True)
            Y = batch['labels'].to(device, non_blocking=True)
            
            # For MTL, generate multiple y targets by shifting left (same as in get_batch)
            if multi_token_loss == "shared_heads" and mtl_length > 1:
                y_list = [Y]  # First element is regular Y
                for shift_amount in range(1, mtl_length):
                    # Shift Y to left by shift_amount by dropping first entries and adding -1 to the end
                    y_shifted = torch.cat([Y[:, shift_amount:], torch.full((Y.size(0), shift_amount), -1, dtype=Y.dtype, device=Y.device)], dim=1)
                    y_list.append(y_shifted)
                Y = y_list  # Replace Y with list of targets
            
            with ctx:
                if multi_token_loss == "shared_heads" and mtl_length > 1:
                    logits, loss, individual_losses = model(X, Y, mtl_modules=mtl_modules, mtl_shared_head_nb_blocks=mtl_shared_head_nb_blocks)
                else:
                    logits, loss = model(X, Y)
            val_losses.append(loss.item())
    except StopIteration:
        pass
    
    if val_losses:
        out['val'] = sum(val_losses) / len(val_losses)
    else:
        out['val'] = float('inf')
    
    model.train()
    return out
# %%

# ------------------------------------------------------------------
# Build a deterministic subset of the validation set for accuracy eval
# IMPORTANT: Use the same samples for both loss and accuracy evaluation
if len(val_dataset) > accuracy_eval_samples:
    _gen = torch.Generator()
    _gen.manual_seed(42 + seed_offset)  # fixed seed → stable subset
    _subset_idx = torch.randperm(len(val_dataset), generator=_gen)[:accuracy_eval_samples]
    accuracy_subset = Subset(val_dataset, _subset_idx)
    
    # Create a separate subset that matches the samples used in loss evaluation
    # Use the same samples that estimate_loss() would use from val_loader
    loss_eval_subset = Subset(val_dataset, _subset_idx[:eval_iters * batch_size])
    loss_eval_loader = DataLoader(
        loss_eval_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=(device_type == 'cuda')
    )
else:
    accuracy_subset = val_dataset
    loss_eval_loader = val_loader

accuracy_loader = DataLoader(
    accuracy_subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=custom_collate_fn,
    pin_memory=(device_type == 'cuda')
)

# ------------------------------------------------------------------
# Build a deterministic subset of the *training* set for accuracy eval
if len(train_dataset) > accuracy_eval_samples:
    _train_gen = torch.Generator()
    _train_gen.manual_seed(24 + seed_offset)  # farklı sabit seed
    _train_idx = torch.randperm(len(train_dataset), generator=_train_gen)[:accuracy_eval_samples]
    accuracy_train_subset = Subset(train_dataset, _train_idx)
else:
    accuracy_train_subset = train_dataset

accuracy_train_loader = DataLoader(
    accuracy_train_subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=custom_collate_fn,
    pin_memory=(device_type == 'cuda')
)
# %%
# ------------------------------------------------------------------
# logging
if wandb_log and master_process:
    import wandb
    wandb_config = {
        'project': wandb_project,
        'name': wandb_run_name,
        'config': config
    }
    if wandb_entity:
        wandb_config['entity'] = wandb_entity
    wandb.init(**wandb_config)
# %%

# training loop
t0 = time.time()
current_epoch = 0  # tracks dataset epochs, must exist before first get_batch
X, Y, _, _ = get_batch('train')  # fetch the very first batch
t0 = time.time()
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

@torch.inference_mode()
def evaluate_accuracies(iter_num):
    print("Evaluating path finding accuracies...")
    model.eval()
    metrics = evaluate_path_accuracy(
        model = model,
        dataloader=accuracy_train_loader,
        device=device,
        number_of_hyphens=num_hyphens,
        max_new_tokens=10,
        max_batches=min(32, accuracy_eval_samples // batch_size + 1),
        max_seq_length=block_size
    )
    train_acc = metrics["accuracy"]

    metrics = evaluate_path_accuracy(
        model = model,
        dataloader=accuracy_loader,
        device=device,
        number_of_hyphens=num_hyphens,
        max_new_tokens=10,
        max_batches=min(32, accuracy_eval_samples // batch_size + 1),
        max_seq_length=block_size
    )
    val_acc = metrics["accuracy"]
    model.train()
    accuracy_history.append({
        'iter': iter_num,
        'train_acc':  train_acc,
        'val_acc':    val_acc,
    })
    
    print(f"step {iter_num}: train acc {train_acc:.4f}, val acc {val_acc:.4f}")    
    return train_acc, val_acc
# %%

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    things_to_log = {}
    if ((iter_num % save_interval == 0) or (iter_num > max_iters)) and (master_process) and (iter_num > 0):
        losses = estimate_loss_consistent()  # Use consistent subset
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Optional debug: compare old vs new loss estimation
        if debug_loss_comparison:
            old_losses = estimate_loss()
            print(f"[DEBUG] Old method - train loss {old_losses['train']:.4f}, val loss {old_losses['val']:.4f}")
            print(f"[DEBUG] New method - train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print(f"[DEBUG] Val loss difference: {abs(old_losses['val'] - losses['val']):.4f}")
        
        # Also evaluate path finding accuracy during save intervals
        train_acc, val_acc = evaluate_accuracies(iter_num)

        if wandb_log:
            things_to_log = {
                "LLMtrain/iter": iter_num,
                "LLMtrain/train/loss": losses['train'],
                "LLMtrain/val/loss": losses['val'],
                "LLMtrain/lr": lr,
                "LLMtrain/mfu": running_mfu*100, # convert to percentage
                "LLMtrain/train/acc": train_acc,
                "LLMtrain/val/acc": val_acc,
            }
            
            # For MTL compatibility, also log as main loss metrics
            if multi_token_loss == "shared_heads" and mtl_length > 1:
                things_to_log["LLMtrain/loss"] = losses['train']  # Use train loss as main loss
            else:
                things_to_log["LLMtrain/loss"] = losses['train']

        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'accuracy_history': accuracy_history,  # Save accuracy history
                }
                
                # Save MTL modules if they exist
                if mtl_modules is not None:
                    mtl_state_dicts = []
                    for mtl_module in mtl_modules:
                        mtl_state_dict = {
                            'blocks': mtl_module['blocks'].state_dict()
                            # ln_f and lm_head are shared, so only save blocks
                        }
                        mtl_state_dicts.append(mtl_state_dict)
                    checkpoint['mtl_modules'] = mtl_state_dicts
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    elif master_process and iter_num > 0:
        if (iter_num % eval_interval == 0):
            losses = estimate_loss_consistent()  # Use consistent subset
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Also evaluate path finding accuracy at the same interval
            train_acc, val_acc = evaluate_accuracies(iter_num)
            
            if wandb_log:
                things_to_log["LLMtrain/iter"] = iter_num
                things_to_log["LLMtrain/train/loss"] = losses['train']
                things_to_log["LLMtrain/val/loss"] = losses['val']
                things_to_log["LLMtrain/train/acc"] = train_acc
                things_to_log["LLMtrain/val/acc"] = val_acc
                things_to_log["LLMtrain/lr"] = lr
                things_to_log["LLMtrain/mfu"] = running_mfu*100
                
                # For MTL compatibility, also log as main loss metrics
                things_to_log["LLMtrain/loss"] = losses['train']  # Use train loss as main loss
                
                wandb.log(things_to_log)

    if iter_num == 0 and eval_only:
        # Also evaluate accuracy when in eval_only mode
        if master_process:
            print("Evaluating path finding accuracy (eval_only mode)...")
            accuracy_metrics = evaluate_path_accuracy(
                model=model,
                dataloader=accuracy_loader,
                device=device,
                number_of_hyphens=num_hyphens,
                max_new_tokens=15,
                max_batches=None,
                max_seq_length=block_size
            )
            print(f"Final results - accuracy: {accuracy_metrics['accuracy']:.4f} "
                  f"({accuracy_metrics['correct_matches']}/{accuracy_metrics['total_samples']})")
            
            # Print some example predictions for analysis
            if 'results' in accuracy_metrics and accuracy_metrics['results']:
                print("\nExample predictions:")
                for i, result in enumerate(accuracy_metrics['results'][:3]):  # Show first 3
                    print(f"  Sample {i+1}:")
                    print(f"    Generated: {result['generated_tokens']}")
                    print(f"    Correct options: {result['correct_tokens']}")
                    print(f"    Match: {'✓' if result['is_correct'] else '✗'}")
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            if multi_token_loss == "shared_heads" and mtl_length > 1:
                logits, loss, individual_losses = model(X, Y, mtl_modules=mtl_modules, mtl_shared_head_nb_blocks=mtl_shared_head_nb_blocks)
                # Store individual losses for logging
                if micro_step == 0:  # Only store on first micro step to avoid accumulation
                    current_individual_losses = [l.item() for l in individual_losses]
            else:
                logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, _, _ = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        log_message = f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        wandb_log_dict = {
            "LLMtrain/iter": iter_num,
            "LLMtrain/lr": lr
        }
        
        # Add MTL individual losses to logging
        if multi_token_loss == "shared_heads" and mtl_length > 1 and 'current_individual_losses' in locals():
            # For MTL: log next token loss as main loss for comparison with non-MTL runs
            next_token_loss = current_individual_losses[0]
            wandb_log_dict["LLMtrain/loss (train)"] = next_token_loss
            wandb_log_dict["LLMtrain/loss"] = next_token_loss  # For easier comparison
            
            # Log average MTL loss
            avg_mtl_loss = sum(current_individual_losses) / len(current_individual_losses)
            wandb_log_dict["LLMtrain/loss_mtl_avg"] = avg_mtl_loss
            
            # Log individual token losses
            for i, ind_loss in enumerate(current_individual_losses):
                token_name = "next" if i == 0 else f"next+{i}"
                log_message += f", {token_name}: {ind_loss:.4f}"
                wandb_log_dict[f"LLMtrain/loss_{token_name}"] = ind_loss
            
            # Add average to log message
            log_message += f", avg: {avg_mtl_loss:.4f}"
        else:
            # For non-MTL: use the standard loss
            wandb_log_dict["LLMtrain/loss (train)"] = lossf
            wandb_log_dict["LLMtrain/loss"] = lossf
        
        print(log_message)

        if wandb_log:
            things_to_log.update(wandb_log_dict)
            wandb.log(things_to_log)
    
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# %%

# Final evaluation and results saving when training ends
if master_process:
    print("\n" + "="*60)
    print("TRAINING COMPLETED - RUNNING FINAL EVALUATION")
    print("="*60)
    
    # Run final evaluation on both training and validation sets
    print("Evaluating final path finding accuracies...")
    final_train_acc, final_val_acc = evaluate_accuracies(iter_num)
    
    # Also get more detailed metrics for the final report
    print("Running detailed evaluation on validation set...")
    detailed_val_metrics = evaluate_path_accuracy(
        model=model,
        dataloader=accuracy_loader,
        device=device,
        number_of_hyphens=num_hyphens,
        max_new_tokens=15,
        max_batches=None,  # Evaluate all samples in the subset
        max_seq_length=block_size
    )
    
    print("Running detailed evaluation on training set...")
    detailed_train_metrics = evaluate_path_accuracy(
        model=model,
        dataloader=accuracy_train_loader,
        device=device,
        number_of_hyphens=num_hyphens,
        max_new_tokens=15,
        max_batches=None,  # Evaluate all samples in the subset
        max_seq_length=block_size
    )
    
    # Save final results to text file
    from datetime import datetime
    results_path = os.path.join(out_dir, 'final_accuracy_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Final Path Finding Accuracy Results\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training completed at iteration: {iter_num}\n")
        f.write(f"Model: {run_name}\n")
        f.write(f"Number of hyphens: {num_hyphens}\n")
        f.write("="*50 + "\n\n")
        
        f.write("VALIDATION SET RESULTS:\n")
        f.write(f"  Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)\n")
        f.write(f"  Correct matches: {detailed_val_metrics['correct_matches']}\n")
        f.write(f"  Total samples: {detailed_val_metrics['total_samples']}\n")
        f.write(f"  Evaluation subset size: {len(accuracy_subset) if len(val_dataset) > accuracy_eval_samples else len(val_dataset)}\n\n")
        
        f.write("TRAINING SET RESULTS:\n")
        f.write(f"  Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)\n")
        f.write(f"  Correct matches: {detailed_train_metrics['correct_matches']}\n")
        f.write(f"  Total samples: {detailed_train_metrics['total_samples']}\n")
        f.write(f"  Evaluation subset size: {len(accuracy_train_subset) if len(train_dataset) > accuracy_eval_samples else len(train_dataset)}\n\n")
        
        f.write("TRAINING CONFIGURATION:\n")
        f.write(f"  Max iterations: {max_iters}\n")
        f.write(f"  Learning rate: {learning_rate}\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Model layers: {n_layer}\n")
        f.write(f"  Model heads: {n_head}\n")
        f.write(f"  Model embedding dim: {n_embd}\n")
        f.write(f"  Block size: {block_size}\n")
        f.write(f"  Vocab size: {vocab_size}\n")
        
        if accuracy_history:
            f.write(f"\nACCURACY HISTORY (last 10 evaluations):\n")
            recent_history = accuracy_history[-10:]
            for entry in recent_history:
                f.write(f"  Iter {entry['iter']:5d}: Train {entry['train_acc']:.4f}, Val {entry['val_acc']:.4f}\n")
        
        # Add sample model outputs
        f.write(f"\nSAMPLE MODEL OUTPUTS (Validation Set):\n")
        f.write("="*50 + "\n")
        
        # Show up to 10 examples: mix of correct and incorrect predictions
        val_results = detailed_val_metrics.get('results', [])
        correct_examples = [r for r in val_results if r['is_correct']]
        incorrect_examples = [r for r in val_results if not r['is_correct']]
        
        # Show up to 5 correct examples
        f.write("\nCORRECT PREDICTIONS:\n")
        for i, result in enumerate(correct_examples[:5]):
            f.write(f"\nExample {i+1} (Sample {result['sample_idx']}):\n")
            f.write(f"  Generated: {' '.join(result['generated_tokens'])}\n")
            f.write(f"  Correct paths:\n")
            for j, correct_path in enumerate(result['correct_tokens']):
                f.write(f"    {j+1}: {' '.join(correct_path)}\n")
        
        # Show up to 5 incorrect examples
        f.write(f"\nINCORRECT PREDICTIONS:\n")
        for i, result in enumerate(incorrect_examples[:5]):
            f.write(f"\nExample {i+1} (Sample {result['sample_idx']}):\n")
            f.write(f"  Generated: {' '.join(result['generated_tokens'])}\n")
            f.write(f"  Correct paths:\n")
            for j, correct_path in enumerate(result['correct_tokens']):
                f.write(f"    {j+1}: {' '.join(correct_path)}\n")
        
        if not correct_examples and not incorrect_examples:
            f.write("  No detailed examples available.\n")
        
        # Add sample training outputs too
        f.write(f"\nSAMPLE MODEL OUTPUTS (Training Set):\n")
        f.write("="*50 + "\n")
        
        train_results = detailed_train_metrics.get('results', [])
        train_correct_examples = [r for r in train_results if r['is_correct']]
        train_incorrect_examples = [r for r in train_results if not r['is_correct']]
        
        # Show up to 3 correct and 3 incorrect from training set
        f.write("\nCORRECT PREDICTIONS:\n")
        for i, result in enumerate(train_correct_examples[:3]):
            f.write(f"\nExample {i+1} (Sample {result['sample_idx']}):\n")
            f.write(f"  Generated: {' '.join(result['generated_tokens'])}\n")
            f.write(f"  Correct paths:\n")
            for j, correct_path in enumerate(result['correct_tokens']):
                f.write(f"    {j+1}: {' '.join(correct_path)}\n")
        
        f.write(f"\nINCORRECT PREDICTIONS:\n")
        for i, result in enumerate(train_incorrect_examples[:3]):
            f.write(f"\nExample {i+1} (Sample {result['sample_idx']}):\n")
            f.write(f"  Generated: {' '.join(result['generated_tokens'])}\n")
            f.write(f"  Correct paths:\n")
            for j, correct_path in enumerate(result['correct_tokens']):
                f.write(f"    {j+1}: {' '.join(correct_path)}\n")
    
    print(f"\nFinal results saved to: {results_path}")
    print(f"Final validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"Final training accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")

if ddp:
    destroy_process_group()
    
# %%
