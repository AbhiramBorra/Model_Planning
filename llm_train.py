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

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, Block, LayerNorm

from llm_train_utils import get_lr, save_cfg_accuracy_log, save_training_metadata, evaluate_cfg_accuracy, load_cfg_from_dataset
from data.context_free_grammar.CFG_data_generation import CFG

run_name = "CFG_trial_accuracy" # name of the run, used for wandb and output directory

# -----------------------------------------------------------------------------
save_interval = 20000
eval_interval = 500
log_interval = 1
eval_iters = 200
base_seed = 1337
# CFG accuracy evaluation
accuracy_eval_iters = 2000
accuracy_eval_samples = 250
# Whether to save generated samples in accuracy evaluation
save_accuracy_samples = True
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'MinimalLLM'
wandb_run_name = run_name # 'run' + str(time.time())
wandb_entity = 'llm_analysis' # 'HyperAttention' # the team name in wandb (the team may be in different organization)

# Multi-token loss (MTL) settings
multi_token_loss = None # None for standard training, "shared_heads" for MTL
mtl_length = 1 # how many tokens to predict (>=1)
mtl_shared_head_nb_blocks = 1 # how many blocks are shared (>=1)

out_dir = 'out/'+run_name
dataset = 'openwebtext' 
gradient_accumulation_steps = 2  # used to simulate larger batch sizes
batch_size = 12*8 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 512
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
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

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

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(base_seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)

# attempt to derive things "meta variables" from the dataset 
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
meta_dtype = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    meta_dtype = meta['dtype']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

if meta_dtype is None:
    # if no meta data, we assume the data is in the same format as GPT-2
    # which is uint16 for the tokens, and float16 for the embeddings
    meta_dtype = np.uint16
    print(f"assuming vocab_size = 50257 (GPT-2 default) and dtype = {meta_dtype}")
    
# poor man's data loader
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=meta_dtype, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=meta_dtype, mode='r')
    
    # Adjust data length constraint for MTL
    if multi_token_loss == "shared_heads" and mtl_length > 1:
        max_start_idx = len(data) - block_size - mtl_length + 1
    else:
        max_start_idx = len(data) - block_size
    
    ix = torch.randint(max_start_idx, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    # For MTL, generate multiple y targets
    if multi_token_loss == "shared_heads" and mtl_length > 1:
        y_list = []
        for shift in range(1, mtl_length + 1):
            y_shift = torch.stack([torch.from_numpy((data[i+shift:i+shift+block_size]).astype(np.int64)) for i in ix])
            y_list.append(y_shift)
        y = y_list  # List of target tensors
    else:
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        if isinstance(y, list):
            y = [y_i.pin_memory().to(device, non_blocking=True) for y_i in y]
        else:
            y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        if isinstance(y, list):
            y = [y_i.to(device) for y_i in y]
        else:
            y = y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# For tracking CFG accuracy history
accuracy_history = {
    'iterations': [],
    'zero_cut': [],
    'half_cut': []
}

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=meta_vocab_size, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
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
        print(f"Loaded accuracy history with {len(accuracy_history['iterations'])} data points")
    
    # If old format checkpoint has accuracy2_history, migrate to accuracy_history
    if 'accuracy2_history' in checkpoint and not 'accuracy_history' in checkpoint:
        accuracy_history = checkpoint['accuracy2_history']
        print(f"Migrated from old accuracy2_history with {len(accuracy_history['iterations'])} data points")
    
    # Load MTL modules if they were saved and current config uses MTL
    saved_mtl_modules = None
    if 'mtl_modules' in checkpoint:
        saved_mtl_modules = checkpoint['mtl_modules']
        print(f"Found saved MTL modules: {len(saved_mtl_modules)} modules")

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

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
    if init_from == 'resume' and 'saved_mtl_modules' in locals() and saved_mtl_modules is not None:
        if len(saved_mtl_modules) == len(mtl_modules):
            print("Loading MTL modules from checkpoint...")
            for i, (mtl_module, saved_state) in enumerate(zip(mtl_modules, saved_mtl_modules)):
                mtl_module['blocks'].load_state_dict(saved_state['blocks'])
                # ln_f and lm_head are shared, so no need to load them separately
            print("MTL modules loaded successfully")
        else:
            print(f"Warning: MTL module count mismatch. Expected {len(mtl_modules)}, found {len(saved_mtl_modules)}. Using fresh initialization.")
    
    # Move MTL modules to device
    for mtl_module in mtl_modules:
        mtl_module['blocks'].to(device)
        # ln_f and lm_head are part of the main model, so they're already on device

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

# Load CFG if dataset is a context_free_grammar dataset
cfg = None
if 'context_free_grammar' in dataset:
    try:
        print(f"Loading CFG from dataset {dataset}...")
        cfg = load_cfg_from_dataset(dataset)
        print("CFG loaded successfully.")
    except Exception as e:
        print(f"Error loading CFG: {e}")
        cfg = None

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                if multi_token_loss == "shared_heads" and mtl_length > 1:
                    logits, loss, individual_losses = model(X, Y, mtl_modules=mtl_modules, mtl_shared_head_nb_blocks=mtl_shared_head_nb_blocks)
                else:
                    logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if ((iter_num % save_interval == 0) or (iter_num > max_iters)) and (master_process) and (iter_num > 0):
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "LLMtrain/iter": iter_num,
                "LLMtrain/train/loss": losses['train'],
                "LLMtrain/val/loss": losses['val'],
                "LLMtrain/lr": lr,
                "LLMtrain/mfu": running_mfu*100, # convert to percentage
            })
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
    elif iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "LLMtrain/iter": iter_num,
                "LLMtrain/train/loss": losses['train'],
                "LLMtrain/val/loss": losses['val'],
                "LLMtrain/lr": lr,
                "LLMtrain/mfu": running_mfu*100, # convert to percentage
            })
    
    # Evaluate CFG accuracy
    if cfg is not None and iter_num % accuracy_eval_iters == 0 and iter_num > 0 and master_process:
        print(f"Evaluating CFG accuracy at step {iter_num}...")
        accuracy_results = evaluate_cfg_accuracy(
            raw_model, cfg, device, 
            num_samples=accuracy_eval_samples, 
            temperature=1.0,
            batch_size=batch_size,
            save_samples=save_accuracy_samples
        )
        
        # Extract accuracies
        zero_cut_acc = accuracy_results['zero_cut']
        half_cut_acc = accuracy_results['half_cut']
        
        # Update accuracy history
        accuracy_history['iterations'].append(iter_num)
        accuracy_history['zero_cut'].append(zero_cut_acc)
        accuracy_history['half_cut'].append(half_cut_acc)
        
        print(f"CFG accuracy - cut=0: {zero_cut_acc*100:.2f}%, half-cut: {half_cut_acc*100:.2f}%")
        
        if wandb_log:
            wandb.log({
                "LLMtrain/iter": iter_num,
                "LLMtrain/cfg_accuracy_zero_cut": zero_cut_acc,
                "LLMtrain/cfg_accuracy_half_cut": half_cut_acc,
            })
        
        # Save accuracy log file with samples
        if save_accuracy_samples:
            save_cfg_accuracy_log(out_dir, iter_num, accuracy_results, cfg)
    
    if iter_num == 0 and eval_only:
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
        X, Y = get_batch('train')
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
            "LLMtrain/loss (train)": lossf,
            "LLMtrain/lr": lr
        }
        
        # Add MTL individual losses to logging
        if multi_token_loss == "shared_heads" and mtl_length > 1 and 'current_individual_losses' in locals():
            for i, ind_loss in enumerate(current_individual_losses):
                token_name = "next" if i == 0 else f"next+{i}"
                log_message += f", {token_name}: {ind_loss:.4f}"
                wandb_log_dict[f"LLMtrain/loss_{token_name}"] = ind_loss
        
        print(log_message)

        if wandb_log:
            wandb.log(wandb_log_dict)
    
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
    