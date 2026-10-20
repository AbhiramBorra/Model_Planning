save_interval = 10000
eval_interval = 500
log_interval = 1
eval_iters = 200
# CFG accuracy evaluation
accuracy_eval_iters = 10000
accuracy_eval_samples = 120

run_name = "LLMtrain_CFG_GPT" # name of the run, used for wandb and output directory
base_seed = 11

cfg_instance = 'cfg_s1444-64-_rd3456_rl23_4000k'
init_from = 'scratch' # 'scratch' or 'resume'
# wandb logging
wandb_log = True # disabled by default
wandb_run_name = run_name # 'run' + str(time.time())
wandb_entity = 'llm_analysis' 

# Multi-token loss (MTL) settings
multi_token_loss = "shared_heads"  # Enable MTL with shared heads approach
mtl_length = 4                     # Predict next 4 tokens (next, next+1, next+2, next+3)
mtl_shared_head_nb_blocks = 2      # Use last 2 blocks as shared part
out_dir = f'exp_{cfg_instance}_mtl4_seed11/LLMout/'+run_name
wandb_project = f'fullexp_{cfg_instance}_mtl4_seed11'

dataset = f'context_free_grammar/{cfg_instance}' 
gradient_accumulation_steps = 2  # used to simulate larger batch sizes
batch_size = 16 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 512
# model
n_layer = 12
n_head = 12
n_embd = 768

# adamw optimizer
learning_rate = 6e-3 # max learning rate
max_iters = 200000 # total number of training iterations
decay_lr = True # whether to decay the learning rate
warmup_iters = 200 # how many steps to warm up for
lr_decay_iters = 25000 # should be ~= max_iters per Chinchilla
min_lr = 6e-4 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

compile = True # use PyTorch 2.0 to compile the model to be faster