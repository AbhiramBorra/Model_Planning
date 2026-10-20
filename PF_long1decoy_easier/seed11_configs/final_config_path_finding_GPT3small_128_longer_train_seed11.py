init_from = 'scratch' # 'scratch' or 'resume'
# model
n_layer = 12
n_head = 12
n_embd = 768

base_seed = 11

padding_avare_RoPE = False
# batch size related things 
gradient_accumulation_steps = 16*2  # used to simulate larger batch sizes
batch_size = 8*8*8 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 128
num_hyphens = 0

batch_size_in_tokens = batch_size * gradient_accumulation_steps * block_size

# I specified leraning rate according to mamba and gpt3 papers
# I specified D according to Chinchilla and mamba papers
# !!!!!!!!!! YOU NEED TO SPECIFY THESE TWO THINGS !!!!!!!!!! the rest is derived from these two.
D = 2.5*8 # 2.5 # training horizon in terms of billion num tokens. CHINCHILLA SAYS CHOOSE D = 20 * N
learning_rate = 6e-4 # max learning rate. -> choosen according to gpt3 and mamba papers
# !!!!!!!!!! YOU NEED TO SPECIFY THESE TWO THINGS !!!!!!!!!!

max_iters = 2000 #int((D *1e9) // batch_size_in_tokens + 1)
max_iters_original = int((D *1e9) // batch_size_in_tokens + 1)
decay_lr = True # whether to decay the learning rate
min_lr = learning_rate/10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
warmup_iters = int(max_iters_original*0.05) # how many steps to warm up for (it is similar to how they done it in chinchilla paper, but it is not mentioned explictly as something with great importance)
lr_decay_iters = max_iters_original #~= max_iters per Chinchilla

# wandb things
save_interval = max_iters//10
eval_interval = max_iters//50 
log_interval = 1
eval_iters = 1024
accuracy_eval_samples = 1024


the_name = 'long1decoy_easier' # !!!!!!!!!! YOU NEED TO SPECIFY THIS !!!!!!!!!!

dataset_dir = 'PF_'+the_name + '/samples' 
num_samples = batch_size*gradient_accumulation_steps*max_iters_original
run_name = f"FINAL_{the_name}_block_size_{block_size}_num_samples_{num_samples}_padding_avare_{padding_avare_RoPE}_seed{base_seed}" # name of the run, used for wandb and output directory
# wandb logging
wandb_log = True 
wandb_project = 'PF_long1decoy_easier_Experiments'
wandb_run_name = run_name # 'run' + str(time.time())
wandb_entity = 'llm_analysis' 
out_dir = 'PF_'+the_name + '/LLMout/'+run_name





