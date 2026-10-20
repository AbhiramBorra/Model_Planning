# this is like Chinchilla augmented GPT3 small config.

init_from = 'scratch' # 'scratch' or 'resume'

block_type = "vanilla" # PP or vanilla

# model architecture is the same as GPT3 small 
# 125M parameter model
n_layer = 8
n_head = 8
n_embd = 8*64

# batch size related things 
# -> GPT3 for models oround our size (gpt3 small, medium, large 125M-1B params) they use 0.5M tokens per batch and 2048 block size
# -> Chinchilla does not say much about batch size but they use block size of 2048
# -> In the transformer baselines of mamba paper as batch size of 0.5M tokens and block size of 2048
gradient_accumulation_steps = 16 *2 # used to simulate larger batch sizes 
batch_size = 8*8 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256 #1024
#batch size in num tokens = batch_size * gradient_accumulation_steps * block_size

batch_size_in_tokens = batch_size * gradient_accumulation_steps * block_size

# I specified leraning rate according to mamba and gpt3 papers
# I specified D according to Chinchilla and mamba papers
# !!!!!!!!!! YOU NEED TO SPECIFY THESE TWO THINGS !!!!!!!!!! the rest is derived from these two.
D = 2.5 # training horizon in terms of billion num tokens. CHINCHILLA SAYS CHOOSE D = 20 * N
learning_rate = 6e-4 # max learning rate. -> choosen according to gpt3 and mamba papers
# !!!!!!!!!! YOU NEED TO SPECIFY THESE TWO THINGS !!!!!!!!!!

max_iters = int((D *1e9) // batch_size_in_tokens + 1)
# learning rate related things
# -> Chinchilla says max_lr/min_lr = 10 and lr_decay iters = max_iters
min_lr = learning_rate/10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
decay_lr = True 
warmup_iters = int(max_iters*0.05) # how many steps to warm up for (it is similar to how they done it in chinchilla paper, but it is not mentioned explictly as something with great importance)
lr_decay_iters = max_iters #~= max_iters per Chinchilla

# wandb things
save_interval = 2000
eval_interval = max_iters + 10 # so no eval during training
log_interval = 1
eval_iters = 200

# wandb logging
run_name = "llm_train_{}layers_{}heads_{}embd_{}batchsize_{}blocksize".format(n_layer, n_head, n_embd, batch_size, block_size)
wandb_log = True # disabled by default
wandb_entity = 'llm_analysis'
wandb_project = 'NLP_openwebtext'
wandb_run_name = run_name # 'run' + str(time.time())
out_dir = wandb_project+'/'+'LLMout'+'/'+run_name