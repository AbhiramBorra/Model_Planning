To generate the datasets:

CFG:
cd data/context_free_grammar
python CFG_data_generation.py
python CFG_full_sequence_data_generation.py
python CFG_prefix_data_generation.py

openwebtext
python data/openwebtext/prepare.py

Path Finding:
python data/path_finding/path_finding_generate_multi_cpu --config [separately run for all config files in the folder]

-------------------

To run a sample CFG Experiment with NTP loss (for MTP loss, use the same paths/files with _mtl4 version)

train model:
python llm_train.py exp_cfg_s1444-64-_rd3456_rl23_4000k_seed11/llmconfig_CFG_GPT3small.py

train VQVAEs:
python train_minimal_vqvae1.py --config exp_cfg_s1444-64-_rd3456_rl23_4000k_seed11/vqvae1_config.json
python train_minimal_vqvae2.py --config exp_cfg_s1444-64-_rd3456_rl23_4000k_seed11/vqvae2_config.json
python train_minimal_vqvae_single.py --config exp_cfg_s1444-64-_rd3456_rl23_4000k_seed11/vqvae_single_config.json

calculate MI:
python mi_calc_minimal.py [please adjust the arguments depending on where you save the previous runs outs]

-----------------
To run a sample Path Finding Experiment with short task (PF-Short) and NTP loss 
(for MTP loss, use the same paths/files with mtl4_ version)

train model:
python path_finding_llm_train.py PF_15XL/seed11_configs/final_config_path_finding_GPT3small_128_longer_train.py

train VQVAEs:
python path_finding_vqvae1_train.py --config PF_15XL/seed11_configs/vqvae1_config_seed11.json
python path_finding_vqvae2_train.py --config PF_15XL/seed11_configs/vqvae2_config_seed11.json
python path_finding_vqvae_single_train.py --config PF_15XL/seed11_configs/vqvae_single_config_seed11.json
python path_vqvae_train.py --config PF_15XL/seed11_configs/vqvae_path_config_seed11.json

calculate MI:
python path_finding_mi_calc.py [please adjust the arguments depending on where you save the previous runs outs]
python mi_calc_full_path.py [please adjust the arguments depending on where you save the previous runs outs]

-----------------
To run a sample Path Finding Experiment with long task (PF-Long) and NTP loss 
(for MTP loss, use the same paths/files with mtl_ version)

train model:
python path_finding_llm_train.py PF_long1decoy_easier/seed11_configs/final_config_path_finding_GPT3small_128_longer_train_seed11.py

train VQVAEs:
python path_finding_vqvae61_train.py --config PF_long1decoy_easier/seed11_configs/vqvae61_config_seed11.json
python path_finding_vqvae_single_train.py --config PF_long1decoy_easier/seed11_configs/vqvae_single_config_seed11.json

calculate MI:
python path_finding_mi_calc.py [please adjust the arguments depending on where you save the previous runs outs]
python mi_calc_full_path_long2_61.py [please adjust the arguments depending on where you save the previous runs outs]

-----------------
To run a sample openwebtext NLP Experiment with long task (PF-Long) and NTP loss 

train model:
python NLP_llm_train.py NLP_openwebtext/configGPT3Xsmall.py

train VQVAEs:
python NLP_vqvae_layer_and_block_train.py --config NLP_openwebtext/layer_and_block_config_256.json
python NLP_vqvae_last_train.py --config NLP_openwebtext/VQVAE_last_train_config07s.json

calculate MI:
python NLP_mi_calc_v6\ 2.py --config NLP_openwebtext/mi_calc_v6_config\ 2.json