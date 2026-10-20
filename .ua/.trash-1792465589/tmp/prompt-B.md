Analyze these files and produce GraphNode and GraphEdge objects.
Project root: `/Users/abhi/Desktop/Masters/ModelPlanning`
Project: `ModelPlanning` — research codebase that builds CFG, path-finding and OpenWebText datasets, trains GPT-style models with next-token (NTP) or multi-token (MTP/"mtl") loss, trains VQ-VAEs on hidden representations, and computes mutual information between learned codes.
Languages: json, markdown, python, txt
Skill directory (for bundled scripts): `/Users/abhi/.claude/plugins/cache/understand-anything/understand-anything/2.9.7/skills/understand`
Data dir (UA_DIR): `/Users/abhi/Desktop/Masters/ModelPlanning/.ua`

This dispatch covers 1 original batch(es): [2] of 12. Process each batch SEPARATELY (its own input JSON, temp files keyed by batchIndex) and write one output per batchIndex: `.ua/intermediate/batch-<batchIndex>.json` (or `batch-<batchIndex>-part-<k>.json` in split mode). Never use any other filename. Verify each output exists before finishing.

**Additional context from main session:**
Project: `ModelPlanning`. Languages: json, markdown, python, txt.
> **Language directive**: Generate all textual content (summaries, descriptions, tags, titles, languageNotes, languageLesson) in **English**.

## Batch 2/12
Output: `.ua/intermediate/batch-2.json`

Pre-resolved import data for this batch (use directly — do NOT re-resolve imports from source):
```json
{"NLP_llm_train.py": ["llm_train_utils.py", "model.py"], "NLP_vqvae_layer_and_block_train.py": ["minimal_VQVAEs.py", "model.py", "vqvae_utils.py"], "llm_train.py": ["data/context_free_grammar/CFG_data_generation.py", "llm_train_utils.py", "model.py"], "llm_train_utils.py": [], "model.py": [], "path_finding_llm_train.py": ["llm_train_utils.py", "model.py", "path_finding_llm_utilities.py"]}
```

Cross-batch neighbors with their exported symbols:
```json
{"NLP_vqvae_layer_and_block_train.py": [{"path": "minimal_VQVAEs.py", "batchIndex": 1, "symbols": ["ResidualBlock", "CustomSequential", "VQVAE_single", "VQVAE1", "Encoder1", "Decoder1", "Encoder3", "Decoder3", "VQVAE3", "VQVAE2", "Encoder2", "Decoder2", "VQVAELastToken", "VQVAE3_Modified", "Path_Encoder", "Path_Decoder", "Path_VQVAE", "Encoder3v2", "Decoder3v2", "VQVAE3v2", "Encoder_layer", "Decoder_layer", "VQVAE_layer", "Encoder61", "Decoder61", "VQVAE61"]}, {"path": "vqvae_utils.py", "batchIndex": 3, "symbols": ["get_logger", "configure_optimizers", "nanoLLM", "get_lr", "get_beta", "compute_masked_nrmse", "compute_masked_nrmse_per_vector", "compute_masked_nrmse_per_element", "checkModelLoadCorrect", "CFGDataset", "create_cfg_dataloader", "convert_left_to_right_padding", "convert_right_to_left_padding", "test_padding_conversions", "path_finding_generate", "path_finding_generate_with_hidden_states", "log_and_plot_usage_for_model"]}], "llm_train.py": [{"path": "data/context_free_grammar/CFG_data_generation.py", "batchIndex": 5, "symbols": ["CFG", "plot_simple_tree", "main"]}], "model.py": [{"path": "NLP_mi_calc_v6 2.py", "batchIndex": 1, "symbols": ["load_state_dict_remove_ddp_prefix", "build_vqvae3v2_from_config", "compute_vq3v2_index", "compute_vq3v2_index_batch", "get_batch", "save_streaming_info_counts", "collect_block_mi", "load_vqvae_last_token_model", "sync_and_update_streaming_info", "calculate_and_report_metrics", "main"]}, {"path": "NLP_vqvae_last_train.py", "batchIndex": 3, "symbols": ["get_batch", "get_lr", "process_hidden_states_for_vqvae_last"]}, {"path": "mi_calc_full_path.py", "batchIndex": 1, "symbols": ["normalize_first_stage", "denormalize_first_stage", "sync_and_update_streaming_info", "calculate_and_report_metrics", "create_plots_and_save"]}, {"path": "mi_calc_full_path_long2_61.py", "batchIndex": 1, "symbols": ["sync_and_update_streaming_info", "calculate_and_report_metrics", "create_plots_and_save"]}, {"path": "path_finding_mi_calc.py", "batchIndex": 1, "symbols": ["normalize_first_stage", "denormalize_first_stage", "normalize_single_stage", "denormalize_first_stage", "sync_and_update_streaming_info", "calculate_and_report_metrics", "create_plots_and_save"]}, {"path": "path_finding_mi_calc_61.py", "batchIndex": 1, "symbols": ["normalize_single_stage", "denormalize_first_stage", "sync_and_update_streaming_info", "calculate_and_report_metrics", "create_plots_and_save"]}, {"path": "path_finding_vqvae1_train.py", "batchIndex": 4, "symbols": []}, {"path": "path_finding_vqvae2_train.py", "batchIndex": 4, "symbols": ["normalize_first_stage", "denormalize_first_stage", "count_params"]}, {"path": "path_finding_vqvae61_train.py", "batchIndex": 3, "symbols": ["count_params"]}, {"path": "path_finding_vqvae_single_train.py", "batchIndex": 4, "symbols": []}, {"path": "path_vqvae_train.py", "batchIndex": 4, "symbols": []}, {"path": "vqvae_utils.py", "batchIndex": 3, "symbols": ["get_logger", "configure_optimizers", "nanoLLM", "get_lr", "get_beta", "compute_masked_nrmse", "compute_masked_nrmse_per_vector", "compute_masked_nrmse_per_element", "checkModelLoadCorrect", "CFGDataset", "create_cfg_dataloader", "convert_left_to_right_padding", "convert_right_to_left_padding", "test_padding_conversions", "path_finding_generate", "path_finding_generate_with_hidden_states", "log_and_plot_usage_for_model"]}], "path_finding_llm_train.py": [{"path": "path_finding_llm_utilities.py", "batchIndex": 4, "symbols": ["PathFindingDataset", "custom_collate_fn", "compute_loss", "crop_prompt_and_attention_mask", "skip_time_steps"]}]}
```

Files (pass every entry to `batchFiles` with all four fields):
1. `NLP_llm_train.py` (343 lines, language: `python`, fileCategory: `code`)
2. `NLP_vqvae_layer_and_block_train.py` (1009 lines, language: `python`, fileCategory: `code`)
3. `llm_train.py` (573 lines, language: `python`, fileCategory: `code`)
4. `llm_train_utils.py` (355 lines, language: `python`, fileCategory: `code`)
5. `model.py` (555 lines, language: `python`, fileCategory: `code`)
6. `path_finding_llm_train.py` (960 lines, language: `python`, fileCategory: `code`)
