Analyze these files and produce GraphNode and GraphEdge objects.
Project root: `/Users/abhi/Desktop/Masters/ModelPlanning`
Project: `ModelPlanning` — research codebase that builds CFG, path-finding and OpenWebText datasets, trains GPT-style models with next-token (NTP) or multi-token (MTP/"mtl") loss, trains VQ-VAEs on hidden representations, and computes mutual information between learned codes.
Languages: json, markdown, python, txt
Skill directory (for bundled scripts): `/Users/abhi/.claude/plugins/cache/understand-anything/understand-anything/2.9.7/skills/understand`
Data dir (UA_DIR): `/Users/abhi/Desktop/Masters/ModelPlanning/.ua`

This dispatch covers 2 original batch(es): [5, 12] of 12. Process each batch SEPARATELY (its own input JSON, temp files keyed by batchIndex) and write one output per batchIndex: `.ua/intermediate/batch-<batchIndex>.json` (or `batch-<batchIndex>-part-<k>.json` in split mode). Never use any other filename. Verify each output exists before finishing.

**Additional context from main session:**
Project: `ModelPlanning`. Languages: json, markdown, python, txt.
> **Language directive**: Generate all textual content (summaries, descriptions, tags, titles, languageNotes, languageLesson) in **English**.

## Batch 5/12
Output: `.ua/intermediate/batch-5.json`

Pre-resolved import data for this batch (use directly — do NOT re-resolve imports from source):
```json
{"data/context_free_grammar/CFG_data_generation.py": [], "data/context_free_grammar/CFG_full_sequence_data_generation.py": ["data/context_free_grammar/CFG_data_generation.py"], "data/context_free_grammar/CFG_prefix_data_generation.py": ["data/context_free_grammar/CFG_data_generation.py"], "data/context_free_grammar/check_CGF_implementation.py": ["data/context_free_grammar/CFG_data_generation.py"]}
```

Cross-batch neighbors with their exported symbols:
```json
{"data/context_free_grammar/CFG_data_generation.py": [{"path": "llm_train.py", "batchIndex": 2, "symbols": ["get_batch", "estimate_loss"]}]}
```

Files (pass every entry to `batchFiles` with all four fields):
1. `data/context_free_grammar/CFG_data_generation.py` (567 lines, language: `python`, fileCategory: `code`)
2. `data/context_free_grammar/CFG_full_sequence_data_generation.py` (91 lines, language: `python`, fileCategory: `code`)
3. `data/context_free_grammar/CFG_prefix_data_generation.py` (128 lines, language: `python`, fileCategory: `code`)
4. `data/context_free_grammar/check_CGF_implementation.py` (117 lines, language: `python`, fileCategory: `code`)

## Batch 12/12
Output: `.ua/intermediate/batch-12.json`

Pre-resolved import data for this batch (use directly — do NOT re-resolve imports from source):
```json
{"NLP_openwebtext/configGPT3Xsmall.py": [], "PF_15XL/seed11_configs/final_config_path_finding_GPT3small_128_longer_train.py": [], "PF_15XL/seed11_configs/mtl_final_config_path_finding_GPT3small_128_longer_train.py": [], "PF_long1decoy_easier/seed11_configs/final_config_path_finding_GPT3small_128_longer_train_seed11.py": [], "PF_long1decoy_easier/seed11_configs/mtl_final_config_path_finding_GPT3small_128_longer_train_seed11.py": [], "README.txt": [], "data/context_free_grammar/peeking_dataset.py": [], "data/openwentext/prepare.py": [], "data/openwentext/readme.md": [], "data/path_finding/path_finding_generate.py": [], "data/path_finding/path_finding_generate_multi_cpu.py": [], "data/path_finding/visualize_saved_graphs.py": ["data/path_finding/path_finding_generate.py"], "exp_cfg_s1444-64-_rd3456_rl23_4000k_mtl4_seed11/llmconfig_CFG_GPT3small.py": [], "exp_cfg_s1444-64-_rd3456_rl23_4000k_seed11/llmconfig_CFG_GPT3small.py": []}
```

Cross-batch neighbors with their exported symbols:
```json
{}
```

Files (pass every entry to `batchFiles` with all four fields):
1. `NLP_openwebtext/configGPT3Xsmall.py` (50 lines, language: `python`, fileCategory: `code`)
2. `PF_15XL/seed11_configs/final_config_path_finding_GPT3small_128_longer_train.py` (52 lines, language: `python`, fileCategory: `code`)
3. `PF_15XL/seed11_configs/mtl_final_config_path_finding_GPT3small_128_longer_train.py` (58 lines, language: `python`, fileCategory: `code`)
4. `PF_long1decoy_easier/seed11_configs/final_config_path_finding_GPT3small_128_longer_train_seed11.py` (55 lines, language: `python`, fileCategory: `code`)
5. `PF_long1decoy_easier/seed11_configs/mtl_final_config_path_finding_GPT3small_128_longer_train_seed11.py` (59 lines, language: `python`, fileCategory: `code`)
6. `README.txt` (72 lines, language: `txt`, fileCategory: `docs`)
7. `data/context_free_grammar/peeking_dataset.py` (105 lines, language: `python`, fileCategory: `code`)
8. `data/openwentext/prepare.py` (82 lines, language: `python`, fileCategory: `code`)
9. `data/openwentext/readme.md` (15 lines, language: `markdown`, fileCategory: `docs`)
10. `data/path_finding/path_finding_generate.py` (480 lines, language: `python`, fileCategory: `code`)
11. `data/path_finding/path_finding_generate_multi_cpu.py` (702 lines, language: `python`, fileCategory: `code`)
12. `data/path_finding/visualize_saved_graphs.py` (401 lines, language: `python`, fileCategory: `code`)
13. `exp_cfg_s1444-64-_rd3456_rl23_4000k_mtl4_seed11/llmconfig_CFG_GPT3small.py` (42 lines, language: `python`, fileCategory: `code`)
14. `exp_cfg_s1444-64-_rd3456_rl23_4000k_seed11/llmconfig_CFG_GPT3small.py` (37 lines, language: `python`, fileCategory: `code`)
