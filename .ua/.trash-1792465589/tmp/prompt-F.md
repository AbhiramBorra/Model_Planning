Analyze these files and produce GraphNode and GraphEdge objects.
Project root: `/Users/abhi/Desktop/Masters/ModelPlanning`
Project: `ModelPlanning` — research codebase that builds CFG, path-finding and OpenWebText datasets, trains GPT-style models with next-token (NTP) or multi-token (MTP/"mtl") loss, trains VQ-VAEs on hidden representations, and computes mutual information between learned codes.
Languages: json, markdown, python, txt
Skill directory (for bundled scripts): `/Users/abhi/.claude/plugins/cache/understand-anything/understand-anything/2.9.7/skills/understand`
Data dir (UA_DIR): `/Users/abhi/Desktop/Masters/ModelPlanning/.ua`

This dispatch covers 6 original batch(es): [6, 7, 8, 9, 10, 11] of 12. Process each batch SEPARATELY (its own input JSON, temp files keyed by batchIndex) and write one output per batchIndex: `.ua/intermediate/batch-<batchIndex>.json` (or `batch-<batchIndex>-part-<k>.json` in split mode). Never use any other filename. Verify each output exists before finishing.

**Additional context from main session:**
Project: `ModelPlanning`. Languages: json, markdown, python, txt.
> **Language directive**: Generate all textual content (summaries, descriptions, tags, titles, languageNotes, languageLesson) in **English**.

## Batch 6/12
Output: `.ua/intermediate/batch-6.json`

Pre-resolved import data for this batch (use directly — do NOT re-resolve imports from source):
```json
{"NLP_openwebtext/VQVAE_last_train_config07s.json": [], "NLP_openwebtext/layer_and_block_config_256.json": [], "NLP_openwebtext/mi_calc_v6_config 2.json": []}
```

Cross-batch neighbors with their exported symbols:
```json
{}
```

Files (pass every entry to `batchFiles` with all four fields):
1. `NLP_openwebtext/VQVAE_last_train_config07s.json` (65 lines, language: `json`, fileCategory: `config`)
2. `NLP_openwebtext/layer_and_block_config_256.json` (129 lines, language: `json`, fileCategory: `config`)
3. `NLP_openwebtext/mi_calc_v6_config 2.json` (20 lines, language: `json`, fileCategory: `config`)

## Batch 7/12
Output: `.ua/intermediate/batch-7.json`

Pre-resolved import data for this batch (use directly — do NOT re-resolve imports from source):
```json
{"PF_15XL/seed11_configs/mtl_vqvae1_config_seed11.json": [], "PF_15XL/seed11_configs/mtl_vqvae2_config_seed11.json": [], "PF_15XL/seed11_configs/mtl_vqvae_single_config_seed11.json": [], "PF_15XL/seed11_configs/vqvae1_config_seed11.json": [], "PF_15XL/seed11_configs/vqvae2_config_seed11.json": [], "PF_15XL/seed11_configs/vqvae_path_config_seed11.json": [], "PF_15XL/seed11_configs/vqvae_single_config_seed11.json": []}
```

Cross-batch neighbors with their exported symbols:
```json
{}
```

Files (pass every entry to `batchFiles` with all four fields):
1. `PF_15XL/seed11_configs/mtl_vqvae1_config_seed11.json` (76 lines, language: `json`, fileCategory: `config`)
2. `PF_15XL/seed11_configs/mtl_vqvae2_config_seed11.json` (84 lines, language: `json`, fileCategory: `config`)
3. `PF_15XL/seed11_configs/mtl_vqvae_single_config_seed11.json` (58 lines, language: `json`, fileCategory: `config`)
4. `PF_15XL/seed11_configs/vqvae1_config_seed11.json` (76 lines, language: `json`, fileCategory: `config`)
5. `PF_15XL/seed11_configs/vqvae2_config_seed11.json` (84 lines, language: `json`, fileCategory: `config`)
6. `PF_15XL/seed11_configs/vqvae_path_config_seed11.json` (63 lines, language: `json`, fileCategory: `config`)
7. `PF_15XL/seed11_configs/vqvae_single_config_seed11.json` (58 lines, language: `json`, fileCategory: `config`)

## Batch 8/12
Output: `.ua/intermediate/batch-8.json`

Pre-resolved import data for this batch (use directly — do NOT re-resolve imports from source):
```json
{"PF_long1decoy_easier/seed11_configs/mtl_vqvae1_config_seed11.json": [], "PF_long1decoy_easier/seed11_configs/mtl_vqvae2_config_seed11.json": [], "PF_long1decoy_easier/seed11_configs/mtl_vqvae61_config_seed11.json": [], "PF_long1decoy_easier/seed11_configs/mtl_vqvae_single_config_seed11.json": [], "PF_long1decoy_easier/seed11_configs/vqvae1_config_seed11.json": [], "PF_long1decoy_easier/seed11_configs/vqvae2_config_seed11.json": [], "PF_long1decoy_easier/seed11_configs/vqvae61_config_seed11.json": [], "PF_long1decoy_easier/seed11_configs/vqvae_path_config_seed11.json": [], "PF_long1decoy_easier/seed11_configs/vqvae_single_config_seed11.json": []}
```

Cross-batch neighbors with their exported symbols:
```json
{}
```

Files (pass every entry to `batchFiles` with all four fields):
1. `PF_long1decoy_easier/seed11_configs/mtl_vqvae1_config_seed11.json` (78 lines, language: `json`, fileCategory: `config`)
2. `PF_long1decoy_easier/seed11_configs/mtl_vqvae2_config_seed11.json` (79 lines, language: `json`, fileCategory: `config`)
3. `PF_long1decoy_easier/seed11_configs/mtl_vqvae61_config_seed11.json` (80 lines, language: `json`, fileCategory: `config`)
4. `PF_long1decoy_easier/seed11_configs/mtl_vqvae_single_config_seed11.json` (53 lines, language: `json`, fileCategory: `config`)
5. `PF_long1decoy_easier/seed11_configs/vqvae1_config_seed11.json` (78 lines, language: `json`, fileCategory: `config`)
6. `PF_long1decoy_easier/seed11_configs/vqvae2_config_seed11.json` (79 lines, language: `json`, fileCategory: `config`)
7. `PF_long1decoy_easier/seed11_configs/vqvae61_config_seed11.json` (80 lines, language: `json`, fileCategory: `config`)
8. `PF_long1decoy_easier/seed11_configs/vqvae_path_config_seed11.json` (59 lines, language: `json`, fileCategory: `config`)
9. `PF_long1decoy_easier/seed11_configs/vqvae_single_config_seed11.json` (53 lines, language: `json`, fileCategory: `config`)

## Batch 9/12
Output: `.ua/intermediate/batch-9.json`

Pre-resolved import data for this batch (use directly — do NOT re-resolve imports from source):
```json
{"data/path_finding/PF_15XL_config.json": [], "data/path_finding/PF_15XL_config_mi.json": [], "data/path_finding/PF_15XL_config_train.json": [], "data/path_finding/PF_15XL_config_train_small.json": [], "data/path_finding/PF_15XL_config_train_vocab.json": [], "data/path_finding/PF_15XL_config_vqvae.json": [], "data/path_finding/PF_long1decoy_easier_config.json": [], "data/path_finding/PF_long1decoy_easier_config_mi.json": [], "data/path_finding/PF_long1decoy_easier_config_observe.json": [], "data/path_finding/PF_long1decoy_easier_config_train.json": [], "data/path_finding/PF_long1decoy_easier_config_train_small.json": [], "data/path_finding/PF_long1decoy_easier_config_train_vocab.json": [], "data/path_finding/PF_long1decoy_easier_config_val.json": [], "data/path_finding/PF_long1decoy_easier_config_vqvae.json": []}
```

Cross-batch neighbors with their exported symbols:
```json
{}
```

Files (pass every entry to `batchFiles` with all four fields):
1. `data/path_finding/PF_15XL_config.json` (15 lines, language: `json`, fileCategory: `config`)
2. `data/path_finding/PF_15XL_config_mi.json` (15 lines, language: `json`, fileCategory: `config`)
3. `data/path_finding/PF_15XL_config_train.json` (15 lines, language: `json`, fileCategory: `config`)
4. `data/path_finding/PF_15XL_config_train_small.json` (15 lines, language: `json`, fileCategory: `config`)
5. `data/path_finding/PF_15XL_config_train_vocab.json` (15 lines, language: `json`, fileCategory: `config`)
6. `data/path_finding/PF_15XL_config_vqvae.json` (15 lines, language: `json`, fileCategory: `config`)
7. `data/path_finding/PF_long1decoy_easier_config.json` (15 lines, language: `json`, fileCategory: `config`)
8. `data/path_finding/PF_long1decoy_easier_config_mi.json` (15 lines, language: `json`, fileCategory: `config`)
9. `data/path_finding/PF_long1decoy_easier_config_observe.json` (15 lines, language: `json`, fileCategory: `config`)
10. `data/path_finding/PF_long1decoy_easier_config_train.json` (15 lines, language: `json`, fileCategory: `config`)
11. `data/path_finding/PF_long1decoy_easier_config_train_small.json` (15 lines, language: `json`, fileCategory: `config`)
12. `data/path_finding/PF_long1decoy_easier_config_train_vocab.json` (15 lines, language: `json`, fileCategory: `config`)
13. `data/path_finding/PF_long1decoy_easier_config_val.json` (15 lines, language: `json`, fileCategory: `config`)
14. `data/path_finding/PF_long1decoy_easier_config_vqvae.json` (15 lines, language: `json`, fileCategory: `config`)

## Batch 10/12
Output: `.ua/intermediate/batch-10.json`

Pre-resolved import data for this batch (use directly — do NOT re-resolve imports from source):
```json
{"exp_cfg_s1444-64-_rd3456_rl23_4000k_mtl4_seed11/vqvae1_config.json": [], "exp_cfg_s1444-64-_rd3456_rl23_4000k_mtl4_seed11/vqvae2_config.json": [], "exp_cfg_s1444-64-_rd3456_rl23_4000k_mtl4_seed11/vqvae_single_config.json": []}
```

Cross-batch neighbors with their exported symbols:
```json
{}
```

Files (pass every entry to `batchFiles` with all four fields):
1. `exp_cfg_s1444-64-_rd3456_rl23_4000k_mtl4_seed11/vqvae1_config.json` (75 lines, language: `json`, fileCategory: `config`)
2. `exp_cfg_s1444-64-_rd3456_rl23_4000k_mtl4_seed11/vqvae2_config.json` (84 lines, language: `json`, fileCategory: `config`)
3. `exp_cfg_s1444-64-_rd3456_rl23_4000k_mtl4_seed11/vqvae_single_config.json` (56 lines, language: `json`, fileCategory: `config`)

## Batch 11/12
Output: `.ua/intermediate/batch-11.json`

Pre-resolved import data for this batch (use directly — do NOT re-resolve imports from source):
```json
{"exp_cfg_s1444-64-_rd3456_rl23_4000k_seed11/vqvae1_config.json": [], "exp_cfg_s1444-64-_rd3456_rl23_4000k_seed11/vqvae2_config.json": [], "exp_cfg_s1444-64-_rd3456_rl23_4000k_seed11/vqvae_single_config.json": []}
```

Cross-batch neighbors with their exported symbols:
```json
{}
```

Files (pass every entry to `batchFiles` with all four fields):
1. `exp_cfg_s1444-64-_rd3456_rl23_4000k_seed11/vqvae1_config.json` (75 lines, language: `json`, fileCategory: `config`)
2. `exp_cfg_s1444-64-_rd3456_rl23_4000k_seed11/vqvae2_config.json` (84 lines, language: `json`, fileCategory: `config`)
3. `exp_cfg_s1444-64-_rd3456_rl23_4000k_seed11/vqvae_single_config.json` (56 lines, language: `json`, fileCategory: `config`)
