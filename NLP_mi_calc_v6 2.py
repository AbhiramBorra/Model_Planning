

import argparse
import os
import json
import logging
import math
import time
import gc
import pickle
from collections import Counter, defaultdict
from contextlib import nullcontext
from typing import Dict, Any, Tuple, Optional

import torch
import numpy as np
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from minimal_VQVAEs import Encoder3v2, Decoder3v2, VQVAE3v2, VQVAELastToken
from model import GPTConfig, GPT
from information_theory_utils import StreamingInfo

logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    force=True,
)


def load_state_dict_remove_ddp_prefix(model: torch.nn.Module, state_dict: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Load a checkpoint state_dict robustly by:
    - Removing DistributedDataParallel 'module.' prefixes
    - Adapting decoder projection keys when tied-weight checkpoints are used
    - Handling historical orientation/shape differences for proj weights and biases
    - Ignoring extraneous buffers (e.g. precomputed normalization values)

    Returns (missing_keys, unexpected_keys) for visibility.
    """

    unwanted_prefix = 'module.'

    # 1) Strip DDP prefix
    cleaned_state: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        cleaned_key = k[len(unwanted_prefix):] if k.startswith(unwanted_prefix) else k
        cleaned_state[cleaned_key] = v

    # 2) Adapt keys/shapes to current model definition
    model_state = model.state_dict()
    adapted: Dict[str, torch.Tensor] = {}

    decoder_keys = [k for k in model_state.keys() if 'decoder' in k and 'proj' in k]
    logging.info(f"Model decoder projection keys: {decoder_keys}")

    for k, v in list(cleaned_state.items()):
        if k == 'normalization_values' and 'normalization_values' not in model_state:
            continue

        if k == 'decoder.projbias':
            if 'decoder.projbias' in model_state:
                if isinstance(v, torch.Tensor) and model_state['decoder.projbias'].shape == v.shape:
                    adapted['decoder.projbias'] = v
                    logging.info(f"Using decoder.projbias for tied weights (shape: {v.shape})")
                else:
                    logging.warning(
                        f"Shape mismatch for decoder.projbias: checkpoint {tuple(v.shape)} vs model {tuple(model_state['decoder.projbias'].shape)}"
                    )
            elif 'decoder.proj.bias' in model_state:
                if isinstance(v, torch.Tensor) and model_state['decoder.proj.bias'].shape == v.shape:
                    adapted['decoder.proj.bias'] = v
                    logging.info(f"Mapped decoder.projbias -> decoder.proj.bias (shape: {v.shape})")
                else:
                    logging.warning(
                        f"Shape mismatch for decoder.proj.bias: checkpoint {tuple(v.shape)} vs model {tuple(model_state['decoder.proj.bias'].shape)}"
                    )
            else:
                logging.warning('Neither decoder.projbias nor decoder.proj.bias found in model state; dropping key')
            continue

        adapted[k] = v

    if 'decoder.proj.weight' in adapted and 'decoder.proj.weight' in model_state:
        w = adapted['decoder.proj.weight']
        try:
            expected_shape = model_state['decoder.proj.weight'].shape
            if hasattr(w, 'shape') and w.shape != expected_shape:
                if w.ndim == 2 and w.t().shape == expected_shape:
                    logging.warning('Transposing decoder.proj.weight to match model shape')
                    adapted['decoder.proj.weight'] = w.t()
                else:
                    logging.warning(
                        f"Replacing mismatched key 'decoder.proj.weight' with model init: ckpt {tuple(w.shape)} vs model {tuple(expected_shape)}"
                    )
                    adapted['decoder.proj.weight'] = model_state['decoder.proj.weight']
        except Exception:
            adapted['decoder.proj.weight'] = model_state['decoder.proj.weight']

    if 'decoder.proj.bias' in adapted and 'decoder.proj.bias' in model_state:
        b = adapted['decoder.proj.bias']
        expected_b_shape = model_state['decoder.proj.bias'].shape
        if hasattr(b, 'shape') and b.shape != expected_b_shape:
            logging.warning(
                f"Replacing mismatched key 'decoder.proj.bias' with model init: ckpt {tuple(b.shape)} vs model {tuple(expected_b_shape)}"
            )
            adapted['decoder.proj.bias'] = model_state['decoder.proj.bias']

    # First try strict loading
    try:
        incompatible = model.load_state_dict(adapted, strict=True)
        try:
            missing_keys = list(incompatible.missing_keys)
            unexpected_keys = list(incompatible.unexpected_keys)
        except Exception:
            missing_keys, unexpected_keys = incompatible
    except RuntimeError as e:
        # If strict loading fails, try to handle codebook_counters specifically
        if "codebook_counters" in str(e):
            logging.warning("Strict loading failed due to codebook_counters, attempting flexible loading")
            # Remove codebook_counters from adapted dict and try again
            if 'codebook_counters' in adapted:
                del adapted['codebook_counters']
            incompatible = model.load_state_dict(adapted, strict=True)
            try:
                missing_keys = list(incompatible.missing_keys)
                unexpected_keys = list(incompatible.unexpected_keys)
            except Exception:
                missing_keys, unexpected_keys = incompatible
        else:
            # Re-raise if it's not a codebook_counters issue
            raise e

    if missing_keys:
        logging.warning(f'Missing keys when loading state dict: {missing_keys}')
    if unexpected_keys:
        logging.warning(f'Unexpected keys when loading state dict: {unexpected_keys}')
    return missing_keys, unexpected_keys


def build_vqvae3v2_from_config(cfg: Dict[str, Any], state_dict: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[VQVAE3v2, Dict[str, int]]:
    L = cfg.get('L')
    d = cfg.get('d')
    d2 = cfg.get('d2')
    T = cfg.get('T', cfg.get('T_max'))
    if any(v is None for v in [L, d, d2, T]):
        raise ValueError('VQVAE3v2 config must include L, d, d2, and T (or T_max).')

    n_ll = cfg.get('num_layers_layerwise_stage', 1)
    n_ag = cfg.get('num_layers_aggregate_stage', 3)
    cfg_ll = cfg.get('config_layerwise_stage', {})
    cfg_ag = cfg.get('config_aggregate_stage', {})

    tied_encoder_proj = None
    enc = Encoder3v2(
        L=L, d=d, d2=d2, T=T,
        num_layers_layerwise_stage=n_ll,
        num_layers_aggregate_stage=n_ag,
        config_layerwise_stage=cfg_ll,
        config_aggregate_stage=cfg_ag
    )
    if state_dict is not None:
        if any('decoder.projbias' in k for k in state_dict.keys()):
            logging.info('Detected tied encoder projection from checkpoint')
            tied_encoder_proj = enc.proj
    elif cfg_ll.get('tied_encoder_proj', False):
        tied_encoder_proj = enc.proj

    dec = Decoder3v2(
        L=L, d=d, d2=d2, T=T,
        num_layers_aggregate_stage=n_ag,
        num_layers_layerwise_stage=n_ll,
        config_aggregate_stage=cfg_ag,
        config_layerwise_stage=cfg_ll,
        tied_encoder_proj=tied_encoder_proj
    )

    model = VQVAE3v2(enc, dec, cfg)

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        for buffer in model.buffers():
            if buffer.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                buffer.zero_()

    logging.info('Initialized VQVAE3v2 parameters to zero before loading checkpoint weights')
    return model, {'L': L, 'd': d, 'd2': d2, 'T': T}


def compute_vq3v2_index(model: VQVAE3v2, x: torch.Tensor, padding_mask: torch.Tensor) -> int:
    if model.normalization_values is None:
        norm_vals = model._compute_normalization_values(x, padding_mask)
        model.register_buffer('normalization_values', norm_vals)
    x = model.normalize(x)
    z_e = model.encoder(x, padding_mask=padding_mask)
    if model.codebook is None:
        return 0
    distances = (
        torch.sum(z_e ** 2, dim=1, keepdim=True)
        + torch.sum(model.codebook.weight ** 2, dim=1)
        - 2 * torch.matmul(z_e, model.codebook.weight.t())
    )
    return torch.argmin(distances, dim=1).item()


def compute_vq3v2_index_batch(model: VQVAE3v2, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """
    Batch version of compute_vq3v2_index that processes multiple samples at once.
    
    Args:
        model: VQVAE3v2 model
        x: Input tensor of shape (batch_size, L, T, d) 
        padding_mask: Padding mask of shape (batch_size, T)
    
    Returns:
        Tensor of VQ indices of shape (batch_size,)
    """
    if model.normalization_values is None:
        # Use first sample to compute normalization values
        norm_vals = model._compute_normalization_values(x[:1], padding_mask[:1])
        model.register_buffer('normalization_values', norm_vals)
    
    x = model.normalize(x)
    z_e = model.encoder(x, padding_mask=padding_mask)  # Shape: (batch_size, d2)
    
    if model.codebook is None:
        return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    
    # Compute distances for all samples at once
    distances = (
        torch.sum(z_e ** 2, dim=1, keepdim=True)  # (batch_size, 1)
        + torch.sum(model.codebook.weight ** 2, dim=1)  # (codebook_size,)
        - 2 * torch.matmul(z_e, model.codebook.weight.t())  # (batch_size, codebook_size)
    )
    return torch.argmin(distances, dim=1)  # (batch_size,)


def get_batch(data_dir: str, meta_dtype: Any, seq_len: int, batch_size: int, device: str, split: str = 'train') -> torch.Tensor:
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=meta_dtype, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=meta_dtype, mode='r')
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + seq_len]).astype(np.int64)) for i in ix])
    if device == 'cuda' or device.startswith('cuda:'):
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x


def save_streaming_info_counts(smi: StreamingInfo, output_dir: str, suffix: str = "") -> Tuple[Optional[str], Optional[str]]:
    try:
        counts_dir = os.path.join(output_dir, 'counts')
        os.makedirs(counts_dir, exist_ok=True)
        count_data = {
            'metadata': {
                'variables': list(smi.variables) if hasattr(smi, 'variables') else [],
                'combos': list(smi.combos) if hasattr(smi, 'combos') else [],
                'N': smi.N,
                'base': smi.base,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'counts': {}
        }
        for combo, counter in smi.counts.items():
            serialized_counter: Dict[str, int] = {}
            for key, value in counter.items():
                if isinstance(key, tuple):
                    key_str = json.dumps(key)
                else:
                    key_str = str(key)
                serialized_counter[key_str] = value
            count_data['counts'][str(combo)] = serialized_counter
        counts_file = os.path.join(counts_dir, f'streaming_info_counts{suffix}.json')
        with open(counts_file, 'w') as f:
            json.dump(count_data, f, indent=2)
        pickle_file = os.path.join(counts_dir, f'streaming_info_counts{suffix}.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'variables': list(smi.variables) if hasattr(smi, 'variables') else [],
                'combos': list(smi.combos) if hasattr(smi, 'combos') else [],
                'N': smi.N,
                'base': smi.base,
                'counts': smi.counts,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f)
        logging.info(f"StreamingInfo count data saved to: {counts_file}")
        logging.info(f"StreamingInfo count data (pickle) saved to: {pickle_file}")
        return counts_file, pickle_file
    except Exception as exc:
        logging.error(f'Error saving StreamingInfo count data: {exc}')
        return None, None


def collect_block_mi(
    smi: StreamingInfo,
    token_variables: Tuple[str, ...],
    max_layer: int,
    num_blocks: int,
    delta_b: int
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for layer_idx in range(max_layer):
        layer_key = f'layer_{layer_idx}'
        results[layer_key] = {}
        for block_idx in range(num_blocks):
            start = block_idx * delta_b
            end = start + delta_b
            var_name = f'block_l{layer_idx}_s{start}_e{end}'
            if var_name in smi.variables:
                block_key = f'block_{start}_{end}'
                block_entry: Dict[str, Dict[str, float]] = {}
                for token_idx, token_var in enumerate(token_variables):
                    mi_val = smi.mutual_information(var_name, token_var)
                    block_entry[f'token_{token_idx}'] = {
                        'mi_block_token': mi_val,
                        'block_var': var_name,
                        'token_var': token_var,
                        'token_index': token_idx,
                    }
                results[layer_key][block_key] = block_entry
    return results


def load_vqvae_last_token_model(
    checkpoint_path: str,
    config_path: str,
    device: torch.device
) -> Tuple[VQVAELastToken, Dict[str, Any]]:
    logging.info(f'Loading VQVAELastToken config from {config_path}')
    if config_path.endswith('.pt'):
        tmp = torch.load(config_path, map_location='cpu')
        if 'vqvae_last_config' in tmp:
            vqvae_last_config = tmp['vqvae_last_config']
        elif 'config' in tmp:
            vqvae_last_config = tmp['config']
        else:
            raise ValueError('Cannot find vqvae_last_config in checkpoint')
        del tmp
        gc.collect()
    else:
        with open(config_path, 'r') as f:
            vqvae_last_full_config = json.load(f)
        
        # Extract vqvae_last_config from the nested structure
        if 'vqvae_last_config' in vqvae_last_full_config:
            vqvae_last_config = vqvae_last_full_config['vqvae_last_config']
        else:
            # If no nested structure, use the config directly
            vqvae_last_config = vqvae_last_full_config
    
    model = VQVAELastToken(
        input_dim=vqvae_last_config['input_dim'],
        hidden_dim=vqvae_last_config['hidden_dim'],
        codebook_size=vqvae_last_config['codebook_size'],
        beta=vqvae_last_config['beta'],
        config=vqvae_last_config
    )
    model.to(device).eval()
    
    logging.info(f'Loading VQVAELastToken checkpoint from {checkpoint_path}')
    vq_last_ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Debug: print available keys in checkpoint
    logging.info(f"VQVAELastToken checkpoint keys: {list(vq_last_ckpt.keys())}")
    
    last_state = None
    
    # Try to find the correct state dict
    for k in ['vqvae_last', 'vqvae_last_token', 'last_model_state_dict', 'models']:
        if k in vq_last_ckpt:
            if k == 'models':
                last_state = vq_last_ckpt['models'].get('vqvae_last_token', vq_last_ckpt['models'])
            else:
                last_state = vq_last_ckpt[k]
            break
    
    # Only use model_state_dict if we haven't found anything else and it looks valid
    if last_state is None and 'model_state_dict' in vq_last_ckpt:
        model_state = vq_last_ckpt['model_state_dict']
        # Check if this has the expected keys for VQVAELastToken
        if isinstance(model_state, dict) and len(model_state) > 0:
            # Check for typical VQVAELastToken patterns
            sample_keys = list(model_state.keys())[:5]
            logging.info(f"model_state_dict sample keys: {sample_keys}")
            # Only use it if it has actual model weights
            if any('encoder' in k or 'decoder' in k or 'codebook' in k for k in model_state.keys()):
                last_state = model_state
            else:
                logging.warning("model_state_dict doesn't contain VQVAELastToken keys, skipping")
    
    # Last resort - use the whole checkpoint
    if last_state is None:
        logging.warning("Could not find specific state dict, using whole checkpoint")
        last_state = vq_last_ckpt
    
    # Load the state dict
    missing_keys, unexpected_keys = load_state_dict_remove_ddp_prefix(model, last_state)
    
    # If we have any missing keys, something is wrong
    if len(missing_keys) > 0:
        raise RuntimeError(f"Failed to load VQVAELastToken model - missing keys ({len(missing_keys)}): {missing_keys[:10]}...")
    
    for param in model.parameters():
        param.requires_grad = False
    
    del vq_last_ckpt, last_state
    gc.collect()
    return model, vqvae_last_config




def sync_and_update_streaming_info(local_samples, smi_dict, processed_samples_count, ddp, ddp_world_size, master_process):
    """Sync samples from all GPUs and update StreamingInfo on master process"""
    if ddp:
        # Gather samples from all processes
        all_samples = [None] * ddp_world_size
        dist.all_gather_object(all_samples, local_samples)
        
        if master_process:
            # Combine samples from all processes
            combined_samples = defaultdict(list)
            for proc_samples in all_samples:
                for var_name, samples in proc_samples.items():
                    combined_samples[var_name].extend(samples)
            
            # Update StreamingInfo with new samples
            for var_name, samples in combined_samples.items():
                for sample in samples:
                    smi_dict.update(sample)
            
            return dict(combined_samples)
    else:
        # Single GPU - directly update StreamingInfo
        for var_name, samples in local_samples.items():
            for sample in samples:
                smi_dict.update(sample)
        
        return local_samples
    
    return {}


def calculate_and_report_metrics(smi, token_variables, max_layers_used, num_blocks, delta_b, processed_samples_count, batch_idx, total_batches, logger, master_process):
    """Calculate and report current metrics"""
    if not master_process:
        return {}
    
    logger.info(f"\n{'='*80}")
    logger.info(f"PROGRESS REPORT - Batch {batch_idx+1}/{total_batches}")
    logger.info(f"Sequences processed so far: {processed_samples_count}")
    logger.info(f"{'='*80}")
    
    # Calculate block MI results
    results = collect_block_mi(smi, token_variables, max_layers_used, num_blocks, delta_b)
    
    # Report summary statistics
    total_pairs = 0
    total_mi = 0.0
    for layer_key, blocks_dict in results.items():
        for block_key, tokens_dict in blocks_dict.items():
            for token_key, data in tokens_dict.items():
                total_pairs += 1
                total_mi += data['mi_block_token']
    
    if total_pairs > 0:
        avg_mi = total_mi / total_pairs
        logger.info(f'Average MI across all block-token pairs: {avg_mi:.6f} bits')
        logger.info(f'Total block-token pairs analyzed: {total_pairs}')
    
    logger.info(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='MI between block VQVAE codes and last hidden state codes.')
    parser.add_argument('--config', type=str, required=True, help='Path to JSON configuration file.')
    parser.add_argument('--sync_interval', type=int, default=100, help='How often to sync and report results (in batches)')
    parser.add_argument('--log_interval', type=int, default=10, help='How often to log progress (in batches)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # ------------- DDP Setup ------------------------------
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device_str = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device_str)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        torch.manual_seed(1337 + seed_offset)
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_rank = 0
        ddp_local_rank = 0
        device_str = config.get('device', 'cuda')

    # Setup logging - only master process prints to console
    if master_process:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(levelname)s][%(asctime)s]: %(message)s', datefmt='%H:%M:%S')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        # Create a dummy logger for non-master processes
        logger = logging.getLogger('dummy')
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL)

    if master_process:
        logger.info(f"DDP: rank {ddp_rank}/{ddp_world_size}, local_rank {ddp_local_rank}, master: {master_process}")

    block_config_path = config.get('vqvae_block_config_path') or config.get('vqvae3v2_config_path')
    block_checkpoint_path = config.get('vqvae_block_checkpoint_path') or config.get('vqvae3v2_checkpoint_path')
    vqvae_last_checkpoint_path = config.get('vqvae_last_checkpoint_path')
    vqvae_last_config_path = config.get('vqvae_last_config_path')
    
    # Helper: try to auto-discover VQVAELastToken assets under NLP_openwebtext
    def _auto_find_vqvae_last(root='NLP_openwebtext'):
        candidates = []
        for cur_root, _dirs, files in os.walk(root):
            if 'checkpoint_final.pt' in files:
                cfgs = [f for f in files if f.endswith('_vqvae_last_config.json') or f.endswith('vqvae_last_config.json')]
                if cfgs:
                    ckpt = os.path.join(cur_root, 'checkpoint_final.pt')
                    # pick the first matching config (prefer with prefix)
                    cfg_file = sorted(cfgs, key=lambda x: (not x.endswith('_vqvae_last_config.json'), x))[0]
                    cfgp = os.path.join(cur_root, cfg_file)
                    try:
                        mtime = os.path.getmtime(ckpt)
                    except Exception:
                        mtime = 0
                    candidates.append((mtime, ckpt, cfgp))
        if candidates:
            candidates.sort(reverse=True)
            _mtime, ckpt, cfgp = candidates[0]
            return ckpt, cfgp
        return None, None

    if block_config_path is None:
        raise ValueError('Configuration must include vqvae_block_config_path or vqvae3v2_config_path.')
    if block_checkpoint_path is None:
        raise ValueError('Configuration must include vqvae_block_checkpoint_path or vqvae3v2_checkpoint_path.')
    
    # Last-token model: allow omission, try auto-find
    if vqvae_last_checkpoint_path is None or vqvae_last_config_path is None:
        auto_ckpt, auto_cfg = _auto_find_vqvae_last()
        if vqvae_last_checkpoint_path is None:
            vqvae_last_checkpoint_path = auto_ckpt
        if vqvae_last_config_path is None:
            vqvae_last_config_path = auto_cfg
    if vqvae_last_checkpoint_path is None or vqvae_last_config_path is None:
        raise ValueError('VQVAELastToken paths not provided and auto-discovery failed. Please set "vqvae_last_checkpoint_path" and "vqvae_last_config_path" in the config.')
    llm_checkpoint_path = config.get('llm_checkpoint_path')
    if llm_checkpoint_path is None:
        raise ValueError('Configuration must include llm_checkpoint_path.')

    logging.info(f'Loading unified VQVAE config from {block_config_path}')
    with open(block_config_path, 'r') as f:
        unified_cfg = json.load(f)

    block_cfg = unified_cfg.get('vqvae_block_config') or unified_cfg
    layer_cfg = unified_cfg.get('vqvae_layer_config')

    seq_len = config.get('sequence_length')
    if seq_len is None:
        if layer_cfg is not None and layer_cfg.get('T') is not None:
            seq_len = layer_cfg['T']
        elif unified_cfg.get('T_max') is not None:
            seq_len = unified_cfg['T_max']
        else:
            raise ValueError('Unable to determine sequence length. Provide sequence_length or ensure vqvae_layer_config.T exists.')
    seq_len = int(seq_len)

    delta_b = int(config.get('delta_b', unified_cfg.get('delta_b', block_cfg.get('T', 16))))
    dataset = config.get('dataset', unified_cfg.get('dataset', 'openwebtext'))
    num_samples_cfg = config.get('num_samples')
    num_sequences_cfg = config.get('num_sequences')
    batch_size = int(config.get('batch_size', unified_cfg.get('batch_size', 32)))
    # device_str already set by DDP setup above
    if not ddp:
        device_str = config.get('device', unified_cfg.get('device', 'cuda'))
    dtype_str = config.get('dtype', unified_cfg.get('dtype', 'bfloat16'))
    num_layers_to_process = int(config.get('num_layers_to_process', 8))
    save_count_data = bool(config.get('save_count_data', True))
    output_file = config.get('output_file')
    if output_file is None:
        base_dir = os.path.dirname(block_config_path) if block_config_path is not None else '.'
        output_file = os.path.join(base_dir, 'mi_v4_results.json')

    prompt_length = int(config.get('prompt_length', seq_len // 2))
    max_new_tokens = int(config.get('max_new_tokens', seq_len - prompt_length))
    if prompt_length <= 0 or prompt_length >= seq_len:
        raise ValueError('prompt_length must be > 0 and < sequence_length')
    if max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be > 0')
    total_seq_len = prompt_length + max_new_tokens
    if total_seq_len > seq_len:
        raise ValueError(
            f'prompt_length ({prompt_length}) + max_new_tokens ({max_new_tokens}) exceeds sequence_length ({seq_len})'
        )

    if num_sequences_cfg is not None and num_samples_cfg is not None:
        if int(num_sequences_cfg) != int(num_samples_cfg):
            raise ValueError('num_samples and num_sequences must match when both are provided; each sample corresponds to one generated sequence.')
    if num_sequences_cfg is not None:
        num_sequences = int(num_sequences_cfg)
    else:
        num_sequences = int(num_samples_cfg) if num_samples_cfg is not None else 50000
    num_tokens_target = num_sequences * max_new_tokens

    eos_token_id = config.get('eos_token_id')
    if eos_token_id is not None:
        eos_token_id = int(eos_token_id)

    if device_str == 'cuda' and not torch.cuda.is_available():
        if master_process:
            logger.warning('CUDA requested but not available. Falling back to CPU.')
        device_str = 'cpu'
    device_obj = torch.device(device_str)
    device_type = 'cuda' if device_obj.type == 'cuda' else 'cpu'
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    if dtype_str not in dtype_map:
        raise ValueError(f'Unsupported dtype: {dtype_str}')
    ptdtype = dtype_map[dtype_str]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    torch.set_float32_matmul_precision('high')

    if total_seq_len % delta_b != 0:
        if master_process:
            logger.warning(
                f'Total sequence length {total_seq_len} not divisible by delta_b {delta_b}; trailing tokens will be ignored.'
            )
    num_blocks = total_seq_len // delta_b

    if master_process:
        logger.info(f'Loading block VQVAE checkpoint from {block_checkpoint_path}')
    block_model, block_meta = build_vqvae3v2_from_config(block_cfg)
    block_model.to(device_obj).eval()
    for param in block_model.parameters():
        param.requires_grad = False
    block_ckpt = torch.load(block_checkpoint_path, map_location='cpu')
    blk_state = None
    keys_try_blk = ['block_model_state_dict', 'vqvae3v2_block', 'vqvae_block', 'vqvae_model', 'model_state_dict']
    for key in keys_try_blk:
        if key in block_ckpt:
            blk_state = block_ckpt[key]
            if isinstance(blk_state, dict):
                break
    if blk_state is None and 'models' in block_ckpt and isinstance(block_ckpt['models'], dict):
        blk_state = block_ckpt['models'].get('vqvae3v2_block') or block_ckpt['models'].get('vqvae_block')
    if blk_state is None:
        raise ValueError(f'Could not find block model state dict in checkpoint. Available keys: {list(block_ckpt.keys())}')
    missing_blk, unexpected_blk = load_state_dict_remove_ddp_prefix(block_model, blk_state)
    if missing_blk and master_process:
        logger.warning(f'Block VQVAE missing keys: {missing_blk}')
    if unexpected_blk and master_process:
        logger.warning(f'Block VQVAE unexpected keys: {unexpected_blk}')
    del block_ckpt, blk_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if block_meta['T'] != delta_b:
        if master_process:
            logger.warning(f'delta_b ({delta_b}) does not match block model T ({block_meta["T"]}); using model T.')
        delta_b = block_meta['T']
        num_blocks = total_seq_len // delta_b

    if master_process:
        logger.info(f'Loading LLM model from {llm_checkpoint_path}')
    llm_checkpoint = torch.load(llm_checkpoint_path, map_location='cpu')
    model_args = llm_checkpoint['model_args']
    model_args_for_hidden_states = model_args.copy()
    model_args_for_hidden_states['block_size'] = seq_len
    gptconf = GPTConfig(**model_args_for_hidden_states)
    llm_model = GPT(gptconf)
    state_dict = llm_checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    llm_model.load_state_dict(state_dict)
    llm_model.to(device_obj).eval()
    del llm_checkpoint, state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vqvae_last_model, vqvae_last_cfg = load_vqvae_last_token_model(
        vqvae_last_checkpoint_path,
        vqvae_last_config_path,
        device_obj
    )
    if master_process:
        logger.info(
            f'VQVAELastToken loaded: input_dim={vqvae_last_model.input_dim}, hidden_dim={vqvae_last_model.hidden_dim}, '
            f'codebook_size={vqvae_last_model.codebook_size}'
        )

    data_dir = os.path.join('data', dataset)
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_dtype = meta.get('dtype', np.uint16)
    else:
        meta_dtype = np.uint16
    if isinstance(meta_dtype, str):
        meta_dtype = np.dtype(meta_dtype)
    if master_process:
        logger.info(
            f'Dataset: {dataset}, dtype: {meta_dtype}, total_seq_len: {total_seq_len}, '
            f'prompt_length: {prompt_length}, max_new_tokens: {max_new_tokens}, '
            f'delta_b: {delta_b}, num_blocks: {num_blocks}'
        )

    token_variables = tuple(f'last_hidden_tok_{i}' for i in range(max_new_tokens))
    combos = [(var,) for var in token_variables]
    
    # Only master process maintains StreamingInfo objects
    if master_process:
        smi = StreamingInfo(variables=token_variables, combos_to_track=combos, base=2.0, store_samples=False)
    else:
        smi = None

    # Calculate data distribution for DDP
    total_batches_estimate = math.ceil(num_sequences / batch_size)
    if ddp:
        batches_per_process = total_batches_estimate // ddp_world_size
        start_batch = ddp_rank * batches_per_process
        if ddp_rank == ddp_world_size - 1:  # Last process takes remaining batches
            end_batch = total_batches_estimate
        else:
            end_batch = start_batch + batches_per_process
        
        if master_process:
            logger.info(f"Total estimated batches: {total_batches_estimate}")
            logger.info(f"Batches per process: {batches_per_process}")
            for rank in range(ddp_world_size):
                rank_start = rank * batches_per_process
                rank_end = total_batches_estimate if rank == ddp_world_size - 1 else rank_start + batches_per_process
                logger.info(f"Rank {rank}: batches {rank_start}-{rank_end-1}")
    else:
        start_batch = 0
        end_batch = total_batches_estimate

    if master_process:
        logger.info(
            f'Processing {num_sequences} sequences ({num_tokens_target} generated tokens) '
            f'(prompt_length={prompt_length}, max_new_tokens={max_new_tokens}, batch_size={batch_size})'
        )
        logger.info(f"Sync interval: {args.sync_interval} batches")
        logger.info(f"Log interval: {args.log_interval} batches")
    # Sample collection buffer for each process
    local_sample_buffer = defaultdict(list)  # var_name -> list of samples
    sequences_processed = 0
    batch_idx = 0
    save_every_percent = 5
    milestone_sequences = sorted(set([max(1, math.ceil(num_sequences * p / 100)) for p in range(save_every_percent, 100, save_every_percent)]))
    milestone_pointer = 0
    max_layers_used = 0

    while sequences_processed < num_sequences:
        # Skip batches not assigned to this process in DDP mode
        if ddp and (batch_idx < start_batch or batch_idx >= end_batch):
            batch_idx += 1
            continue
        sequences_remaining = num_sequences - sequences_processed
        current_batch_size = min(batch_size, sequences_remaining)

        input_ids = get_batch(data_dir, meta_dtype, prompt_length, current_batch_size, device_str, 'train')
        input_ids = input_ids[:current_batch_size]
        prompt_attn_mask = torch.ones_like(input_ids, device=input_ids.device)

        with torch.no_grad():
            generated_sequences = llm_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                eos_token=eos_token_id,
                attention_mask=prompt_attn_mask
            )
            with ctx:
                hidden_states = llm_model.hidden_states(generated_sequences)

        B, total_layers, T, d = hidden_states.shape
        layers_for_blocks = min(num_layers_to_process, total_layers)
        max_layers_used = max(max_layers_used, layers_for_blocks)

        # Pre-compute block codes using batch operations
        # Initialize storage for all block codes
        block_codes_per_sample = [{} for _ in range(current_batch_size)]
        
        # Process each layer and block combination with batch operations
        for layer_idx in range(layers_for_blocks):
            for block_idx in range(num_blocks):
                start = block_idx * delta_b
                end = start + delta_b
                if end > T:
                    continue
                
                # Extract block data for all samples at once
                blk_x = hidden_states[:current_batch_size, layer_idx:layer_idx + 1, start:end, :].contiguous()
                blk_mask = torch.ones(current_batch_size, end - start, device=device_obj)
                
                # Batch VQ-VAE forward pass for all samples
                blk_codes = compute_vq3v2_index_batch(block_model, blk_x, blk_mask)  # Shape: (current_batch_size,)
                
                # Variable name and StreamingInfo setup (only for master process)
                var_name = f'block_l{layer_idx}_s{start}_e{end}'
                if master_process and smi and var_name not in smi.variables:
                    smi.variables = tuple(list(smi.variables) + [var_name])
                    if (var_name,) not in smi.combos:
                        smi.combos.add((var_name,))
                        smi.counts[(var_name,)] = Counter()
                    for token_var in token_variables:
                        pair_key = tuple(sorted((var_name, token_var)))
                        if pair_key not in smi.combos:
                            smi.combos.add(pair_key)
                            smi.counts[pair_key] = Counter()
                
                # Store codes for each sample
                for sample_idx in range(current_batch_size):
                    block_codes_per_sample[sample_idx][var_name] = int(blk_codes[sample_idx].item())

        # Quantize generated token hidden states
        generated_hidden = hidden_states[:, -2, prompt_length:prompt_length + max_new_tokens, :].contiguous()
        generated_token_count = generated_hidden.shape[1]
        if generated_token_count != max_new_tokens:
            logging.warning(
                f'Expected {max_new_tokens} generated tokens but received {generated_token_count}; '
                'padding with sentinel values for missing positions.'
            )
        
        # Use VQVAELastToken quantization (similar to v3 approach)
        generated_hidden_flat = generated_hidden.view(-1, d)  # (batch_size * seq_len, d)
        with torch.no_grad():
            with ctx:
                z_e = vqvae_last_model.encoder(generated_hidden_flat)  # (batch_size * seq_len, hidden_dim)
                distances_flat = (torch.sum(z_e**2, dim=1, keepdim=True)
                                 + torch.sum(vqvae_last_model.codebook.weight**2, dim=1)
                                 - 2 * torch.matmul(z_e, vqvae_last_model.codebook.weight.t()))
                last_indices_flat = torch.argmin(distances_flat, dim=1)
        last_indices = last_indices_flat.view(current_batch_size, generated_token_count)

        for sample_idx in range(current_batch_size):
            if sequences_processed >= num_sequences:
                break
            sample_block_codes = block_codes_per_sample[sample_idx]
            sample_data: Dict[str, Any] = {}
            for token_idx, token_var in enumerate(token_variables):
                if token_idx < generated_token_count:
                    sample_data[token_var] = int(last_indices[sample_idx, token_idx].item())
                else:
                    sample_data[token_var] = -1
            for var_name, code in sample_block_codes.items():
                sample_data[var_name] = code
                
            # Collect sample in local buffer instead of directly updating StreamingInfo
            local_sample_buffer['samples'].append(sample_data)
            sequences_processed += 1

        del hidden_states
        del generated_sequences
        del input_ids
        del generated_hidden
        del generated_hidden_flat
        del distances_flat
        del last_indices_flat
        del last_indices
        del block_codes_per_sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_idx += 1
        tokens_processed = sequences_processed * max_new_tokens
        
        # Log progress
        if batch_idx % args.log_interval == 0:
            samples_in_buffer = len(local_sample_buffer['samples'])
            if ddp:
                print(f"Rank {ddp_rank}: Batch {batch_idx}, Processed: {sequences_processed}, Buffer: {samples_in_buffer}")
            else:
                print(f"Batch {batch_idx}, Processed: {sequences_processed}, Buffer: {samples_in_buffer}")
        
        # Periodic sync and reporting
        if batch_idx % args.sync_interval == 0 and batch_idx > 0:
            if ddp:
                dist.barrier()  # Synchronize all processes
            
            # Sync samples and update StreamingInfo
            synced_samples = sync_and_update_streaming_info(local_sample_buffer, smi, sequences_processed, ddp, ddp_world_size, master_process)
            
            # Calculate and report metrics on master process
            if master_process and smi:
                results = calculate_and_report_metrics(smi, token_variables, max_layers_used, num_blocks, delta_b, sequences_processed, batch_idx, end_batch, logger, master_process)
            
            # Clear local buffer after sync
            local_sample_buffer.clear()
        
        if master_process:
            logger.info(
                f'Processed {tokens_processed}/{num_tokens_target} generated tokens '
                f'({sequences_processed}/{num_sequences} sequences, batch {batch_idx}, '
                f'current_batch_size={current_batch_size})'
            )

        while milestone_pointer < len(milestone_sequences) and sequences_processed >= milestone_sequences[milestone_pointer] and master_process:
            current_milestone = milestone_sequences[milestone_pointer]
            percent_done = int(round(100 * current_milestone / num_sequences))
            logger.info(
                f'Computing intermediate MI at {percent_done}% ({sequences_processed} sequences, '
                f'{tokens_processed} tokens processed)...'
            )
            results_snapshot = {
                'blocks': collect_block_mi(smi, token_variables, max_layers_used, num_blocks, delta_b)
            }
            base, ext = os.path.splitext(output_file)
            if not ext:
                ext = '.json'
            intermediate_path = f"{base}_samples_{sequences_processed}_pct_{percent_done}{ext}"
            output_dirname = os.path.dirname(intermediate_path)
            if output_dirname:
                os.makedirs(output_dirname, exist_ok=True)
            with open(intermediate_path, 'w') as f:
                json.dump({
                    'config': config,
                    'num_sequences_processed': sequences_processed,
                    'num_tokens_processed': tokens_processed,
                    'percent_sequences_completed': percent_done,
                    'results': results_snapshot
                }, f, indent=2)
            logger.info(f'Intermediate results saved to {intermediate_path}')
            if save_count_data:
                output_dir = os.path.dirname(output_file)
                if output_dir:
                    save_streaming_info_counts(smi, output_dir, suffix=f"_samples_{sequences_processed}_pct_{percent_done}")
            milestone_pointer += 1

    # Final sync
    if ddp:
        dist.barrier()

    # Final sync of remaining samples
    if local_sample_buffer['samples']:
        synced_samples = sync_and_update_streaming_info(local_sample_buffer, smi, sequences_processed, ddp, ddp_world_size, master_process)
        local_sample_buffer.clear()

    if master_process:
        logger.info('Calculating final MI...')
    tokens_processed = sequences_processed * max_new_tokens
    
    # Only calculate final results on master process
    if master_process and smi:
        final_blocks = collect_block_mi(smi, token_variables, max_layers_used, num_blocks, delta_b)
    else:
        final_blocks = {}
    # Only save final results on master process
    if master_process:
        final_results = {
            'config': config,
            'num_sequences_processed': sequences_processed,
            'num_tokens_processed': tokens_processed,
            'sequence_length': seq_len,
            'prompt_length': prompt_length,
            'max_new_tokens': max_new_tokens,
            'delta_b': delta_b,
            'num_blocks': num_blocks,
            'token_variables': list(token_variables),
            'blocks': final_blocks
        }

        output_dirname = os.path.dirname(output_file)
        if output_dirname:
            os.makedirs(output_dirname, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f'Final MI results written to {output_file}')

        if save_count_data and smi:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                save_streaming_info_counts(smi, output_dir, suffix='_final')

        all_pairs = []
        for layer_key, blocks_dict in final_blocks.items():
            for block_key, tokens_dict in blocks_dict.items():
                for token_key, data in tokens_dict.items():
                    all_pairs.append((data['mi_block_token'], layer_key, block_key, token_key))
        if all_pairs:
            all_pairs.sort(key=lambda x: x[0], reverse=True)
            logger.info('Top MI(block; token) pairs:')
            for mi_val, layer_key, block_key, token_key in all_pairs[:5]:
                logger.info(f'  {layer_key}/{block_key}/{token_key}: {mi_val:.6f} bits')

    # Clean up DDP
    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    main()
