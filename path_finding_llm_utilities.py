import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class PathFindingDataset(Dataset):
    """
    Dataset class for path finding tasks with support for loss masking on the edges portion.
    """
    
    def __init__(self, 
                 jsonl_file: str, 
                 tokenizer_vocab: Optional[Dict[str, int]] = None,
                 max_seq_length: int = 512,
                 mask_edges: bool = True,
                 max_nodes: int = 240,
                 number_of_hyphens: Optional[int] = 0):
        """
        Initialize the path finding dataset.
        
        Args:
            jsonl_file: Path to JSONL file containing path finding data
            tokenizer_vocab: Vocabulary mapping for tokenization (if None, will create from data)
            max_seq_length: Maximum sequence length for padding/truncation
            mask_edges: Whether to mask the edges portion for loss calculation
        """
        self.max_seq_length = max_seq_length
        self.mask_edges = mask_edges
        self.max_nodes = max_nodes
        self.number_of_hyphens = number_of_hyphens
        
        # # Load data from JSONL file
        # self.data = []
        # with open(jsonl_file, 'r') as f:
        #     for line in f:
        #         if line.strip():
        #             self.data.append(json.loads(line))
        import orjson as json  # drop-in replacement for load/loads in this snippet
        import time
        start_time = time.time()
        with open(jsonl_file, "rb") as f:                   # read bytes (faster)
            self.data = [json.loads(line) for line in f if line.strip()]
        print(f"Time taken: {time.time() - start_time} seconds")
        print(f"Loaded {len(self.data)} samples from {jsonl_file}")
        # Build vocabulary if not provided
        if tokenizer_vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = tokenizer_vocab
            
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"Vocabulary size: {self.vocab_size}")
        
    def _build_vocab(self) -> Dict[str, int]:
        """
        Build vocabulary from the dataset.
        Node indices 1-max_nodes map directly to vocabulary indices 1-max_nodes.
        Node 0 is not used by the path generator, so we skip it.
        Special tokens use indices starting from max_nodes + 1.
        """
        vocab = {}
        
        # Reserve indices 1 to max_nodes for nodes (skip 0 since it's never used)
        # Node "1" gets vocab ID 1, Node "2" gets vocab ID 2, etc.
        for i in range(1, self.max_nodes + 1):
            vocab[str(i)] = i
        
        # Special tokens start from index max_nodes + 1
        special_tokens = {
            "<PAD>": 0,                    # Use the unused 0 slot for padding
            "-": self.max_nodes + 1,       # max_nodes + 5
            ",": self.max_nodes + 2,       # max_nodes + 6
            ":": self.max_nodes + 3,       # max_nodes + 7
            #"<BOS>": self.max_nodes + 4,   # max_nodes + 1
            #"<EOS>": self.max_nodes + 5,   # max_nodes + 2
            #"<EDGES>": self.max_nodes + 6, # max_nodes + 3
            #"<PATHS>": self.max_nodes + 7, # max_nodes + 4
        }
        
        vocab.update(special_tokens)
        
        # Collect any additional unique tokens from the dataset (if any)
        all_tokens = set()
        
        for sample in self.data:
            # Add node IDs from edges
            for edge in sample["edges"]:
                nodes = edge.split("-")
                all_tokens.update(nodes)
            
            # Add node IDs from paths
            for path_list in [sample["correct_paths"], sample["decoy_paths"]]:
                for path in path_list:
                    all_tokens.update([str(node) for node in path])
        
        # Check for any unexpected tokens and warn about potential conflicts
        unexpected_tokens = []
        for token in all_tokens:
            if token not in vocab:
                try:
                    token_num = int(token)
                    if token_num == 0:
                        print(f"Warning: Found node '0' in data, but path generator shouldn't create node 0!")
                    elif token_num > self.max_nodes:
                        print(f"Warning: Found node '{token}' > max_nodes ({self.max_nodes})!")
                except ValueError:
                    pass  # Non-numeric token
                unexpected_tokens.append(token)
        
        # Add any unexpected tokens at the end
        token_id = len(vocab)
        for token in sorted(unexpected_tokens, key=lambda x: int(x) if x.isdigit() else float('inf')):
            print(f"Warning: Found unexpected token '{token}' - adding at index {token_id}")
            vocab[token] = token_id
            token_id += 1
                
        return vocab
    
    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize text into token IDs."""
        tokens = []
        words = text.split()
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Handle unknown tokens - for simplicity, skip them
                # In a real implementation, you might want to use <UNK> token
                print(f"Warning: Found unknown token '{word}' - skipping")
                pass
                
        return tokens
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
            - input_ids: Token IDs for the input sequence
            - attention_mask: Attention mask for padding  
            - labels: Target labels for training (-1 for ignored tokens)
        """
        sample = self.data[idx]
        
        edges = sample["edges"]

        prompt_ids = []
        for edge in edges[:-1]:
            node1, node2 = edge.split("-")
            prompt_ids.append(self.vocab[node1])
            if self.number_of_hyphens is not None:
                for _ in range(self.number_of_hyphens):
                    prompt_ids.append(self.vocab["-"])
            else:
                prompt_ids.append(self.vocab["-"])
            prompt_ids.append(self.vocab[node2])
            prompt_ids.append(self.vocab[","])
        
        node1, node2 = edges[-1].split("-")
        prompt_ids.append(self.vocab[node1])
        if self.number_of_hyphens is not None:
            for _ in range(self.number_of_hyphens):
                prompt_ids.append(self.vocab["-"])
        else:
            prompt_ids.append(self.vocab["-"])
        prompt_ids.append(self.vocab[node2])
        prompt_ids.append(self.vocab[":"])

        original_prompt_len = len(prompt_ids)

        # Ensure there is at least one correct path
        if not sample["correct_paths"]:
            raise ValueError(f"Sample {idx} has no correct paths. Each sample must have at least one correct path.")
        
        # Select the first correct path (this is the path that will be in input_ids/labels)
        selected_path_index = 0
        correct_path = sample["correct_paths"][selected_path_index]
        path_ids = []
        for sym in correct_path[:-1]:
            if sym in self.vocab:
                path_ids.append(self.vocab[sym])
                if self.number_of_hyphens is not None:
                    for _ in range(self.number_of_hyphens):
                        path_ids.append(self.vocab["-"])
                # When number_of_hyphens is None, add no hyphens to the path (old behavior)
            else:
                print(f"Warning: Token '{sym}' not in vocabulary, skipping")
                continue
        if correct_path[-1] in self.vocab:
            path_ids.append(self.vocab[correct_path[-1]])
        else:
            print(f"Warning: Token '{correct_path[-1]}' not in vocabulary, skipping")

        input_ids = prompt_ids + path_ids
        
        # Capture the length of the path used in input_ids BEFORE path_ids gets reused
        selected_path_length = len(path_ids)

        # Create labels BEFORE truncation to maintain proper structure
        # Labels: -1 for ignored tokens (prompt), actual tokens for path
        original_labels = [-1] * len(prompt_ids) + path_ids
        
        # Handle sequence length gracefully - truncate BOTH input_ids and labels together
        if len(input_ids) > (self.max_seq_length+1):
            # Truncate from the beginning (keep recent tokens)
            input_ids = input_ids[-(self.max_seq_length+1):]
            original_labels = original_labels[-(self.max_seq_length+1):]
        
        # Create padded prompt_ids tensor (for evaluation use)
        prompt_pad_length = self.max_seq_length - len(prompt_ids)
        if prompt_pad_length > 0:
            padded_prompt_ids = [self.vocab["<PAD>"]] * prompt_pad_length + prompt_ids
            # Create attention mask for prompt: 0 for padding, 1 for real tokens
            prompt_attention_mask = [0] * prompt_pad_length + [1] * len(prompt_ids)
        else:
            # If prompt is too long, truncate from the beginning
            padded_prompt_ids = prompt_ids[-self.max_seq_length:]
            prompt_attention_mask = [1] * self.max_seq_length
        
        # Left padding - pad to max_seq_length + 1 to be consistent with truncation
        # Both will be reduced by 1 later for next-token prediction
        target_length = self.max_seq_length + 1
        pad_length = target_length - len(input_ids)
        if pad_length > 0:
            # Add padding to the left
            input_ids = [self.vocab["<PAD>"]] * pad_length + input_ids
            original_labels = [-1] * pad_length + original_labels
        
        # Create attention mask (0 for padding, 1 for real tokens)
        attention_mask = [0] * pad_length + [1] * (target_length - pad_length)
        
        # Create labels for next token prediction (your approach - more efficient!)
        input_ids = input_ids[:-1]  # Remove last token
        labels = original_labels[1:]  # Shift labels left
        attention_mask = attention_mask[:-1]  # Adjust attention mask
        
        # Also adjust padded_prompt_ids and its attention mask to match the shifted sequence length
        padded_prompt_ids = padded_prompt_ids[1:]
        prompt_attention_mask = prompt_attention_mask[1:]

        correct_path_ids = []
        for cp in sample["correct_paths"]:
            path_ids = []
            for sym in cp:
                if sym in self.vocab:
                    path_ids.append(self.vocab[sym])
                else:
                    print(f"Warning: Token '{sym}' not in vocabulary, skipping")
            correct_path_ids.append(path_ids)
            
        decoy_path_ids = []
        for decoy_path in sample["decoy_paths"]:
            path_ids = []
            for sym in decoy_path:
                if sym in self.vocab:
                    path_ids.append(self.vocab[sym])
                else:
                    print(f"Warning: Token '{sym}' not in vocabulary, skipping")
                    continue
            decoy_path_ids.append(path_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long), 
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_ids": torch.tensor(padded_prompt_ids, dtype=torch.long),
            "prompt_attention_mask": torch.tensor(prompt_attention_mask, dtype=torch.long),
            "correct_path_ids": correct_path_ids,
            "decoy_path_ids": decoy_path_ids,
            "original_prompt_len": original_prompt_len,
            "output_length": selected_path_length,  # Length of the specific path used in input_ids/labels
            "selected_path_index": selected_path_index  # Which correct path was selected
        }

def custom_collate_fn(batch):
    """Custom collate function to handle variable-length path lists"""
    # Handle tensors normally
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    prompt_ids = torch.stack([item['prompt_ids'] for item in batch])
    prompt_attention_mask = torch.stack([item['prompt_attention_mask'] for item in batch])
    
    # Handle variable-length lists
    correct_path_ids = [item['correct_path_ids'] for item in batch]
    decoy_path_ids = [item['decoy_path_ids'] for item in batch]
    original_prompt_len = [item['original_prompt_len'] for item in batch]
    output_length = [item['output_length'] for item in batch]
    selected_path_index = [item['selected_path_index'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'prompt_ids': prompt_ids,
        'prompt_attention_mask': prompt_attention_mask,
        'correct_path_ids': correct_path_ids,
        'decoy_path_ids': decoy_path_ids,
        'original_prompt_len': original_prompt_len,
        'output_length': output_length,
        'selected_path_index': selected_path_index
    }

def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss with automatic masking via ignore_index=-1.
    
    Args:
        logits: Model outputs [batch_size, seq_length, vocab_size]
        labels: Target labels [batch_size, seq_length] (-1 for ignored tokens)
        
    Returns:
        Loss tensor
    """
    # Flatten everything
    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = labels.view(-1)
    
    # PyTorch automatically ignores tokens where labels == -1
    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-1)
    
    return loss

def crop_prompt_and_attention_mask(prompt_ids: torch.Tensor, prompt_attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crop prompt_ids and prompt_attention_mask to remove time indices where all samples have attention_mask = 0.
    
    Args:
        prompt_ids: Tensor of shape [batch_size, seq_length]
        prompt_attention_mask: Tensor of shape [batch_size, seq_length]
    
    Returns:
        Tuple of (cropped_prompt_ids, cropped_prompt_attention_mask)
    """
    # Find positions where at least one sample has attention_mask = 1
    # Sum across batch dimension (dim=0) to get per-position counts
    attention_sum = prompt_attention_mask.sum(dim=0)  # Shape: [seq_length]
    
    # Keep positions where at least one sample has attention
    valid_positions = attention_sum > 0
    
    # Crop both tensors to only include valid positions
    cropped_prompt_ids = prompt_ids[:, valid_positions]
    cropped_prompt_attention_mask = prompt_attention_mask[:, valid_positions]
    
    return cropped_prompt_ids, cropped_prompt_attention_mask


def skip_time_steps(prompt_ids, prompt_attention_mask, H, max_nodes):    # Create mask for each batch element - keep tokens that are NOT max_nodes+1 or max_nodes+2
    time_mask = ~((prompt_ids == max_nodes + 1) | (prompt_ids == max_nodes + 2))
    
    # Apply mask to each row and collect filtered sequences
    filtered_prompt_ids = []
    filtered_attention_masks = []
    filtered_H = []
    
    for i in range(prompt_ids.shape[0]):
        # Apply mask to this batch element
        filtered_ids = prompt_ids[i][time_mask[i]]
        filtered_mask = prompt_attention_mask[i][time_mask[i]]
        # Apply same mask to H tensor on T dimension (H shape: B, L, T, d)
        filtered_h = H[i, :, time_mask[i], :]  # Shape: (L, filtered_T, d)
        
        filtered_prompt_ids.append(filtered_ids)
        filtered_attention_masks.append(filtered_mask)
        filtered_H.append(filtered_h)
    
    # Find the maximum length after filtering
    max_filtered_len = max(len(seq) for seq in filtered_prompt_ids)
    
    # Pad sequences from the left with zeros
    padded_prompt_ids = torch.zeros((prompt_ids.shape[0], max_filtered_len), dtype=prompt_ids.dtype, device=prompt_ids.device)
    padded_attention_mask = torch.zeros((prompt_ids.shape[0], max_filtered_len), dtype=prompt_attention_mask.dtype, device=prompt_attention_mask.device)
    padded_H = torch.zeros((H.shape[0], H.shape[1], max_filtered_len, H.shape[3]), dtype=H.dtype, device=H.device)
    
    for i, (ids, mask, h) in enumerate(zip(filtered_prompt_ids, filtered_attention_masks, filtered_H)):
        seq_len = len(ids)
        if seq_len > 0:
            # Pad from left: place the sequence at the end of the tensor
            padded_prompt_ids[i, -seq_len:] = ids
            padded_attention_mask[i, -seq_len:] = mask
            padded_H[i, :, -seq_len:, :] = h
    
    return padded_prompt_ids, padded_attention_mask, padded_H