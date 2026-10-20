
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# === Rotary Positional Embedding helpers ===================================
def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of values (x_0,x_1) -> (-x_1,x_0) along the last dimension."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)

def apply_rotary_pos_emb(x: torch.Tensor,
                         cos: torch.Tensor,
                         sin: torch.Tensor) -> torch.Tensor:
    """
    Apply pre‑computed rotary cos/sin embeddings to query/key tensors.
    Expected shapes
      x   : (batch, n_head, seq_len, head_dim)
      cos : (1, 1, seq_len, head_dim)
      sin : (1, 1, seq_len, head_dim)
    """
    return (x * cos) + (rotate_every_two(x) * sin)
# ===========================================================================


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # --- Rotary positional embedding tables --------------------------------
        head_dim = self.n_embd // self.n_head
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE."
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(config.block_size)
        freqs = torch.outer(t, inv_freq)                     # (seq, head_dim/2)
        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)  # (seq, head_dim)
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)
        # add two leading singleton dims for broadcasting: (1,1,seq,head_dim)
        self.register_buffer("rotary_cos", cos[None, None, :, :], persistent=False)
        self.register_buffer("rotary_sin", sin[None, None, :, :], persistent=False)

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

        # Take this out of the if statement in case we switch to slow attention after initialization
        bias = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias",
            bias.view(1, 1, config.block_size, config.block_size),
            persistent=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        cos = self.rotary_cos[:, :, :T, :]
        sin = self.rotary_sin[:, :, :T, :]
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # Create merged mask combining causal and attention masks
            attn_mask = self._create_merged_mask(B, T, attention_mask, device=q.device)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=False  # We handle causality in our merged mask
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            if attention_mask is not None:
                # attention_mask: (B, T) -> (B, 1, 1, T)
                mask = attention_mask[:, None, None, :].to(dtype=att.dtype)
                att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        # zero out padded positions to avoid nan
        if attention_mask is not None:
            # attention_mask: (B, T) of bool or 0/1
            mask = attention_mask.view(B, T, 1).to(torch.bool)          # shape (B, T, 1)
            zero = torch.zeros_like(y)
            # where(mask) keeps y at True positions, takes zero where mask is False
            y = torch.where(mask, y, zero)                            # zero out padded positions

        return y

    def _create_merged_mask(self, B, T, attention_mask, device):
        """
        Create a merged mask that combines causal masking and attention masking.
        
        Args:
            B: batch size
            T: sequence length
            attention_mask: (B, T) tensor where 1 = valid token, 0 = padded token
            device: torch device
            
        Returns:
            merged_mask: (B, 1, T, T) boolean tensor where True = can attend, False = cannot attend
        """
        # Create causal mask: (T, T) where mask[i, j] = True if i >= j (can attend to past/current)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        
        if attention_mask is not None:
            # Convert attention_mask to boolean and reshape for broadcasting
            # attention_mask: (B, T) -> (B, 1, 1, T)
            attn_mask = attention_mask.to(torch.bool).view(B, 1, 1, T)
            
            # Broadcast causal mask to batch dimension: (1, 1, T, T)
            causal_mask = causal_mask.view(1, 1, T, T)
            
            # Combine masks: can attend if both causal allows it AND the key position is not padded
            # merged_mask[b, 0, i, j] = causal_mask[0, 0, i, j] AND attn_mask[b, 0, 0, j]
            merged_mask = causal_mask & attn_mask
        else:
            # No attention mask, just use causal mask broadcasted to batch
            merged_mask = causal_mask.view(1, 1, T, T).expand(B, 1, T, T)
            
        return merged_mask

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 64 
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    rope_theta: float = 10000.0   # base used in RoPE frequency calculation

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        #if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, attention_mask=None, mtl_modules=None, mtl_shared_head_nb_blocks=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)

        # For MTL, we need to split the forward pass
        if mtl_modules is not None and mtl_shared_head_nb_blocks is not None:
            # Process through shared blocks (all blocks except the last mtl_shared_head_nb_blocks)
            shared_blocks = len(self.transformer.h) - mtl_shared_head_nb_blocks
            
            # Forward through shared blocks
            for i in range(shared_blocks):
                x = self.transformer.h[i](x, attention_mask=attention_mask)
            
            # Get shared representation
            shared_repr = x
            
            # Forward through remaining original blocks
            for i in range(shared_blocks, len(self.transformer.h)):
                x = self.transformer.h[i](x, attention_mask=attention_mask)
            
            # Apply ln_f and get logits for original prediction
            x_orig = self.transformer.ln_f(x)
            logits_orig = self.lm_head(x_orig)
            
            # Forward through MTL modules
            mtl_logits = []
            for mtl_module in mtl_modules:
                x_mtl = shared_repr
                # Forward through MTL blocks
                # Handle DDP wrapping - if blocks are wrapped with DDP, access the underlying module
                blocks = mtl_module['blocks']
                if hasattr(blocks, 'module'):  # DDP wrapped
                    blocks = blocks.module
                
                for block in blocks:
                    x_mtl = block(x_mtl, attention_mask=attention_mask)
                # Apply shared ln_f and get logits using shared lm_head
                x_mtl = self.transformer.ln_f(x_mtl)
                logits_mtl = self.lm_head(x_mtl)
                mtl_logits.append(logits_mtl)
            
            # Combine all logits
            all_logits = [logits_orig] + mtl_logits
            
            if targets is not None:
                # Calculate loss for each prediction
                if not isinstance(targets, list):
                    raise ValueError(f"MTL mode requires targets to be a list of {len(all_logits)} tensors, but got {type(targets)}")
                if len(targets) != len(all_logits):
                    raise ValueError(f"MTL mode requires {len(all_logits)} target tensors, but got {len(targets)}")
                
                losses = []
                for i, (logits, target) in enumerate(zip(all_logits, targets)):
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
                    losses.append(loss)
                
                # Average all losses
                total_loss = sum(losses) / len(losses)
                return all_logits, total_loss, losses  # Return individual losses for logging
            else:
                # Inference: return only original model's prediction (last token)
                logits = self.lm_head(x_orig[:, [-1], :])
                return logits, None
        else:
            # Standard forward pass (backward compatibility)
            for block in self.transformer.h:
                x = block(x, attention_mask=attention_mask)
            x = self.transformer.ln_f(x)

            if targets is not None:
                # if we are given some desired targets also calculate the loss
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                # inference-time mini-optimization: only forward the lm_head on the very last position
                logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
                loss = None

            return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token=None, attention_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        
        Args:
            idx: Input token indices (b, t)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
            top_k: If specified, restricts sampling to the k most likely tokens
            eos_token: If specified, stops generation for each sequence when this token is generated
            
        Returns:
            Tensor of shape (b, t+max_new_tokens) with generated tokens
        """
        b, t = idx.size()
        device = idx.device
        
        # Create a tensor to keep track of which sequences have finished
        if eos_token is not None:
            # Check if any sequences have already generated the EOS token
            finished = (idx == eos_token).any(dim=1)
        else:
            finished = torch.zeros(b, dtype=torch.bool, device=device)
            
        for _ in range(max_new_tokens):
            # If all sequences are finished, stop generation
            if finished.all():
                break
                
            # Only generate for unfinished sequences
            # but we still need to use the full idx for context
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # forward the model to get the logits for the index in the sequence
            # TODO: check if this is correct !! important
            mask_cond = None
            if attention_mask is not None:
                mask_cond = attention_mask if attention_mask.size(1) == idx_cond.size(1) else attention_mask[:, -idx_cond.size(1):]
            logits, _ = self(idx_cond, attention_mask=mask_cond)
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # For finished sequences, just use a dummy token (doesn't matter which one)
            # We'll overwrite these positions anyway
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # For finished sequences, don't change them
            if eos_token is not None:
                # Mark sequences that just generated an EOS token as finished
                just_finished = (idx_next == eos_token) & ~finished
                finished = finished | just_finished
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

            if attention_mask is not None:
                pad = torch.ones((b, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((attention_mask, pad), dim=1)

        return idx

    @torch.no_grad()
    def hidden_states(self, input_ids, attention_mask=None):
        """
        Generate hidden states from all transformer layers for input sequences.
        Designed for VQ-VAE training where we need hidden representations.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing token IDs
            attention_mask: Tensor of shape (batch_size, seq_len) with 1 for valid tokens, 0 for padding
            
        Returns:
            Tensor of shape (batch_size, n_layer+1, seq_len, n_embd) containing hidden states from each layer
            (+1 because we include the final layer norm output)
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.size()
        
        # Create default attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Ensure sequence length doesn't exceed block size
        assert seq_len <= self.config.block_size, f"Sequence length {seq_len} exceeds block size {self.config.block_size}"
        
        # Set model to eval mode for hidden state generation
        was_training = self.training
        self.eval()
        
        try:
            # Initialize hidden states storage
            all_hidden_states = []
            
            # Forward the GPT model - start with embeddings
            tok_emb = self.transformer.wte(input_ids)  # (batch_size, seq_len, n_embd)
            x = self.transformer.drop(tok_emb)
            
            # Forward through each transformer block, collecting hidden states
            for block in self.transformer.h:
                x = block(x, attention_mask=attention_mask)
                all_hidden_states.append(x.clone())  # Store hidden state from this layer
            
            # Apply final layer norm and store
            x = self.transformer.ln_f(x)
            all_hidden_states.append(x.clone())  # Final layer norm output
            
            # Stack hidden states: (batch_size, n_layers+1, seq_len, n_embd)
            hidden_states = torch.stack(all_hidden_states, dim=1)
            
            return hidden_states
            
        finally:
            # Restore original training mode
            if was_training:
                self.train()

    @torch.no_grad()
    def iter_hidden_states(self, input_ids, attention_mask=None, include_final_norm=False):
        """
        Stream hidden states one layer at a time to avoid holding all layers in memory.

        Yields tuples of (layer_index, hidden), where hidden has shape (batch_size, seq_len, n_embd).
        If include_final_norm is True, also yields ("final", ln_f(hidden)).
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.size()

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        assert seq_len <= self.config.block_size, f"Sequence length {seq_len} exceeds block size {self.config.block_size}"

        was_training = self.training
        self.eval()

        try:
            tok_emb = self.transformer.wte(input_ids)
            x = self.transformer.drop(tok_emb)

            for li, block in enumerate(self.transformer.h):
                x = block(x, attention_mask=attention_mask)
                # Do not clone to keep memory low; caller should consume before next yield
                yield li, x

            if include_final_norm:
                x_ln = self.transformer.ln_f(x)
                yield "final", x_ln

        finally:
            if was_training:
                self.train()
