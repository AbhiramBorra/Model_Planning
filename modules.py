# %%
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#########################
# Basic Modules
#########################

class LayerNorm(nn.Module):
    """LayerNorm with an optional bias."""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

#########################
# Rotary Positional Embedding
#########################

class RotaryPositionalEmbedding(nn.Module):
    """
    Computes rotary positional embeddings and applies them to query and key.
    """
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        # Compute inverse frequencies for rotary embedding
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float)
        # Compute sinusoidal frequencies for all positions
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # shape: (max_seq_len, dim)
        self.register_buffer("cos_emb", emb.cos())
        self.register_buffer("sin_emb", emb.sin())

    def forward(self, q, k):
        """
        Applies rotary embedding to queries and keys.
        q, k: tensors of shape (B, n_head, T, head_dim)
        """
        T = q.size(-2)
        # Slice the precomputed cos and sin embeddings for the current sequence length
        cos = self.cos_emb[:T, :].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin = self.sin_emb[:T, :].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        # Apply rotary transformation
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k

    @staticmethod
    def rotate_half(x):
        # Splits last dimension in half and rotates them.
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

#########################
# Self-Attention with Rotary Embedding and Flash Attention
#########################

class RotarySelfAttention(nn.Module):
    """
    Self-attention module supporting rotary positional embeddings and flash attention.
    The module can function as either encoder-only (bidirectional) or decoder-only (causal)
    based on config["is_decoder"].
    """
    def __init__(self, config):
        super().__init__()
        assert config["n_embd"] % config["n_head"] == 0, "Embedding dim must be divisible by number of heads"
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.head_dim = self.n_embd // self.n_head
        self.is_decoder = config.get("is_decoder", True)
        self.dropout = config["dropout"]

        # Linear projection for q, k, v
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config["bias"])
        # Output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config["bias"])
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Rotary positional embedding module (if enabled)
        self.use_rotary = config.get("use_rotary", True)
        if self.use_rotary:
            self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_seq_len=config["max_seq_len"])
        else:
            self.rotary_emb = None

        # Use flash attention if available and enabled in config.
        self.use_flash = config.get("use_flash", True) and hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        if self.is_decoder:
            # Precompute a causal mask for slower attention versions.
            self.register_buffer("look_ahead", torch.tril(torch.ones(config["block_size"], config["block_size"]))
                                          .view(1, 1, config["block_size"], config["block_size"]))

    def forward(self, x, padding_mask=None):
        """
        x: tensor of shape (B, T, n_embd)
        """
        B, T, C = x.size()
        qkv = self.c_attn(x)  # (B, T, 3*n_embd)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings to q and k if enabled
        if self.use_rotary:
            q, k = self.rotary_emb(q, k)

        # Use flash attention if available
        if self.use_flash:
            if self.is_decoder and (padding_mask is not None):
                #print("self.look_ahead.shape:", self.look_ahead.shape)
                causal_mask = self.look_ahead[:, :, :T, :T]
                #print("causal_mask.shape:", causal_mask.shape)
                # Expand causal mask to shape (B, 1, T, T)
                #causal_mask = causal_mask.expand(B, -1, -1, -1)
                causal_mask = causal_mask.expand(B, 1, T, T)
                #print("causal_mask.shape (expanded):", causal_mask.shape)
                # Expand padding_mask: (B, T) -> (B, 1, 1, T)
                pad_mask = padding_mask.unsqueeze(1).unsqueeze(2) # shape: (B, 1, 1, T)
                #print("pad_mask:",pad_mask.shape)
                # Combine: a key position is allowed if it's both within the causal window and not padded.
                combined_mask = causal_mask.bool() & pad_mask.bool()
                #print("combined_mask:",combined_mask.shape)
                # Create the final attention mask: allowed positions = 0, disallowed = -inf
                attn_mask = torch.zeros(B, 1, T, T, device=x.device, dtype=q.dtype)
                #print("attn_mask:",attn_mask.shape)
                attn_mask = attn_mask.masked_fill(~combined_mask, float('-inf'))
                #print("attn_mask (filled):",attn_mask.shape)
                # Unsqueeze to shape (B, 1, T, T) so it broadcasts across heads.
                #attn_mask = attn_mask.unsqueeze(1)
                # Now, call scaled_dot_product_attention with is_causal=False because we already incorporated the causal mask.
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False,
                    attn_mask=attn_mask,
                )

            elif self.is_decoder and (padding_mask is None):
                # No padding mask provided; we can use the built-in causal mechanism.
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True,
                    attn_mask=None,
                )
            elif (not self.is_decoder) and (padding_mask is not None):
                # For encoder (bidirectional) mode: padding_mask is of shape (B, T) with valid tokens=1 and padded tokens=0.
                # Unsqueeze once to shape (B, 1, T)
                pad_mask = padding_mask.unsqueeze(1)  # shape: (B, 1, T)
                # Create an attention mask of shape (B, T, T)
                attn_mask = torch.zeros(B, T, T, device=x.device, dtype=q.dtype)
                attn_mask = attn_mask.masked_fill(pad_mask.expand(B, T, T) == 0, float('-inf'))
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, T, T)
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False,
                    attn_mask=attn_mask,
                )

            elif (not self.is_decoder) and (padding_mask is None):
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False,
                    attn_mask=None,
                )
            else:
                raise ValueError("Invalid combination of is_decoder and padding_mask")
        else:
            # Manual attention computation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if self.is_decoder:
                att = att.masked_fill(self.look_ahead[:, :, :T, :T] == 0, float('-inf'))
            if padding_mask is not None:
                # Expand padding_mask: (B, T) --> (B, 1, 1, T)
                pad_mask = padding_mask.unsqueeze(1).unsqueeze(2)
                att = att.masked_fill(pad_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reshape back to (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

#########################
# MLP Module
#########################

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.get("mlp_hidden_dim", 4 * config["n_embd"])
        self.fc = nn.Linear(config["n_embd"], hidden_dim, bias=config["bias"])
        self.gelu = nn.GELU()
        self.proj = nn.Linear(hidden_dim, config["n_embd"], bias=config["bias"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x

#########################
# Transformer Block
#########################

class TransformerBlock(nn.Module):
    """
    Transformer block composed of:
      - A pre-attention layer norm
      - Self-attention (with rotary embeddings and flash attention support)
      - A post-attention layer norm and MLP with residual connections.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config["n_embd"], bias=config["bias"])
        self.attn = RotarySelfAttention(config)
        self.ln_2 = LayerNorm(config["n_embd"], bias=config["bias"])
        self.mlp = MLP(config)

    def forward(self, x, padding_mask=None):
        x = x + self.attn(self.ln_1(x), padding_mask=padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

# %%
#########################
# Example Config & Usage
#########################

# Large configuration dictionary with sub-dictionary style parameters.
config = {
    "n_embd": 768,
    "n_head": 12,
    "dropout": 0.1,
    "bias": False,
    "block_size": 1024,    # maximum block size for causal mask (if not using flash attention)
    "max_seq_len": 1024,   # maximum sequence length for rotary embeddings
    "is_decoder": False,    # Set to False for encoder-only (bidirectional) attention
    "use_flash": True,     # Use flash attention if available (requires PyTorch >= 2.0)
    "use_rotary": True,    # Enable rotary positional embeddings
    # Optionally, you can specify MLP hidden dimension:
    "mlp_hidden_dim": 4 * 768,
}

## TEST CODE ############################################################
def test_attention_combinations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 2
    T = 16
    n_embd = 32
    n_head = 4

    # Create a dummy input tensor of shape (B, T, n_embd)
    x = torch.randn(B, T, n_embd, device=device)
    
    # Create a dummy padding mask of shape (B, T)
    # Assume valid tokens are marked with 1 (True) and padded tokens with 0 (False).
    # For example, mark the last 4 tokens in each sequence as padded.
    padding_mask = torch.ones(B, T, device=device, dtype=torch.bool)
    padding_mask[:, -4:] = 0  # Last 4 tokens are padded

    # We'll test all combinations of:
    # - use_flash: True or False
    # - is_decoder: True (causal) or False (bidirectional)
    # - padding_mask provided: True or False
    combinations = []
    for use_flash in [True, False]:
        for is_decoder in [True, False]:
            for pad_mask_provided in [True, False]:
                combinations.append((use_flash, is_decoder, pad_mask_provided))
    
    print("Starting tests over 8 combinations:\n")
    for idx, (use_flash, is_decoder, pad_mask_provided) in enumerate(combinations, start=1):
        # Build configuration dictionary.
        config = {
            "n_embd": n_embd,
            "n_head": n_head,
            "dropout": 0.0,        # Set dropout to 0.0 for deterministic behavior
            "bias": False,
            "block_size": 64,      # Must be >= T
            "max_seq_len": 128,
            "is_decoder": is_decoder,
            "use_flash": use_flash,
            "use_rotary": True,
        }
        # Instantiate the RotarySelfAttention module with the given config.
        attn_module = RotarySelfAttention(config).to(device)
        
        # Determine whether to pass a padding mask.
        mask = padding_mask if pad_mask_provided else None
        
        try:
            out = attn_module(x, padding_mask=mask)
            assert out.shape == (B, T, n_embd), f"Output shape mismatch: got {out.shape}"
            print(f"Combination {idx}: use_flash={use_flash}, is_decoder={is_decoder}, padding_mask_provided={pad_mask_provided} -> Output shape: {out.shape}")
        except Exception as e:
            print(f"Combination {idx}: use_flash={use_flash}, is_decoder={is_decoder}, padding_mask_provided={pad_mask_provided} -> Error: {e}")

def test_rotary_norm_preservation():
    """
    Test that the rotary positional embedding applies a rotation and preserves the vector norms.
    """
    # Parameters
    B = 2          # Batch size
    n_head = 2     # Number of heads
    T = 10         # Sequence length
    head_dim = 16  # Dimension per head

    # Instantiate the rotary embedding module.
    rotary = RotaryPositionalEmbedding(head_dim, max_seq_len=100)
    
    # Create a dummy query tensor of shape (B, n_head, T, head_dim)
    q = torch.randn(B, n_head, T, head_dim)
    
    # We can simply use q for both queries and keys (the operation is applied independently).
    q_rot, _ = rotary(q, q)
    
    # Compute norms along the last dimension.
    norm_orig = torch.norm(q, dim=-1)      # shape: (B, n_head, T)
    norm_rot = torch.norm(q_rot, dim=-1)     # shape: (B, n_head, T)
    
    # Calculate the maximum absolute difference.
    max_diff = torch.abs(norm_orig - norm_rot).max().item()
    print("Maximum norm difference after rotary rotation:", max_diff)
    
    # Check that the difference is negligible (within tolerance).
    assert max_diff < 1e-5, "Rotary positional embedding does not preserve vector norms!"
    print("Test passed: Norm preservation verified.")

def test_gpu_flash():
    """
    Test the RotarySelfAttention module on GPU with flash attention activated.
    This function explicitly checks that:
      - A CUDA device is available.
      - The configuration forces flash attention.
      - The module and inputs are moved to GPU.
      - A forward pass runs without error and produces the expected output shape.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU test.")
        return

    device = torch.device("cuda")
    print("Running GPU test with flash attention activated on device:", device)

    B = 2
    T = 16
    n_embd = 32
    n_head = 4
    config = {
        "n_embd": n_embd,
        "n_head": n_head,
        "dropout": 0.0,
        "bias": False,
        "block_size": 64,     # Must be >= T
        "max_seq_len": 128,
        "is_decoder": True,   # Testing decoder (causal) mode
        "use_flash": True,    # Force flash attention usage
        "use_rotary": True,
    }
    attn_module = RotarySelfAttention(config).to(device)
    x = torch.randn(B, T, n_embd, device=device)
    # Create a padding mask on GPU: valid tokens are 1 and padded tokens are 0.
    padding_mask = torch.ones(B, T, device=device, dtype=torch.bool)
    padding_mask[:, -4:] = 0

    # Check that our configuration really uses flash attention.
    print("Flash attention activated:", attn_module.use_flash)

    try:
        out = attn_module(x, padding_mask=padding_mask)
        assert out.shape == (B, T, n_embd), f"Output shape mismatch: got {out.shape}"
        print("GPU test with flash attention passed. Output shape:", out.shape)
    except Exception as e:
        print("GPU test with flash attention failed with error:", e)

if __name__ == "__main__":
    
    # Create a transformer block instance using the config.
    block = TransformerBlock(config)
    # Create a dummy input: (batch, sequence length, embedding dim)
    x = torch.randn(2, 128, config["n_embd"])
    # Forward pass through the block
    y = block(x)
    print("Transformer block output shape:", y.shape)

    # Define a large configuration dictionary.
    config = {
        "n_embd": 768,
        "n_head": 12,
        "dropout": 0.1,
        "bias": False,
        "block_size": 1024,
        "max_seq_len": 1024,
        "is_decoder": True,    # Change to False for encoder-only (bidirectional) mode.
        "use_flash": True,     # Enable flash attention if available.
        "use_rotary": True,    # Enable rotary positional embeddings.
        "mlp_hidden_dim": 4 * 768,
    }
    
    # Set a fixed random seed for reproducibility.
    torch.manual_seed(42)
    
    # Test LayerNorm.
    ln = LayerNorm(config["n_embd"], bias=config["bias"])
    x_ln = torch.randn(2, 128, config["n_embd"])
    ln_out = ln(x_ln)
    print("LayerNorm output shape:", ln_out.shape)
    assert ln_out.shape == x_ln.shape, "LayerNorm output shape mismatch"
    
    # Test RotaryPositionalEmbedding.
    B, T = 2, 128
    n_head = config["n_head"]
    head_dim = config["n_embd"] // n_head
    rotary = RotaryPositionalEmbedding(head_dim, max_seq_len=config["max_seq_len"])
    # Dummy queries and keys: shape (B, n_head, T, head_dim)
    q_dummy = torch.randn(B, n_head, T, head_dim)
    k_dummy = torch.randn(B, n_head, T, head_dim)
    q_rot, k_rot = rotary(q_dummy, k_dummy)
    print("RotaryPositionalEmbedding output shapes:", q_rot.shape, k_rot.shape)
    assert q_rot.shape == q_dummy.shape and k_rot.shape == k_dummy.shape, "Rotary output shape mismatch"
    
    # Test RotarySelfAttention.
    self_attn = RotarySelfAttention(config)
    x_att = torch.randn(2, 128, config["n_embd"])
    attn_out = self_attn(x_att)
    print("RotarySelfAttention output shape:", attn_out.shape)
    assert attn_out.shape == x_att.shape, "SelfAttention output shape mismatch"
    
    # Test MLP.
    mlp = MLP(config)
    x_mlp = torch.randn(2, 128, config["n_embd"])
    mlp_out = mlp(x_mlp)
    print("MLP output shape:", mlp_out.shape)
    assert mlp_out.shape == x_mlp.shape, "MLP output shape mismatch"
    
    # Test TransformerBlock.
    block = TransformerBlock(config)
    x_block = torch.randn(2, 128, config["n_embd"])
    block_out = block(x_block)
    print("TransformerBlock output shape:", block_out.shape)
    assert block_out.shape == x_block.shape, "TransformerBlock output shape mismatch"
    
    print("All module tests passed!")


    test_attention_combinations()
    test_rotary_norm_preservation()
    test_gpu_flash()

# %%
