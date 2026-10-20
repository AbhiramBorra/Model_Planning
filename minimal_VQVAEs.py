import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules import *

class ResidualBlock(nn.Module):
    """
    Residual block for MLP with configurable activation and normalization.
    Implements: output = x + MLP(x)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation='gelu', dropout=0.1, layer_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_norm = layer_norm
        
        # MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Layer normalization
        if layer_norm:
            self.norm = nn.LayerNorm(input_dim)
        
        # If dimensions don't match, we need a projection for the residual
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
    
    def forward(self, x):
        # Store residual
        residual = x
        
        # Pre-norm if using layer norm
        if self.layer_norm:
            x = self.norm(x)
        
        # MLP
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        # Project residual if needed
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        # Add residual connection
        return x + residual

class CustomSequential(nn.Sequential):
    def forward(self, input, padding_mask):
        for module in self:
            input = module(input, padding_mask)
        return input

class VQVAE_single(nn.Module):
    def __init__(self, model_dim, hidden_dim, codebook_size, beta, codebook_reset_counter_multiplier, config):
        """
        Args:
        
        """
        super().__init__()
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.beta = beta
        self.codebook_reset_counter_multiplier = codebook_reset_counter_multiplier
        
        # NEW for codebook collapse - Store experimental feature settings from config
        if config is not None:
            self.cosine_push_weight = config['cosine_push_weight']
            self.entropy_loss_weight = config['entropy_loss_weight']
            self.mask_prob = config['mask_prob']
            self.entropy_temperature = config['entropy_temperature']
            self.usage_tracking_window = config['usage_tracking_window']
        else:
            # Default values for backward compatibility
            self.cosine_push_weight = 0.0
            self.entropy_loss_weight = 0.0
            self.mask_prob = 0.0
            self.entropy_temperature = 1.0
            self.usage_tracking_window = 0  # Disabled by default
        
        # For vector usage tracking
        if self.usage_tracking_window > 0:
            self.register_buffer('usage_history', torch.zeros(self.usage_tracking_window, dtype=torch.long))
            self.register_buffer('usage_ptr', torch.tensor(0, dtype=torch.long))
            self.register_buffer('usage_full', torch.tensor(False, dtype=torch.bool))
            self.register_buffer('total_vectors_processed', torch.tensor(0, dtype=torch.long))

        self.forward_proj = nn.Sequential(
            nn.Linear(model_dim, int(math.sqrt(model_dim*hidden_dim))),
            nn.ReLU(),
            nn.Linear(int(math.sqrt(model_dim*hidden_dim)), int(math.sqrt(model_dim*hidden_dim))),
            nn.ReLU(),
            nn.Linear(int(math.sqrt(model_dim*hidden_dim)), hidden_dim)
        )

        self.backward_proj = nn.Sequential(
            nn.Linear(hidden_dim, int(math.sqrt(model_dim*hidden_dim))),
            nn.ReLU(),
            nn.Linear(int(math.sqrt(model_dim*hidden_dim)), int(math.sqrt(model_dim*hidden_dim))),
            nn.ReLU(),
            nn.Linear(int(math.sqrt(model_dim*hidden_dim)), model_dim)
        )

        # Assuming the encoder outputs a latent with dimension d2.
        # We create a codebook (embedding table) of size (K, d2)
        self.codebook_size = codebook_size
        if self.codebook_size > 0:
            self.codebook = nn.Embedding(self.codebook_size, hidden_dim)
            # Initialize the codebook weights (you can choose a different init if desired)
            nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
        else:
            self.codebook = None

        # Initialize counters for codebook vectors as a buffer (non-trainable)
        if self.codebook_reset_counter_multiplier > 0:
            self.register_buffer(
                "codebook_counters", 
                torch.full((self.codebook_size,), 0)
            )
        
        # Add buffer for normalization value (single value for VQVAE_single)
        self.register_buffer('normalization_value', None)
    
    def normalize_codebook_vectors(self):
        """
        Normalize each codebook vector to unit length.
        
        This is applied after each parameter update when using cosine-push regularization.
        Helps ensure the cosine-push loss focuses on angular relationships rather than magnitudes.
        """
        if self.cosine_push_weight > 0 and self.codebook is not None:
            with torch.no_grad():
                # Normalize each row (codebook vector) to unit length
                self.codebook.weight.data = F.normalize(self.codebook.weight.data, p=2, dim=1)
    
    def compute_cosine_push_loss(self):
        """
        Compute cosine-push regularization loss on the codebook matrix.
        
        The cosine-push loss encourages codebook vectors to be orthogonal by penalizing
        the squared cosine similarities between different vectors. This replaces the
        gram-matrix Frobenius norm with a more direct angular penalty.
        
        If usage tracking is enabled, only considers used vectors and weights each
        vector's contribution by its usage frequency.
        
        L_push = sum_{i≠j} w_i * w_j * (dot(e_i, e_j))^2
        
        Where e_i and e_j are normalized codebook vectors, and w_i is the usage weight.
        
        Returns:
            cosine_push_loss: Cosine-push loss scalar
        """
        if self.cosine_push_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # If usage tracking is enabled, use only tracked vectors with usage-based weighting
        if self.usage_tracking_window > 0:
            usage_stats = self.get_usage_statistics()
            usage_counts = usage_stats['usage_counts']
            used_mask = usage_counts > 0
            used_indices = torch.where(used_mask)[0]
            
            if len(used_indices) <= 1:
                # Need at least 2 vectors to compute similarities
                return torch.tensor(0.0, device=next(self.parameters()).device)
            
            # Get used codebook vectors and normalize them
            E_used = self.codebook.weight[used_indices]  # (num_used, hidden_dim)
            E_used_normalized = F.normalize(E_used, p=2, dim=1)  # (num_used, hidden_dim)
            
            # Compute cosine similarity matrix for used vectors: (num_used, num_used)
            cosine_sim_matrix = torch.mm(E_used_normalized, E_used_normalized.t())
            
            # Get usage weights for used vectors and normalize them
            usage_weights = usage_counts[used_indices].float()  # (num_used,)
            usage_weights = usage_weights / usage_weights.sum()  # Normalize to sum to 1
            
            # Create weight matrix: w_i * w_j for all pairs
            weight_matrix = usage_weights.unsqueeze(1) * usage_weights.unsqueeze(0)  # (num_used, num_used)
            
            # Zero out diagonal (self-similarities) and apply weights
            num_used = len(used_indices)
            mask = ~torch.eye(num_used, device=E_used.device, dtype=torch.bool)
            
            # Apply weights and compute weighted sum of squared cosine similarities
            weighted_cosines_squared = (cosine_sim_matrix ** 2) * weight_matrix
            cosine_push_loss = weighted_cosines_squared[mask].sum()
            
        else:
            # Original implementation when usage tracking is disabled
            # Get codebook matrix E: (K, hidden_dim)
            E = self.codebook.weight  # (codebook_size, hidden_dim)
            
            # Normalize vectors to unit length for cosine computation
            E_normalized = F.normalize(E, p=2, dim=1)  # (K, hidden_dim)
            
            # Compute cosine similarity matrix: (K, K)
            cosine_sim_matrix = torch.mm(E_normalized, E_normalized.t())  # (K, K)
            
            # Zero out diagonal (self-similarities) and compute sum of squared off-diagonal elements
            mask = ~torch.eye(self.codebook_size, device=E.device, dtype=torch.bool)
            off_diagonal_cosines = cosine_sim_matrix[mask]  # Get all off-diagonal elements
            
            # Sum of squared cosine similarities between different vectors
            cosine_push_loss = torch.sum(off_diagonal_cosines ** 2)
        
        return cosine_push_loss
    
    def compute_soft_entropy_loss(self, distances, padding_mask=None):
        """
        Compute entropy loss using soft assignments from current batch.
        This maintains gradient flow through the assignment probabilities.
        
        Args:
            distances: Distance tensor (B*T, K) between encoder outputs and codebook
            padding_mask: Optional padding mask to exclude padded positions (B, T)
            
        Returns:
            entropy_loss: Soft entropy loss scalar
        """
        if self.entropy_loss_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=distances.device)
        
        # Convert distances to probabilities using softmax
        # Negative distances because smaller distance = higher probability
        assignment_probs = F.softmax(-distances / self.entropy_temperature, dim=-1)  # (B*T, K)
        
        # Apply padding mask if provided
        if padding_mask is not None:
            B, T = padding_mask.shape
            mask_flat = padding_mask.view(-1)  # (B*T,)
            # Only consider valid (non-padded) positions
            valid_probs = assignment_probs[mask_flat == 1]  # (valid_positions, K)
            if valid_probs.numel() == 0:
                return torch.tensor(0.0, device=distances.device)
            avg_probs = valid_probs.mean(dim=0)  # (K,)
        else:
            avg_probs = assignment_probs.mean(dim=0)  # (K,)
        
        # Compute entropy: -sum(p * log(p))
        # We want to maximize entropy (uniform distribution), so minimize negative entropy
        epsilon = 1e-10
        avg_probs_safe = torch.clamp(avg_probs, min=epsilon)
        entropy = -torch.sum(avg_probs_safe * torch.log(avg_probs_safe))
        
        # Convert to "equal usage" loss: penalize low entropy
        max_entropy = torch.log(torch.tensor(self.codebook_size, device=distances.device, dtype=avg_probs.dtype))
        entropy_loss = max_entropy - entropy  # Higher when entropy is low
        
        return entropy_loss
    
    def update_usage_tracking(self, encoding_indices, padding_mask=None):
        """
        Update the sliding window of vector usage for statistics tracking.
        
        Args:
            encoding_indices: Tensor of assignment indices (B*T,)
            padding_mask: Optional padding mask to exclude padded positions (B, T)
        """
        if self.usage_tracking_window <= 0:
            return
            
        # Filter out padded positions if mask is provided
        if padding_mask is not None:
            B, T = padding_mask.shape
            mask_flat = padding_mask.view(-1)  # (B*T,)
            valid_indices = encoding_indices[mask_flat == 1]
        else:
            valid_indices = encoding_indices
            
        # Update total vectors processed
        self.total_vectors_processed += valid_indices.size(0)
            
        # Add valid assignments to circular buffer
        n_valid = valid_indices.size(0)
        if n_valid > 0:
            # Handle case where we have more assignments than buffer size
            if n_valid >= self.usage_tracking_window:
                # Fill entire buffer with most recent assignments
                self.usage_history[:] = valid_indices[-self.usage_tracking_window:]
                self.usage_ptr.fill_(0)
                self.usage_full.fill_(True)
            else:
                # Add assignments to circular buffer
                end_ptr = (self.usage_ptr + n_valid) % self.usage_tracking_window
                
                if end_ptr > self.usage_ptr:
                    # No wraparound
                    self.usage_history[self.usage_ptr:end_ptr] = valid_indices
                else:
                    # Wraparound case
                    n_until_end = self.usage_tracking_window - self.usage_ptr
                    self.usage_history[self.usage_ptr:] = valid_indices[:n_until_end]
                    if n_valid > n_until_end:
                        self.usage_history[:end_ptr] = valid_indices[n_until_end:]
                
                self.usage_ptr.copy_(end_ptr)
                if self.usage_ptr == 0 or self.usage_full:
                    self.usage_full.fill_(True)
    
    def get_usage_statistics(self):
        """
        Get current usage statistics.
        
        Returns:
            dict: Dictionary containing usage statistics
        """
        if self.usage_tracking_window <= 0 or self.codebook is None:
            return {
                'total_vectors_processed': 0,
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size if self.codebook is not None else 0),
                'usage_percentages': torch.zeros(self.codebook_size if self.codebook is not None else 0)
            }
        
        # Get the valid portion of usage history
        if self.usage_full:
            valid_history = self.usage_history
        else:
            valid_history = self.usage_history[:self.usage_ptr]
            
        if valid_history.numel() == 0:
            return {
                'total_vectors_processed': self.total_vectors_processed.item(),
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size),
                'usage_percentages': torch.zeros(self.codebook_size)
            }
        
        # Compute usage statistics
        usage_counts = torch.bincount(valid_history, minlength=self.codebook_size)
        unique_vectors_used = (usage_counts > 0).sum().item()
        usage_percentages = usage_counts.float() / valid_history.numel()
        
        return {
            'total_vectors_processed': self.total_vectors_processed.item(),
            'unique_vectors_used': unique_vectors_used,
            'usage_counts': usage_counts,
            'usage_percentages': usage_percentages
        }
    
    def compute_codebook_similarities(self):
        """
        Compute cosine similarities between used codebook vectors.
        
        Returns:
            dict: Dictionary containing similarity statistics
        """
        if self.codebook is None:
            return {
                'similarities': None,
                'used_indices': None,
                'num_used_vectors': 0
            }
        
        # Get usage statistics to find which vectors are used
        usage_stats = self.get_usage_statistics()
        used_mask = usage_stats['usage_counts'] > 0
        used_indices = torch.where(used_mask)[0]
        
        if len(used_indices) <= 1:
            return {
                'similarities': None,
                'used_indices': used_indices,
                'num_used_vectors': len(used_indices)
            }
        
        # Get used codebook vectors
        with torch.no_grad():
            used_codebook = self.codebook.weight[used_indices]
            # Normalize for cosine similarity
            used_codebook_norm = used_codebook / used_codebook.norm(dim=1, keepdim=True)
            # Compute similarity matrix
            similarities = torch.mm(used_codebook_norm, used_codebook_norm.t())
        
        return {
            'similarities': similarities,
            'used_indices': used_indices,
            'num_used_vectors': len(used_indices)
        }
    
    def apply_stochastic_mask(self, distances, training=True):
        """
        Apply stochastic masking to assignment distances for exploration.
        
        With probability mask_prob, randomly masks a fraction of the distance
        values to very large numbers, forcing the model to explore unused codes.
        This helps prevent codebook collapse by encouraging diversity.
        
        Args:
            distances: Distance tensor (B*T, K) between encoder outputs and codebook
            training: Whether model is in training mode
            
        Returns:
            masked_distances: Distances with stochastic masking applied
        """
        if self.mask_prob <= 0 or not training:
            return distances
            
        # Apply masking with probability mask_prob
        if torch.rand(1).item() < self.mask_prob:
            B_T, K = distances.shape
            
            # Create random mask - randomly select fraction mask_prob of entries to mask
            num_mask = int(self.mask_prob * B_T * K)
            mask_indices = torch.randperm(B_T * K, device=distances.device)[:num_mask]
            
            # Convert flat indices to 2D indices
            mask_rows = mask_indices // K
            mask_cols = mask_indices % K
            
            # Create masked distances by setting selected entries to large value
            masked_distances = distances.clone()
            masked_distances[mask_rows, mask_cols] = 1e10
            
            return masked_distances
        
        return distances
        
    def normalize(self, x):
        """
        Normalize input using the computed normalization value.
        
        Args:
            x: Input tensor of shape (B, T, d)
            
        Returns:
            Normalized tensor of the same shape
        """
        if self.normalization_value is not None:
            return x / (self.normalization_value + 1e-8)
        else:
            return x
    
    def denormalize(self, x):
        """
        Denormalize input using the computed normalization value.
        
        Args:
            x: Input tensor of shape (B, T, d)
            
        Returns:
            Denormalized tensor of the same shape
        """
        if self.normalization_value is not None:
            return x * (self.normalization_value + 1e-8)
        else:
            return x
        
    def _compute_normalization_value(self, x, padding_mask=None):
        """
        Compute normalization value from the first batch.
        x: (B, T, d)
        padding_mask: (B, T)
        Returns: scalar normalization value
        """
        # Calculate norm in the last dimension: (B, T)
        norms = torch.norm(x, dim=-1)  # (B, T)
        
        if padding_mask is not None:
            # Apply mask to exclude padded positions
            mask = padding_mask.float()  # (B, T)
            masked_norms = norms * mask  # (B, T)
            
            # Calculate mean over valid positions
            sum_norms = masked_norms.sum()  # scalar
            count_valid = mask.sum()  # scalar
            
            # Calculate mean, avoiding division by zero
            norm_value = sum_norms / (count_valid + 1e-8)  # scalar
        else:
            # No mask, simple mean over all positions
            norm_value = norms.mean()  # scalar
            
        return norm_value
        
    def forward(self, x, padding_mask=None, beta=None):
        # Compute normalization value if not already computed
        if self.normalization_value is None:
            norm_value = self._compute_normalization_value(x, padding_mask)
            self.register_buffer('normalization_value', norm_value)
        
        # Apply normalization
        x_normalized = self.normalize(x)

        # Encode: get continuous latent representation z_e from the encoder.
        # Expected shape: (B, T, hidden_dim)
        z_e = self.forward_proj(x_normalized)
        unique_count = torch.tensor(0, device=x.device)
        if self.codebook is not None:
            # Flatten z_e to shape (B*T, d2) for vector quantization.
            B, T, hidden_dim = z_e.shape
            z_e_flat = z_e.view(-1, hidden_dim)  # shape: (B*T, hidden_dim)
            
            # Get the codebook (embedding table): shape (K, d2)
            codebook = self.codebook.weight  # shape: (codebook_size, hidden_dim)
            
            # Compute L2 distances between each encoder output and each codebook entry.
            # distances: shape (B*T, codebook_size)
            distances = torch.sum(z_e_flat**2, dim=1, keepdim=True) + torch.sum(codebook**2, dim=1) - 2 * torch.matmul(z_e_flat, codebook.t())
            
            # Apply stochastic masking for exploration (if enabled)
            distances = self.apply_stochastic_mask(distances, training=self.training)
            
            # Compute soft entropy loss BEFORE argmin (to maintain gradients)
            entropy_loss = self.compute_soft_entropy_loss(distances, padding_mask)
            
            # For each latent vector, find the nearest codebook entry.
            encoding_indices = torch.argmin(distances, dim=1)  # shape: (B*T,)
            
            # Update usage tracking
            self.update_usage_tracking(encoding_indices, padding_mask)
            if self.codebook_reset_counter_multiplier > 0:
                # Decrement all counters by the batch size.
                BT = B * T
                self.codebook_counters -= BT
                # Reset counters for the used indices.
                self.codebook_counters[encoding_indices] = self.codebook_reset_counter_multiplier * self.codebook_size
                # Check for any collapsed codebook vectors.
                collapsed = self.codebook_counters <= 0
                if collapsed.any():
                    num_collapsed = int(collapsed.sum().item())
                    collapsed_indices = torch.where(collapsed)[0]
                    if num_collapsed > BT:
                        # If there are more collapsed vectors than BT,
                        # Select BT vectors from z_e_flat without replacement.
                        new_batch_vectors = z_e_flat[torch.randperm(BT)[:BT]]
                        # For the remaining collapsed vectors, compute overall mean and std per vector.
                        # Calculate the mean of each row in z_e_flat and then average over rows.
                        overall_mean = z_e_flat.mean(dim=1).mean()
                        # Calculate the standard deviation of each row and then average over rows.
                        overall_std = z_e_flat.std(dim=1).mean()
                        rem = num_collapsed - BT
                        random_vectors = torch.normal(mean=overall_mean.item(), std=overall_std.item(), size=(rem, z_e_flat.shape[1]), device=z_e_flat.device)
                        # To maintain a consistent assignment order, sort the collapsed indices.
                        sorted_indices = collapsed_indices[torch.argsort(collapsed_indices)]
                        self.codebook.weight.data[sorted_indices[:BT]] = new_batch_vectors.to(self.codebook.weight.dtype)
                        self.codebook.weight.data[sorted_indices[BT:]] = random_vectors.to(self.codebook.weight.dtype)
                    else:
                        # If number of collapsed vectors is less than or equal to BT,
                        # update them with randomly selected vectors from z_e_flat without replacement.
                        new_vectors = z_e_flat[torch.randperm(BT)[:num_collapsed]]
                        self.codebook.weight.data[collapsed] = new_vectors.to(self.codebook.weight.dtype)
                    # Reset the counters for these codebook entries.
                    self.codebook_counters[collapsed] = self.codebook_reset_counter_multiplier * self.codebook_size
            unique_count = torch.unique(encoding_indices).numel()
            # Quantize: lookup the codebook entries.
            z_q_flat = codebook[encoding_indices]  # shape: (B*T, hidden_dim)
            z_q = z_q_flat.view(B, T, hidden_dim)  # shape: (B, T, hidden_dim)
            
            # Compute the losses.
            # The codebook loss encourages the codebook vectors to move towards the encoder outputs.
            if padding_mask is not None:
                # Expand padding mask to match dimensions (B,T,d2)
                active_mask = padding_mask.unsqueeze(-1).float()
                # Compute MSE only over active positions and average over them
                codebook_loss = (F.mse_loss(z_q.detach() * active_mask, z_e * active_mask, reduction='sum') / 
                               (active_mask.sum() * z_e.size(-1)))
                # The commitment loss encourages the encoder outputs to commit to a code.
                if beta is None:
                    commitment_loss = self.beta * (F.mse_loss(z_q * active_mask, z_e.detach() * active_mask, reduction='sum') /
                                                 (active_mask.sum() * z_e.size(-1)))
                else:
                    commitment_loss = beta * (F.mse_loss(z_q * active_mask, z_e.detach() * active_mask, reduction='sum') /
                                            (active_mask.sum() * z_e.size(-1)))
            else:
                codebook_loss = F.mse_loss(z_q.detach(), z_e)
                if beta is None:
                    commitment_loss = self.beta * F.mse_loss(z_q, z_e.detach())
                else:
                    commitment_loss = beta * F.mse_loss(z_q, z_e.detach())
            
            # Use the straight-through estimator:
            z_q = z_e + (z_q - z_e).detach()
        else:
            # No-codebook mode: bypass quantization.
            z_q = z_e
            codebook_loss = torch.tensor(0.0, device=z_e.device)
            commitment_loss = torch.tensor(0.0, device=z_e.device)
            entropy_loss = torch.tensor(0.0, device=z_e.device)
            
        
        x_recon = self.backward_proj(z_q)
        
        if padding_mask is not None:
            # Expand padding_mask to match x_recon/x dimensions
            active_mask = padding_mask.unsqueeze(-1).unsqueeze(1).float()
            # Compute MSE only over active (non-padded) positions and average over them
            recon_loss = F.mse_loss(x_recon * active_mask, x_normalized * active_mask, reduction='sum') / active_mask.sum()
        else:
            recon_loss = F.mse_loss(x_recon, x_normalized)
        
        # Compute cosine-push regularization loss
        cosine_push_loss = self.compute_cosine_push_loss()
        
        # entropy_loss is already computed in the quantization section above
        if self.codebook is None:
            entropy_loss = torch.tensor(0.0, device=z_e.device)
        
        total_loss = recon_loss + codebook_loss + commitment_loss + \
                    self.cosine_push_weight * cosine_push_loss + \
                    self.entropy_loss_weight * entropy_loss
        
        return x_recon, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count, cosine_push_loss, entropy_loss


class VQVAE1(nn.Module):
    def __init__(self, encoder, decoder, config):
        """
        Args:
            encoder (nn.Module): An instance of Encoder1 that outputs latents of shape (B, T, d2).
            decoder (nn.Module): An instance of Decoder1 that takes latents of shape (B, T, d2) and produces reconstructions of shape (B, L, T, d).
            codebook_size (int): The number of discrete latent vectors (K).
            beta (float): Commitment loss coefficient.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codebook_size = config["codebook_size"]
        self.beta = config["beta"]
        #self.codebook_size = codebook_size
        #self.beta = beta


        # Assuming the encoder outputs a latent with dimension d2.
        # We create a codebook (embedding table) of size (K, d2)
        self.codebook_size = config["codebook_size"]
        if self.codebook_size > 0:
            self.codebook = nn.Embedding(self.codebook_size, encoder.d2)
            # Initialize the codebook weights (you can choose a different init if desired)
            nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
        else:
            self.codebook = None


        self.codebook_reset_counter_multiplier = config["codebook_reset_counter_multiplier"]
        # Initialize counters for codebook vectors as a buffer (non-trainable)
        if self.codebook_reset_counter_multiplier > 0:
            self.register_buffer(
                "codebook_counters", 
                torch.full((self.codebook_size,), 0)
            )
        
        # Add buffer for normalization values (L values for VQVAE1)
        self.register_buffer('normalization_values', None)
        
        # NEW for codebook collape
        # Store experimental feature settings from config
        self.cosine_push_weight = config['cosine_push_weight']  # Replaces ortho_loss_weight
        self.entropy_loss_weight = config['entropy_loss_weight']
        self.mask_prob = config['mask_prob']
        
        # For soft entropy loss: temperature parameter for softmax
        self.entropy_temperature = config['entropy_temperature']  # Temperature for soft assignments
        
        # For vector usage tracking
        self.usage_tracking_window = config.get('usage_tracking_window', 2000)  # Window size for tracking usage
        if self.usage_tracking_window > 0:
            self.register_buffer('usage_history', torch.zeros(self.usage_tracking_window, dtype=torch.long))
            self.register_buffer('usage_ptr', torch.tensor(0, dtype=torch.long))
            self.register_buffer('usage_full', torch.tensor(False, dtype=torch.bool))
            self.register_buffer('total_vectors_processed', torch.tensor(0, dtype=torch.long))
        

        
    def normalize_codebook_vectors(self):
        """
        Normalize each codebook vector to unit length.
        
        This is applied after each parameter update when using cosine-push regularization.
        Helps ensure the cosine-push loss focuses on angular relationships rather than magnitudes.
        """
        if self.cosine_push_weight > 0 and self.codebook is not None:
            with torch.no_grad():
                # Normalize each row (codebook vector) to unit length
                self.codebook.weight.data = F.normalize(self.codebook.weight.data, p=2, dim=1)
    
    def compute_cosine_push_loss(self):
        """
        Compute cosine-push regularization loss on the codebook matrix.
        
        The cosine-push loss encourages codebook vectors to be orthogonal by penalizing
        the squared cosine similarities between different vectors. This replaces the
        gram-matrix Frobenius norm with a more direct angular penalty.
        
        If usage tracking is enabled, only considers used vectors and weights each
        vector's contribution by its usage frequency.
        
        L_push = sum_{i≠j} w_i * w_j * (dot(e_i, e_j))^2
        
        Where e_i and e_j are normalized codebook vectors, and w_i is the usage weight.
        
        Returns:
            cosine_push_loss: Cosine-push loss scalar
        """
        if self.cosine_push_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # If usage tracking is enabled, use only tracked vectors with usage-based weighting
        if self.usage_tracking_window > 0:
            usage_stats = self.get_usage_statistics()
            usage_counts = usage_stats['usage_counts']
            used_mask = usage_counts > 0
            used_indices = torch.where(used_mask)[0]
            
            if len(used_indices) <= 1:
                # Need at least 2 vectors to compute similarities
                return torch.tensor(0.0, device=next(self.parameters()).device)
            
            # Get used codebook vectors and normalize them
            E_used = self.codebook.weight[used_indices]  # (num_used, d2)
            E_used_normalized = F.normalize(E_used, p=2, dim=1)  # (num_used, d2)
            
            # Compute cosine similarity matrix for used vectors: (num_used, num_used)
            cosine_sim_matrix = torch.mm(E_used_normalized, E_used_normalized.t())
            
            # Get usage weights for used vectors and normalize them
            usage_weights = usage_counts[used_indices].float()  # (num_used,)
            usage_weights = usage_weights / usage_weights.sum()  # Normalize to sum to 1
            
            # Create weight matrix: w_i * w_j for all pairs
            weight_matrix = usage_weights.unsqueeze(1) * usage_weights.unsqueeze(0)  # (num_used, num_used)
            
            # Zero out diagonal (self-similarities) and apply weights
            num_used = len(used_indices)
            mask = ~torch.eye(num_used, device=E_used.device, dtype=torch.bool)
            
            # Apply weights and compute weighted sum of squared cosine similarities
            weighted_cosines_squared = (cosine_sim_matrix ** 2) * weight_matrix
            cosine_push_loss = weighted_cosines_squared[mask].sum()
            
        else:
            # Original implementation when usage tracking is disabled
            # Get codebook matrix E: (K, d2)
            E = self.codebook.weight  # (codebook_size, d2)
            
            # Normalize vectors to unit length for cosine computation
            E_normalized = F.normalize(E, p=2, dim=1)  # (K, d2)
            
            # Compute cosine similarity matrix: (K, K)
            cosine_sim_matrix = torch.mm(E_normalized, E_normalized.t())  # (K, K)
            
            # Zero out diagonal (self-similarities) and compute sum of squared off-diagonal elements
            mask = ~torch.eye(self.codebook_size, device=E.device, dtype=torch.bool)
            off_diagonal_cosines = cosine_sim_matrix[mask]  # Get all off-diagonal elements
            
            # Sum of squared cosine similarities between different vectors
            cosine_push_loss = torch.sum(off_diagonal_cosines ** 2)
        
        return cosine_push_loss
    
    def compute_soft_entropy_loss(self, distances, padding_mask=None):
        """
        Compute entropy loss using soft assignments from current batch.
        This maintains gradient flow through the assignment probabilities.
        
        Args:
            distances: Distance tensor (B*T, K) between encoder outputs and codebook
            padding_mask: Optional padding mask to exclude padded positions (B, T)
            
        Returns:
            entropy_loss: Soft entropy loss scalar
        """
        if self.entropy_loss_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=distances.device)
        
        # Convert distances to probabilities using softmax
        # Negative distances because smaller distance = higher probability
        assignment_probs = F.softmax(-distances / self.entropy_temperature, dim=-1)  # (B*T, K)
        
        # Apply padding mask if provided
        if padding_mask is not None:
            B, T = padding_mask.shape
            mask_flat = padding_mask.view(-1)  # (B*T,)
            # Only consider valid (non-padded) positions
            valid_probs = assignment_probs[mask_flat == 1]  # (valid_positions, K)
            if valid_probs.numel() == 0:
                return torch.tensor(0.0, device=distances.device)
            avg_probs = valid_probs.mean(dim=0)  # (K,)
        else:
            avg_probs = assignment_probs.mean(dim=0)  # (K,)
        
        # Compute entropy: -sum(p * log(p))
        # We want to maximize entropy (uniform distribution), so minimize negative entropy
        epsilon = 1e-10
        avg_probs_safe = torch.clamp(avg_probs, min=epsilon)
        entropy = -torch.sum(avg_probs_safe * torch.log(avg_probs_safe))
        
        # Convert to "equal usage" loss: penalize low entropy
        max_entropy = torch.log(torch.tensor(self.codebook_size, device=distances.device, dtype=avg_probs.dtype))
        entropy_loss = max_entropy - entropy  # Higher when entropy is low
        
        return entropy_loss
    
    def update_usage_tracking(self, encoding_indices, padding_mask=None):
        """
        Update the sliding window of vector usage for statistics tracking.
        
        Args:
            encoding_indices: Tensor of assignment indices (B*T,)
            padding_mask: Optional padding mask to exclude padded positions (B, T)
        """
        if self.usage_tracking_window <= 0:
            return
            
        # Filter out padded positions if mask is provided
        if padding_mask is not None:
            B, T = padding_mask.shape
            mask_flat = padding_mask.view(-1)  # (B*T,)
            valid_indices = encoding_indices[mask_flat == 1]
        else:
            valid_indices = encoding_indices
            
        # Update total vectors processed
        self.total_vectors_processed += valid_indices.size(0)
            
        # Add valid assignments to circular buffer
        n_valid = valid_indices.size(0)
        if n_valid > 0:
            # Handle case where we have more assignments than buffer size
            if n_valid >= self.usage_tracking_window:
                # Fill entire buffer with most recent assignments
                self.usage_history[:] = valid_indices[-self.usage_tracking_window:]
                self.usage_ptr.fill_(0)
                self.usage_full.fill_(True)
            else:
                # Add assignments to circular buffer
                end_ptr = (self.usage_ptr + n_valid) % self.usage_tracking_window
                
                if end_ptr > self.usage_ptr:
                    # No wraparound
                    self.usage_history[self.usage_ptr:end_ptr] = valid_indices
                else:
                    # Wraparound case
                    n_until_end = self.usage_tracking_window - self.usage_ptr
                    self.usage_history[self.usage_ptr:] = valid_indices[:n_until_end]
                    if n_valid > n_until_end:
                        self.usage_history[:end_ptr] = valid_indices[n_until_end:]
                
                self.usage_ptr.copy_(end_ptr)
                if self.usage_ptr == 0 or self.usage_full:
                    self.usage_full.fill_(True)
    
    def get_usage_statistics(self):
        """
        Get current usage statistics.
        
        Returns:
            dict: Dictionary containing usage statistics
        """
        if self.usage_tracking_window <= 0 or self.codebook is None:
            return {
                'total_vectors_processed': 0,
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size if self.codebook is not None else 0),
                'usage_percentages': torch.zeros(self.codebook_size if self.codebook is not None else 0)
            }
        
        # Get the valid portion of usage history
        if self.usage_full:
            valid_history = self.usage_history
        else:
            valid_history = self.usage_history[:self.usage_ptr]
            
        if valid_history.numel() == 0:
            return {
                'total_vectors_processed': self.total_vectors_processed.item(),
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size),
                'usage_percentages': torch.zeros(self.codebook_size)
            }
        
        # Compute usage statistics
        usage_counts = torch.bincount(valid_history, minlength=self.codebook_size)
        unique_vectors_used = (usage_counts > 0).sum().item()
        usage_percentages = usage_counts.float() / valid_history.numel()
        
        return {
            'total_vectors_processed': self.total_vectors_processed.item(),
            'unique_vectors_used': unique_vectors_used,
            'usage_counts': usage_counts,
            'usage_percentages': usage_percentages
        }
    
    def compute_codebook_similarities(self):
        """
        Compute cosine similarities between used codebook vectors.
        
        Returns:
            dict: Dictionary containing similarity statistics
        """
        if self.codebook is None:
            return {
                'similarities': None,
                'used_indices': None,
                'num_used_vectors': 0
            }
        
        # Get usage statistics to find which vectors are used
        usage_stats = self.get_usage_statistics()
        used_mask = usage_stats['usage_counts'] > 0
        used_indices = torch.where(used_mask)[0]
        
        if len(used_indices) <= 1:
            return {
                'similarities': None,
                'used_indices': used_indices,
                'num_used_vectors': len(used_indices)
            }
        
        # Get used codebook vectors
        with torch.no_grad():
            used_codebook = self.codebook.weight[used_indices]
            # Normalize for cosine similarity
            used_codebook_norm = used_codebook / used_codebook.norm(dim=1, keepdim=True)
            # Compute similarity matrix
            similarities = torch.mm(used_codebook_norm, used_codebook_norm.t())
        
        return {
            'similarities': similarities,
            'used_indices': used_indices,
            'num_used_vectors': len(used_indices)
        }
        
    def apply_stochastic_mask(self, distances, training=True):
        """
        Apply stochastic masking to assignment distances for exploration.
        
        With probability mask_prob, randomly masks a fraction of the distance
        values to very large numbers, forcing the model to explore unused codes.
        This helps prevent codebook collapse by encouraging diversity.
        
        Args:
            distances: Distance tensor (B*T, K) between encoder outputs and codebook
            training: Whether model is in training mode
            
        Returns:
            masked_distances: Distances with stochastic masking applied
        """
        if self.mask_prob <= 0 or not training:
            return distances
            
        # Apply masking with probability mask_prob
        if torch.rand(1).item() < self.mask_prob:
            B_T, K = distances.shape
            
            # Create random mask - randomly select fraction mask_prob of entries to mask
            num_mask = int(self.mask_prob * B_T * K)
            mask_indices = torch.randperm(B_T * K, device=distances.device)[:num_mask]
            
            # Convert flat indices to 2D indices
            mask_rows = mask_indices // K
            mask_cols = mask_indices % K
            
            # Create masked distances by setting selected entries to large value
            masked_distances = distances.clone()
            masked_distances[mask_rows, mask_cols] = 1e10
            
            return masked_distances
        
        return distances
    
    def normalize(self, x):
        """
        Normalize input using the computed normalization values.
        
        Args:
            x: Input tensor of shape (B, L, T, d)
            
        Returns:
            Normalized tensor of the same shape
        """
        if self.normalization_values is not None:
            # Expand normalization values to match x shape: (1, L, 1, 1)
            norm_values_expanded = self.normalization_values.view(1, -1, 1, 1)
            return x / (norm_values_expanded + 1e-8)
        else:
            return x
    
    def denormalize(self, x):
        """
        Denormalize input using the computed normalization values.
        
        Args:
            x: Input tensor of shape (B, L, T, d)
            
        Returns:
            Denormalized tensor of the same shape
        """
        if self.normalization_values is not None:
            # Expand normalization values to match x shape: (1, L, 1, 1)
            norm_values_expanded = self.normalization_values.view(1, -1, 1, 1)
            return x * (norm_values_expanded + 1e-8)
        else:
            return x
        
    def _compute_normalization_values(self, x, padding_mask=None):
        """
        Compute normalization values from the first batch.
        x: (B, L, T, d)
        padding_mask: (B, T)
        Returns: (L,) tensor of normalization values
        """
        B, L, T, d = x.shape
        
        # Calculate norm in the last dimension: (B, L, T)
        norms = torch.norm(x, dim=-1)  # (B, L, T)
        
        if padding_mask is not None:
            # Expand padding mask to (B, L, T)
            mask = padding_mask.unsqueeze(1).expand(B, L, T).float()  # (B, L, T)
            
            # Apply mask and sum over B and T dimensions
            masked_norms = norms * mask  # (B, L, T)
            sum_norms = masked_norms.sum(dim=(0, 2))  # (L,)
            count_valid = mask.sum(dim=(0, 2))  # (L,)
            
            # Calculate mean, avoiding division by zero
            norm_values = sum_norms / (count_valid + 1e-8)  # (L,)
        else:
            # No mask, simple mean over B and T
            norm_values = norms.mean(dim=(0, 2))  # (L,)
            
        return norm_values
        
    def forward(self, x, padding_mask=None, beta=None):
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, T, d).
            padding_mask (Tensor, optional): Padding mask to be passed to encoder and decoder.
        
        Returns:
            x_recon (Tensor): Reconstruction output from the decoder (B, L, T, d).
            total_loss (Tensor): Sum of reconstruction, codebook, and commitment losses.
            recon_loss (Tensor): Reconstruction loss.
            codebook_loss (Tensor): Codebook loss.
            commitment_loss (Tensor): Commitment loss.
        """
        # Compute normalization values if not already computed
        if self.normalization_values is None:
            norm_values = self._compute_normalization_values(x, padding_mask)
            self.register_buffer('normalization_values', norm_values)
        
        # Apply normalization
        x_normalized = self.normalize(x)
        
        # Encode: get continuous latent representation z_e from the encoder.
        # Expected shape: (B, T, d2)
        z_e = self.encoder(x_normalized, padding_mask=padding_mask)
        unique_count = torch.tensor(0, device=x.device)
        if self.codebook is not None:
            # Flatten z_e to shape (B*T, d2) for vector quantization.
            B, T, d2 = z_e.shape
            z_e_flat = z_e.view(-1, d2)  # shape: (B*T, d2)
            
            # Get the codebook (embedding table): shape (K, d2)
            codebook = self.codebook.weight  # shape: (codebook_size, d2)
            
            # Compute L2 distances between each encoder output and each codebook entry.
            # distances: shape (B*T, codebook_size)
            distances = torch.sum(z_e_flat**2, dim=1, keepdim=True) + torch.sum(codebook**2, dim=1) - 2 * torch.matmul(z_e_flat, codebook.t())
            
            # NEW for codebook collape
            # Apply stochastic masking for exploration (if enabled)
            distances = self.apply_stochastic_mask(distances, training=self.training)
            
            # Compute soft entropy loss BEFORE argmin (to maintain gradients)
            entropy_loss = self.compute_soft_entropy_loss(distances, padding_mask)
            
            # For each latent vector, find the nearest codebook entry.
            encoding_indices = torch.argmin(distances, dim=1)  # shape: (B*T,)
            
            # Update usage tracking
            self.update_usage_tracking(encoding_indices, padding_mask)
            
            if self.codebook_reset_counter_multiplier > 0:
                # Decrement all counters by the batch size.
                BT = B * T
                self.codebook_counters -= BT
                # Reset counters for the used indices.
                self.codebook_counters[encoding_indices] = self.codebook_reset_counter_multiplier * self.codebook_size
                # Check for any collapsed codebook vectors.
                collapsed = self.codebook_counters <= 0
                if collapsed.any():
                    num_collapsed = int(collapsed.sum().item())
                    collapsed_indices = torch.where(collapsed)[0]
                    if num_collapsed > BT:
                        # If there are more collapsed vectors than BT,
                        # Select BT vectors from z_e_flat without replacement.
                        new_batch_vectors = z_e_flat[torch.randperm(BT)[:BT]]
                        # For the remaining collapsed vectors, compute overall mean and std per vector.
                        # Calculate the mean of each row in z_e_flat and then average over rows.
                        overall_mean = z_e_flat.mean(dim=1).mean()
                        # Calculate the standard deviation of each row and then average over rows.
                        overall_std = z_e_flat.std(dim=1).mean()
                        rem = num_collapsed - BT
                        random_vectors = torch.normal(mean=overall_mean.item(), std=overall_std.item(), size=(rem, z_e_flat.shape[1]), device=z_e_flat.device)
                        # To maintain a consistent assignment order, sort the collapsed indices.
                        sorted_indices = collapsed_indices[torch.argsort(collapsed_indices)]
                        self.codebook.weight.data[sorted_indices[:BT]] = new_batch_vectors.to(self.codebook.weight.dtype)
                        self.codebook.weight.data[sorted_indices[BT:]] = random_vectors.to(self.codebook.weight.dtype)
                    else:
                        # If number of collapsed vectors is less than or equal to BT,
                        # update them with randomly selected vectors from z_e_flat without replacement.
                        new_vectors = z_e_flat[torch.randperm(BT)[:num_collapsed]]
                        self.codebook.weight.data[collapsed] = new_vectors.to(self.codebook.weight.dtype)
                    # Reset the counters for these codebook entries.
                    self.codebook_counters[collapsed] = self.codebook_reset_counter_multiplier * self.codebook_size
            unique_count = torch.unique(encoding_indices).numel()
            # Quantize: lookup the codebook entries.
            z_q_flat = codebook[encoding_indices]  # shape: (B*T, d2)
            z_q = z_q_flat.view(B, T, d2)  # shape: (B, T, d2)
            
            # Compute the losses.
            # The codebook loss encourages the codebook vectors to move towards the encoder outputs.
            if padding_mask is not None:
                # Expand padding_mask to match z_q/z_e dimensions
                active_mask = padding_mask.unsqueeze(-1).float()
                # Compute MSE only over active (non-padded) positions and average over them
                codebook_loss = F.mse_loss(z_q.detach() * active_mask, z_e * active_mask, reduction='sum') / active_mask.sum()
                # The commitment loss encourages the encoder outputs to commit to a code.
                if beta is None:
                    commitment_loss = self.beta * F.mse_loss(z_q * active_mask, z_e.detach() * active_mask, reduction='sum') / active_mask.sum()
                else:
                    commitment_loss = beta * F.mse_loss(z_q * active_mask, z_e.detach() * active_mask, reduction='sum') / active_mask.sum()
            else:
                codebook_loss = F.mse_loss(z_q.detach(), z_e)
                # The commitment loss encourages the encoder outputs to commit to a code.
                if beta is None:
                    commitment_loss = self.beta * F.mse_loss(z_q, z_e.detach())
                else:
                    commitment_loss = beta * F.mse_loss(z_q, z_e.detach())
            
            # Use the straight-through estimator:
            z_q = z_e + (z_q - z_e).detach()
        else:
            # No-codebook mode: bypass quantization.
            z_q = z_e
            codebook_loss = torch.tensor(0.0, device=z_e.device)
            commitment_loss = torch.tensor(0.0, device=z_e.device)
            entropy_loss = torch.tensor(0.0, device=z_e.device)
            
        
        # Decode: pass the quantized latent through the decoder.
        # Expected output shape: (B, L, T, d)
        x_recon = self.decoder(z_q, padding_mask=padding_mask)
        
        if padding_mask is not None:
            # Expand padding_mask to match x_recon/x dimensions
            active_mask = padding_mask.unsqueeze(-1).unsqueeze(1).float()
            # Compute MSE only over active (non-padded) positions and average over them
            recon_loss = F.mse_loss(x_recon * active_mask, x_normalized * active_mask, reduction='sum') / active_mask.sum()
        else:
            recon_loss = F.mse_loss(x_recon, x_normalized)
        
        # NEW for codebook collape - Replaced orthogonality loss with cosine-push and entropy losses
        # Compute cosine-push regularization loss (replaces orthogonality loss)
        cosine_push_loss = self.compute_cosine_push_loss()
        
        # entropy_loss is already computed in the quantization section above
        if self.codebook is None:
            entropy_loss = torch.tensor(0.0, device=z_e.device)
        
        total_loss = recon_loss + codebook_loss + commitment_loss + \
                    self.cosine_push_weight * cosine_push_loss + \
                    self.entropy_loss_weight * entropy_loss
        
        return x_recon, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count, cosine_push_loss, entropy_loss
 
class Encoder1(nn.Module):
    """
    Encoder for a vector-quantized VAE.

    This encoder processes an input tensor of shape (B, L, T, d) in two stages:

      Stage 1:
        For each l in [L], a separate transformer network (a stack of one or more TransformerBlock)
        processes the input slice of shape (B, T, d) independently.
      
      Projection:
        The outputs from stage 1 (shape: B, L, T, d) are re-arranged by concatenating the L and d
        dimensions, forming a tensor of shape (B, T, L*d), which is then projected to (B, T, d₂).
      
      Stage 2:
        The projected tensor is refined by additional transformer blocks, producing the final output
        of shape (B, T, d₂).x

    Note: All transformer blocks use flash attention, rotary positional embeddings, and are set in
    encoder-only mode.
    """
    def __init__(self, L, d, d2, num_layers_layerwise_stage=1, num_layers_aggregate_stage=3, config_layerwise_stage=None, config_aggregate_stage=None):
        """
        Args:
            L (int): Number of slices along the first dimension.
            d (int): Input feature dimension for each slice.
            d2 (int): Output feature dimension after projection.
            num_layers_layerwise_stage (int): Number of transformer blocks per transformer network in stage 1.
            num_layers_aggregate_stage (int): Number of transformer blocks in stage 2.
            config_layerwise_stage (dict, optional): Configuration dictionary for stage 1 transformer blocks.
            config_aggregate_stage (dict, optional): Configuration dictionary for stage 2 transformer blocks.
        """
        super().__init__()
        self.L = L
        self.d = d
        self.d2 = d2
        

        config_layerwise_stage['n_embd'] = d
        config_layerwise_stage['mlp_hidden_dim'] = 4 * d

        # Create L independent transformer networks for stage 1.
        self.layerwise_stage_transformers = nn.ModuleList([
            CustomSequential(*[TransformerBlock(config_layerwise_stage) for _ in range(num_layers_layerwise_stage)])
            for _ in range(L)
        ])
        
        # Projection layer: from concatenated (L*d) to d2.
        self.proj = nn.Linear(L * d, d2)
        
        config_aggregate_stage['n_embd'] = d2
        config_aggregate_stage['mlp_hidden_dim'] = 4 * d2
        # Create transformer blocks for stage 2.
        self.aggregate_stage_blocks = nn.ModuleList([TransformerBlock(config_aggregate_stage) for _ in range(num_layers_aggregate_stage)])
    
    def forward(self, x, padding_mask=None):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, L, T, d).

        Returns:
            Tensor: Output tensor of shape (B, T, d2).
        """
        B, L, T, d = x.size()
        assert L == self.L, f"Expected L={self.L} but got {L}"
        
        # Stage 1: Process each slice independently.
        layerwise_stage_outputs = []
        for l in range(self.L):
            # Each slice x[:, l, :, :] has shape (B, T, d)
            out_l = self.layerwise_stage_transformers[l](x[:, l, :, :], padding_mask=padding_mask)
            layerwise_stage_outputs.append(out_l)
        # Stack along the L dimension -> (B, L, T, d)
        x_layerwise_stage = torch.stack(layerwise_stage_outputs, dim=1)
        
        # Concatenate the L and d dimensions:
        # Permute to (B, T, L, d) and then reshape to (B, T, L*d)
        x_cat = x_layerwise_stage.permute(0, 2, 1, 3).contiguous().view(B, T, L * d)
        
        # Project to the desired dimension d2.
        x_proj = self.proj(x_cat)  # (B, T, d2)
        
        # Stage 2: Process with additional transformer blocks.
        x_out = x_proj
        for block in self.aggregate_stage_blocks:
            x_out = block(x_out, padding_mask=padding_mask)  # (B, T, d2)
        
        return x_out

class Decoder1(nn.Module):
    """
    Decoder for a vector-quantized VAE.

    This decoder processes an input tensor of shape (B, T, d2) in two stages:

      Aggregate Stage:
        The input is processed by a series of transformer blocks (with flash attention, rotary 
        positional embeddings, and encoder-only mode), maintaining shape (B, T, d2).
      
      Projection & Reshaping:
        A linear layer maps the d2-dimension to (L*d) so that each token becomes a vector of 
        length L*d. The tensor is then reshaped to (B, L, T, d).
      
      Layerwise Stage:
        For each l in [L], a separate transformer network processes the slice (B, T, d) independently.
        The final output is of shape (B, L, T, d).
    """
    def __init__(self, L, d, d2, num_layers_aggregate_stage=3, num_layers_layerwise_stage=1, 
                 config_aggregate_stage=None, config_layerwise_stage=None, tied_encoder_proj = None):
        """
        Args:
            L (int): Number of slices for the final output.
            d (int): Feature dimension for each slice in the final output.
            d2 (int): Input feature dimension for the aggregate stage.
            num_layers_aggregate_stage (int): Number of transformer blocks for the aggregate stage.
            num_layers_layerwise_stage (int): Number of transformer blocks per slice in the layerwise stage.
            config_aggregate_stage (dict, optional): Configuration dictionary for aggregate stage blocks.
            config_layerwise_stage (dict, optional): Configuration dictionary for layerwise stage blocks.
        """
        super().__init__()
        self.L = L
        self.d = d
        self.d2 = d2


        config_aggregate_stage['n_embd'] = d2
        config_aggregate_stage['mlp_hidden_dim'] = 4 * d2
        
        self.aggregate_stage_blocks = nn.ModuleList([
            TransformerBlock(config_aggregate_stage) for _ in range(num_layers_aggregate_stage)
        ])
        
        self.tied = False
        if tied_encoder_proj is None:
            self.proj = nn.Linear(d2, L * d)
        else:
            self.proj = tied_encoder_proj
            self.projbias = nn.Parameter(torch.zeros(L * d))
            self.tied = True

        
        config_layerwise_stage['n_embd'] = d
        config_layerwise_stage['mlp_hidden_dim'] = 4 * d
        
        # Create L independent transformer networks for the layerwise stage.
        self.layerwise_stage_transformers = nn.ModuleList([
            CustomSequential(*[TransformerBlock(config_layerwise_stage) for _ in range(num_layers_layerwise_stage)])
            for _ in range(L)
        ])
    
    def forward(self, x, padding_mask=None):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, T, d2).

        Returns:
            Tensor: Output tensor of shape (B, L, T, d).
        """
        B, T, d2_ = x.size()
        assert d2_ == self.d2, f"Expected last dimension to be {self.d2} but got {d2_}"
        
        # Aggregate Stage: process with transformer blocks.
        for block in self.aggregate_stage_blocks:
            x = block(x, padding_mask=padding_mask)  # shape remains (B, T, d2)
        
        # Project each token from d2 to (L*d); result shape: (B, T, L*d).
        if self.tied:
            x = F.linear(x, self.proj.weight.t(), bias=self.projbias)
        else:
            x = self.proj(x)
        
        # Reshape: from (B, T, L*d) to (B, T, L, d) then permute to (B, L, T, d).
        x = x.view(B, T, self.L, self.d).permute(0, 2, 1, 3).contiguous()
        
        # Layerwise Stage: process each of the L slices independently.
        layerwise_outputs = []
        for l in range(self.L):
            # Each slice: (B, T, d)
            out_l = self.layerwise_stage_transformers[l](x[:, l, :, :], padding_mask=padding_mask)
            layerwise_outputs.append(out_l)
        # Stack along the L dimension to recover shape (B, L, T, d)
        x_out = torch.stack(layerwise_outputs, dim=1)
        
        return x_out

class Encoder3(nn.Module):
    """
    """
    def __init__(self, L, d, d2, T, num_layers_layerwise_stage=1, num_layers_aggregate_stage=3, config_layerwise_stage=None, config_aggregate_stage=None):
        """
        Args:
            L (int): Number of slices along the first dimension.
            d (int): Input feature dimension for each slice.
            d2 (int): Output feature dimension after projection.
            num_layers_layerwise_stage (int): Number of transformer blocks per transformer network in stage 1.
            num_layers_aggregate_stage (int): Number of transformer blocks in stage 2.
            config_layerwise_stage (dict, optional): Configuration dictionary for stage 1 transformer blocks.
            config_aggregate_stage (dict, optional): Configuration dictionary for stage 2 transformer blocks.
            T (int): sequence length.
        """
        super().__init__()
        self.L = L
        self.d = d
        self.d2 = d2
        self.T = T

        config_layerwise_stage['n_embd'] = d
        config_layerwise_stage['mlp_hidden_dim'] = 4 * d

        # Create L independent transformer networks for stage 1.
        self.layerwise_stage_transformers = nn.ModuleList([
            CustomSequential(*[TransformerBlock(config_layerwise_stage) for _ in range(num_layers_layerwise_stage)])
            for _ in range(L)
        ])
        
        # Projection layer: from concatenated (L*d) to d2.
        self.proj = nn.Linear(L * d, d2)
        
        config_aggregate_stage['n_embd'] = d2
        config_aggregate_stage['mlp_hidden_dim'] = 4 * d2
        # Create transformer blocks for stage 2.
        self.aggregate_stage_blocks = nn.ModuleList([TransformerBlock(config_aggregate_stage) for _ in range(num_layers_aggregate_stage)])

        self.last_proj = nn.Sequential(
            nn.Linear(T * d2, int(math.sqrt(T * d2* d2))),
            nn.ReLU(),
            nn.Linear(int(math.sqrt(T * d2* d2)), d2)
        )
    
    def forward(self, x, padding_mask=None):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, L, T, d).

        Returns:
            Tensor: Output tensor of shape (B, T, d2).
        """
        B, L, T, d = x.size()
        assert L == self.L, f"Expected L={self.L} but got {L}"
        
        # Stage 1: Process each slice independently.
        layerwise_stage_outputs = []
        for l in range(self.L):
            # Each slice x[:, l, :, :] has shape (B, T, d)
            out_l = self.layerwise_stage_transformers[l](x[:, l, :, :], padding_mask=padding_mask)
            layerwise_stage_outputs.append(out_l)
        # Stack along the L dimension -> (B, L, T, d)
        x_layerwise_stage = torch.stack(layerwise_stage_outputs, dim=1)
        
        # Concatenate the L and d dimensions:
        # Permute to (B, T, L, d) and then reshape to (B, T, L*d)
        x_cat = x_layerwise_stage.permute(0, 2, 1, 3).contiguous().view(B, T, L * d)
        
        # Project to the desired dimension d2.
        x_proj = self.proj(x_cat)  # (B, T, d2)
        
        # Stage 2: Process with additional transformer blocks.
        x_out = x_proj
        for block in self.aggregate_stage_blocks:
            x_out = block(x_out, padding_mask=padding_mask)  # (B, T, d2)

        x_out = x_out.view(B, self.T * self.d2)
        x_out = self.last_proj(x_out)
        
        return x_out

class Decoder3(nn.Module):
    """
    """
    def __init__(self, L, d, d2, T, num_layers_aggregate_stage=3, num_layers_layerwise_stage=1, 
                 config_aggregate_stage=None, config_layerwise_stage=None, tied_encoder_proj = None):
        """
        Args:
            L (int): Number of slices for the final output.
            d (int): Feature dimension for each slice in the final output.
            d2 (int): Input feature dimension for the aggregate stage.
            num_layers_aggregate_stage (int): Number of transformer blocks for the aggregate stage.
            num_layers_layerwise_stage (int): Number of transformer blocks per slice in the layerwise stage.
            config_aggregate_stage (dict, optional): Configuration dictionary for aggregate stage blocks.
            config_layerwise_stage (dict, optional): Configuration dictionary for layerwise stage blocks.
            T (int): sequence length.
        """
        super().__init__()
        self.L = L
        self.d = d
        self.d2 = d2
        self.T = T

        config_aggregate_stage['n_embd'] = d2
        config_aggregate_stage['mlp_hidden_dim'] = 4 * d2
        
        self.aggregate_stage_blocks = nn.ModuleList([
            TransformerBlock(config_aggregate_stage) for _ in range(num_layers_aggregate_stage)
        ])
        
        self.tied = False
        if tied_encoder_proj is None:
            self.proj = nn.Linear(d2, L * d)
        else:
            self.proj = tied_encoder_proj
            self.projbias = nn.Parameter(torch.zeros(L * d))
            self.tied = True

        
        config_layerwise_stage['n_embd'] = d
        config_layerwise_stage['mlp_hidden_dim'] = 4 * d
        
        # Create L independent transformer networks for the layerwise stage.
        self.layerwise_stage_transformers = nn.ModuleList([
            CustomSequential(*[TransformerBlock(config_layerwise_stage) for _ in range(num_layers_layerwise_stage)])
            for _ in range(L)
        ])

        self.proj_sequence = nn.Sequential(
            nn.Linear(d2, int(math.sqrt(d2 * self.T * d2))),
            nn.ReLU(),
            nn.Linear(int(math.sqrt(d2 * self.T * d2)), self.T * d2)
        )
    
    def forward(self, x, padding_mask=None):
        """
        Forward pass.
        """

        B, d2_ = x.size()
        assert d2_ == self.d2, f"Expected last dimension to be {self.d2} but got {d2_}"
        d2 = self.d2
        T = self.T
        
        # Generate main sequence: (B, T, d2)
        x = self.proj_sequence(x)  # (B, T * d2)
        
        x = x.view(B, self.T, self.d2)

        
        # Aggregate Stage: process with transformer blocks.
        for block in self.aggregate_stage_blocks:
            x = block(x, padding_mask=padding_mask)  # shape remains (B, T, d2)
        
        # Project each token from d2 to (L*d); result shape: (B, T, L*d).
        if self.tied:
            x = F.linear(x, self.proj.weight.t(), bias=self.projbias)
        else:
            x = self.proj(x)
        
        # Reshape: from (B, T, L*d) to (B, T, L, d) then permute to (B, L, T, d).
        x = x.view(B, T, self.L, self.d).permute(0, 2, 1, 3).contiguous()
        
        # Layerwise Stage: process each of the L slices independently.
        layerwise_outputs = []
        for l in range(self.L):
            # Each slice: (B, T, d)
            out_l = self.layerwise_stage_transformers[l](x[:, l, :, :], padding_mask=padding_mask)
            layerwise_outputs.append(out_l)
        # Stack along the L dimension to recover shape (B, L, T, d)
        x_out = torch.stack(layerwise_outputs, dim=1)
        
        return x_out

class VQVAE3(nn.Module):
    def __init__(self, encoder, decoder, config):
        """
        """
        super().__init__()
        self.encoder = encoder  # Should output shape (B, D)
        self.decoder = decoder  # Should accept input (B, D) and produce output (B, T, d2)
        self.codebook_size = config["codebook_size"]
        self.codebook_reset_counter_multiplier = config.get("codebook_reset_counter_multiplier", 0)
        self.beta = config["beta"]

        # Add buffer for normalization values (L values for VQVAE3)
        self.register_buffer('normalization_values', None)

        if self.codebook_size > 0:
            # Create a codebook (embedding table) of size (K, D) where D is the latent dimension of encoder.
            self.codebook = nn.Embedding(self.codebook_size, encoder.d2)
            nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
        else:
            self.codebook = None
        
        if self.codebook_reset_counter_multiplier > 0:
            self.register_buffer(
            "codebook_counters",
            torch.full((self.codebook_size,), 0)
        )
        
        # NEW for codebook collape - Store experimental feature settings from config
        self.cosine_push_weight = config.get('cosine_push_weight', 0.0)  # Replaces ortho_loss_weight
        self.entropy_loss_weight = config.get('entropy_loss_weight', 0.0)
        self.mask_prob = config.get('mask_prob', 0.0)
        
        # For soft entropy loss: temperature parameter for softmax
        self.entropy_temperature = config.get('entropy_temperature', 1.0)  # Temperature for soft assignments
        
        # For vector usage tracking
        self.usage_tracking_window = config.get('usage_tracking_window', 5000)  # Window size for tracking usage
        if self.usage_tracking_window > 0:
            self.register_buffer('usage_history', torch.zeros(self.usage_tracking_window, dtype=torch.long))
            self.register_buffer('usage_ptr', torch.tensor(0, dtype=torch.long))
            self.register_buffer('usage_full', torch.tensor(False, dtype=torch.bool))
            self.register_buffer('total_vectors_processed', torch.tensor(0, dtype=torch.long))

    def normalize_codebook_vectors(self):
        """
        Normalize each codebook vector to unit length.
        
        This is applied after each parameter update when using cosine-push regularization.
        Helps ensure the cosine-push loss focuses on angular relationships rather than magnitudes.
        """
        if self.cosine_push_weight > 0 and self.codebook is not None:
            with torch.no_grad():
                # Normalize each row (codebook vector) to unit length
                self.codebook.weight.data = F.normalize(self.codebook.weight.data, p=2, dim=1)
    
    def compute_cosine_push_loss(self):
        """
        Compute cosine-push regularization loss on the codebook matrix.
        
        The cosine-push loss encourages codebook vectors to be orthogonal by penalizing
        the squared cosine similarities between different vectors. This replaces the
        gram-matrix Frobenius norm with a more direct angular penalty.
        
        If usage tracking is enabled, only considers used vectors and weights each
        vector's contribution by its usage frequency.
        
        L_push = sum_{i≠j} w_i * w_j * (dot(e_i, e_j))^2
        
        Where e_i and e_j are normalized codebook vectors, and w_i is the usage weight.
        
        Returns:
            cosine_push_loss: Cosine-push loss scalar
        """
        if self.cosine_push_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # If usage tracking is enabled, use only tracked vectors with usage-based weighting
        if self.usage_tracking_window > 0:
            usage_stats = self.get_usage_statistics()
            usage_counts = usage_stats['usage_counts']
            used_mask = usage_counts > 0
            used_indices = torch.where(used_mask)[0]
            
            if len(used_indices) <= 1:
                # Need at least 2 vectors to compute similarities
                return torch.tensor(0.0, device=next(self.parameters()).device)
            
            # Get used codebook vectors and normalize them
            E_used = self.codebook.weight[used_indices]  # (num_used, D)
            E_used_normalized = F.normalize(E_used, p=2, dim=1)  # (num_used, D)
            
            # Compute cosine similarity matrix for used vectors: (num_used, num_used)
            cosine_sim_matrix = torch.mm(E_used_normalized, E_used_normalized.t())
            
            # Get usage weights for used vectors and normalize them
            usage_weights = usage_counts[used_indices].float()  # (num_used,)
            usage_weights = usage_weights / usage_weights.sum()  # Normalize to sum to 1
            
            # Create weight matrix: w_i * w_j for all pairs
            weight_matrix = usage_weights.unsqueeze(1) * usage_weights.unsqueeze(0)  # (num_used, num_used)
            
            # Zero out diagonal (self-similarities) and apply weights
            num_used = len(used_indices)
            mask = ~torch.eye(num_used, device=E_used.device, dtype=torch.bool)
            
            # Apply weights and compute weighted sum of squared cosine similarities
            weighted_cosines_squared = (cosine_sim_matrix ** 2) * weight_matrix
            cosine_push_loss = weighted_cosines_squared[mask].sum()
            
        else:
            # Original implementation when usage tracking is disabled
            # Get codebook matrix E: (K, D)
            E = self.codebook.weight  # (codebook_size, D)
            
            # Normalize vectors to unit length for cosine computation
            E_normalized = F.normalize(E, p=2, dim=1)  # (K, D)
            
            # Compute cosine similarity matrix: (K, K)
            cosine_sim_matrix = torch.mm(E_normalized, E_normalized.t())  # (K, K)
            
            # Zero out diagonal (self-similarities) and compute sum of squared off-diagonal elements
            mask = ~torch.eye(self.codebook_size, device=E.device, dtype=torch.bool)
            off_diagonal_cosines = cosine_sim_matrix[mask]  # Get all off-diagonal elements
            
            # Sum of squared cosine similarities between different vectors
            cosine_push_loss = torch.sum(off_diagonal_cosines ** 2)
        
        return cosine_push_loss
    
    def compute_soft_entropy_loss(self, distances):
        """
        Compute entropy loss using soft assignments from current batch.
        This maintains gradient flow through the assignment probabilities.
        
        For VQVAE2, distances have shape (B, K) since we have one assignment per batch item.
        
        Args:
            distances: Distance tensor (B, K) between encoder outputs and codebook
            
        Returns:
            entropy_loss: Soft entropy loss scalar
        """
        if self.entropy_loss_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=distances.device)
        
        # Convert distances to probabilities using softmax
        # Negative distances because smaller distance = higher probability
        assignment_probs = F.softmax(-distances / self.entropy_temperature, dim=-1)  # (B, K)
        
        # Average assignment probabilities across batch
        avg_probs = assignment_probs.mean(dim=0)  # (K,)
        
        # Compute entropy: -sum(p * log(p))
        # We want to maximize entropy (uniform distribution), so minimize negative entropy
        epsilon = 1e-10
        avg_probs_safe = torch.clamp(avg_probs, min=epsilon)
        entropy = -torch.sum(avg_probs_safe * torch.log(avg_probs_safe))
        
        # Convert to "equal usage" loss: penalize low entropy
        max_entropy = torch.log(torch.tensor(self.codebook_size, device=distances.device, dtype=avg_probs.dtype))
        entropy_loss = max_entropy - entropy  # Higher when entropy is low
        
        return entropy_loss
    
    def update_usage_tracking(self, encoding_indices):
        """
        Update the sliding window of vector usage for statistics tracking.
        
        For VQVAE2, encoding_indices has shape (B,) since we have one assignment per batch item.
        
        Args:
            encoding_indices: Tensor of assignment indices (B,)
        """
        if self.usage_tracking_window <= 0:
            return
            
        # For VQVAE2, all assignments are valid (no padding mask needed)
        valid_indices = encoding_indices
        
        # Update total vectors processed
        self.total_vectors_processed += valid_indices.size(0)
            
        # Add valid assignments to circular buffer
        n_valid = valid_indices.size(0)
        if n_valid > 0:
            # Handle case where we have more assignments than buffer size
            if n_valid >= self.usage_tracking_window:
                # Fill entire buffer with most recent assignments
                self.usage_history[:] = valid_indices[-self.usage_tracking_window:]
                self.usage_ptr.fill_(0)
                self.usage_full.fill_(True)
            else:
                # Add assignments to circular buffer
                end_ptr = (self.usage_ptr + n_valid) % self.usage_tracking_window
                
                if end_ptr > self.usage_ptr:
                    # No wraparound
                    self.usage_history[self.usage_ptr:end_ptr] = valid_indices
                else:
                    # Wraparound case
                    n_until_end = self.usage_tracking_window - self.usage_ptr
                    self.usage_history[self.usage_ptr:] = valid_indices[:n_until_end]
                    if n_valid > n_until_end:
                        self.usage_history[:end_ptr] = valid_indices[n_until_end:]
                
                self.usage_ptr.copy_(end_ptr)
                if self.usage_ptr == 0 or self.usage_full:
                    self.usage_full.fill_(True)
    
    def get_usage_statistics(self):
        """
        Get current usage statistics.
        
        Returns:
            dict: Dictionary containing usage statistics
        """
        if self.usage_tracking_window <= 0 or self.codebook is None:
            return {
                'total_vectors_processed': 0,
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size if self.codebook is not None else 0),
                'usage_percentages': torch.zeros(self.codebook_size if self.codebook is not None else 0)
            }
        
        # Get the valid portion of usage history
        if self.usage_full:
            valid_history = self.usage_history
        else:
            valid_history = self.usage_history[:self.usage_ptr]
            
        if valid_history.numel() == 0:
            return {
                'total_vectors_processed': self.total_vectors_processed.item(),
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size),
                'usage_percentages': torch.zeros(self.codebook_size)
            }
        
        # Compute usage statistics
        usage_counts = torch.bincount(valid_history, minlength=self.codebook_size)
        unique_vectors_used = (usage_counts > 0).sum().item()
        usage_percentages = usage_counts.float() / valid_history.numel()
        
        return {
            'total_vectors_processed': self.total_vectors_processed.item(),
            'unique_vectors_used': unique_vectors_used,
            'usage_counts': usage_counts,
            'usage_percentages': usage_percentages
        }
    
    def compute_codebook_similarities(self):
        """
        Compute cosine similarities between used codebook vectors.
        
        Returns:
            dict: Dictionary containing similarity statistics
        """
        if self.codebook is None:
            return {
                'similarities': None,
                'used_indices': None,
                'num_used_vectors': 0
            }
        
        # Get usage statistics to find which vectors are used
        usage_stats = self.get_usage_statistics()
        used_mask = usage_stats['usage_counts'] > 0
        used_indices = torch.where(used_mask)[0]
        
        if len(used_indices) <= 1:
            return {
                'similarities': None,
                'used_indices': used_indices,
                'num_used_vectors': len(used_indices)
            }
        
        # Get used codebook vectors
        with torch.no_grad():
            used_codebook = self.codebook.weight[used_indices]
            # Normalize for cosine similarity
            used_codebook_norm = used_codebook / used_codebook.norm(dim=1, keepdim=True)
            # Compute similarity matrix
            similarities = torch.mm(used_codebook_norm, used_codebook_norm.t())
        
        return {
            'similarities': similarities,
            'used_indices': used_indices,
            'num_used_vectors': len(used_indices)
        }
        
    def apply_stochastic_mask(self, distances, training=True):
        """
        Apply stochastic masking to assignment distances for exploration.
        
        With probability mask_prob, randomly masks a fraction of the distance
        values to very large numbers, forcing the model to explore unused codes.
        This helps prevent codebook collapse by encouraging diversity.
        
        For VQVAE2, distances have shape (B, K) instead of (B*T, K).
        
        Args:
            distances: Distance tensor (B, K) between encoder outputs and codebook
            training: Whether model is in training mode
            
        Returns:
            masked_distances: Distances with stochastic masking applied
        """
        if self.mask_prob <= 0 or not training:
            return distances
            
        # Apply masking with probability mask_prob
        if torch.rand(1).item() < self.mask_prob:
            B, K = distances.shape
            
            # Create random mask - randomly select fraction mask_prob of entries to mask
            num_mask = int(self.mask_prob * B * K)
            mask_indices = torch.randperm(B * K, device=distances.device)[:num_mask]
            
            # Convert flat indices to 2D indices
            mask_rows = mask_indices // K
            mask_cols = mask_indices % K
            
            # Create masked distances by setting selected entries to large value
            masked_distances = distances.clone()
            masked_distances[mask_rows, mask_cols] = 1e10
            
            return masked_distances
        
        return distances

    def normalize(self, x):
        """
        Normalize input using the computed normalization values.
        
        Args:
            x: Input tensor of shape (B, L, T, d)
            
        Returns:
            Normalized tensor of the same shape
        """
        if self.normalization_values is not None:
            # Expand normalization values to match x shape: (1, L, 1, 1)
            norm_values_expanded = self.normalization_values.view(1, -1, 1, 1)
            return x / (norm_values_expanded + 1e-8)
        else:
            return x
    
    def denormalize(self, x):
        """
        Denormalize input using the computed normalization values.
        
        Args:
            x: Input tensor of shape (B, L, T, d)
            
        Returns:
            Denormalized tensor of the same shape
        """
        if self.normalization_values is not None:
            # Expand normalization values to match x shape: (1, L, 1, 1)
            norm_values_expanded = self.normalization_values.view(1, -1, 1, 1)
            return x * (norm_values_expanded + 1e-8)
        else:
            return x
        
    def _compute_normalization_values(self, x, padding_mask=None):
        """
        Compute normalization values from the first batch.
        x: (B, L, T, d)
        padding_mask: (B, T)
        Returns: (L,) tensor of normalization values
        """
        B, L, T, d = x.shape
        
        # Calculate norm in the last dimension: (B, L, T)
        norms = torch.norm(x, dim=-1)  # (B, L, T)
        
        if padding_mask is not None:
            # Expand padding mask to (B, L, T)
            mask = padding_mask.unsqueeze(1).expand(B, L, T).float()  # (B, L, T)
            
            # Apply mask and sum over B and T dimensions
            masked_norms = norms * mask  # (B, L, T)
            sum_norms = masked_norms.sum(dim=(0, 2))  # (L,)
            count_valid = mask.sum(dim=(0, 2))  # (L,)
            
            # Calculate mean, avoiding division by zero
            norm_values = sum_norms / (count_valid + 1e-8)  # (L,)
        else:
            # No mask, simple mean over B and T
            norm_values = norms.mean(dim=(0, 2))  # (L,)
            
        return norm_values

    def forward(self, x, padding_mask=None, beta = None):
        """
        """
        # Compute normalization values if not already computed
        if self.normalization_values is None:
            norm_values = self._compute_normalization_values(x, padding_mask)
            self.register_buffer('normalization_values', norm_values)
        # Apply normalization
        x = self.normalize(x)
        

        # Encode: get continuous latent representation z_e from Encoder2.
        # z_e shape: (B, D)
        z_e = self.encoder(x, padding_mask=padding_mask)

        if self.codebook is not None:
            # Vector quantization: z_e shape is (B, D)
            codebook = self.codebook.weight  # shape: (codebook_size, D)
            # Compute squared L2 distances: shape (B, codebook_size)
            distances = torch.sum(z_e**2, dim=1, keepdim=True) + torch.sum(codebook**2, dim=1) - 2 * torch.matmul(z_e, codebook.t())
            
            # NEW for codebook collape - Apply stochastic masking for exploration (if enabled)
            distances = self.apply_stochastic_mask(distances, training=self.training)
            
            # Compute soft entropy loss BEFORE argmin (to maintain gradients)
            entropy_loss = self.compute_soft_entropy_loss(distances)
            
            # Find nearest codebook index for each latent vector in the batch.
            encoding_indices = torch.argmin(distances, dim=1)  # shape: (B,)
            
            # Update usage tracking
            self.update_usage_tracking(encoding_indices)
            
            unique_count = torch.unique(encoding_indices).numel()
            if self.codebook_reset_counter_multiplier > 0:
                # For VQVAE2, z_e has shape (B, D) so B is the number of latent vectors.
                B = z_e.shape[0]
                # Decrement all counters by the batch size.
                self.codebook_counters -= B
                # Reset counters for the used indices.
                self.codebook_counters[encoding_indices] = self.codebook_reset_counter_multiplier * self.codebook_size
                # Check for any collapsed codebook vectors.
                collapsed = self.codebook_counters <= 0
                if collapsed.any():
                    num_collapsed = int(collapsed.sum().item())
                    collapsed_indices = torch.where(collapsed)[0]
                    if num_collapsed > B:
                        # If there are more collapsed vectors than B,
                        # first randomly select B collapsed indices.
                        # Select B vectors from z_e without replacement.
                        new_batch_vectors = z_e[torch.randperm(B)[:B]]
                        # For the remaining collapsed vectors, compute overall mean and std per vector.
                        # Calculate the mean of each row in z_e and then average over rows.
                        overall_mean = z_e.mean(dim=1).mean()
                        # Calculate the standard deviation of each row and then average over rows.
                        overall_std = z_e.std(dim=1).mean()
                        rem = num_collapsed - B
                        random_vectors = torch.normal(mean=overall_mean.item(), std=overall_std.item(), size=(rem, z_e.shape[1]), device=z_e.device)
                        # To maintain a consistent assignment order, sort the collapsed indices.
                        sorted_indices = collapsed_indices[torch.argsort(collapsed_indices)]
                        self.codebook.weight.data[sorted_indices[:B]] = new_batch_vectors.to(self.codebook.weight.dtype)
                        self.codebook.weight.data[sorted_indices[B:]] = random_vectors.to(self.codebook.weight.dtype)
                    else:
                        # If number of collapsed vectors is less than or equal to B,
                        # update them with randomly selected vectors from z_e without replacement.
                        new_vectors = z_e[torch.randperm(B)[:num_collapsed]]
                        self.codebook.weight.data[collapsed] = new_vectors.to(self.codebook.weight.dtype)
                    # Reset the counters for these codebook entries.
                    self.codebook_counters[collapsed] = self.codebook_reset_counter_multiplier * self.codebook_size
            # Quantize: lookup codebook entries.
            z_q = self.codebook(encoding_indices)  # shape: (B, D)

            # Compute losses.
            codebook_loss = F.mse_loss(z_q.detach(), z_e)
            if beta is None:
                commitment_loss = self.beta * F.mse_loss(z_q, z_e.detach())
            else:
                commitment_loss = beta * F.mse_loss(z_q, z_e.detach())

            # Use straight-through estimator: substitute z_q in forward pass but allow gradients to flow into encoder.
            z_q = z_e + (z_q - z_e).detach()
        else:
            # No-codebook mode: bypass quantization.
            z_q = z_e
            codebook_loss = torch.tensor(0.0, device=z_e.device)
            commitment_loss = torch.tensor(0.0, device=z_e.device)
            entropy_loss = torch.tensor(0.0, device=z_e.device)

        # Decode:
        # Decoder now expects input of shape (B, D) and supports padding mask.
            
        x_recon = self.decoder(z_q, padding_mask=padding_mask)

        if padding_mask is not None:
            # Expand padding_mask to match x_recon/x dimensions
            active_mask = padding_mask.unsqueeze(-1).unsqueeze(1).float()
            # Compute MSE only over active (non-padded) positions and average over them
            recon_loss = F.mse_loss(x_recon * active_mask, x * active_mask, reduction='sum') / active_mask.sum()
        else:
            recon_loss = F.mse_loss(x_recon, x)
        
        # NEW for codebook collape - Replaced orthogonality loss with cosine-push and entropy losses
        # Compute cosine-push regularization loss (replaces orthogonality loss)
        cosine_push_loss = self.compute_cosine_push_loss()
        
        # entropy_loss is already computed in the quantization section above
        if self.codebook is None:
            entropy_loss = torch.tensor(0.0, device=z_e.device)
        
        total_loss = recon_loss + codebook_loss + commitment_loss + \
                    self.cosine_push_weight * cosine_push_loss + \
                    self.entropy_loss_weight * entropy_loss
        
        return x_recon, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count, cosine_push_loss, entropy_loss

## ASSUMES MINIMUM SEQUENCE LENGTH IS >= T_STAR ##################
class VQVAE2(nn.Module):
    def __init__(self, encoder, decoder, config):
        """
        Args:
            encoder (nn.Module): An instance of Encoder2 that accepts inputs of shape (B, T, d2)
                                 and outputs latent representations of shape (B, D).
            decoder (nn.Module): An instance of Decoder2 that accepts a latent of shape (B, D) and,
                                 in teacher forcing mode, teacher_token_vecs of shape (B, T, d2)
                                 or in autoregressive mode, a token count T, and outputs a reconstruction
                                 of shape (B, T, d2).
            codebook_size (int): Number of discrete latent vectors.
            beta (float): Commitment loss coefficient.
        """
        super().__init__()
        self.encoder = encoder  # Should output shape (B, D)
        self.decoder = decoder  # Should accept input (B, D) and produce output (B, T, d2)
        self.codebook_size = config["codebook_size"]
        self.codebook_reset_counter_multiplier = config.get("codebook_reset_counter_multiplier", 0)
        self.beta = config["beta"]

        if self.codebook_size > 0:
            # Create a codebook (embedding table) of size (K, D) where D is the latent dimension of encoder.
            self.codebook = nn.Embedding(self.codebook_size, encoder.D)
            nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
        else:
            self.codebook = None
        
        if self.codebook_reset_counter_multiplier > 0:
            self.register_buffer(
            "codebook_counters",
            torch.full((self.codebook_size,), 0)
        )
        
        # NEW for codebook collape - Store experimental feature settings from config
        self.cosine_push_weight = config.get('cosine_push_weight', 0.0)  # Replaces ortho_loss_weight
        self.entropy_loss_weight = config.get('entropy_loss_weight', 0.0)
        self.mask_prob = config.get('mask_prob', 0.0)
        
        # For soft entropy loss: temperature parameter for softmax
        self.entropy_temperature = config.get('entropy_temperature', 1.0)  # Temperature for soft assignments
        
        # For vector usage tracking
        self.usage_tracking_window = config.get('usage_tracking_window', 5000)  # Window size for tracking usage
        if self.usage_tracking_window > 0:
            self.register_buffer('usage_history', torch.zeros(self.usage_tracking_window, dtype=torch.long))
            self.register_buffer('usage_ptr', torch.tensor(0, dtype=torch.long))
            self.register_buffer('usage_full', torch.tensor(False, dtype=torch.bool))
            self.register_buffer('total_vectors_processed', torch.tensor(0, dtype=torch.long))

    def normalize_codebook_vectors(self):
        """
        Normalize each codebook vector to unit length.
        
        This is applied after each parameter update when using cosine-push regularization.
        Helps ensure the cosine-push loss focuses on angular relationships rather than magnitudes.
        """
        if self.cosine_push_weight > 0 and self.codebook is not None:
            with torch.no_grad():
                # Normalize each row (codebook vector) to unit length
                self.codebook.weight.data = F.normalize(self.codebook.weight.data, p=2, dim=1)
    
    def compute_cosine_push_loss(self):
        """
        Compute cosine-push regularization loss on the codebook matrix.
        
        The cosine-push loss encourages codebook vectors to be orthogonal by penalizing
        the squared cosine similarities between different vectors. This replaces the
        gram-matrix Frobenius norm with a more direct angular penalty.
        
        If usage tracking is enabled, only considers used vectors and weights each
        vector's contribution by its usage frequency.
        
        L_push = sum_{i≠j} w_i * w_j * (dot(e_i, e_j))^2
        
        Where e_i and e_j are normalized codebook vectors, and w_i is the usage weight.
        
        Returns:
            cosine_push_loss: Cosine-push loss scalar
        """
        if self.cosine_push_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # If usage tracking is enabled, use only tracked vectors with usage-based weighting
        if self.usage_tracking_window > 0:
            usage_stats = self.get_usage_statistics()
            usage_counts = usage_stats['usage_counts']
            used_mask = usage_counts > 0
            used_indices = torch.where(used_mask)[0]
            
            if len(used_indices) <= 1:
                # Need at least 2 vectors to compute similarities
                return torch.tensor(0.0, device=next(self.parameters()).device)
            
            # Get used codebook vectors and normalize them
            E_used = self.codebook.weight[used_indices]  # (num_used, D)
            E_used_normalized = F.normalize(E_used, p=2, dim=1)  # (num_used, D)
            
            # Compute cosine similarity matrix for used vectors: (num_used, num_used)
            cosine_sim_matrix = torch.mm(E_used_normalized, E_used_normalized.t())
            
            # Get usage weights for used vectors and normalize them
            usage_weights = usage_counts[used_indices].float()  # (num_used,)
            usage_weights = usage_weights / usage_weights.sum()  # Normalize to sum to 1
            
            # Create weight matrix: w_i * w_j for all pairs
            weight_matrix = usage_weights.unsqueeze(1) * usage_weights.unsqueeze(0)  # (num_used, num_used)
            
            # Zero out diagonal (self-similarities) and apply weights
            num_used = len(used_indices)
            mask = ~torch.eye(num_used, device=E_used.device, dtype=torch.bool)
            
            # Apply weights and compute weighted sum of squared cosine similarities
            weighted_cosines_squared = (cosine_sim_matrix ** 2) * weight_matrix
            cosine_push_loss = weighted_cosines_squared[mask].sum()
            
        else:
            # Original implementation when usage tracking is disabled
            # Get codebook matrix E: (K, D)
            E = self.codebook.weight  # (codebook_size, D)
            
            # Normalize vectors to unit length for cosine computation
            E_normalized = F.normalize(E, p=2, dim=1)  # (K, D)
            
            # Compute cosine similarity matrix: (K, K)
            cosine_sim_matrix = torch.mm(E_normalized, E_normalized.t())  # (K, K)
            
            # Zero out diagonal (self-similarities) and compute sum of squared off-diagonal elements
            mask = ~torch.eye(self.codebook_size, device=E.device, dtype=torch.bool)
            off_diagonal_cosines = cosine_sim_matrix[mask]  # Get all off-diagonal elements
            
            # Sum of squared cosine similarities between different vectors
            cosine_push_loss = torch.sum(off_diagonal_cosines ** 2)
        
        return cosine_push_loss
    
    def compute_soft_entropy_loss(self, distances):
        """
        Compute entropy loss using soft assignments from current batch.
        This maintains gradient flow through the assignment probabilities.
        
        For VQVAE2, distances have shape (B, K) since we have one assignment per batch item.
        
        Args:
            distances: Distance tensor (B, K) between encoder outputs and codebook
            
        Returns:
            entropy_loss: Soft entropy loss scalar
        """
        if self.entropy_loss_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=distances.device)
        
        # Convert distances to probabilities using softmax
        # Negative distances because smaller distance = higher probability
        assignment_probs = F.softmax(-distances / self.entropy_temperature, dim=-1)  # (B, K)
        
        # Average assignment probabilities across batch
        avg_probs = assignment_probs.mean(dim=0)  # (K,)
        
        # Compute entropy: -sum(p * log(p))
        # We want to maximize entropy (uniform distribution), so minimize negative entropy
        epsilon = 1e-10
        avg_probs_safe = torch.clamp(avg_probs, min=epsilon)
        entropy = -torch.sum(avg_probs_safe * torch.log(avg_probs_safe))
        
        # Convert to "equal usage" loss: penalize low entropy
        max_entropy = torch.log(torch.tensor(self.codebook_size, device=distances.device, dtype=avg_probs.dtype))
        entropy_loss = max_entropy - entropy  # Higher when entropy is low
        
        return entropy_loss
    
    def update_usage_tracking(self, encoding_indices):
        """
        Update the sliding window of vector usage for statistics tracking.
        
        For VQVAE2, encoding_indices has shape (B,) since we have one assignment per batch item.
        
        Args:
            encoding_indices: Tensor of assignment indices (B,)
        """
        if self.usage_tracking_window <= 0:
            return
            
        # For VQVAE2, all assignments are valid (no padding mask needed)
        valid_indices = encoding_indices
        
        # Update total vectors processed
        self.total_vectors_processed += valid_indices.size(0)
            
        # Add valid assignments to circular buffer
        n_valid = valid_indices.size(0)
        if n_valid > 0:
            # Handle case where we have more assignments than buffer size
            if n_valid >= self.usage_tracking_window:
                # Fill entire buffer with most recent assignments
                self.usage_history[:] = valid_indices[-self.usage_tracking_window:]
                self.usage_ptr.fill_(0)
                self.usage_full.fill_(True)
            else:
                # Add assignments to circular buffer
                end_ptr = (self.usage_ptr + n_valid) % self.usage_tracking_window
                
                if end_ptr > self.usage_ptr:
                    # No wraparound
                    self.usage_history[self.usage_ptr:end_ptr] = valid_indices
                else:
                    # Wraparound case
                    n_until_end = self.usage_tracking_window - self.usage_ptr
                    self.usage_history[self.usage_ptr:] = valid_indices[:n_until_end]
                    if n_valid > n_until_end:
                        self.usage_history[:end_ptr] = valid_indices[n_until_end:]
                
                self.usage_ptr.copy_(end_ptr)
                if self.usage_ptr == 0 or self.usage_full:
                    self.usage_full.fill_(True)
    
    def get_usage_statistics(self):
        """
        Get current usage statistics.
        
        Returns:
            dict: Dictionary containing usage statistics
        """
        if self.usage_tracking_window <= 0 or self.codebook is None:
            return {
                'total_vectors_processed': 0,
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size if self.codebook is not None else 0),
                'usage_percentages': torch.zeros(self.codebook_size if self.codebook is not None else 0)
            }
        
        # Get the valid portion of usage history
        if self.usage_full:
            valid_history = self.usage_history
        else:
            valid_history = self.usage_history[:self.usage_ptr]
            
        if valid_history.numel() == 0:
            return {
                'total_vectors_processed': self.total_vectors_processed.item(),
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size),
                'usage_percentages': torch.zeros(self.codebook_size)
            }
        
        # Compute usage statistics
        usage_counts = torch.bincount(valid_history, minlength=self.codebook_size)
        unique_vectors_used = (usage_counts > 0).sum().item()
        usage_percentages = usage_counts.float() / valid_history.numel()
        
        return {
            'total_vectors_processed': self.total_vectors_processed.item(),
            'unique_vectors_used': unique_vectors_used,
            'usage_counts': usage_counts,
            'usage_percentages': usage_percentages
        }
    
    def compute_codebook_similarities(self):
        """
        Compute cosine similarities between used codebook vectors.
        
        Returns:
            dict: Dictionary containing similarity statistics
        """
        if self.codebook is None:
            return {
                'similarities': None,
                'used_indices': None,
                'num_used_vectors': 0
            }
        
        # Get usage statistics to find which vectors are used
        usage_stats = self.get_usage_statistics()
        used_mask = usage_stats['usage_counts'] > 0
        used_indices = torch.where(used_mask)[0]
        
        if len(used_indices) <= 1:
            return {
                'similarities': None,
                'used_indices': used_indices,
                'num_used_vectors': len(used_indices)
            }
        
        # Get used codebook vectors
        with torch.no_grad():
            used_codebook = self.codebook.weight[used_indices]
            # Normalize for cosine similarity
            used_codebook_norm = used_codebook / used_codebook.norm(dim=1, keepdim=True)
            # Compute similarity matrix
            similarities = torch.mm(used_codebook_norm, used_codebook_norm.t())
        
        return {
            'similarities': similarities,
            'used_indices': used_indices,
            'num_used_vectors': len(used_indices)
        }
        
    def apply_stochastic_mask(self, distances, training=True):
        """
        Apply stochastic masking to assignment distances for exploration.
        
        With probability mask_prob, randomly masks a fraction of the distance
        values to very large numbers, forcing the model to explore unused codes.
        This helps prevent codebook collapse by encouraging diversity.
        
        For VQVAE2, distances have shape (B, K) instead of (B*T, K).
        
        Args:
            distances: Distance tensor (B, K) between encoder outputs and codebook
            training: Whether model is in training mode
            
        Returns:
            masked_distances: Distances with stochastic masking applied
        """
        if self.mask_prob <= 0 or not training:
            return distances
            
        # Apply masking with probability mask_prob
        if torch.rand(1).item() < self.mask_prob:
            B, K = distances.shape
            
            # Create random mask - randomly select fraction mask_prob of entries to mask
            num_mask = int(self.mask_prob * B * K)
            mask_indices = torch.randperm(B * K, device=distances.device)[:num_mask]
            
            # Convert flat indices to 2D indices
            mask_rows = mask_indices // K
            mask_cols = mask_indices % K
            
            # Create masked distances by setting selected entries to large value
            masked_distances = distances.clone()
            masked_distances[mask_rows, mask_cols] = 1e10
            
            return masked_distances
        
        return distances

    def forward(self, x, padding_mask=None, beta = None):
        """
        Args:
            first_stage_tokens_right_padded: Tokens from the first stage encoder.
            padding_mask (Tensor, optional): Padding mask for Encoder2 and Decoder2.
            teacher_token_vecs (Tensor, optional): In teacher forcing mode, ground-truth tokens for the decoder,
                                                 of shape (B, T_out, d2).
            T (int, optional): Number of tokens to generate in autoregressive mode (if teacher_token_vecs is None).
        
        Returns:
            x_recon (Tensor): Reconstruction from the decoder (B, T_out, d2).
            total_loss (Tensor): Sum of reconstruction, codebook, and commitment losses.
            recon_loss (Tensor): Reconstruction loss.
            codebook_loss (Tensor): Codebook loss.
            commitment_loss (Tensor): Commitment loss.
        """

        B, T, d2 = x.shape

        # Encode: get continuous latent representation z_e from Encoder2.
        # z_e shape: (B, D)
        z_e = self.encoder(x, padding_mask=padding_mask)

        if self.codebook is not None:
            # Vector quantization: z_e shape is (B, D)
            codebook = self.codebook.weight  # shape: (codebook_size, D)
            # Compute squared L2 distances: shape (B, codebook_size)
            distances = torch.sum(z_e**2, dim=1, keepdim=True) + torch.sum(codebook**2, dim=1) - 2 * torch.matmul(z_e, codebook.t())
            
            # NEW for codebook collape - Apply stochastic masking for exploration (if enabled)
            distances = self.apply_stochastic_mask(distances, training=self.training)
            
            # Compute soft entropy loss BEFORE argmin (to maintain gradients)
            entropy_loss = self.compute_soft_entropy_loss(distances)
            
            # Find nearest codebook index for each latent vector in the batch.
            encoding_indices = torch.argmin(distances, dim=1)  # shape: (B,)
            
            # Update usage tracking
            self.update_usage_tracking(encoding_indices)
            
            unique_count = torch.unique(encoding_indices).numel()
            if self.codebook_reset_counter_multiplier > 0:
                # For VQVAE2, z_e has shape (B, D) so B is the number of latent vectors.
                B = z_e.shape[0]
                # Decrement all counters by the batch size.
                self.codebook_counters -= B
                # Reset counters for the used indices.
                self.codebook_counters[encoding_indices] = self.codebook_reset_counter_multiplier * self.codebook_size
                # Check for any collapsed codebook vectors.
                collapsed = self.codebook_counters <= 0
                if collapsed.any():
                    num_collapsed = int(collapsed.sum().item())
                    collapsed_indices = torch.where(collapsed)[0]
                    if num_collapsed > B:
                        # If there are more collapsed vectors than B,
                        # first randomly select B collapsed indices.
                        # Select B vectors from z_e without replacement.
                        new_batch_vectors = z_e[torch.randperm(B)[:B]]
                        # For the remaining collapsed vectors, compute overall mean and std per vector.
                        # Calculate the mean of each row in z_e and then average over rows.
                        overall_mean = z_e.mean(dim=1).mean()
                        # Calculate the standard deviation of each row and then average over rows.
                        overall_std = z_e.std(dim=1).mean()
                        rem = num_collapsed - B
                        random_vectors = torch.normal(mean=overall_mean.item(), std=overall_std.item(), size=(rem, z_e.shape[1]), device=z_e.device)
                        # To maintain a consistent assignment order, sort the collapsed indices.
                        sorted_indices = collapsed_indices[torch.argsort(collapsed_indices)]
                        self.codebook.weight.data[sorted_indices[:B]] = new_batch_vectors.to(self.codebook.weight.dtype)
                        self.codebook.weight.data[sorted_indices[B:]] = random_vectors.to(self.codebook.weight.dtype)
                    else:
                        # If number of collapsed vectors is less than or equal to B,
                        # update them with randomly selected vectors from z_e without replacement.
                        new_vectors = z_e[torch.randperm(B)[:num_collapsed]]
                        self.codebook.weight.data[collapsed] = new_vectors.to(self.codebook.weight.dtype)
                    # Reset the counters for these codebook entries.
                    self.codebook_counters[collapsed] = self.codebook_reset_counter_multiplier * self.codebook_size
            # Quantize: lookup codebook entries.
            z_q = self.codebook(encoding_indices)  # shape: (B, D)

            # Compute losses.
            codebook_loss = F.mse_loss(z_q.detach(), z_e)
            if beta is None:
                commitment_loss = self.beta * F.mse_loss(z_q, z_e.detach())
            else:
                commitment_loss = beta * F.mse_loss(z_q, z_e.detach())

            # Use straight-through estimator: substitute z_q in forward pass but allow gradients to flow into encoder.
            z_q = z_e + (z_q - z_e).detach()
        else:
            # No-codebook mode: bypass quantization.
            z_q = z_e
            codebook_loss = torch.tensor(0.0, device=z_e.device)
            commitment_loss = torch.tensor(0.0, device=z_e.device)
            entropy_loss = torch.tensor(0.0, device=z_e.device)

        # Decode:
        # Decoder2 now expects input of shape (B, D) and supports padding mask.
            
        x_recon = self.decoder(z_q, teacher_token_vecs=x, padding_mask=padding_mask)

        if padding_mask is not None:
            # Expand padding_mask to match x_recon/x dimensions
            active_mask = padding_mask.unsqueeze(-1).unsqueeze(1).float()
            # Compute MSE only over active (non-padded) positions and average over them
            recon_loss = F.mse_loss(x_recon * active_mask, x * active_mask, reduction='sum') / active_mask.sum()
        else:
            recon_loss = F.mse_loss(x_recon, x)
        
        # NEW for codebook collape - Compute new regularization losses  
        # total_loss = recon_loss + codebook_loss + commitment_loss
        # Compute cosine-push regularization loss (replaces orthogonality loss)
        cosine_push_loss = self.compute_cosine_push_loss()
        
        # entropy_loss is already computed in the quantization section above
        if self.codebook is None:
            entropy_loss = torch.tensor(0.0, device=z_e.device)
        
        total_loss = recon_loss + codebook_loss + commitment_loss + \
                    self.cosine_push_weight * cosine_push_loss + \
                    self.entropy_loss_weight * entropy_loss

        # return x_recon, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count
        return x_recon, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count, cosine_push_loss, entropy_loss
        
class Encoder2(nn.Module):
    """
    Encoder for a VQVAE variant.

    This encoder takes an input tensor of shape (B, T, d2) and processes it as follows:
      1. It passes the entire sequence through several transformer blocks (encoder-only).
      2. From the resulting (B, T, d2) tensor, only the first T* tokens are selected,
         resulting in a tensor of shape (B, T*, d2).
      3. The selected tokens are flattened to form a vector of dimension (T* * d2).
      4. A linear projection reduces the flattened vector to dimension D.

    The final output has shape (B, D).
    """
    def __init__(self, d2, min_seq_len, D, num_layers=3, config=None):
        """
        Args:
            d2 (int): Embedding dimension of the input and transformer outputs.
            T_star (int): Number of tokens (from the end of the sequence) to consider.
            D (int): Desired output dimension after projection.
            num_layers (int): Number of transformer blocks to use.
            config (dict, optional): Configuration dictionary for the transformer blocks.
                                     If not provided, a default configuration is used.
        """
        super().__init__()
        self.d2 = d2
        self.D = D
        self.min_seq_len = min_seq_len
        
        config['n_embd'] = d2
        config['mlp_hidden_dim'] = 4 * d2

        # Transformer blocks to process the input sequence.
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(num_layers)
        ])


        # Do proj + relu + proj to get D
        self.proj = nn.Sequential(
            nn.Linear(min_seq_len * d2, int(math.sqrt(min_seq_len * d2* D))),
            nn.ReLU(),
            nn.Linear(int(math.sqrt(min_seq_len * d2* D)), D)
        )

    def forward(self, x, padding_mask=None):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, T, d2).

        Returns:
            Tensor: Output tensor of shape (B, D).
        """
        B, T, d2_input = x.shape
        assert d2_input == self.d2, f"Expected last dimension to be {self.d2} but got {d2_input}"
        assert T >= self.min_seq_len, f"Input sequence length T={T} is smaller than min_seq_len={self.min_seq_len}"
        
        # Pass input through the transformer blocks.
        for block in self.transformer_blocks:
            x = block(x, padding_mask=padding_mask)  # Shape remains (B, T, d2)
        

        # Only supports right padding
        x_first = x[:, :self.min_seq_len, :]
        
        # Flatten the selected tokens into a vector of size (min_seq_len * d2).
        x_flat = x_first.view(B, self.min_seq_len * self.d2)  # Shape: (B, min_seq_len * d2)
        
        # Project the flattened vector to dimension D.
        output = self.proj(x_flat)  # Shape: (B, D)
        
        return output

class Decoder2(nn.Module):
    """
    Non-autoregressive Decoder for a VQVAE variant.

    This decoder takes an input tensor of shape (B, D) and proceeds as follows:

      Upscaling:
        Two linear layers map the input from dimension D to:
        1. (T_star * d₂) for the prefix
        2. (T * d₂) for the main sequence

      Encoder-like Processing:
        The full sequence of shape (B, T_star + T, d₂) is processed through encoder transformer blocks
        (non-causal attention) with padding mask support.

      Output Selection:
        Only the parts corresponding to the original sequence length are considered for the final output.
    """
    def __init__(self, d2, max_prefix_len, D, num_layers=3, config=None):
        """
        Args:
            d2 (int): Embedding dimension used in the transformer blocks.
            D (int): Input dimension.
            num_layers (int): Number of transformer blocks.
            config (dict, optional): Configuration dictionary for the transformer blocks.
                                     If not provided, a default configuration is used.
        """
        super().__init__()
        self.d2 = d2
        self.D = D

        
        # Upscaling layer for main sequence: projects input from dimension D to (T_max * d2)
        # We'll use a reasonable maximum T for the projection
        self.T_max = max_prefix_len  # You can adjust this based on your expected maximum sequence length
        self.proj_sequence = nn.Sequential(
            nn.Linear(D, int(math.sqrt(D * self.T_max * d2))),
            nn.ReLU(),
            nn.Linear(int(math.sqrt(D * self.T_max * d2)), self.T_max * d2)
        )

        # Update config for encoder-like transformer blocks (non-causal)
        config = config.copy()
        config['n_embd'] = d2
        config['mlp_hidden_dim'] = 4 * d2
        config['is_decoder'] = False  # Use encoder-like attention (non-causal)

        # Create a stack of encoder-like transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(num_layers)
        ])

    def forward(self, x, teacher_token_vecs=None, padding_mask=None):
        """
        Forward pass supporting both teacher forcing (training) and generation (inference).

        Args:
            x (Tensor): Input tensor of shape (B, D).
            T (int, optional): Number of new tokens to generate in inference mode.
                             Must be provided if teacher_token_vecs is None.
            teacher_token_vecs (Tensor, optional): Ground truth tokens for teacher forcing.
                                                 Expected shape is (B, T, d2).
            padding_mask (Tensor, optional): Padding mask for the sequence.

        Returns:
            Tensor: If teacher_token_vecs is provided, returns output of shape (B, T, codebook_len)
                    corresponding to softmax probabilities.
                    Otherwise, returns generated sequence of shape (B, T, d2).
        """

        B, T, _ = teacher_token_vecs.shape
        
        # Generate main sequence: (B, T, d2)
        sequence = self.proj_sequence(x)  # (B, T_max * d2)
        sequence = sequence.view(B, self.T_max, self.d2)
        out = sequence[:, :T, :]  # Take only T tokens for efficiency in case T<T_max

        # Process through transformer blocks
        for block in self.transformer_blocks:
            out = block(out, padding_mask=padding_mask)  # Shape remains (B, T, d2)
        return out

# VQVAE class for single token (last layer last token)
class VQVAELastToken(nn.Module):
    def __init__(self, input_dim, hidden_dim, codebook_size, beta, config):
        """
        VQVAE for single token (no temporal dimension)
        Similar to VQVAE_single but for a single vector
        
        Args:
            input_dim: Input dimension (n_embd of the LLM)
            hidden_dim: Hidden/latent dimension
            codebook_size: Number of discrete codes
            beta: Commitment loss coefficient
            config: Configuration dictionary with additional parameters
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.beta = beta
        
        # Store experimental feature settings from config
        self.cosine_push_weight = config.get('cosine_push_weight', 0.0)
        self.entropy_loss_weight = config.get('entropy_loss_weight', 0.0)
        self.entropy_temperature = config.get('entropy_temperature', 1.0)
        
        # Architecture configuration
        self.num_encoder_layers = config.get('num_encoder_layers')
        self.num_decoder_layers = config.get('num_decoder_layers')
        self.mlp_ratio = config.get('mlp_ratio')  # Hidden dimension multiplier for MLP blocks
        self.use_residual = config.get('use_residual', True)
        self.activation = config.get('activation', 'gelu')  # 'relu' or 'gelu'
        self.dropout = config.get('dropout', 0.1)
        self.layer_norm = config.get('layer_norm', True)
        
        # For vector usage tracking
        self.usage_tracking_window = config.get('usage_tracking_window', 5000)  # Window size for tracking usage
        if self.usage_tracking_window > 0:
            self.register_buffer('usage_history', torch.zeros(self.usage_tracking_window, dtype=torch.long))
            self.register_buffer('usage_ptr', torch.tensor(0, dtype=torch.long))
            self.register_buffer('usage_full', torch.tensor(False, dtype=torch.bool))
            self.register_buffer('total_vectors_processed', torch.tensor(0, dtype=torch.long))
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Codebook
        if self.codebook_size > 0:
            self.codebook = nn.Embedding(self.codebook_size, hidden_dim)
            nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
        else:
            self.codebook = None
    
    def _get_activation(self):
        """Get activation function based on config."""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _build_encoder(self):
        """Build encoder with configurable layers and residual connections."""
        layers = []
        current_dim = self.input_dim
        
        # Input projection layer
        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        if self.layer_norm:
            layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(self._get_activation())
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        
        # Hidden layers with residual connections
        for i in range(self.num_encoder_layers):
            if self.use_residual:
                layers.append(ResidualBlock(
                    self.hidden_dim, 
                    int(self.hidden_dim * self.mlp_ratio),
                    self.hidden_dim,
                    activation=self.activation,
                    dropout=self.dropout,
                    layer_norm=self.layer_norm
                ))
            else:
                # Standard MLP block without residual
                layers.append(nn.Linear(self.hidden_dim, int(self.hidden_dim * self.mlp_ratio)))
                if self.layer_norm:
                    layers.append(nn.LayerNorm(int(self.hidden_dim * self.mlp_ratio)))
                layers.append(self._get_activation())
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
                layers.append(nn.Linear(int(self.hidden_dim * self.mlp_ratio), self.hidden_dim))
                if self.layer_norm and i < self.num_encoder_layers - 1:
                    layers.append(nn.LayerNorm(self.hidden_dim))
                if i < self.num_encoder_layers - 1:
                    layers.append(self._get_activation())
                    if self.dropout > 0:
                        layers.append(nn.Dropout(self.dropout))
        
        # Final layer norm before quantization
        if self.layer_norm:
            layers.append(nn.LayerNorm(self.hidden_dim))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self):
        """Build decoder with configurable layers and residual connections."""
        layers = []
        
        # Hidden layers with residual connections
        for i in range(self.num_decoder_layers):
            if self.use_residual:
                layers.append(ResidualBlock(
                    self.hidden_dim, 
                    int(self.hidden_dim * self.mlp_ratio),
                    self.hidden_dim,
                    activation=self.activation,
                    dropout=self.dropout,
                    layer_norm=self.layer_norm
                ))
            else:
                # Standard MLP block without residual
                layers.append(nn.Linear(self.hidden_dim, int(self.hidden_dim * self.mlp_ratio)))
                if self.layer_norm:
                    layers.append(nn.LayerNorm(int(self.hidden_dim * self.mlp_ratio)))
                layers.append(self._get_activation())
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
                layers.append(nn.Linear(int(self.hidden_dim * self.mlp_ratio), self.hidden_dim))
                if self.layer_norm:
                    layers.append(nn.LayerNorm(self.hidden_dim))
                layers.append(self._get_activation())
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
        
        # Output projection layer
        layers.append(nn.Linear(self.hidden_dim, self.input_dim))
        
        return nn.Sequential(*layers)
            
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (B, input_dim)
        Returns:
            x_recon: Reconstructed tensor
            total_loss: Total loss
            recon_loss: Reconstruction loss
            codebook_loss: Codebook loss
            commitment_loss: Commitment loss
            unique_count: Number of unique codes used
        """
        B = x.shape[0]
        
        # Encode
        z_e = self.encoder(x)  # (B, hidden_dim)
        
        if self.codebook is not None:
            # Compute distances to codebook vectors
            codebook_weights = self.codebook.weight  # (codebook_size, hidden_dim)
            distances = torch.sum(z_e**2, dim=1, keepdim=True) + \
                       torch.sum(codebook_weights**2, dim=1) - \
                       2 * torch.matmul(z_e, codebook_weights.t())  # (B, codebook_size)
            
            # Find nearest codebook entries
            encoding_indices = torch.argmin(distances, dim=1)  # (B,)
            unique_count = torch.unique(encoding_indices).numel()
            
            # Update usage tracking
            self.update_usage_tracking(encoding_indices)
            
            # Quantize
            z_q = self.codebook(encoding_indices)  # (B, hidden_dim)
            
            # Compute losses
            codebook_loss = F.mse_loss(z_q.detach(), z_e)
            commitment_loss = self.beta * F.mse_loss(z_q, z_e.detach())
            
            # Straight-through estimator
            z_q = z_e + (z_q - z_e).detach()
            
            # Compute entropy loss if enabled
            if self.entropy_loss_weight > 0:
                assignment_probs = F.softmax(-distances / self.entropy_temperature, dim=-1)
                avg_probs = assignment_probs.mean(dim=0)
                epsilon = 1e-10
                avg_probs_safe = torch.clamp(avg_probs, min=epsilon)
                entropy = -torch.sum(avg_probs_safe * torch.log(avg_probs_safe))
                max_entropy = torch.log(torch.tensor(self.codebook_size, device=distances.device, dtype=avg_probs.dtype))
                entropy_loss = max_entropy - entropy
            else:
                entropy_loss = torch.tensor(0.0, device=z_e.device)
        else:
            z_q = z_e
            codebook_loss = torch.tensor(0.0, device=z_e.device)
            commitment_loss = torch.tensor(0.0, device=z_e.device)
            entropy_loss = torch.tensor(0.0, device=z_e.device)
            unique_count = 0
        
        # Decode
        x_recon = self.decoder(z_q)  # (B, input_dim)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # Compute cosine-push regularization loss
        cosine_push_loss = self.compute_cosine_push_loss()
        
        # Total loss
        total_loss = recon_loss + codebook_loss + commitment_loss + \
                    self.cosine_push_weight * cosine_push_loss + \
                    self.entropy_loss_weight * entropy_loss
        
        return x_recon, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count, cosine_push_loss, entropy_loss
    
    def normalize_codebook_vectors(self):
        """
        Normalize each codebook vector to unit length.
        
        This is applied after each parameter update when using cosine-push regularization.
        Helps ensure the cosine-push loss focuses on angular relationships rather than magnitudes.
        """
        if self.cosine_push_weight > 0 and self.codebook is not None:
            with torch.no_grad():
                # Normalize each row (codebook vector) to unit length
                self.codebook.weight.data = F.normalize(self.codebook.weight.data, p=2, dim=1)
    
    def compute_cosine_push_loss(self):
        """
        Compute cosine-push regularization loss on the codebook matrix.
        
        The cosine-push loss encourages codebook vectors to be orthogonal by penalizing
        the squared cosine similarities between different vectors. This replaces the
        gram-matrix Frobenius norm with a more direct angular penalty.
        
        If usage tracking is enabled, only considers used vectors and weights each
        vector's contribution by its usage frequency.
        
        L_push = sum_{i≠j} w_i * w_j * (dot(e_i, e_j))^2
        
        Where e_i and e_j are normalized codebook vectors, and w_i is the usage weight.
        
        Returns:
            cosine_push_loss: Cosine-push loss scalar
        """
        if self.cosine_push_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # If usage tracking is enabled, use only tracked vectors with usage-based weighting
        if self.usage_tracking_window > 0:
            usage_stats = self.get_usage_statistics()
            usage_counts = usage_stats['usage_counts']
            used_mask = usage_counts > 0
            used_indices = torch.where(used_mask)[0]
            
            if len(used_indices) <= 1:
                # Need at least 2 vectors to compute similarities
                return torch.tensor(0.0, device=next(self.parameters()).device)
            
            # Get used codebook vectors and normalize them
            E_used = self.codebook.weight[used_indices]  # (num_used, hidden_dim)
            E_used_normalized = F.normalize(E_used, p=2, dim=1)  # (num_used, hidden_dim)
            
            # Compute cosine similarity matrix for used vectors: (num_used, num_used)
            cosine_sim_matrix = torch.mm(E_used_normalized, E_used_normalized.t())
            
            # Get usage weights for used vectors and normalize them
            usage_weights = usage_counts[used_indices].float()  # (num_used,)
            usage_weights = usage_weights / usage_weights.sum()  # Normalize to sum to 1
            
            # Create weight matrix: w_i * w_j for all pairs
            weight_matrix = usage_weights.unsqueeze(1) * usage_weights.unsqueeze(0)  # (num_used, num_used)
            
            # Zero out diagonal (self-similarities) and apply weights
            num_used = len(used_indices)
            mask = ~torch.eye(num_used, device=E_used.device, dtype=torch.bool)
            
            # Apply weights and compute weighted sum of squared cosine similarities
            weighted_cosines_squared = (cosine_sim_matrix ** 2) * weight_matrix
            cosine_push_loss = weighted_cosines_squared[mask].sum()
            
        else:
            # Original implementation when usage tracking is disabled
            # Get codebook matrix E: (K, hidden_dim)
            E = self.codebook.weight  # (codebook_size, hidden_dim)
            
            # Normalize vectors to unit length for cosine computation
            E_normalized = F.normalize(E, p=2, dim=1)  # (K, hidden_dim)
            
            # Compute cosine similarity matrix: (K, K)
            cosine_sim_matrix = torch.mm(E_normalized, E_normalized.t())  # (K, K)
            
            # Zero out diagonal (self-similarities) and compute sum of squared off-diagonal elements
            mask = ~torch.eye(self.codebook_size, device=E.device, dtype=torch.bool)
            off_diagonal_cosines = cosine_sim_matrix[mask]  # Get all off-diagonal elements
            
            # Sum of squared cosine similarities between different vectors
            cosine_push_loss = torch.sum(off_diagonal_cosines ** 2)
        
        return cosine_push_loss
    
    def update_usage_tracking(self, encoding_indices):
        """
        Update the sliding window of vector usage for statistics tracking.
        
        For VQVAELastToken, encoding_indices has shape (B,) since we have one assignment per batch item.
        
        Args:
            encoding_indices: Tensor of assignment indices (B,)
        """
        if self.usage_tracking_window <= 0:
            return
            
        # For VQVAELastToken, all assignments are valid (no padding mask needed)
        valid_indices = encoding_indices
        
        # Update total vectors processed
        self.total_vectors_processed += valid_indices.size(0)
            
        # Add valid assignments to circular buffer
        n_valid = valid_indices.size(0)
        if n_valid > 0:
            # Handle case where we have more assignments than buffer size
            if n_valid >= self.usage_tracking_window:
                # Fill entire buffer with most recent assignments
                self.usage_history[:] = valid_indices[-self.usage_tracking_window:]
                self.usage_ptr.fill_(0)
                self.usage_full.fill_(True)
            else:
                # Add assignments to circular buffer
                end_ptr = (self.usage_ptr + n_valid) % self.usage_tracking_window
                
                if end_ptr > self.usage_ptr:
                    # No wraparound
                    self.usage_history[self.usage_ptr:end_ptr] = valid_indices
                else:
                    # Wraparound case
                    n_until_end = self.usage_tracking_window - self.usage_ptr
                    self.usage_history[self.usage_ptr:] = valid_indices[:n_until_end]
                    if n_valid > n_until_end:
                        self.usage_history[:end_ptr] = valid_indices[n_until_end:]
                
                self.usage_ptr.copy_(end_ptr)
                if self.usage_ptr == 0 or self.usage_full:
                    self.usage_full.fill_(True)
    
    def get_usage_statistics(self):
        """
        Get current usage statistics.
        
        Returns:
            dict: Dictionary containing usage statistics
        """
        if self.usage_tracking_window <= 0 or self.codebook is None:
            return {
                'total_vectors_processed': 0,
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size if self.codebook is not None else 0),
                'usage_percentages': torch.zeros(self.codebook_size if self.codebook is not None else 0)
            }
        
        # Get the valid portion of usage history
        if self.usage_full:
            valid_history = self.usage_history
        else:
            valid_history = self.usage_history[:self.usage_ptr]
            
        if valid_history.numel() == 0:
            return {
                'total_vectors_processed': self.total_vectors_processed.item(),
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size),
                'usage_percentages': torch.zeros(self.codebook_size)
            }
        
        # Compute usage statistics
        usage_counts = torch.bincount(valid_history, minlength=self.codebook_size)
        unique_vectors_used = (usage_counts > 0).sum().item()
        usage_percentages = usage_counts.float() / valid_history.numel()
        
        return {
            'total_vectors_processed': self.total_vectors_processed.item(),
            'unique_vectors_used': unique_vectors_used,
            'usage_counts': usage_counts,
            'usage_percentages': usage_percentages
        }
    
    def compute_codebook_similarities(self):
        """
        Compute cosine similarities between used codebook vectors.
        
        Returns:
            dict: Dictionary containing similarity statistics
        """
        if self.codebook is None:
            return {
                'similarities': None,
                'used_indices': None,
                'num_used_vectors': 0
            }
        
        # Get usage statistics to find which vectors are used
        usage_stats = self.get_usage_statistics()
        used_mask = usage_stats['usage_counts'] > 0
        used_indices = torch.where(used_mask)[0]
        
        if len(used_indices) <= 1:
            return {
                'similarities': None,
                'used_indices': used_indices,
                'num_used_vectors': len(used_indices)
            }
        
        # Get used codebook vectors
        with torch.no_grad():
            used_codebook = self.codebook.weight[used_indices]
            # Normalize for cosine similarity
            used_codebook_norm = used_codebook / used_codebook.norm(dim=1, keepdim=True)
            # Compute similarity matrix
            similarities = torch.mm(used_codebook_norm, used_codebook_norm.t())
        
        return {
            'similarities': similarities,
            'used_indices': used_indices,
            'num_used_vectors': len(used_indices)
        }

class VQVAE3_Modified(VQVAE3):
    """
    Modified VQVAE3 class with alternative cosine push loss calculation methods.
    
    Two different approaches:
    1. 'log_scaling': Uses log scaling to dampen quadratic growth (as previously implemented)
    2. 'count_division': Ignores usage weights and divides by number of used vectors
    
    Set cosine_push_method in config to choose the method.
    """
    
    def __init__(self, encoder, decoder, config):
        super().__init__(encoder, decoder, config)
        
        # Choose the cosine push loss calculation method
        self.cosine_push_method = config.get('cosine_push_method', 'log_scaling')
        assert self.cosine_push_method in ['log_scaling', 'count_division'], \
            f"cosine_push_method must be 'log_scaling' or 'count_division', got {self.cosine_push_method}"
    
    def compute_cosine_push_loss(self):
        """
        Compute cosine-push regularization loss with alternative methods.
        """
        if self.cosine_push_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        if self.cosine_push_method == 'log_scaling':
            return self._compute_cosine_push_loss_log_scaling()
        elif self.cosine_push_method == 'count_division':
            return self._compute_cosine_push_loss_count_division()
    
    def _compute_cosine_push_loss_log_scaling(self):
        """
        Method 1: Uses the original logic but applies log scaling at the end.
        """
        # If usage tracking is enabled, use only tracked vectors with usage-based weighting
        if self.usage_tracking_window > 0:
            usage_stats = self.get_usage_statistics()
            usage_counts = usage_stats['usage_counts']
            used_mask = usage_counts > 0
            used_indices = torch.where(used_mask)[0]
            
            if len(used_indices) <= 1:
                return torch.tensor(0.0, device=next(self.parameters()).device)
            
            # Get used codebook vectors and normalize them
            E_used = self.codebook.weight[used_indices]  # (num_used, D)
            E_used_normalized = F.normalize(E_used, p=2, dim=1)  # (num_used, D)
            
            # Compute cosine similarity matrix for used vectors: (num_used, num_used)
            cosine_sim_matrix = torch.mm(E_used_normalized, E_used_normalized.t())
            
            # Get usage weights for used vectors and normalize them
            usage_weights = usage_counts[used_indices].float()  # (num_used,)
            usage_weights = usage_weights / usage_weights.sum()  # Normalize to sum to 1
            
            # Create weight matrix: w_i * w_j for all pairs
            weight_matrix = usage_weights.unsqueeze(1) * usage_weights.unsqueeze(0)  # (num_used, num_used)
            
            # Zero out diagonal (self-similarities) and apply weights
            num_used = len(used_indices)
            mask = ~torch.eye(num_used, device=E_used.device, dtype=torch.bool)
            
            # Apply weights and compute weighted sum of squared cosine similarities
            weighted_cosines_squared = (cosine_sim_matrix ** 2) * weight_matrix
            cosine_push_loss = weighted_cosines_squared[mask].sum()
            
            # Apply log scaling based on number of used vectors
            scaling_factor = 1.0 / torch.log(torch.tensor(max(num_used, 2), dtype=torch.float, device=cosine_push_loss.device))
            
        else:
            # Original implementation when usage tracking is disabled
            E = self.codebook.weight  # (codebook_size, D)
            E_normalized = F.normalize(E, p=2, dim=1)  # (K, D)
            cosine_sim_matrix = torch.mm(E_normalized, E_normalized.t())  # (K, K)
            
            mask = ~torch.eye(self.codebook_size, device=E.device, dtype=torch.bool)
            off_diagonal_cosines = cosine_sim_matrix[mask]
            cosine_push_loss = torch.sum(off_diagonal_cosines ** 2)
            
            # Apply log scaling based on codebook size
            scaling_factor = 1.0 / torch.log(torch.tensor(self.codebook_size, dtype=torch.float, device=cosine_push_loss.device))
        
        return cosine_push_loss * scaling_factor
    
    def _compute_cosine_push_loss_count_division(self):
        """
        Method 2: Ignores usage frequency weights and simply divides by number of used vectors.
        """
        # If usage tracking is enabled, use only tracked vectors but ignore usage weights
        if self.usage_tracking_window > 0:
            usage_stats = self.get_usage_statistics()
            usage_counts = usage_stats['usage_counts']
            used_mask = usage_counts > 0
            used_indices = torch.where(used_mask)[0]
            
            if len(used_indices) <= 1:
                return torch.tensor(0.0, device=next(self.parameters()).device)
            
            # Get used codebook vectors and normalize them
            E_used = self.codebook.weight[used_indices]  # (num_used, D)
            E_used_normalized = F.normalize(E_used, p=2, dim=1)  # (num_used, D)
            
            # Compute cosine similarity matrix for used vectors: (num_used, num_used)
            cosine_sim_matrix = torch.mm(E_used_normalized, E_used_normalized.t())
            
            # Zero out diagonal (self-similarities) - NO usage weights applied
            num_used = len(used_indices)
            mask = ~torch.eye(num_used, device=E_used.device, dtype=torch.bool)
            
            # Sum squared cosine similarities WITHOUT usage weights
            cosine_push_loss = torch.sum((cosine_sim_matrix ** 2)[mask])
            
            # Divide by number of used vectors (not pairs)
            cosine_push_loss = cosine_push_loss / num_used
            
        else:
            # When usage tracking is disabled, use all vectors
            E = self.codebook.weight  # (codebook_size, D)
            E_normalized = F.normalize(E, p=2, dim=1)  # (K, D)
            cosine_sim_matrix = torch.mm(E_normalized, E_normalized.t())  # (K, K)
            
            mask = ~torch.eye(self.codebook_size, device=E.device, dtype=torch.bool)
            off_diagonal_cosines = cosine_sim_matrix[mask]
            cosine_push_loss = torch.sum(off_diagonal_cosines ** 2)
            
            # Divide by codebook size
            cosine_push_loss = cosine_push_loss / self.codebook_size
        
        return cosine_push_loss

class Path_Encoder(nn.Module):
    """
    Encoder for path sequences of integer node labels with fixed length T.

    Input: x (B, T) long tensor of token ids
    Output: z_e (B, T*d_model) continuous latent vector
    """
    def __init__(self, vocab_size, d_model, T, num_layers=3, config=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.T = T

        self.token_embed = nn.Embedding(vocab_size, d_model)

        if config is None:
            raise ValueError("config is required for path encoder transformer blocks")
        else:
            config["n_embd"] = d_model
            config["mlp_hidden_dim"] = 4 * d_model

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(num_layers)])

    def forward(self, x, padding_mask=None):
        # x: (B, T) long
        B, T = x.shape
        assert T == self.T, f"Expected T={self.T} but got {T}"
        h = self.token_embed(x)  # (B, T, d_model)
        for blk in self.blocks:
            h = blk(h, padding_mask=padding_mask)
        z_e = h.reshape(B, self.T * self.d_model)  # (B, T*d_model)
        return z_e

class Path_Decoder(nn.Module):
    """
    Decoder for path sequences.

    Input: z (B, T*d_model)
    Output: logits over tokens (B, T, vocab_size)
    """
    def __init__(self, vocab_size, d_model, T, num_layers=3, config=None, tied_token_embedding_weight=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.T = T

        if config is None:
            raise ValueError("config is required for path decoder transformer blocks")
        else:
            config["n_embd"] = d_model
            config["mlp_hidden_dim"] = 4 * d_model

        self.aggregate_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(num_layers)])

        self.tied = tied_token_embedding_weight is not None
        if self.tied:
            self.tied_weight = tied_token_embedding_weight  # expected shape (vocab_size, d_model)
            self.token_head_bias = nn.Parameter(torch.zeros(vocab_size))
            self.token_head = None
        else:
            self.token_head = nn.Linear(d_model, vocab_size)

    def forward(self, z, padding_mask=None):
        # z: (B, T*d_model)
        B, z_dim = z.shape
        expected_dim = self.T * self.d_model
        assert z_dim == expected_dim, f"Expected T*d_model={expected_dim} but got {z_dim}"
        x = z.view(B, self.T, self.d_model)  # (B, T, d_model)
        for blk in self.aggregate_blocks:
            x = blk(x, padding_mask=padding_mask)
        if self.tied:
            logits = F.linear(x, self.tied_weight, bias=self.token_head_bias)
        else:
            logits = self.token_head(x)
        return logits  # (B, T, V)


class Path_VQVAE(nn.Module):
    """
    VQ-VAE for fixed-length integer paths.

    - Encoder maps (B, T) tokens -> (B, d_model) latent
    - Quantization over codebook in R^d_model
    - Decoder maps (B, d_model) -> logits (B, T, V)
    - Reconstruction loss: token cross-entropy
    """
    def __init__(self, encoder, decoder, config):
        super().__init__()
        self.encoder = encoder   # Path_encoder
        self.decoder = decoder   # Path_Decoder

        self.codebook_size = config.get("codebook_size", 0)
        self.codebook_reset_counter_multiplier = config.get("codebook_reset_counter_multiplier", 0)
        self.beta = config.get("beta", 0.25)

        if self.codebook_size > 0:
            self.codebook = nn.Embedding(self.codebook_size, encoder.T * encoder.d_model)
            nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
        else:
            self.codebook = None

        if self.codebook_reset_counter_multiplier > 0 and self.codebook_size > 0:
            self.register_buffer("codebook_counters", torch.full((self.codebook_size,), 0))

        # Regularization and exploration
        self.cosine_push_weight = config.get('cosine_push_weight', 0.0)
        self.entropy_loss_weight = config.get('entropy_loss_weight', 0.0)
        self.mask_prob = config.get('mask_prob', 0.0)
        self.entropy_temperature = config.get('entropy_temperature', 1.0)

        # Usage tracking
        self.usage_tracking_window = config.get('usage_tracking_window', 5000)
        if self.usage_tracking_window > 0 and self.codebook_size > 0:
            self.register_buffer('usage_history', torch.zeros(self.usage_tracking_window, dtype=torch.long))
            self.register_buffer('usage_ptr', torch.tensor(0, dtype=torch.long))
            self.register_buffer('usage_full', torch.tensor(False, dtype=torch.bool))
            self.register_buffer('total_vectors_processed', torch.tensor(0, dtype=torch.long))

    def normalize_codebook_vectors(self):
        if self.cosine_push_weight > 0 and self.codebook is not None:
            with torch.no_grad():
                self.codebook.weight.data = F.normalize(self.codebook.weight.data, p=2, dim=1)

    def compute_soft_entropy_loss(self, distances):
        # distances: (B, K)
        if self.entropy_loss_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=distances.device)
        assignment_probs = F.softmax(-distances / self.entropy_temperature, dim=-1)  # (B, K)
        avg_probs = assignment_probs.mean(dim=0)  # (K,)
        epsilon = 1e-10
        avg_probs_safe = torch.clamp(avg_probs, min=epsilon)
        entropy = -torch.sum(avg_probs_safe * torch.log(avg_probs_safe))
        max_entropy = torch.log(torch.tensor(self.codebook_size, device=distances.device, dtype=avg_probs.dtype))
        entropy_loss = max_entropy - entropy
        return entropy_loss

    def apply_stochastic_mask(self, distances, training=True):
        if self.mask_prob <= 0 or not training:
            return distances
        if torch.rand(1).item() < self.mask_prob:
            B, K = distances.shape
            num_mask = int(self.mask_prob * B * K)
            if num_mask > 0:
                idx = torch.randperm(B * K, device=distances.device)[:num_mask]
                rows = idx // K
                cols = idx % K
                masked = distances.clone()
                masked[rows, cols] = 1e10
                return masked
        return distances
        
    def compute_codebook_similarities(self):
        """
        Compute cosine similarities between used codebook vectors.
        
        Returns:
            dict: Dictionary containing similarity statistics
        """
        if self.codebook is None:
            return {
                'similarities': None,
                'used_indices': None,
                'num_used_vectors': 0
            }
        
        # Get usage statistics to find which vectors are used
        usage_stats = self.get_usage_statistics()
        used_mask = usage_stats['usage_counts'] > 0
        used_indices = torch.where(used_mask)[0]
        
        if len(used_indices) <= 1:
            return {
                'similarities': None,
                'used_indices': used_indices,
                'num_used_vectors': len(used_indices)
            }
        
        # Get used codebook vectors
        with torch.no_grad():
            used_codebook = self.codebook.weight[used_indices]
            # Normalize for cosine similarity
            used_codebook_norm = used_codebook / used_codebook.norm(dim=1, keepdim=True)
            # Compute similarity matrix
            similarities = torch.mm(used_codebook_norm, used_codebook_norm.t())
        
        return {
            'similarities': similarities,
            'used_indices': used_indices,
            'num_used_vectors': len(used_indices)
        }

    def update_usage_tracking(self, encoding_indices):
        if self.usage_tracking_window <= 0 or self.codebook is None:
            return
        valid_indices = encoding_indices  # (B,)
        self.total_vectors_processed += valid_indices.size(0)
        n_valid = valid_indices.size(0)
        if n_valid > 0:
            if n_valid >= self.usage_tracking_window:
                self.usage_history[:] = valid_indices[-self.usage_tracking_window:]
                self.usage_ptr.fill_(0)
                self.usage_full.fill_(True)
            else:
                end_ptr = (self.usage_ptr + n_valid) % self.usage_tracking_window
                if end_ptr > self.usage_ptr:
                    self.usage_history[self.usage_ptr:end_ptr] = valid_indices
                else:
                    n_until_end = self.usage_tracking_window - self.usage_ptr
                    self.usage_history[self.usage_ptr:] = valid_indices[:n_until_end]
                    if n_valid > n_until_end:
                        self.usage_history[:end_ptr] = valid_indices[n_until_end:]
                self.usage_ptr.copy_(end_ptr)
                if self.usage_ptr == 0 or self.usage_full:
                    self.usage_full.fill_(True)

    def get_usage_statistics(self):
        if self.usage_tracking_window <= 0 or self.codebook is None:
            return {
                'total_vectors_processed': 0,
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size if self.codebook is not None else 0),
                'usage_percentages': torch.zeros(self.codebook_size if self.codebook is not None else 0)
            }
        if self.usage_full:
            valid_history = self.usage_history
        else:
            valid_history = self.usage_history[:self.usage_ptr]
        if valid_history.numel() == 0:
            return {
                'total_vectors_processed': int(self.total_vectors_processed.item()) if isinstance(self.total_vectors_processed, torch.Tensor) else 0,
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size),
                'usage_percentages': torch.zeros(self.codebook_size)
            }
        usage_counts = torch.bincount(valid_history, minlength=self.codebook_size)
        unique_vectors_used = (usage_counts > 0).sum().item()
        usage_percentages = usage_counts.float() / valid_history.numel()
        return {
            'total_vectors_processed': int(self.total_vectors_processed.item()),
            'unique_vectors_used': unique_vectors_used,
            'usage_counts': usage_counts,
            'usage_percentages': usage_percentages
        }

    def compute_cosine_push_loss(self):
        if self.cosine_push_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        if self.usage_tracking_window > 0:
            usage_stats = self.get_usage_statistics()
            usage_counts = usage_stats['usage_counts']
            used_mask = usage_counts > 0
            used_indices = torch.where(used_mask)[0]
            if len(used_indices) <= 1:
                return torch.tensor(0.0, device=next(self.parameters()).device)
            E_used = self.codebook.weight[used_indices]
            E_used_normalized = F.normalize(E_used, p=2, dim=1)
            cosine_sim_matrix = torch.mm(E_used_normalized, E_used_normalized.t())
            usage_weights = usage_counts[used_indices].float()
            usage_weights = usage_weights / usage_weights.sum()
            weight_matrix = usage_weights.unsqueeze(1) * usage_weights.unsqueeze(0)
            num_used = len(used_indices)
            mask = ~torch.eye(num_used, device=E_used.device, dtype=torch.bool)
            weighted_cosines_squared = (cosine_sim_matrix ** 2) * weight_matrix
            cosine_push_loss = weighted_cosines_squared[mask].sum()
        else:
            E = self.codebook.weight
            E_normalized = F.normalize(E, p=2, dim=1)
            cosine_sim_matrix = torch.mm(E_normalized, E_normalized.t())
            mask = ~torch.eye(self.codebook_size, device=E.device, dtype=torch.bool)
            off_diagonal_cosines = cosine_sim_matrix[mask]
            cosine_push_loss = torch.sum(off_diagonal_cosines ** 2)
        return cosine_push_loss

    def forward(self, x, padding_mask=None, beta=None):
        # x: (B, T) long tokens
        z_e = self.encoder(x, padding_mask=padding_mask)  # (B, T*d_model)
        unique_count = torch.tensor(0, device=z_e.device)

        if self.codebook is not None:
            B, latent_dim = z_e.shape
            codebook = self.codebook.weight  # (K, T*d_model)
            # distances: (B, K)
            distances = torch.sum(z_e**2, dim=1, keepdim=True) + torch.sum(codebook**2, dim=1) - 2 * torch.matmul(z_e, codebook.t())
            distances = self.apply_stochastic_mask(distances, training=self.training)
            entropy_loss = self.compute_soft_entropy_loss(distances)
            encoding_indices = torch.argmin(distances, dim=1)  # (B,)
            self.update_usage_tracking(encoding_indices)

            if self.codebook_reset_counter_multiplier > 0:
                self.codebook_counters -= B
                self.codebook_counters[encoding_indices] = self.codebook_reset_counter_multiplier * self.codebook_size
                collapsed = self.codebook_counters <= 0
                if collapsed.any():
                    num_collapsed = int(collapsed.sum().item())
                    collapsed_indices = torch.where(collapsed)[0]
                    if num_collapsed > B:
                        new_batch_vectors = z_e[torch.randperm(B)[:B]]
                        overall_mean = z_e.mean(dim=1).mean()
                        overall_std = z_e.std(dim=1).mean()
                        rem = num_collapsed - B
                        random_vectors = torch.normal(mean=overall_mean.item(), std=overall_std.item(), size=(rem, z_e.shape[1]), device=z_e.device)
                        sorted_indices = collapsed_indices[torch.argsort(collapsed_indices)]
                        self.codebook.weight.data[sorted_indices[:B]] = new_batch_vectors.to(self.codebook.weight.dtype)
                        self.codebook.weight.data[sorted_indices[B:]] = random_vectors.to(self.codebook.weight.dtype)
                    else:
                        new_vectors = z_e[torch.randperm(B)[:num_collapsed]]
                        self.codebook.weight.data[collapsed] = new_vectors.to(self.codebook.weight.dtype)
                    self.codebook_counters[collapsed] = self.codebook_reset_counter_multiplier * self.codebook_size

            unique_count = torch.unique(encoding_indices).numel()
            z_q = self.codebook(encoding_indices)  # (B, D)

            codebook_loss = F.mse_loss(z_q.detach(), z_e)
            if beta is None:
                commitment_loss = self.beta * F.mse_loss(z_q, z_e.detach())
            else:
                commitment_loss = beta * F.mse_loss(z_q, z_e.detach())
            z_q = z_e + (z_q - z_e).detach()
        else:
            z_q = z_e
            codebook_loss = torch.tensor(0.0, device=z_e.device)
            commitment_loss = torch.tensor(0.0, device=z_e.device)
            entropy_loss = torch.tensor(0.0, device=z_e.device)

        logits = self.decoder(z_q, padding_mask=padding_mask)  # (B, T, V)

        # Cross-entropy reconstruction loss over tokens
        B, T, V = logits.shape
        x_flat = x.view(-1)
        logits_flat = logits.view(B * T, V)
        if padding_mask is not None:
            mask_flat = padding_mask.view(-1).to(logits_flat.dtype)
            ce = F.cross_entropy(logits_flat, x_flat, reduction='none')
            recon_loss = (ce * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        else:
            recon_loss = F.cross_entropy(logits_flat, x_flat)

        cosine_push_loss = self.compute_cosine_push_loss()
        if self.codebook is None:
            entropy_loss = torch.tensor(0.0, device=z_e.device)

        total_loss = recon_loss + codebook_loss + commitment_loss + \
                     self.cosine_push_weight * cosine_push_loss + \
                     self.entropy_loss_weight * entropy_loss

        return logits, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count, cosine_push_loss, entropy_loss

# ==============================
# Encoder3v2 / Decoder3v2 / VQVAE3v2
# ==============================

class Encoder3v2(nn.Module):
    """
    Encoder3 variant that outputs a flattened latent of size T*d2 (no final projection).
    Input:  x (B, L, T, d)
    Output: z (B, T*d2)
    """
    def __init__(self, L, d, d2, T, num_layers_layerwise_stage=1, num_layers_aggregate_stage=3,
                 config_layerwise_stage=None, config_aggregate_stage=None):
        super().__init__()
        self.L = L
        self.d = d
        self.d2 = d2
        self.T = T

        # Stage 1 (per-slice transformers)
        config_layerwise_stage = dict(config_layerwise_stage or {})
        config_layerwise_stage['n_embd'] = d
        config_layerwise_stage['mlp_hidden_dim'] = 4 * d
        self.layerwise_stage_transformers = nn.ModuleList([
            CustomSequential(*[TransformerBlock(config_layerwise_stage) for _ in range(num_layers_layerwise_stage)])
            for _ in range(L)
        ])

        # Projection from concatenated (L*d) to d2 per token
        self.proj = nn.Linear(L * d, d2)

        # Stage 2 (aggregate transformers)
        config_aggregate_stage = dict(config_aggregate_stage or {})
        config_aggregate_stage['n_embd'] = d2
        config_aggregate_stage['mlp_hidden_dim'] = 4 * d2
        self.aggregate_stage_blocks = nn.ModuleList([
            TransformerBlock(config_aggregate_stage) for _ in range(num_layers_aggregate_stage)
        ])

    def forward(self, x, padding_mask=None):
        B, L, T, d = x.size()
        assert L == self.L, f"Expected L={self.L} but got {L}"

        # Stage 1: process each slice independently (B, T, d)
        layerwise_stage_outputs = []
        for l in range(self.L):
            out_l = self.layerwise_stage_transformers[l](x[:, l, :, :], padding_mask=padding_mask)
            layerwise_stage_outputs.append(out_l)
        x_layerwise_stage = torch.stack(layerwise_stage_outputs, dim=1)  # (B, L, T, d)

        # Concatenate L and d, then project to d2 per token
        x_cat = x_layerwise_stage.permute(0, 2, 1, 3).contiguous().view(B, T, L * d)  # (B, T, L*d)
        x_proj = self.proj(x_cat)  # (B, T, d2)

        # Stage 2: aggregate blocks over tokens
        x_out = x_proj
        for block in self.aggregate_stage_blocks:
            x_out = block(x_out, padding_mask=padding_mask)  # (B, T, d2)

        # Flatten without final projection
        x_out = x_out.view(B, self.T * self.d2)  # (B, T*d2)
        return x_out


class Decoder3v2(nn.Module):
    """
    Decoder3 variant that consumes a flattened latent of size T*d2.
    Input:  z (B, T*d2)
    Output: x_recon (B, L, T, d)
    """
    def __init__(self, L, d, d2, T, num_layers_aggregate_stage=3, num_layers_layerwise_stage=1,
                 config_aggregate_stage=None, config_layerwise_stage=None, tied_encoder_proj=None):
        super().__init__()
        self.L = L
        self.d = d
        self.d2 = d2
        self.T = T

        # Aggregate stage (operates on (B, T, d2))
        config_aggregate_stage = dict(config_aggregate_stage or {})
        config_aggregate_stage['n_embd'] = d2
        config_aggregate_stage['mlp_hidden_dim'] = 4 * d2
        self.aggregate_stage_blocks = nn.ModuleList([
            TransformerBlock(config_aggregate_stage) for _ in range(num_layers_aggregate_stage)
        ])

        # Per-token projection from d2 -> L*d, optionally tied to encoder proj
        self.tied = False
        if tied_encoder_proj is None:
            self.proj = nn.Linear(d2, L * d)
        else:
            self.proj = tied_encoder_proj
            self.projbias = nn.Parameter(torch.zeros(L * d))
            self.tied = True

        # Layerwise stage (L independent transformers on (B, T, d))
        config_layerwise_stage = dict(config_layerwise_stage or {})
        config_layerwise_stage['n_embd'] = d
        config_layerwise_stage['mlp_hidden_dim'] = 4 * d
        self.layerwise_stage_transformers = nn.ModuleList([
            CustomSequential(*[TransformerBlock(config_layerwise_stage) for _ in range(num_layers_layerwise_stage)])
            for _ in range(L)
        ])

    def forward(self, x, padding_mask=None):
        # x is (B, T*d2)
        B, D_in = x.size()
        assert D_in == self.T * self.d2, f"Expected latent dim {self.T * self.d2} but got {D_in}"

        # Reshape to sequence (B, T, d2)
        x = x.view(B, self.T, self.d2)

        # Aggregate stage
        for block in self.aggregate_stage_blocks:
            x = block(x, padding_mask=padding_mask)  # (B, T, d2)

        # Project per token to (L*d)
        if self.tied:
            x = F.linear(x, self.proj.weight.t(), bias=self.projbias)
        else:
            x = self.proj(x)  # (B, T, L*d)

        # Reshape to (B, L, T, d)
        x = x.view(B, self.T, self.L, self.d).permute(0, 2, 1, 3).contiguous()

        # Layerwise stage per slice
        layerwise_outputs = []
        for l in range(self.L):
            out_l = self.layerwise_stage_transformers[l](x[:, l, :, :], padding_mask=padding_mask)
            layerwise_outputs.append(out_l)
        x_out = torch.stack(layerwise_outputs, dim=1)  # (B, L, T, d)
        return x_out


class VQVAE3v2(nn.Module):
    """
    VQVAE using Encoder3v2/Decoder3v2 where the latent is a flattened (T*d2) vector.
    - Codebook vectors have dimension T*d2
    - Quantization is per-example (one code per batch item)
    """
    def __init__(self, encoder, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codebook_size = config["codebook_size"]
        self.codebook_reset_counter_multiplier = config.get("codebook_reset_counter_multiplier", 0)
        self.beta = config["beta"]

        # Normalization buffer (L values)
        self.register_buffer('normalization_values', None)

        # Latent dimension is T*d2
        self.latent_dim = self.encoder.T * self.encoder.d2

        if self.codebook_size > 0:
            self.codebook = nn.Embedding(self.codebook_size, self.latent_dim)
            nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
        else:
            self.codebook = None

        if self.codebook_reset_counter_multiplier > 0:
            self.register_buffer(
                "codebook_counters",
                torch.full((self.codebook_size,), 0)
            )

        # Experimental features
        self.cosine_push_weight = config.get('cosine_push_weight', 0.0)
        self.entropy_loss_weight = config.get('entropy_loss_weight', 0.0)
        self.mask_prob = config.get('mask_prob', 0.0)
        self.entropy_temperature = config.get('entropy_temperature', 1.0)

        # Usage tracking
        self.usage_tracking_window = config.get('usage_tracking_window', 5000)
        if self.usage_tracking_window > 0:
            self.register_buffer('usage_history', torch.zeros(self.usage_tracking_window, dtype=torch.long))
            self.register_buffer('usage_ptr', torch.tensor(0, dtype=torch.long))
            self.register_buffer('usage_full', torch.tensor(False, dtype=torch.bool))
            self.register_buffer('total_vectors_processed', torch.tensor(0, dtype=torch.long))

    def normalize_codebook_vectors(self):
        if self.cosine_push_weight > 0 and self.codebook is not None:
            with torch.no_grad():
                self.codebook.weight.data = F.normalize(self.codebook.weight.data, p=2, dim=1)

    def compute_cosine_push_loss(self):
        if self.cosine_push_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        if self.usage_tracking_window > 0:
            usage_stats = self.get_usage_statistics()
            usage_counts = usage_stats['usage_counts']
            used_mask = usage_counts > 0
            used_indices = torch.where(used_mask)[0]
            if len(used_indices) <= 1:
                return torch.tensor(0.0, device=next(self.parameters()).device)
            E_used = self.codebook.weight[used_indices]
            E_used_normalized = F.normalize(E_used, p=2, dim=1)
            cosine_sim_matrix = torch.mm(E_used_normalized, E_used_normalized.t())
            usage_weights = usage_counts[used_indices].float()
            usage_weights = usage_weights / usage_weights.sum()
            weight_matrix = usage_weights.unsqueeze(1) * usage_weights.unsqueeze(0)
            num_used = len(used_indices)
            mask = ~torch.eye(num_used, device=E_used.device, dtype=torch.bool)
            weighted_cosines_squared = (cosine_sim_matrix ** 2) * weight_matrix
            return weighted_cosines_squared[mask].sum()
        else:
            E = self.codebook.weight
            E_normalized = F.normalize(E, p=2, dim=1)
            cosine_sim_matrix = torch.mm(E_normalized, E_normalized.t())
            mask = ~torch.eye(self.codebook_size, device=E.device, dtype=torch.bool)
            off_diagonal_cosines = cosine_sim_matrix[mask]
            return torch.sum(off_diagonal_cosines ** 2)

    def compute_soft_entropy_loss(self, distances):
        if self.entropy_loss_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=distances.device)
        assignment_probs = F.softmax(-distances / self.entropy_temperature, dim=-1)  # (B, K)
        avg_probs = assignment_probs.mean(dim=0)
        epsilon = 1e-10
        avg_probs_safe = torch.clamp(avg_probs, min=epsilon)
        entropy = -torch.sum(avg_probs_safe * torch.log(avg_probs_safe))
        max_entropy = torch.log(torch.tensor(self.codebook_size, device=distances.device, dtype=avg_probs.dtype))
        return max_entropy - entropy

    def update_usage_tracking(self, encoding_indices):
        if self.usage_tracking_window <= 0:
            return
        valid_indices = encoding_indices
        self.total_vectors_processed += valid_indices.size(0)
        n_valid = valid_indices.size(0)
        if n_valid > 0:
            if n_valid >= self.usage_tracking_window:
                self.usage_history[:] = valid_indices[-self.usage_tracking_window:]
                self.usage_ptr.fill_(0)
                self.usage_full.fill_(True)
            else:
                end_ptr = (self.usage_ptr + n_valid) % self.usage_tracking_window
                if end_ptr > self.usage_ptr:
                    self.usage_history[self.usage_ptr:end_ptr] = valid_indices
                else:
                    n_until_end = self.usage_tracking_window - self.usage_ptr
                    self.usage_history[self.usage_ptr:] = valid_indices[:n_until_end]
                    if n_valid > n_until_end:
                        self.usage_history[:end_ptr] = valid_indices[n_until_end:]
                self.usage_ptr.copy_(end_ptr)
                if self.usage_ptr == 0 or self.usage_full:
                    self.usage_full.fill_(True)

    def get_usage_statistics(self):
        if self.usage_tracking_window <= 0 or self.codebook is None:
            return {
                'total_vectors_processed': 0,
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size if self.codebook is not None else 0),
                'usage_percentages': torch.zeros(self.codebook_size if self.codebook is not None else 0)
            }
        if self.usage_full:
            valid_history = self.usage_history
        else:
            valid_history = self.usage_history[:self.usage_ptr]
        if valid_history.numel() == 0:
            return {
                'total_vectors_processed': self.total_vectors_processed.item(),
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size),
                'usage_percentages': torch.zeros(self.codebook_size)
            }
        usage_counts = torch.bincount(valid_history, minlength=self.codebook_size)
        unique_vectors_used = (usage_counts > 0).sum().item()
        usage_percentages = usage_counts.float() / valid_history.numel()
        return {
            'total_vectors_processed': self.total_vectors_processed.item(),
            'unique_vectors_used': unique_vectors_used,
            'usage_counts': usage_counts,
            'usage_percentages': usage_percentages
        }

    def compute_codebook_similarities(self):
        if self.codebook is None:
            return {'similarities': None, 'used_indices': None, 'num_used_vectors': 0}
        usage_stats = self.get_usage_statistics()
        used_mask = usage_stats['usage_counts'] > 0
        used_indices = torch.where(used_mask)[0]
        if len(used_indices) <= 1:
            return {'similarities': None, 'used_indices': used_indices, 'num_used_vectors': len(used_indices)}
        with torch.no_grad():
            used_codebook = self.codebook.weight[used_indices]
            used_codebook_norm = used_codebook / used_codebook.norm(dim=1, keepdim=True)
            similarities = torch.mm(used_codebook_norm, used_codebook_norm.t())
        return {'similarities': similarities, 'used_indices': used_indices, 'num_used_vectors': len(used_indices)}

    def apply_stochastic_mask(self, distances, training=True):
        if self.mask_prob <= 0 or not training:
            return distances
        if torch.rand(1).item() < self.mask_prob:
            B, K = distances.shape
            num_mask = int(self.mask_prob * B * K)
            mask_indices = torch.randperm(B * K, device=distances.device)[:num_mask]
            mask_rows = mask_indices // K
            mask_cols = mask_indices % K
            masked_distances = distances.clone()
            masked_distances[mask_rows, mask_cols] = 1e10
            return masked_distances
        return distances

    def normalize(self, x):
        if self.normalization_values is not None:
            norm_values_expanded = self.normalization_values.view(1, -1, 1, 1)
            return x / (norm_values_expanded + 1e-8)
        else:
            return x

    def denormalize(self, x):
        if self.normalization_values is not None:
            norm_values_expanded = self.normalization_values.view(1, -1, 1, 1)
            return x * (norm_values_expanded + 1e-8)
        else:
            return x

    def _compute_normalization_values(self, x, padding_mask=None):
        B, L, T, d = x.shape
        norms = torch.norm(x, dim=-1)  # (B, L, T)
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).expand(B, L, T).float()
            masked_norms = norms * mask
            sum_norms = masked_norms.sum(dim=(0, 2))
            count_valid = mask.sum(dim=(0, 2))
            norm_values = sum_norms / (count_valid + 1e-8)
        else:
            norm_values = norms.mean(dim=(0, 2))
        return norm_values

    def forward(self, x, padding_mask=None, beta=None):
        # Compute/Cache normalization values
        if self.normalization_values is None:
            norm_values = self._compute_normalization_values(x, padding_mask)
            self.register_buffer('normalization_values', norm_values)
        x = self.normalize(x)

        # Encode flattened latent (B, T*d2)
        z_e = self.encoder(x, padding_mask=padding_mask)
        unique_count = torch.tensor(0, device=z_e.device)

        if self.codebook is not None:
            B = z_e.size(0)
            codebook = self.codebook.weight  # (K, D)
            distances = torch.sum(z_e**2, dim=1, keepdim=True) + torch.sum(codebook**2, dim=1) - 2 * torch.matmul(z_e, codebook.t())
            distances = self.apply_stochastic_mask(distances, training=self.training)
            entropy_loss = self.compute_soft_entropy_loss(distances)
            encoding_indices = torch.argmin(distances, dim=1)  # (B,)
            self.update_usage_tracking(encoding_indices)

            if self.codebook_reset_counter_multiplier > 0:
                self.codebook_counters -= B
                self.codebook_counters[encoding_indices] = self.codebook_reset_counter_multiplier * self.codebook_size
                collapsed = self.codebook_counters <= 0
                if collapsed.any():
                    num_collapsed = int(collapsed.sum().item())
                    collapsed_indices = torch.where(collapsed)[0]
                    if num_collapsed > B:
                        new_batch_vectors = z_e[torch.randperm(B)[:B]]
                        overall_mean = z_e.mean(dim=1).mean()
                        overall_std = z_e.std(dim=1).mean()
                        rem = num_collapsed - B
                        random_vectors = torch.normal(mean=overall_mean.item(), std=overall_std.item(), size=(rem, z_e.shape[1]), device=z_e.device)
                        sorted_indices = collapsed_indices[torch.argsort(collapsed_indices)]
                        self.codebook.weight.data[sorted_indices[:B]] = new_batch_vectors.to(self.codebook.weight.dtype)
                        self.codebook.weight.data[sorted_indices[B:]] = random_vectors.to(self.codebook.weight.dtype)
                    else:
                        new_vectors = z_e[torch.randperm(B)[:num_collapsed]]
                        self.codebook.weight.data[collapsed] = new_vectors.to(self.codebook.weight.dtype)
                    self.codebook_counters[collapsed] = self.codebook_reset_counter_multiplier * self.codebook_size

            unique_count = torch.unique(encoding_indices).numel()
            z_q = self.codebook(encoding_indices)  # (B, T*d2)

            codebook_loss = F.mse_loss(z_q.detach(), z_e)
            if beta is None:
                commitment_loss = self.beta * F.mse_loss(z_q, z_e.detach())
            else:
                commitment_loss = beta * F.mse_loss(z_q, z_e.detach())
            z_q = z_e + (z_q - z_e).detach()
        else:
            z_q = z_e
            codebook_loss = torch.tensor(0.0, device=z_e.device)
            commitment_loss = torch.tensor(0.0, device=z_e.device)
            entropy_loss = torch.tensor(0.0, device=z_e.device)

        # Decode expects (B, T*d2)
        x_recon = self.decoder(z_q, padding_mask=padding_mask)

        # Reconstruction loss (MSE), masked if padding provided
        if padding_mask is not None:
            active_mask = padding_mask.unsqueeze(-1).unsqueeze(1).float()
            recon_loss = F.mse_loss(x_recon * active_mask, x * active_mask, reduction='sum') / (active_mask.sum() + 1e-8)
        else:
            recon_loss = F.mse_loss(x_recon, x)

        cosine_push_loss = self.compute_cosine_push_loss()
        if self.codebook is None:
            entropy_loss = torch.tensor(0.0, device=z_e.device)

        total_loss = recon_loss + codebook_loss + commitment_loss + \
                     self.cosine_push_weight * cosine_push_loss + \
                     self.entropy_loss_weight * entropy_loss

        return x_recon, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count, cosine_push_loss, entropy_loss



#### VQVAE_layer ##########################################################################################


class Encoder_layer(nn.Module):
    """
    Encoder for variable-length sequences using learnable CLS tokens.
    
    Takes sequences of shape (B, T, d) where T varies from 16 to 384,
    appends T* learnable CLS vectors, processes through transformer blocks,
    and outputs the last T* positions flattened to (B, T*×d).
    """
    def __init__(self, d, T_star, num_layers=3, config=None):
        """
        Args:
            d (int): Feature dimension
            T_star (int): Number of CLS tokens to append
            num_layers (int): Number of transformer blocks
            config (dict): Configuration for transformer blocks
        """
        super().__init__()
        self.d = d
        self.T_star = T_star
        
        # Learnable CLS tokens - shared across batch
        self.cls_tokens = nn.Parameter(torch.randn(T_star, d))
        
        # Update config for transformer blocks
        if config is None:
            config = {}
        config = config.copy()
        config['n_embd'] = d
        config['mlp_hidden_dim'] = config.get('mlp_hidden_dim', 4 * d)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(num_layers)
        ])
        
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: Input tensor of shape (B, T, d) where T can vary
            padding_mask: Optional padding mask of shape (B, T)
            
        Returns:
            z_e: Encoded representation of shape (B, T_star × d)
        """
        B, T, d = x.shape
        assert d == self.d, f"Expected feature dimension {self.d}, got {d}"
        
        # Expand CLS tokens for batch and append to input
        cls_tokens_expanded = self.cls_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, T_star, d)
        x_with_cls = torch.cat([x, cls_tokens_expanded], dim=1)  # (B, T + T_star, d)
        
        # Extend padding mask if provided to include CLS tokens (which are always valid)
        if padding_mask is not None:
            # CLS tokens are always valid (1s)
            cls_mask = torch.ones(B, self.T_star, device=padding_mask.device, dtype=padding_mask.dtype)
            extended_mask = torch.cat([padding_mask, cls_mask], dim=1)  # (B, T + T_star)
        else:
            extended_mask = None
        
        # Process through transformer blocks
        h = x_with_cls
        for block in self.transformer_blocks:
            h = block(h, padding_mask=extended_mask)  # (B, T + T_star, d)
        
        # Extract last T_star positions (the CLS positions)
        cls_output = h[:, -self.T_star:, :]  # (B, T_star, d)
        
        # Flatten to create the latent representation
        z_e = cls_output.reshape(B, self.T_star * self.d)  # (B, T_star × d)
        
        return z_e


class Decoder_layer(nn.Module):
    """
    Decoder for variable-length sequences.
    
    Takes latent representation (B, T_star×d), projects to max length,
    extracts the needed length T, and processes through transformer blocks.
    """
    def __init__(self, d, T_star, T_max, num_layers=3, config=None):
        """
        Args:
            d (int): Feature dimension
            T_star (int): Number of CLS tokens (latent is T_star × d)
            T_max (int): Maximum sequence length (e.g., 384)
            num_layers (int): Number of transformer blocks
            config (dict): Configuration for transformer blocks
        """
        super().__init__()
        self.d = d
        self.T_star = T_star
        self.T_max = T_max
        
        # Projection from T_star×d to T_max×d
        #latent_dim = T_max * d // 16 # int(math.sqrt(T_star * d * T_max * d))
        #self.proj_sequence = nn.Sequential(
        #    nn.Linear(T_star * d, latent_dim),
        #    nn.ReLU(),
        #    nn.Linear(latent_dim, T_max * d)
        #)
        self.proj_sequence = nn.Linear(T_star * d, T_max * d)
        
        # Update config for transformer blocks
        if config is None:
            config = {}
        config = config.copy()
        config['n_embd'] = d
        config['mlp_hidden_dim'] = config.get('mlp_hidden_dim', 4 * d)
        config['is_decoder'] = False  # Non-causal attention like VQVAE2
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(num_layers)
        ])
        
    def forward(self, z, T, padding_mask=None):
        """
        Args:
            z: Latent tensor of shape (B, T_star × d)
            T: Target sequence length for this batch
            padding_mask: Optional padding mask of shape (B, T)
            
        Returns:
            x_recon: Reconstructed sequence of shape (B, T, d)
        """
        B = z.shape[0]
        assert z.shape[1] == self.T_star * self.d, f"Expected latent dimension {self.T_star * self.d}, got {z.shape[1]}"
        assert T <= self.T_max, f"Sequence length {T} exceeds maximum {self.T_max}"
        
        # Project to maximum sequence length
        sequence = self.proj_sequence(z)  # (B, T_max × d)
        sequence = sequence.view(B, self.T_max, self.d)  # (B, T_max, d)
        
        # Extract first T positions
        x = sequence[:, :T, :]  # (B, T, d)
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, padding_mask=padding_mask)  # (B, T, d)
            
        return x


class VQVAE_layer(nn.Module):
    """
    VQVAE for variable-length sequences using CLS tokens.
    
    Maps sequences of any length (16-384) to fixed-size latent codes
    via learnable CLS tokens that aggregate sequence information.
    """
    def __init__(self, encoder, decoder, config):
        """
        Args:
            encoder: Instance of Encoder_layer
            decoder: Instance of Decoder_layer
            config: Configuration dict with VQVAE parameters
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        # VQVAE parameters from config
        self.codebook_size = config["codebook_size"]
        self.beta = config["beta"]
        self.codebook_reset_counter_multiplier = config.get("codebook_reset_counter_multiplier", 0)
        
        # Latent dimension is T_star × d
        self.latent_dim = encoder.T_star * encoder.d
        
        # Codebook
        if self.codebook_size > 0:
            self.codebook = nn.Embedding(self.codebook_size, self.latent_dim)
            nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
        else:
            self.codebook = None
            
        # Codebook reset counters
        if self.codebook_reset_counter_multiplier > 0 and self.codebook_size > 0:
            self.register_buffer(
                "codebook_counters",
                torch.full((self.codebook_size,), 0)
            )
            
        # Experimental features from VQVAE2
        self.cosine_push_weight = config.get('cosine_push_weight', 0.0)
        self.entropy_loss_weight = config.get('entropy_loss_weight', 0.0)
        self.mask_prob = config.get('mask_prob', 0.0)
        self.entropy_temperature = config.get('entropy_temperature', 1.0)
        
        # Usage tracking
        self.usage_tracking_window = config.get('usage_tracking_window', 5000)
        if self.usage_tracking_window > 0 and self.codebook_size > 0:
            self.register_buffer('usage_history', torch.zeros(self.usage_tracking_window, dtype=torch.long))
            self.register_buffer('usage_ptr', torch.tensor(0, dtype=torch.long))
            self.register_buffer('usage_full', torch.tensor(False, dtype=torch.bool))
            self.register_buffer('total_vectors_processed', torch.tensor(0, dtype=torch.long))
    
    def normalize_codebook_vectors(self):
        """Normalize codebook vectors to unit length for cosine-push regularization."""
        if self.cosine_push_weight > 0 and self.codebook is not None:
            with torch.no_grad():
                self.codebook.weight.data = F.normalize(self.codebook.weight.data, p=2, dim=1)
    
    def compute_cosine_push_loss(self):
        """Compute cosine-push regularization loss (same as VQVAE2)."""
        if self.cosine_push_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        if self.usage_tracking_window > 0:
            usage_stats = self.get_usage_statistics()
            usage_counts = usage_stats['usage_counts']
            used_mask = usage_counts > 0
            used_indices = torch.where(used_mask)[0]
            
            if len(used_indices) <= 1:
                return torch.tensor(0.0, device=next(self.parameters()).device)
            
            E_used = self.codebook.weight[used_indices]
            E_used_normalized = F.normalize(E_used, p=2, dim=1)
            cosine_sim_matrix = torch.mm(E_used_normalized, E_used_normalized.t())
            
            usage_weights = usage_counts[used_indices].float()
            usage_weights = usage_weights / usage_weights.sum()
            weight_matrix = usage_weights.unsqueeze(1) * usage_weights.unsqueeze(0)
            
            num_used = len(used_indices)
            mask = ~torch.eye(num_used, device=E_used.device, dtype=torch.bool)
            weighted_cosines_squared = (cosine_sim_matrix ** 2) * weight_matrix
            cosine_push_loss = weighted_cosines_squared[mask].sum()
        else:
            E = self.codebook.weight
            E_normalized = F.normalize(E, p=2, dim=1)
            cosine_sim_matrix = torch.mm(E_normalized, E_normalized.t())
            mask = ~torch.eye(self.codebook_size, device=E.device, dtype=torch.bool)
            off_diagonal_cosines = cosine_sim_matrix[mask]
            cosine_push_loss = torch.sum(off_diagonal_cosines ** 2)
        
        return cosine_push_loss
    
    def compute_soft_entropy_loss(self, distances):
        """Compute entropy loss using soft assignments (same as VQVAE2)."""
        if self.entropy_loss_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=distances.device)
        
        assignment_probs = F.softmax(-distances / self.entropy_temperature, dim=-1)
        avg_probs = assignment_probs.mean(dim=0)
        
        epsilon = 1e-10
        avg_probs_safe = torch.clamp(avg_probs, min=epsilon)
        entropy = -torch.sum(avg_probs_safe * torch.log(avg_probs_safe))
        
        max_entropy = torch.log(torch.tensor(self.codebook_size, device=distances.device, dtype=avg_probs.dtype))
        entropy_loss = max_entropy - entropy
        
        return entropy_loss
    
    def update_usage_tracking(self, encoding_indices):
        """Update usage statistics (same as VQVAE2)."""
        if self.usage_tracking_window <= 0:
            return
            
        valid_indices = encoding_indices
        self.total_vectors_processed += valid_indices.size(0)
        
        n_valid = valid_indices.size(0)
        if n_valid > 0:
            if n_valid >= self.usage_tracking_window:
                self.usage_history[:] = valid_indices[-self.usage_tracking_window:]
                self.usage_ptr.fill_(0)
                self.usage_full.fill_(True)
            else:
                end_ptr = (self.usage_ptr + n_valid) % self.usage_tracking_window
                
                if end_ptr > self.usage_ptr:
                    self.usage_history[self.usage_ptr:end_ptr] = valid_indices
                else:
                    n_until_end = self.usage_tracking_window - self.usage_ptr
                    self.usage_history[self.usage_ptr:] = valid_indices[:n_until_end]
                    if n_valid > n_until_end:
                        self.usage_history[:end_ptr] = valid_indices[n_until_end:]
                
                self.usage_ptr.copy_(end_ptr)
                if self.usage_ptr == 0 or self.usage_full:
                    self.usage_full.fill_(True)
    
    def get_usage_statistics(self):
        """Get usage statistics (same as VQVAE2)."""
        if self.usage_tracking_window <= 0 or self.codebook is None:
            return {
                'total_vectors_processed': 0,
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size if self.codebook is not None else 0),
                'usage_percentages': torch.zeros(self.codebook_size if self.codebook is not None else 0)
            }
        
        if self.usage_full:
            valid_history = self.usage_history
        else:
            valid_history = self.usage_history[:self.usage_ptr]
            
        if valid_history.numel() == 0:
            return {
                'total_vectors_processed': self.total_vectors_processed.item(),
                'unique_vectors_used': 0,
                'usage_counts': torch.zeros(self.codebook_size),
                'usage_percentages': torch.zeros(self.codebook_size)
            }
        
        usage_counts = torch.bincount(valid_history, minlength=self.codebook_size)
        unique_vectors_used = (usage_counts > 0).sum().item()
        usage_percentages = usage_counts.float() / valid_history.numel()
        
        return {
            'total_vectors_processed': self.total_vectors_processed.item(),
            'unique_vectors_used': unique_vectors_used,
            'usage_counts': usage_counts,
            'usage_percentages': usage_percentages
        }
    
    def compute_codebook_similarities(self):
        """Compute cosine similarities between used codebook vectors."""
        if self.codebook is None:
            return {
                'similarities': None,
                'used_indices': None,
                'num_used_vectors': 0
            }
        
        usage_stats = self.get_usage_statistics()
        used_mask = usage_stats['usage_counts'] > 0
        used_indices = torch.where(used_mask)[0]
        
        if len(used_indices) <= 1:
            return {
                'similarities': None,
                'used_indices': used_indices,
                'num_used_vectors': len(used_indices)
            }
        
        with torch.no_grad():
            used_codebook = self.codebook.weight[used_indices]
            used_codebook_norm = used_codebook / used_codebook.norm(dim=1, keepdim=True)
            similarities = torch.mm(used_codebook_norm, used_codebook_norm.t())
        
        return {
            'similarities': similarities,
            'used_indices': used_indices,
            'num_used_vectors': len(used_indices)
        }
    
    def apply_stochastic_mask(self, distances, training=True):
        """Apply stochastic masking for exploration (same as VQVAE2)."""
        if self.mask_prob <= 0 or not training:
            return distances
            
        if torch.rand(1).item() < self.mask_prob:
            B, K = distances.shape
            num_mask = int(self.mask_prob * B * K)
            mask_indices = torch.randperm(B * K, device=distances.device)[:num_mask]
            
            mask_rows = mask_indices // K
            mask_cols = mask_indices % K
            
            masked_distances = distances.clone()
            masked_distances[mask_rows, mask_cols] = 1e10
            
            return masked_distances
        
        return distances
    
    def forward(self, x, padding_mask=None, beta=None):
        """
        Forward pass of VQVAE_layer.
        
        Args:
            x: Input tensor of shape (B, T, d) where T can vary
            padding_mask: Optional padding mask of shape (B, T)
            beta: Optional commitment loss coefficient override
            
        Returns:
            x_recon: Reconstructed sequence (B, T, d)
            total_loss: Total loss
            recon_loss: Reconstruction loss
            codebook_loss: Codebook loss
            commitment_loss: Commitment loss
            unique_count: Number of unique codes used
            cosine_push_loss: Cosine push regularization loss
            entropy_loss: Entropy loss
        """
        B, T, d = x.shape
        
        # Encode to latent representation
        z_e = self.encoder(x, padding_mask=padding_mask)  # (B, latent_dim)
        
        if self.codebook is not None:
            # Vector quantization
            codebook = self.codebook.weight  # (codebook_size, latent_dim)
            
            # Compute distances
            distances = torch.sum(z_e**2, dim=1, keepdim=True) + \
                       torch.sum(codebook**2, dim=1) - \
                       2 * torch.matmul(z_e, codebook.t())  # (B, codebook_size)
            
            # Apply stochastic masking if enabled
            distances = self.apply_stochastic_mask(distances, training=self.training)
            
            # Compute soft entropy loss before argmin
            entropy_loss = self.compute_soft_entropy_loss(distances)
            
            # Find nearest codebook entries
            encoding_indices = torch.argmin(distances, dim=1)  # (B,)
            
            # Update usage tracking
            self.update_usage_tracking(encoding_indices)
            
            unique_count = torch.unique(encoding_indices).numel()
            
            # Handle codebook reset if enabled
            if self.codebook_reset_counter_multiplier > 0:
                self.codebook_counters -= B
                self.codebook_counters[encoding_indices] = self.codebook_reset_counter_multiplier * self.codebook_size
                collapsed = self.codebook_counters <= 0
                
                if collapsed.any():
                    num_collapsed = int(collapsed.sum().item())
                    collapsed_indices = torch.where(collapsed)[0]
                    
                    if num_collapsed > B:
                        new_batch_vectors = z_e[torch.randperm(B)[:B]]
                        overall_mean = z_e.mean(dim=1).mean()
                        overall_std = z_e.std(dim=1).mean()
                        rem = num_collapsed - B
                        random_vectors = torch.normal(
                            mean=overall_mean.item(), 
                            std=overall_std.item(), 
                            size=(rem, z_e.shape[1]), 
                            device=z_e.device
                        )
                        sorted_indices = collapsed_indices[torch.argsort(collapsed_indices)]
                        self.codebook.weight.data[sorted_indices[:B]] = new_batch_vectors.to(self.codebook.weight.dtype)
                        self.codebook.weight.data[sorted_indices[B:]] = random_vectors.to(self.codebook.weight.dtype)
                    else:
                        new_vectors = z_e[torch.randperm(B)[:num_collapsed]]
                        self.codebook.weight.data[collapsed] = new_vectors.to(self.codebook.weight.dtype)
                    
                    self.codebook_counters[collapsed] = self.codebook_reset_counter_multiplier * self.codebook_size
            
            # Quantize
            z_q = self.codebook(encoding_indices)  # (B, latent_dim)
            
            # Compute VQ losses
            codebook_loss = F.mse_loss(z_q.detach(), z_e)
            commitment_loss = (self.beta if beta is None else beta) * F.mse_loss(z_q, z_e.detach())
            
            # Straight-through estimator
            z_q = z_e + (z_q - z_e).detach()
        else:
            # No codebook mode
            z_q = z_e
            codebook_loss = torch.tensor(0.0, device=z_e.device)
            commitment_loss = torch.tensor(0.0, device=z_e.device)
            entropy_loss = torch.tensor(0.0, device=z_e.device)
            unique_count = 0
        
        # Decode
        x_recon = self.decoder(z_q, T=T, padding_mask=padding_mask)  # (B, T, d)
        
        # Reconstruction loss
        if padding_mask is not None:
            # Apply mask to compute loss only on valid positions
            active_mask = padding_mask.unsqueeze(-1).float()  # (B, T, 1)
            recon_loss = F.mse_loss(x_recon * active_mask, x * active_mask, reduction='sum') / active_mask.sum()
        else:
            recon_loss = F.mse_loss(x_recon, x)
        
        # Compute cosine-push regularization
        cosine_push_loss = self.compute_cosine_push_loss()
        
        # Total loss
        total_loss = recon_loss + codebook_loss + commitment_loss + \
                    self.cosine_push_weight * cosine_push_loss + \
                    self.entropy_loss_weight * entropy_loss
        
        return x_recon, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count, cosine_push_loss, entropy_loss

class Encoder61(nn.Module):
    """
    Two-stage encoder: layerwise processing of L transformer layers, followed by a CLS-based aggregate stage.
    """
    def __init__(
        self,
        L,
        d,
        d_aggregate,
        T_star,
        num_layers_layerwise_stage=1,
        num_layers_aggregate_stage=3,
        config_layerwise_stage=None,
        config_aggregate_stage=None,
    ):
        super().__init__()
        self.L = L
        self.d = d
        self.d_aggregate = d_aggregate
        self.T_star = T_star
        self.latent_dim = T_star * d_aggregate

        layerwise_cfg = dict(config_layerwise_stage) if config_layerwise_stage is not None else {}
        aggregate_cfg = dict(config_aggregate_stage) if config_aggregate_stage is not None else {}

        layerwise_cfg['n_embd'] = d
        layerwise_cfg['mlp_hidden_dim'] = layerwise_cfg.get('mlp_hidden_dim', 4 * d)
        self.layerwise_stage_transformers = nn.ModuleList([
            CustomSequential(*[TransformerBlock(layerwise_cfg) for _ in range(num_layers_layerwise_stage)])
            for _ in range(L)
        ])

        self.proj = nn.Linear(L * d, d_aggregate)

        aggregate_cfg['n_embd'] = d_aggregate
        aggregate_cfg['mlp_hidden_dim'] = aggregate_cfg.get('mlp_hidden_dim', 4 * d_aggregate)
        self.aggregate_stage_blocks = nn.ModuleList([
            TransformerBlock(aggregate_cfg) for _ in range(num_layers_aggregate_stage)
        ])

        self.cls_tokens = nn.Parameter(torch.randn(T_star, d_aggregate))

    def forward(self, x, padding_mask=None):
        B, L, T, d = x.shape
        assert L == self.L, f"Expected {self.L} layers, received {L}"
        assert d == self.d, f"Expected feature dim {self.d}, received {d}"

        layerwise_outputs = [
            self.layerwise_stage_transformers[l](x[:, l, :, :], padding_mask=padding_mask)
            for l in range(self.L)
        ]
        x_layerwise = torch.stack(layerwise_outputs, dim=1)                 # (B, L, T, d)

        x_cat = x_layerwise.permute(0, 2, 1, 3).contiguous().view(B, T, self.L * self.d)
        x_proj = self.proj(x_cat)                                           # (B, T, d_aggregate)

        cls_tokens = self.cls_tokens.unsqueeze(0).expand(B, -1, -1)
        x_with_cls = torch.cat([x_proj, cls_tokens], dim=1)                 # (B, T + T_star, d_aggregate)

        if padding_mask is not None:
            cls_mask = torch.ones(B, self.T_star, device=padding_mask.device, dtype=padding_mask.dtype)
            extended_mask = torch.cat([padding_mask, cls_mask], dim=1)
        else:
            extended_mask = None

        h = x_with_cls
        for block in self.aggregate_stage_blocks:
            h = block(h, padding_mask=extended_mask)

        cls_output = h[:, -self.T_star:, :]                                 # (B, T_star, d_aggregate)
        return cls_output.reshape(B, self.latent_dim)

class Decoder61(nn.Module):
    """
    Decoder symmetrical to Encoder61:
      1. Expand the latent CLS tokens into a (T + T_star) sequence and process it with
         the same style aggregate transformer blocks.
      2. Map aggregated tokens back to per-layer representations.
      3. Run layerwise transformer stacks to recover (B, L, T, d).
    """
    def __init__(
        self,
        L,
        d,
        d_aggregate,
        T_star,
        T_max,
        num_layers_aggregate_stage=3,
        num_layers_layerwise_stage=1,
        config_aggregate_stage=None,
        config_layerwise_stage=None,
        tied_encoder_proj=None,
    ):
        super().__init__()
        self.L = L
        self.d = d
        self.d_aggregate = d_aggregate
        self.T_star = T_star
        self.T_max = T_max
        self.latent_dim = T_star * d_aggregate

        aggregate_cfg = dict(config_aggregate_stage) if config_aggregate_stage is not None else {}
        aggregate_cfg['n_embd'] = d_aggregate
        aggregate_cfg['mlp_hidden_dim'] = aggregate_cfg.get('mlp_hidden_dim', 4 * d_aggregate)
        self.aggregate_stage_blocks = nn.ModuleList([
            TransformerBlock(aggregate_cfg) for _ in range(num_layers_aggregate_stage)
        ])

        self.aggregate_seed_proj = nn.Linear(self.latent_dim, T_max * d_aggregate)

        self.tied_proj = tied_encoder_proj
        if self.tied_proj is None:
            self.proj_back = nn.Linear(d_aggregate, L * d)
        else:
            if self.tied_proj:
                self.proj_back_bias = nn.Parameter(torch.zeros(L * d))
            else:
                self.proj_back = nn.Linear(d_aggregate, L * d)

        layerwise_cfg = dict(config_layerwise_stage) if config_layerwise_stage is not None else {}
        layerwise_cfg['n_embd'] = d
        layerwise_cfg['mlp_hidden_dim'] = layerwise_cfg.get('mlp_hidden_dim', 4 * d)
        self.layerwise_stage_transformers = nn.ModuleList([
            CustomSequential(*[TransformerBlock(layerwise_cfg) for _ in range(num_layers_layerwise_stage)])
            for _ in range(L)
        ])

    def forward(self, z, T, padding_mask=None):
        B, latent_dim = z.shape
        assert latent_dim == self.latent_dim, f"Expected latent dim {self.latent_dim}, received {latent_dim}"
        assert T <= self.T_max, f"Sequence length {T} exceeds maximum {self.T_max}"

        cls_state = z.view(B, self.T_star, self.d_aggregate)
        aggregate_seed = self.aggregate_seed_proj(z).view(B, self.T_max, self.d_aggregate)[:, :T, :]
        sequence = torch.cat([aggregate_seed, cls_state], dim=1)

        if padding_mask is not None:
            cls_mask = torch.ones(B, self.T_star, device=padding_mask.device, dtype=padding_mask.dtype)
            extended_mask = torch.cat([padding_mask, cls_mask], dim=1)
        else:
            extended_mask = None

        h = sequence
        for block in self.aggregate_stage_blocks:
            h = block(h, padding_mask=extended_mask)

        aggregate_tokens = h[:, :T, :]

        if self.tied_proj is None:
            per_layer_tokens = self.proj_back(aggregate_tokens)
        else:
            if self.tied_proj:
                per_layer_tokens = F.linear(aggregate_tokens, self.tied_proj.weight.t(), self.proj_back_bias)
            else:
                per_layer_tokens = self.proj_back(aggregate_tokens)

        x_layers = per_layer_tokens.view(B, T, self.L, self.d).permute(0, 2, 1, 3).contiguous()

        layer_outputs = []
        for l in range(self.L):
            layer_outputs.append(self.layerwise_stage_transformers[l](x_layers[:, l, :, :], padding_mask=padding_mask))
        x_recon = torch.stack(layer_outputs, dim=1)
        return x_recon

class VQVAE61(nn.Module):
    """
    VQVAE variant for L layers of hidden states with CLS-based aggregation.
    """
    def __init__(self, encoder, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.codebook_size = config["codebook_size"]
        self.beta = config["beta"]
        self.codebook_reset_counter_multiplier = config["codebook_reset_counter_multiplier"]
        self.latent_dim = encoder.latent_dim

        if self.codebook_size > 0:
            self.codebook = nn.Embedding(self.codebook_size, self.latent_dim)
            nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)
        else:
            self.codebook = None

        if self.codebook_reset_counter_multiplier > 0 and self.codebook_size > 0:
            self.register_buffer("codebook_counters", torch.full((self.codebook_size,), 0))

        self.cosine_push_weight = config["cosine_push_weight"]
        self.entropy_loss_weight = config["entropy_loss_weight"]
        self.mask_prob = config["mask_prob"]
        self.entropy_temperature = config["entropy_temperature"]

        self.usage_tracking_window = config["usage_tracking_window"]
        if self.usage_tracking_window > 0 and self.codebook_size > 0:
            self.register_buffer("usage_history", torch.zeros(self.usage_tracking_window, dtype=torch.long))
            self.register_buffer("usage_ptr", torch.tensor(0, dtype=torch.long))
            self.register_buffer("usage_full", torch.tensor(False, dtype=torch.bool))
            self.register_buffer("total_vectors_processed", torch.tensor(0, dtype=torch.long))

    def normalize_codebook_vectors(self):
        if self.cosine_push_weight > 0 and self.codebook is not None:
            with torch.no_grad():
                self.codebook.weight.data = F.normalize(self.codebook.weight.data, p=2, dim=1)

    def compute_cosine_push_loss(self):
        if self.cosine_push_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        if self.usage_tracking_window > 0:
            stats = self.get_usage_statistics()
            usage_counts = stats["usage_counts"]
            used_indices = torch.where(usage_counts > 0)[0]
            if len(used_indices) <= 1:
                return torch.tensor(0.0, device=next(self.parameters()).device)

            E_used = self.codebook.weight[used_indices]
            E_used_normalized = F.normalize(E_used, p=2, dim=1)
            cosine_sim_matrix = torch.mm(E_used_normalized, E_used_normalized.t())

            usage_weights = usage_counts[used_indices].float()
            usage_weights = usage_weights / usage_weights.sum()
            weight_matrix = usage_weights.unsqueeze(1) * usage_weights.unsqueeze(0)

            mask = ~torch.eye(len(used_indices), device=E_used.device, dtype=torch.bool)
            weighted_cosines_squared = (cosine_sim_matrix ** 2) * weight_matrix
            return weighted_cosines_squared[mask].sum()

        E = self.codebook.weight
        E_normalized = F.normalize(E, p=2, dim=1)
        cosine_sim_matrix = torch.mm(E_normalized, E_normalized.t())
        mask = ~torch.eye(self.codebook_size, device=E.device, dtype=torch.bool)
        return torch.sum(cosine_sim_matrix[mask] ** 2)

    def compute_soft_entropy_loss(self, distances):
        if self.entropy_loss_weight <= 0 or self.codebook is None:
            return torch.tensor(0.0, device=distances.device)

        assignment_probs = F.softmax(-distances / self.entropy_temperature, dim=-1)
        avg_probs = assignment_probs.mean(dim=0)
        epsilon = 1e-10
        avg_probs_safe = torch.clamp(avg_probs, min=epsilon)
        entropy = -torch.sum(avg_probs_safe * torch.log(avg_probs_safe))
        max_entropy = torch.log(torch.tensor(self.codebook_size, device=distances.device, dtype=avg_probs.dtype))
        return max_entropy - entropy

    def update_usage_tracking(self, encoding_indices):
        if self.usage_tracking_window <= 0 or self.codebook is None:
            return

        self.total_vectors_processed += encoding_indices.size(0)
        n_valid = encoding_indices.size(0)
        if n_valid == 0:
            return

        if n_valid >= self.usage_tracking_window:
            self.usage_history[:] = encoding_indices[-self.usage_tracking_window:]
            self.usage_ptr.fill_(0)
            self.usage_full.fill_(True)
            return

        end_ptr = (self.usage_ptr + n_valid) % self.usage_tracking_window
        if end_ptr > self.usage_ptr:
            self.usage_history[self.usage_ptr:end_ptr] = encoding_indices
        else:
            n_until_end = self.usage_tracking_window - self.usage_ptr
            self.usage_history[self.usage_ptr:] = encoding_indices[:n_until_end]
            if n_valid > n_until_end:
                self.usage_history[:end_ptr] = encoding_indices[n_until_end:]

        self.usage_ptr.copy_(end_ptr)
        if self.usage_ptr == 0 or self.usage_full:
            self.usage_full.fill_(True)

    def get_usage_statistics(self):
        if self.usage_tracking_window <= 0 or self.codebook is None:
            zero_counts = torch.zeros(self.codebook_size if self.codebook is not None else 0)
            return {
                "total_vectors_processed": 0,
                "unique_vectors_used": 0,
                "usage_counts": zero_counts,
                "usage_percentages": zero_counts,
            }

        if self.usage_full:
            valid_history = self.usage_history
        else:
            valid_history = self.usage_history[:self.usage_ptr]

        if valid_history.numel() == 0:
            zero_counts = torch.zeros(self.codebook_size)
            return {
                "total_vectors_processed": self.total_vectors_processed.item(),
                "unique_vectors_used": 0,
                "usage_counts": zero_counts,
                "usage_percentages": zero_counts,
            }

        usage_counts = torch.bincount(valid_history, minlength=self.codebook_size)
        usage_percentages = usage_counts.float() / valid_history.numel()
        return {
            "total_vectors_processed": self.total_vectors_processed.item(),
            "unique_vectors_used": (usage_counts > 0).sum().item(),
            "usage_counts": usage_counts,
            "usage_percentages": usage_percentages,
        }

    def compute_codebook_similarities(self):
        if self.codebook is None:
            return {"similarities": None, "used_indices": None, "num_used_vectors": 0}

        stats = self.get_usage_statistics()
        used_indices = torch.where(stats["usage_counts"] > 0)[0]
        if len(used_indices) <= 1:
            return {"similarities": None, "used_indices": used_indices, "num_used_vectors": len(used_indices)}

        with torch.no_grad():
            used_codebook = self.codebook.weight[used_indices]
            used_codebook_norm = used_codebook / used_codebook.norm(dim=1, keepdim=True)
            similarities = torch.mm(used_codebook_norm, used_codebook_norm.t())

        return {"similarities": similarities, "used_indices": used_indices, "num_used_vectors": len(used_indices)}

    def apply_stochastic_mask(self, distances, training=True):
        if self.mask_prob <= 0 or not training:
            return distances

        if torch.rand(1).item() < self.mask_prob:
            B, K = distances.shape
            num_mask = int(self.mask_prob * B * K)
            if num_mask > 0:
                mask_indices = torch.randperm(B * K, device=distances.device)[:num_mask]
                mask_rows = mask_indices // K
                mask_cols = mask_indices % K
                masked_distances = distances.clone()
                masked_distances[mask_rows, mask_cols] = 1e10
                return masked_distances
        return distances

    def forward(self, x, padding_mask=None, beta=None):
        B, L, T, d = x.shape
        entropy_loss = torch.tensor(0.0, device=x.device)
        unique_count = 0

        z_e = self.encoder(x, padding_mask=padding_mask)

        if self.codebook is not None:
            codebook = self.codebook.weight
            distances = (
                torch.sum(z_e ** 2, dim=1, keepdim=True)
                + torch.sum(codebook ** 2, dim=1)
                - 2 * torch.matmul(z_e, codebook.t())
            )

            distances = self.apply_stochastic_mask(distances, training=self.training)
            entropy_loss = self.compute_soft_entropy_loss(distances)

            encoding_indices = torch.argmin(distances, dim=1)
            self.update_usage_tracking(encoding_indices)
            unique_count = torch.unique(encoding_indices).numel()

            if self.codebook_reset_counter_multiplier > 0:
                self.codebook_counters -= B
                self.codebook_counters[encoding_indices] = self.codebook_reset_counter_multiplier * self.codebook_size
                collapsed = self.codebook_counters <= 0
                if collapsed.any():
                    collapsed_indices = torch.where(collapsed)[0]
                    num_collapsed = int(collapsed_indices.numel())
                    if num_collapsed > B:
                        new_batch_vectors = z_e[torch.randperm(B, device=z_e.device)[:B]]
                        overall_mean = z_e.mean(dim=1).mean()
                        overall_std = z_e.std(dim=1).mean()
                        rem = num_collapsed - B
                        random_vectors = torch.normal(
                            mean=overall_mean.item(),
                            std=overall_std.item(),
                            size=(rem, z_e.shape[1]),
                            device=z_e.device,
                        )
                        sorted_indices = collapsed_indices[torch.argsort(collapsed_indices)]
                        self.codebook.weight.data[sorted_indices[:B]] = new_batch_vectors.to(self.codebook.weight.dtype)
                        self.codebook.weight.data[sorted_indices[B:]] = random_vectors.to(self.codebook.weight.dtype)
                    else:
                        new_vectors = z_e[torch.randperm(B, device=z_e.device)[:num_collapsed]]
                        self.codebook.weight.data[collapsed] = new_vectors.to(self.codebook.weight.dtype)
                    self.codebook_counters[collapsed] = self.codebook_reset_counter_multiplier * self.codebook_size

            z_q = self.codebook(encoding_indices)
            codebook_loss = F.mse_loss(z_q.detach(), z_e)
            commitment_weight = self.beta if beta is None else beta
            commitment_loss = commitment_weight * F.mse_loss(z_q, z_e.detach())
            z_q = z_e + (z_q - z_e).detach()
        else:
            z_q = z_e
            codebook_loss = torch.tensor(0.0, device=z_e.device)
            commitment_loss = torch.tensor(0.0, device=z_e.device)

        x_recon = self.decoder(z_q, T=T, padding_mask=padding_mask)

        if padding_mask is not None:
            active_mask = padding_mask.unsqueeze(1).unsqueeze(-1).float()
            recon_loss = F.mse_loss(
                x_recon * active_mask,
                x * active_mask,
                reduction="sum",
            ) / (active_mask.sum() + 1e-8)
        else:
            recon_loss = F.mse_loss(x_recon, x)

        cosine_push_loss = self.compute_cosine_push_loss()
        if self.codebook is None:
            entropy_loss = torch.tensor(0.0, device=z_e.device)

        total_loss = (
            recon_loss
            + codebook_loss
            + commitment_loss
            + self.cosine_push_weight * cosine_push_loss
            + self.entropy_loss_weight * entropy_loss
        )

        return x_recon, total_loss, recon_loss, codebook_loss, commitment_loss, unique_count, cosine_push_loss, entropy_loss