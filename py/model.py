"""
Full definition of a GPT Language Model with optional Linear Attention
and Autocorrelation Attention.

Attention mechanisms available:
1. CausalSelfAttention - Standard softmax attention (default)
2. LinearCausalSelfAttention - Linear attention with O(N) complexity
3. AutocorrelationAttention - FFT-based autocorrelation (inspired by Autoformer)

The autocorrelation mechanism replaces position-specific attention weights with
lag-based correlation weights, computing how the query correlates with keys at
different temporal offsets.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.disabled = False  # Flag to optionally disable normalization

    def forward(self, input):
        if self.disabled:
            return input  # Pass through without normalization
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Standard softmax attention"""

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
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class LinearCausalSelfAttention(nn.Module):
    """
    Linear attention with causal masking.
    
    Instead of softmax(QK^T/sqrt(d))V, we compute:
        φ(Q) @ cumsum(φ(K)^T @ V) / (φ(Q) @ cumsum(φ(K)^T))
    
    where φ is a feature map (here: ELU + 1 for non-negativity).
    
    This achieves O(N * d^2) complexity instead of O(N^2 * d), which is better
    for long sequences but may have reduced expressiveness.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Small epsilon for numerical stability in normalization
        self.eps = 1e-6

    def feature_map(self, x):
        """
        Feature map φ for linear attention.
        Using ELU + 1 to ensure non-negative values, which is important for
        the attention weights to behave like probabilities.
        """
        return F.elu(x) + 1.0

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head: (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply feature map to queries and keys
        q = self.feature_map(q)  # (B, nh, T, hs)
        k = self.feature_map(k)  # (B, nh, T, hs)

        # Causal linear attention via cumulative sum
        # We need to compute for each position i:
        #   output_i = (sum_{j<=i} φ(k_j)^T v_j) @ φ(q_i) / (sum_{j<=i} φ(k_j)) @ φ(q_i)
        #
        # This can be done efficiently with cumsum:
        #   S_i = cumsum(φ(k) ⊗ v)  -- running key-value outer product sum
        #   z_i = cumsum(φ(k))      -- running key sum for normalization

        # Compute key-value outer products: (B, nh, T, hs, hs)
        # k: (B, nh, T, hs) -> (B, nh, T, hs, 1)
        # v: (B, nh, T, hs) -> (B, nh, T, 1, hs)
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)  # (B, nh, T, hs, hs)
        
        # Cumulative sum along sequence dimension for causal masking
        kv_cumsum = torch.cumsum(kv, dim=2)  # (B, nh, T, hs, hs)
        k_cumsum = torch.cumsum(k, dim=2)    # (B, nh, T, hs)

        # Compute attention output
        # For each query, multiply by the accumulated key-value matrix
        # q: (B, nh, T, hs) -> (B, nh, T, 1, hs)
        # kv_cumsum: (B, nh, T, hs, hs)
        # result: (B, nh, T, hs)
        numerator = torch.einsum('bnti,bntij->bntj', q, kv_cumsum)
        
        # Compute normalization factor
        # q: (B, nh, T, hs), k_cumsum: (B, nh, T, hs)
        # result: (B, nh, T)
        denominator = torch.einsum('bnti,bnti->bnt', q, k_cumsum)
        
        # Normalize (with epsilon for stability)
        denominator = denominator.unsqueeze(-1) + self.eps  # (B, nh, T, 1)
        y = numerator / denominator  # (B, nh, T, hs)

        # Reshape back: (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class AutocorrelationAttention(nn.Module):
    """
    Autocorrelation-based attention mechanism inspired by Autoformer (Wu et al., 2021).
    
    Instead of computing position-specific attention weights via softmax(QK^T),
    this computes lag-based correlation weights using FFT:
    
    1. Project input to Q, K, V
    2. Compute autocorrelation R(τ) = IFFT(FFT(Q) * conj(FFT(K)))
    3. For each position t, use correlations at lags 0, 1, ..., t as weights
    4. Aggregate V values from corresponding past positions
    
    The key insight: instead of asking "which specific positions should I attend to?",
    we ask "at what temporal offsets (lags) is my query most correlated with keys?"
    
    This captures periodic and rhythmic dependencies that standard attention might miss,
    while potentially being more parameter-efficient for certain patterns.
    
    Complexity: O(T log T) for the FFT operations.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Q, K, V projections (same as standard attention)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Number of top lags to use (following Autoformer's sparse selection)
        # Default: use top ceil(log(T)) lags, but configurable
        self.top_k = getattr(config, 'autocorr_top_k', None)  # None = use all lags
        
        # Learnable temperature for softmax over correlation weights
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(self.head_dim))
        
        # Small epsilon for numerical stability
        self.eps = 1e-6

    def forward(self, x):
        B, T, C = x.size()
        
        # Project to Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head: (B, T, C) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Compute autocorrelation via FFT
        # This gives us R(τ) for all lags τ in O(T log T)
        # 
        # Autocorrelation: R(τ) = Σ_t Q[t] * K[t-τ]
        # Via convolution theorem: R = IFFT(FFT(Q) * conj(FFT(K)))
        
        # Pad to avoid circular correlation artifacts
        # We need length 2T-1 to get full linear (not circular) correlation
        fft_len = 2 * T
        
        # FFT of Q and K along time dimension
        q_fft = torch.fft.rfft(q, n=fft_len, dim=2)  # (B, nh, fft_len//2+1, hs)
        k_fft = torch.fft.rfft(k, n=fft_len, dim=2)  # (B, nh, fft_len//2+1, hs)
        
        # Cross-correlation in frequency domain
        # Element-wise: Q_f * conj(K_f)
        corr_fft = q_fft * torch.conj(k_fft)  # (B, nh, fft_len//2+1, hs)
        
        # Back to time domain
        corr = torch.fft.irfft(corr_fft, n=fft_len, dim=2)  # (B, nh, fft_len, hs)
        
        # Extract meaningful lags: 0 to T-1 (non-negative lags = past positions)
        # corr[:, :, 0, :] = lag 0 (same position)
        # corr[:, :, 1, :] = lag 1 (one position back)
        # etc.
        corr = corr[:, :, :T, :]  # (B, nh, T, hs)
        
        # Average correlation across head dimension to get scalar weight per lag
        # This gives us how much to weight each lag
        corr_weights = corr.mean(dim=-1)  # (B, nh, T)
        
        # Apply temperature scaling
        corr_weights = corr_weights / (self.temperature + self.eps)
        
        # For causal language modeling, we need position-specific masking:
        # At position t, we can only use lags 0, 1, ..., t (can't look beyond start)
        # 
        # Build output by aggregating V with lag-based weights
        # output[t] = Σ_{τ=0}^{t} softmax(R(τ)) * V[t-τ]
        
        # Create causal lag weights: at position t, only lags 0..t are valid
        # This requires position-dependent softmax
        
        # Expand corr_weights for position-dependent masking
        # We need: for each position t, a distribution over lags 0..t
        
        # Create mask: mask[t, τ] = 1 if τ <= t, else 0
        lag_mask = torch.tril(torch.ones(T, T, device=x.device))  # (T, T)
        
        # Expand weights: (B, nh, T) -> (B, nh, T, T) where last dim is lag
        # Each position t gets the same lag weights, but masked to valid lags
        weights_expanded = corr_weights.unsqueeze(2).expand(B, self.n_head, T, T)  # (B, nh, T, T)
        
        # Apply causal mask: positions can only use lags up to their index
        # mask shape: (T, T) -> (1, 1, T, T)
        weights_masked = weights_expanded.masked_fill(lag_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        # Softmax over valid lags for each position
        attn_weights = F.softmax(weights_masked, dim=-1)  # (B, nh, T, T)
        
        # Handle positions where all lags are masked (shouldn't happen with lag_mask, but safety)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Now we need to gather V values according to lags
        # For position t with lag τ, we want V[t - τ]
        # 
        # Build index tensor: indices[t, τ] = t - τ (clamped to valid range)
        positions = torch.arange(T, device=x.device).unsqueeze(1)  # (T, 1)
        lags = torch.arange(T, device=x.device).unsqueeze(0)       # (1, T)
        gather_indices = (positions - lags).clamp(min=0)           # (T, T)
        
        # Expand for batch and heads: (T, T) -> (B, nh, T, T)
        gather_indices = gather_indices.unsqueeze(0).unsqueeze(0).expand(B, self.n_head, T, T)
        
        # Gather V values: for each (position, lag), get V[position - lag]
        # v shape: (B, nh, T, hs)
        # We need to index into dim 2 (time) using gather_indices
        gather_indices_expanded = gather_indices.unsqueeze(-1).expand(B, self.n_head, T, T, self.head_dim)
        v_expanded = v.unsqueeze(2).expand(B, self.n_head, T, T, self.head_dim)
        v_gathered = torch.gather(v_expanded, dim=3, index=gather_indices_expanded)  # (B, nh, T, T, hs)
        
        # Weighted sum over lags
        # attn_weights: (B, nh, T, T), v_gathered: (B, nh, T, T, hs)
        y = torch.einsum('bntl,bntld->bntd', attn_weights, v_gathered)  # (B, nh, T, hs)
        
        # Reshape back: (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.no_gelu = getattr(config, 'no_gelu', False)

    def forward(self, x):
        x = self.c_fc(x)
        if not self.no_gelu:
            x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # Choose attention type based on config
        if getattr(config, 'use_autocorrelation_attention', False):
            self.attn = AutocorrelationAttention(config)
        elif getattr(config, 'use_linear_attention', False):
            self.attn = LinearCausalSelfAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    use_linear_attention: bool = False      # Set True to use linear attention
    use_autocorrelation_attention: bool = False  # Set True to use autocorrelation attention
    tie_weights: bool = True  # Tie input embeddings to output projection
    no_gelu: bool = False  # Disable GELU nonlinearity in MLP (makes it purely linear)
    autocorr_top_k: int = None  # Number of top lags for autocorrelation (None = all)


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if getattr(config, 'tie_weights', True):
            self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        if getattr(config, 'use_autocorrelation_attention', False):
            attn_type = "autocorrelation"
        elif getattr(config, 'use_linear_attention', False):
            attn_type = "linear"
        else:
            attn_type = "softmax"
        tie_status = "tied" if getattr(config, 'tie_weights', True) else "untied"
        print(f"number of parameters: %.2fM ({attn_type} attention, {tie_status} weights)" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
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
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
