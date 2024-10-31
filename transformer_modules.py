import math


import einops

import torch as t
import torch.nn as nn
from torch import Tensor
from utils import Config, ScaledLinearFunction

device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')

class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))
    
    def forward(self, residual):
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()
        residual = (residual - residual_mean) / residual_std
        return residual * self.w + self.b

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=1)  # Input initialization scaling

    def forward(self, tokens):
        return self.W_E[tokens]  # Input activation scaling is 1

class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=1)  # Input initialization scaling

    def forward(self, tokens):
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)

class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Initialize weights with std=1 for hidden layers
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))

        nn.init.normal_(self.W_Q, std=1)
        nn.init.normal_(self.W_K, std=1)
        nn.init.normal_(self.W_V, std=1)
        nn.init.normal_(self.W_O, std=1)

        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))

        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def forward(self, normalized_resid_pre):
        # Compute query, key, and value vectors with activation scaling
        if self.cfg.apply_UmuP:
            q = (einops.einsum(
                normalized_resid_pre, self.W_Q,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            ) + self.b_Q) * (1 / math.sqrt(self.cfg.d_model))

            k = (einops.einsum(
                normalized_resid_pre, self.W_K,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            ) + self.b_K) * (1 / math.sqrt(self.cfg.d_model))

            v = (einops.einsum(
                normalized_resid_pre, self.W_V,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            ) + self.b_V) * (1 / math.sqrt(self.cfg.d_model))
        else:
            q = einops.einsum(
                normalized_resid_pre, self.W_Q,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            ) + self.b_Q

            k = einops.einsum(
                normalized_resid_pre, self.W_K,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            ) + self.b_K

            v = einops.einsum(
                normalized_resid_pre, self.W_V,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            ) + self.b_V

        # Compute attention scores and apply mask
        attn_scores = einops.einsum(
            q, k,
            "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K",
        )


        if self.cfg.apply_UmuP:
          attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head ** 0.5)
        else:
          attn_scores_masked = self.apply_causal_mask(attn_scores)

        attn_pattern = attn_scores_masked.softmax(-1)

        # Compute attention output
        z = einops.einsum(
            attn_pattern, v,
            "batch nheads posn_Q posn_K, batch posn_K nheads d_head -> batch posn_Q nheads d_head",
        )

        if self.cfg.apply_UmuP:
            attn_out = (einops.einsum(
                z, self.W_O,
                "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
            ) + self.b_O) * (1 / math.sqrt(self.cfg.d_head))  # Activation scaling
        else:
            attn_out = einops.einsum(
                z, self.W_O,
                "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
            ) + self.b_O

        return attn_out

    def apply_causal_mask(self, attn_scores):
        # Apply causal mask to attention scores
        mask = t.triu(t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_in, std=1)
        nn.init.normal_(self.W_out, std=1)
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))

    def forward(self, normalized_resid_mid):
        # Input to hidden layer with activation scaling
        if self.cfg.apply_UmuP:
            pre = (einops.einsum(
                normalized_resid_mid, self.W_in,
                "batch position d_model, d_model d_mlp -> batch position d_mlp",
            ) + self.b_in) * (1 / math.sqrt(self.cfg.d_model))
        else:
            pre = einops.einsum(
                normalized_resid_mid, self.W_in,
                "batch position d_model, d_model d_mlp -> batch position d_mlp",
            ) + self.b_in
        post = t.nn.functional.relu(pre)  # Using ReLU activation function

        # Hidden to output layer with activation scaling
        if self.cfg.apply_UmuP:
          mlp_out = (einops.einsum(
              post, self.W_out,
              "batch position d_mlp, d_mlp d_model -> batch position d_model",
          ) + self.b_out) * (1 / math.sqrt(self.cfg.d_mlp))
        else:
          mlp_out = (einops.einsum(
              post, self.W_out,
              "batch position d_mlp, d_mlp d_model -> batch position d_model",
          ) + self.b_out)

        return mlp_out

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre):
        # Apply attention and residual connection with scaling
        attn_out = self.attn(self.ln1(resid_pre))
        resid_mid = resid_pre + attn_out

        if self.cfg.apply_UmuP:
          resid_mid = resid_mid * (1 / math.sqrt(self.cfg.n_layers))

        # Apply MLP and residual connection with scaling
        mlp_out = self.mlp(self.ln2(resid_mid))
        resid_post = resid_mid + mlp_out

        if self.cfg.apply_UmuP:
          resid_post = resid_post * (1 / math.sqrt(self.cfg.n_layers))

        return resid_post

class Unembed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)  # Standard initialization
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab)), requires_grad=False)

    def forward(
        self, normalized_resid_final: t.Tensor  # Shape: [batch, position, d_model]
    ) -> t.Tensor:  # Shape: [batch, position, d_vocab]
        batch_size, seq_len, _ = normalized_resid_final.shape

        if self.cfg.apply_UmuP:
          input_flat = normalized_resid_final.view(-1, self.cfg.d_model)
          output_flat = ScaledLinearFunction.apply(input_flat, self.W_U, self.b_U)
          logits = output_flat.view(batch_size, seq_len, self.cfg.d_vocab)
        else:
          logits = normalized_resid_final @ self.W_U + self.b_U

        return logits
    
class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.ln_final(residual))
        return logits