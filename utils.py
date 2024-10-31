from torch.autograd import Function
import torch as t
import math
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

@dataclass
class TransformerTrainingArgs:
    lr : float = 1e-3
    weight_decay : float = 1e-2
    wandb_project: str | None = "day1-demotransformer"
    wandb_name: str | None = None
    batch_size : int = 16
    epochs: int= 20
    max_steps_per_epoch:int= 200
    

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    apply_UmuP: bool = False
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


def get_adaptive_optimizer(model, eta=1e-3):
    """
    Returns an Adam optimizer with adaptive learning rates for different layers
    of the model based on the specified scaling factors.

    Args:
        model: The transformer model instance.
        eta (float): The base learning rate.

    Returns:
        optimizer: A configured Adam optimizer with parameter groups.
    """
    # Initialize parameter groups list
    param_groups = []

    # -- Input Layer Parameters --
    input_params = []
    # Assuming model.embed and model.pos_embed are your embedding layers
    input_params.extend([model.embed.W_E, model.pos_embed.W_pos])

    # Fan-out for input layer is the output dimension of the embedding
    fan_out_input = model.cfg.d_model
    scaling_input = eta / math.sqrt(fan_out_input)

    param_groups.append({
        'params': input_params,
        'lr': scaling_input
    })

    # -- Hidden Layer Parameters --
    hidden_params = []
    for block in model.blocks:
        # Attention parameters
        hidden_params.extend([
            block.attn.W_Q, block.attn.b_Q,
            block.attn.W_K, block.attn.b_K,
            block.attn.W_V, block.attn.b_V,
            block.attn.W_O, block.attn.b_O
        ])
        # MLP parameters
        hidden_params.extend([
            block.mlp.W_in, block.mlp.b_in,
            block.mlp.W_out, block.mlp.b_out
        ])

    # Fan-in for hidden layers is the input dimension to the layers
    fan_in_hidden = model.cfg.d_model
    scaling_hidden = eta / math.sqrt(fan_in_hidden)

    param_groups.append({
        'params': hidden_params,
        'lr': scaling_hidden
    })

    # -- Output Layer Parameters --
    output_params = [model.unembed.W_U]
    # Learning rate for output layer is eta
    param_groups.append({
        'params': output_params,
        'lr': eta
    })

    # -- Residual Connections (LayerNorm Parameters) --
    residual_params = []
    for block in model.blocks:
        residual_params.extend([
            block.ln1.w, block.ln1.b,
            block.ln2.w, block.ln2.b
        ])

    # Depth is the number of layers
    depth = model.cfg.n_layers
    scaling_residual = eta / math.sqrt(depth)

    param_groups.append({
        'params': residual_params,
        'lr': scaling_residual
    })

    # -- Initialize the Adam Optimizer --
    optimizer = optim.Adam(param_groups)

    return optimizer

class ScaledLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        """
        Forward pass for the linear layer.
        """
        ctx.save_for_backward(input, weight, bias)
        fan_in = input.size(-1)
        scaling_factor = 1 / fan_in
        output = input.matmul(weight) + bias
        return output * scaling_factor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass where we scale the gradients by 1/sqrt(fan-in).
        """
        input, weight, bias = ctx.saved_tensors
        fan_in = input.size(-1)
        scaling_factor = 1 / math.sqrt(fan_in)

        # Compute gradients w.r.t. input, weight, and bias
        grad_input = grad_output.matmul(weight.t()) * scaling_factor
        grad_weight = input.t().matmul(grad_output) * scaling_factor
        grad_bias = grad_output.sum(dim=0) * scaling_factor

        return grad_input, grad_weight, grad_bias


def get_log_probs(
    logits,
    tokens
):

    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens