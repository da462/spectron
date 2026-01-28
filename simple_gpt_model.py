import torch
import torch.nn as nn
import math
import random


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class GPTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = nn.Dropout(config["attention_dropout"])
        self.resid_dropout = nn.Dropout(config["resid_dropout"])

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.resid_dropout(self.out_proj(attn_output))
        return output


class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_epsilon"])
        self.attn = GPTSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_epsilon"])
        self.mlp = nn.Sequential(
            nn.Linear(config["hidden_size"], 2048),
            GELU(),
            nn.Linear(2048, config["hidden_size"]),
            nn.Dropout(config["resid_dropout"])
        )
        self.masked = False

    def forward(self, x):
        if self.masked:
            return x  # skip the block
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.embed_dim = config["hidden_size"]
        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config["max_position_embeddings"], self.embed_dim)
        self.drop = nn.Dropout(config["embed_dropout"])
        self.h = nn.ModuleList([GPTBlock(config) for _ in range(config["num_layers"])])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config["layer_norm_epsilon"])

    def forward(self, input_ids):
        B, T = input_ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)
        for i, block in enumerate(self.h):
            x = block(x)
        x = self.ln_f(x)
        return x

    def set_mask_from_vector(self, mask_vec):
        """
        Sets each block's `masked` attribute according to the provided mask vector.
        `mask_vec` is a list or tensor of 0s and 1s with length = number of blocks.
        """
        assert len(mask_vec) == len(self.h), "Mask vector length must match number of transformer blocks"
        for i, (block, val) in enumerate(zip(self.h, mask_vec)):
            block.masked = (val == 0)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        self.used_params = set()

    def forward(self, input_ids):
        hidden_states = self.transformer(input_ids)
        logits = self.lm_head(hidden_states)
        return logits

    def set_mask_from_vector(self, mask_vec):
        """
        Sets each block's `masked` attribute according to the provided mask vector.
        `mask_vec` is a list or tensor of 0s and 1s with length = number of blocks.
        """
        self.transformer.set_mask_from_vector(mask_vec)
        
        # Compute used_params once when mask is set
        self.used_params = set()
        self.used_params.update(['transformer.wte.weight', 'transformer.wpe.weight'])
        for i, block in enumerate(self.transformer.h):
            if not block.masked:
                self.used_params.update([f'transformer.h.{i}.{name}' for name, _ in block.named_parameters()])
        self.used_params.update(['transformer.ln_f.weight', 'transformer.ln_f.bias'])
        self.used_params.update(['lm_head.weight'])

    def report_parameter_usage(self):
        for name, _ in self.named_parameters():
            if name in self.used_params:
                print(f"✅ {name}")
            else:
                print(f"❌ {name}")


class NetGPT(nn.Module):
    def __init__(self, model, gpu_id):
        super().__init__()
        self.model = model
        self.gpu_id = gpu_id
        self.get_weights()  # Initialize size_learnable_params

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def get_weights(self):
        """Returns a 1D tensor containing all model parameters."""
        learnable_params = nn.utils.parameters_to_vector(self.model.parameters())
        self.size_learnable_params = len(learnable_params)
        return learnable_params

    @torch.no_grad()
    def set_weights(self, weights):
        """Sets model parameters from a 1D tensor."""
        nn.utils.vector_to_parameters(weights, self.model.parameters())

    @torch.no_grad()
    def get_parameter_mask(self):
        """Returns a mask indicating which parameters are used."""
        mask = torch.zeros(self.size_learnable_params, dtype=torch.bool)
        param_idx = 0
        for name, param in self.model.named_parameters():
            if name in self.model.used_params:
                mask[param_idx:param_idx + param.numel()] = True
            param_idx += param.numel()
        return mask


def create_model_gpt(gpu_id, args):
    """Create a GPT model with the specified configuration."""
    config = {
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "vocab_size": args.vocab_size,
        "max_position_embeddings": args.max_position_embeddings,
        "embed_dropout": args.embed_dropout,
        "resid_dropout": args.resid_dropout,
        "attention_dropout": args.attention_dropout,
        "layer_norm_epsilon": args.layer_norm_epsilon,
    }
    
    model = GPT(config)
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(gpu_id)
    criterion = criterion.to(gpu_id)

    if gpu_id == 0:
        print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("Model architecture:", config)

    return NetGPT(model, gpu_id), criterion 