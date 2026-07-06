import os
import torch
import torch.distributed as dist
import wandb
import numpy as np
import argparse
import copy
import random
import math
from titan_gpt import TitanGPT, TitanModelArgs
from simple_gpt_dataloader import create_dataloaders
from low_rank_linear import replace_linear_with_lowrank, frobdecay, get_frobenius_regularization_grads, count_parameters
from self_guided_linear import replace_linear_with_selfguided, CosineTempDecay, LowRankWithSelfGuided
from model_analysis import calculate_transformer_flops
import time
from datetime import datetime
from pathlib import Path
from power_iter import get_lowrank_spectral_norm_scaling
from matrix_analysis import compute_matrix_metrics
# Base seed for reproducibility. Keep the historical default unless a run
# explicitly overrides it.
BASE_SEED = 1337

# Conditional import for Muon optimizer
try:
    from muon_local import MuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    print("Warning: Muon optimizer not available. Install with: pip install muon")


def wsd_schedule(
    n_iterations,
    final_lr_factor=0.0,
    n_warmup=1000,
    init_div_factor=100,
    fract_decay=0.1,
    decay_type="linear",
    sqrt_power=0.5,
    linear_pw_subdivisions=[],
    cooldown_start_lr_factor=1.0,
):
    """Warmup, hold, and decay schedule.
    Args:
        n_iterations: total number of iterations
        final_lr_factor: factor by which to reduce max_lr at the end
        n_warmup: number of iterations used for warmup
        init_div_factor: initial division factor for warmup
        fract_decay: fraction of iterations used for decay
        decay_type: type of decay ('linear', 'linear_pw', 'exp', 'cosine', 'miror_cosine', 'square', 'sqrt')
        sqrt_power: power for sqrt decay type
        linear_pw_subdivisions: subdivisions for piecewise linear decay
        cooldown_start_lr_factor: starting lr factor for cooldown phase
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    n_anneal_steps = int(fract_decay * n_iterations)
    n_hold = n_iterations - n_anneal_steps
    linear_pw_subdivisions = linear_pw_subdivisions or []

    def schedule(step):
        if step < n_warmup:
            return (step / n_warmup) + (1 - step / n_warmup) / init_div_factor
        elif step < n_hold:
            return 1.0
        elif step < n_iterations:
            if decay_type == "linear" or decay_type == "linear_pw":
                subdivisions = [cooldown_start_lr_factor] + linear_pw_subdivisions + [final_lr_factor]
                division_step = 1 / (len(subdivisions) - 1)

                cooldown_fraction = (step - n_hold) / n_anneal_steps
                now_subdivision = math.floor(cooldown_fraction / division_step)
                left_frac, right_frac = subdivisions[now_subdivision], subdivisions[now_subdivision + 1]
                local_fraction = (cooldown_fraction - division_step * now_subdivision) / division_step
                return left_frac + (right_frac - left_frac) * local_fraction
            elif decay_type == "exp":
                return final_lr_factor ** ((step - n_hold) / n_anneal_steps)
            elif decay_type == "cosine":
                return (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
            elif decay_type == "miror_cosine":
                cosine_value = (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
                linear_value = final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
                return linear_value * 2 - cosine_value
            elif decay_type == "square":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - ((step - n_hold) / n_anneal_steps) ** 2
                )

            elif decay_type == "sqrt":
                return final_lr_factor + (cooldown_start_lr_factor - final_lr_factor) * (
                    1 - ((step - n_hold) / n_anneal_steps) ** sqrt_power
                )

            else:
                raise ValueError(
                    f"decay type {decay_type} is not in ['cosine','miror_cosine','linear','exp','square','sqrt']"
                )

        else:
            return final_lr_factor

    return schedule


def save_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    model_args: TitanModelArgs,
    shared_optimizer: torch.optim.Optimizer,
    private_optimizers: dict,
    private_param_store: dict,
    scheduler,
    step: int,
    total_tokens_rank0: int,
    total_tokens_world: int,
    total_flops: int,
    total_flops_forward: int,
    total_flops_backward: int,
    args: argparse.Namespace,
    best_val_loss: float = float('inf'),
    is_final: bool = False,
    is_best: bool = False,
    alpha_scheduler = None
):
    """
    Save a training checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoints
        model: The model to save
        model_args: Model configuration
        shared_optimizer: Shared parameter optimizer
        private_optimizers: Dictionary of private parameter optimizers per virtual worker
        private_param_store: Dictionary of private parameter states per virtual worker
        scheduler: Learning rate scheduler
        step: Current training step
        total_tokens_rank0: Total tokens processed by rank 0
        total_tokens_world: Total tokens processed globally
        total_flops: Total FLOPs computed
        total_flops_forward: Forward pass FLOPs
        total_flops_backward: Backward pass FLOPs
        args: Training arguments
        is_final: Whether this is the final checkpoint
        is_best: Whether this is the best validation checkpoint
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Create checkpoint filename
    if is_best:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
    elif is_final:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}_{timestamp}.pt")

    # Prepare checkpoint data
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'model_args': vars(model_args) if hasattr(model_args, '__dict__') else model_args,
        'shared_optimizer_state_dict': shared_optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'total_tokens_rank0': total_tokens_rank0,
        'total_tokens_world': total_tokens_world,
        'total_flops': total_flops,
        'total_flops_forward': total_flops_forward,
        'total_flops_backward': total_flops_backward,
        'best_val_loss': best_val_loss,
        'args': vars(args),
        'timestamp': datetime.now().isoformat(),
    }

    # Save self-guided state if enabled
    if hasattr(args, 'self_guided') and args.self_guided and alpha_scheduler is not None:
        checkpoint['self_guided_phase'] = 1 if step < args.guided_steps else 2
        checkpoint['current_alpha'] = alpha_scheduler.get_alpha(step).item()

    # Save wandb run ID if wandb is initialized
    if wandb.run is not None:
        checkpoint['wandb_run_id'] = wandb.run.id
        checkpoint['wandb_run_name'] = wandb.run.name

    # Save private optimizers state
    private_optimizers_state = {}
    for vw, opt in private_optimizers.items():
        if opt is not None:
            private_optimizers_state[vw] = opt.state_dict()
    checkpoint['private_optimizers_state_dict'] = private_optimizers_state

    # Save private parameter store
    private_param_store_serializable = {}
    for vw, params_dict in private_param_store.items():
        private_param_store_serializable[vw] = {k: v.cpu() for k, v in params_dict.items()}
    checkpoint['private_param_store'] = private_param_store_serializable

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path} at step {step}")

    return checkpoint_path


def prune_step_checkpoints(checkpoint_dir: str, keep_latest_k: int) -> None:
    if keep_latest_k < 0:
        return

    checkpoint_paths = sorted(
        Path(checkpoint_dir).glob("checkpoint_step_*.pt"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )

    for checkpoint_path in checkpoint_paths[keep_latest_k:]:
        try:
            checkpoint_path.unlink()
            print(f"Removed old step checkpoint: {checkpoint_path}")
        except OSError as exc:
            print(f"Warning: failed to remove old step checkpoint {checkpoint_path}: {exc}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    shared_optimizer: torch.optim.Optimizer,
    private_optimizers: dict,
    private_param_store: dict,
    scheduler,
    device: torch.device
):
    """
    Load a training checkpoint and resume training.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load weights into
        shared_optimizer: Shared parameter optimizer to load state into
        private_optimizers: Dictionary of private parameter optimizers
        private_param_store: Dictionary of private parameter states
        scheduler: Learning rate scheduler
        device: Device to load tensors to

    Returns:
        Dictionary containing checkpoint metadata (step, tokens, flops, etc.)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return None

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{device}")

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model state loaded from step {checkpoint['step']}")

    # Load shared optimizer state
    shared_optimizer.load_state_dict(checkpoint['shared_optimizer_state_dict'])
    print("Shared optimizer state loaded")

    # Load scheduler state
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print("Scheduler state loaded")

    # Load private optimizers state
    if 'private_optimizers_state_dict' in checkpoint:
        for vw, opt in private_optimizers.items():
            if opt is not None and vw in checkpoint['private_optimizers_state_dict']:
                opt.load_state_dict(checkpoint['private_optimizers_state_dict'][vw])
        print("Private optimizers state loaded")

    # Load private parameter store
    if 'private_param_store' in checkpoint:
        for vw, params_dict in checkpoint['private_param_store'].items():
            if vw in private_param_store:
                for param_name, param_tensor in params_dict.items():
                    if param_name in private_param_store[vw]:
                        private_param_store[vw][param_name].copy_(param_tensor.to(device))
        print("Private parameter store loaded")

    # Return metadata
    metadata = {
        'start_step': checkpoint['step'] + 1,  # Resume from next step
        'total_tokens_rank0': checkpoint.get('total_tokens_rank0', 0),
        'total_tokens_world': checkpoint.get('total_tokens_world', 0),
        'total_flops': checkpoint.get('total_flops', 0),
        'total_flops_forward': checkpoint.get('total_flops_forward', 0),
        'total_flops_backward': checkpoint.get('total_flops_backward', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'checkpoint_timestamp': checkpoint.get('timestamp', 'unknown'),
        'wandb_run_id': checkpoint.get('wandb_run_id', None),
        'wandb_run_name': checkpoint.get('wandb_run_name', None),
        'self_guided_phase': checkpoint.get('self_guided_phase', None),
        'current_alpha': checkpoint.get('current_alpha', None),
    }

    print(f"Resuming from step {metadata['start_step']}")
    if metadata['self_guided_phase'] is not None:
        print(f"Self-guided training state:")
        print(f"  Phase: {metadata['self_guided_phase']}")
        print(f"  Alpha: {metadata['current_alpha']:.4f}")
    print(f"Previous training: {metadata['total_tokens_world']:,} tokens, {metadata['total_flops']/1e12:.2f} TFLOPs")
    if metadata['wandb_run_id']:
        print(f"Wandb run ID: {metadata['wandb_run_id']}")

    return metadata
def seed_everything(seed: int, rank: int):
    """Seed all random number generators for reproducibility."""
    # Add rank to seed to ensure different data order per rank while keeping model init consistent
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if rank == 0:
        print(f"Seeded all RNGs with base seed: {seed}")


def broadcast_model_parameters(model, rank: int, src: int = 0):
    """Broadcast all model parameters and buffers from src rank to all other ranks."""
    if rank == 0:
        print("Broadcasting model parameters and buffers from rank 0 to all workers...")

    # Broadcast parameters
    for param in model.parameters():
        dist.broadcast(param.data, src=src)

    # Broadcast buffers (e.g., running stats in batch norm, if any)
    for buffer in model.buffers():
        dist.broadcast(buffer, src=src)

    if rank == 0:
        print("Model broadcast complete. All workers have synchronized weights.")


def generate_spread_with_min_col_constraint(n_workers: int, n_layers: int, num_overlaps: int, seed: int = None) -> torch.Tensor:
    """Build (n_workers x n_layers) mask where each worker has exactly num_overlaps layers.
    First and last blocks are always shared by all workers."""
    assert 0 <= num_overlaps <= n_layers
    if seed is not None:
        # Use a consistent generator for reproducibility
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    M = torch.zeros((n_workers, n_layers), dtype=torch.int)
    
    # Always share first and last blocks (layers 0 and n_layers-1)
    M[:, 0] = 1  # First block shared by all
    M[:, n_layers-1] = 1  # Last block shared by all
    
    # Calculate remaining layers to assign per worker
    remaining_layers_per_worker = num_overlaps - 2  # Subtract the 2 always-shared layers
    
    if remaining_layers_per_worker > 0:
        # Simple approach: assign middle layers to workers in a round-robin fashion
        # Skip the first and last layers (indices 0 and n_layers-1)
        middle_layers = list(range(1, n_layers-1))  # Layers 1 to n_layers-2
        rng.shuffle(middle_layers) # Shuffle to make the distribution non-sequential
        
        layer_idx = 0
        for worker in range(n_workers):
            layers_assigned = 0
            while layers_assigned < remaining_layers_per_worker and layer_idx < len(middle_layers):
                M[worker, middle_layers[layer_idx]] = 1
                layers_assigned += 1
                layer_idx = (layer_idx + 1) % len(middle_layers)
    
    print(f"Column sums: {M.sum(dim=0).tolist()}")
    print(f"Row sums: {M.sum(dim=1).tolist()}")
    return M


def get_parameter_sharing_info(model, block_selection_matrix, device):
    """Get detailed information about which workers share each parameter."""
    param_sharing = {}
    
    # Iterate through all model parameters
    for name, param in model.named_parameters():
        is_shared = False
        sharing_workers = []
        
        # Check for transformer block parameters
        if '.layers.' in name:
            try:
                # Extract layer index from the parameter name (e.g., 'layers.5.attention.wq.weight')
                layer_idx = int(name.split('.layers.')[1].split('.')[0])
                workers_with_block = torch.where(block_selection_matrix[:, layer_idx] == 1)[0].tolist()
                if len(workers_with_block) > 1:
                    is_shared = True
                    sharing_workers = workers_with_block
            except (ValueError, IndexError):
                # This can happen for parameters that aren't part of a numbered layer
                pass
        
        # Embeddings and final normalization are always shared
        elif 'tok_embeddings.' in name or 'norm.' in name or 'output.' in name:
            is_shared = True
            sharing_workers = list(range(block_selection_matrix.shape[0]))  # All workers
        
        if is_shared:
            param_key = name
            param_sharing[param_key] = {
                'name': name,
                'sharing_workers': sharing_workers,
                'sharing_count': len(sharing_workers)
            }
            
    return param_sharing


def communicate_gradients(model, param_sharing, rank, world_size):
    """All-reduce gradients for shared parameters across only the workers that share them."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if name in param_sharing:
                sharing_info = param_sharing[name]
                sharing_workers = sharing_info['sharing_workers']
                sharing_count = sharing_info['sharing_count']
                
                grad_to_reduce = param.grad.clone()
                if rank not in sharing_workers:
                    grad_to_reduce.zero_()
                
                dist.all_reduce(grad_to_reduce, op=dist.ReduceOp.SUM)
                
                if rank in sharing_workers:
                    param.grad = grad_to_reduce / sharing_count
                else:
                    param.grad.zero_()


def communicate_lowrank_gradients(model, world_size):
    """Communicate low-rank A and B gradients by averaging across all ranks"""
    backend = dist.get_backend()
    with torch.no_grad():
        for name, param in model.named_parameters():
            name_parts = name.split(".")
            if len(name_parts) > 1:
                param_type = name_parts[-1]
                if param_type in ['A', 'B'] and param.grad is not None:
                    # Average gradients across all ranks
                    if param.is_cuda and backend == "gloo":
                        cpu_grad = param.grad.cpu().contiguous()
                        dist.all_reduce(cpu_grad, op=dist.ReduceOp.SUM)
                        cpu_grad /= world_size
                        param.grad.copy_(cpu_grad.cuda())
                    else:
                        if not param.grad.is_contiguous():
                            param.grad = param.grad.contiguous()
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                        param.grad /= world_size


def get_lowrank_param_names(model):
    """Get parameter names for low-rank A and B matrices"""
    lowrank_names = []
    for name, param in model.named_parameters():
        name_parts = name.split(".")
        if len(name_parts) > 1 and name_parts[-1] in ['A', 'B']:
            lowrank_names.append(name)
    return lowrank_names


def register_stochastic_depth_hooks(model, worker_mask):
    """Register hooks to implement stochastic depth (backward masking semantics).
    For masked layers on this worker:
    - Treat the layer as identity wrt upstream gradient (pass grad_output -> grad_input)
    - Zero out parameter gradients so masked layers are not updated
    Notes:
    - We use a full backward hook to modify grad_input (identity behavior upstream)
    - We also attach per-parameter gradient hooks that consult the mutable worker_mask
    - The worker_mask tensor is mutated in-place per virtual worker before backward
    """

    def create_backward_identity_hook(layer_idx):
        def hook(module, grad_input, grad_output):
            if worker_mask[layer_idx] == 0:
                num_inputs = len(grad_input)
                return (grad_output[0],) + (None,) * (num_inputs - 1)
            return None
        return hook

    def create_param_grad_mask_hook(layer_idx):
        def param_hook(grad):
            return grad if worker_mask[layer_idx] == 1 else torch.zeros_like(grad)
        return param_hook

    for i, layer in enumerate(model.layers.values()):
        # Identity gradient upstream when masked
        layer.register_full_backward_hook(create_backward_identity_hook(i))
        # Suppress parameter gradients when masked
        for param in layer.parameters(recurse=True):
            param.register_hook(create_param_grad_mask_hook(i))


def evaluate_model(model, dataloader, criterion, device, args):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Use bf16 autocast if enabled (pass as global or argument)
            if hasattr(args, 'bf16') and args.bf16:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(input_ids, input_batch=input_ids)
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                logits = model(input_ids, input_batch=input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            

            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
            
            # Calculate token accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct_tokens += (predictions == labels).sum().item()
    
    if total_tokens == 0:
        # Handle case where validation set is empty
        return {
            'val_loss': float('inf'),
            'val_perplexity': float('inf'),
            'val_token_accuracy': 0.0
        }

    avg_loss = total_loss / total_tokens
    token_accuracy = correct_tokens / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    
    return {
        'val_loss': avg_loss,
        'val_perplexity': perplexity,
        'val_token_accuracy': token_accuracy
    }


def main():
    parser = argparse.ArgumentParser(description='Simplified GPT Subnet Training')

    # Training config
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--micro_batch_size', type=int, default=None)  # For gradient accumulation
    parser.add_argument('--max_lr', type=float, default=0.0001)  # Renamed from lr_inner
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--nh_weight_decay', type=float, default=0.1, help='Non-hidden weight decay (default: 0.0)')

    parser.add_argument('--total_steps', type=int, default=1000)
    parser.add_argument('--lr_schedule_steps', type=int, default=None,
                        help='Number of steps used to shape warmup/cosine LR schedule. Defaults to total_steps.')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Warmup ratio (warmup_steps = warmup_ratio * lr_schedule_steps)')
    parser.add_argument('--warmup_start_factor', type=float, default=0.1,
                        help='Initial warmup LR as a fraction of max_lr for AdamW cosine schedule')
    parser.add_argument('--seed', type=int, default=BASE_SEED, help='Base random seed for model init and data-worker RNGs')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=None,
                        help='Evaluate every N steps. Defaults to log_interval. Set 0 to disable periodic evaluation.')
    parser.add_argument('--skip_final_eval', action='store_true',
                        help='Skip final validation evaluation at the end of training')
    parser.add_argument('--max_val_samples', type=int, default=1000)  # Limit validation samples for faster eval

    # Optimizer config
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'muon'],
                        help='Optimizer to use: adamw or muon')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1 parameter')
    parser.add_argument('--adam_beta2', type=float, default=0.95, help='Adam beta2 parameter')
    parser.add_argument('--adjust_muon_lr', type=str, default='original', choices=['original', 'match_rms_adamw', 'none'],
                        help='Muon learning rate adjustment function: original (sqrt(max(1, A/B))), match_rms_adamw (0.2*sqrt(max(A,B))), or none (no adjustment)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'wsd'],
                        help='Learning rate scheduler to use: cosine or wsd (warmup-stable-decay)')
    parser.add_argument('--min_lr_factor', type=float, default=0.1,
                        help='Final LR as a fraction of max_lr for cosine AdamW schedule')

    # Model config
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--max_position_embeddings', type=int, default=1024)
    parser.add_argument('--embed_dropout', type=float, default=0.0)
    parser.add_argument('--resid_dropout', type=float, default=0.0)
    parser.add_argument('--attention_dropout', type=float, default=0.0)
    parser.add_argument('--layer_norm_epsilon', type=float, default=1e-5)
    parser.add_argument('--n_kv_heads', type=int, default=None)
    parser.add_argument('--multiple_of', type=int, default=256)
    parser.add_argument('--ffn_dim_multiplier', type=float, default=None)
    parser.add_argument('--rope_theta', type=float, default=10000.0)
    parser.add_argument(
        '--tt_style_init',
        action='store_true',
        help='Initialize weights with TorchTitan Llama3-style rules for ablations',
    )

    # Subnet config
    parser.add_argument('--stochastic_depth', action='store_true')
    parser.add_argument('--num_overlaps', type=int, default=6)
    parser.add_argument('--block_selection_seed', type=int, default=42)
    parser.add_argument('--stochastic_depth_mode', type=str, default='backward_hook', choices=['backward_hook', 'forward_skip', 'backward_nograd'])
    # Virtual workers per GPU (time-multiplexed workers per physical device)
    parser.add_argument('--virtual_workers_per_gpu', type=int, default=1, help='Number of virtual workers to simulate per GPU')

    # Low-rank training config
    parser.add_argument('--low_rank', action='store_true', help='Enable low-rank linear layer decomposition')
    parser.add_argument('--low_rank_ratio', type=float, default=0.25, help='Low-rank ratio for calculating rank as ratio * in_features (default: 0.25)')
    parser.add_argument('--max_layers', type=int, default=None, help='Maximum number of layers to apply low-rank decomposition (default: all eligible layers)')
    parser.add_argument('--layer_indices', type=int, nargs='+', default=None, help='Specific layer indices to apply low-rank decomposition (0-based indexing)')
    parser.add_argument('--exclude_modules', type=str, nargs='+', default=['tok_embeddings','output'], help='Modules to exclude from low-rank decomposition')
    parser.add_argument('--disable_c', action='store_true', help='Disable C matrix in ACB factorization (use AB factorization instead)')
    parser.add_argument('--frobenius_coef', type=float, default=0.0, help='Frobenius decay regularization coefficient (default: 0.0, disabled)')
    parser.add_argument('--communicate_lowrank_freq', type=int, default=None, help='Frequency of low-rank A/B parameter communication (steps). If None, syncs every step')
    parser.add_argument('--sanity_check_lowrank', action='store_true', help='Run low-rank sanity check with A=W, C=B=I')
    parser.add_argument('--spectral_lr_scaling', action='store_true', help='Scale learning rate of A and B matrices by 1/(spectral_norm(A) + spectral_norm(B)) (requires --low_rank)')
    parser.add_argument('--spectral_lr_scaling_offset', type=float, default=1.0,
                        help='Additive offset for spectral LR scaling denominator (default: 1.0 preserves current behavior)')
    parser.add_argument('--spectral_weight_decay', type=float, default=0.0,
                        help='Spectral norm weight decay coefficient (default: 0.0, disabled)')
    parser.add_argument('--swd_type', type=str, default='standard', choices=['standard', 'product'],
                        help='Type of spectral weight decay: standard (sum of squares) or product (default: standard)')

    # Self-guided training arguments
    parser.add_argument('--self_guided', action='store_true',
                        help='Enable self-guided training (mutually exclusive with --low_rank)')
    parser.add_argument('--guided_steps_ratio', type=float, default=0.5,
                        help='Ratio of total steps for guided training (default: 0.3 for 30%%)')
    parser.add_argument('--guided_steps', type=int, default=None,
                        help='Number of guided steps (overrides guided_steps_ratio if specified)')
    parser.add_argument('--reduce_flop', action='store_true', default=True,
                        help='Use stochastic guide computation to reduce FLOPs (default: True)')
    parser.add_argument('--sg_warmup_steps', type=int, default=0,
                        help='Warmup steps for self-guided alpha scheduler (default: 0)')

    # Add flex_attn flag
    parser.add_argument('--use_flex_attn', action='store_true', help="Enable flex attention.")
    # Add bf16 flag
    parser.add_argument('--bf16', action='store_true', help="Enable bf16 mixed precision training.")
    parser.add_argument('--rescale_lr', action='store_true', help="Rescale learning rate based on num_overlaps / num_layers.")
    parser.add_argument('--track_stable_rank', action='store_true', help="Track stable rank of weight matrices in layers 4 and 7.")

    # Data config (modded-nanogpt style)
    parser.add_argument('--train_files', type=str, default="fineweb/fineweb_train_*.bin")
    parser.add_argument('--val_files', type=str, default="fineweb/fineweb_val_*.bin")
    parser.add_argument('--train_seq_len', type=int, default=1024)
    parser.add_argument('--val_seq_len', type=int, default=1024)

    # Wandb config
    parser.add_argument('--wandb_project', type=str, default='gpt_subnet_training')
    parser.add_argument('--wandb_entity', type=str, default='your_username')
    parser.add_argument('--run_name', type=str, default=None)

    # Checkpoint config
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_interval_hours', type=float, default=2.8,
                        help='Save checkpoint every N hours (default: 2.8)')
    parser.add_argument('--checkpoint_interval_steps', type=int, default=0,
                        help='Save a periodic checkpoint every N optimizer steps without exiting (0 disables)')
    parser.add_argument('--checkpoint_keep_latest_k', type=int, default=2,
                        help='Keep only the latest K periodic step checkpoints; negative keeps all')
    parser.add_argument('--save_final_checkpoint', action='store_true', default=True,
                        help='Save final checkpoint at end of training')
    parser.add_argument('--save_best_checkpoint', action='store_true', default=False,
                        help='Save checkpoint when best validation loss is achieved')
    args = parser.parse_args()

    # Calculate warmup from the schedule horizon. This lets short diagnostic
    # jobs stop early while keeping the same LR values as a longer reference run.
    args.lr_schedule_steps = args.lr_schedule_steps or args.total_steps
    if args.lr_schedule_steps <= 0:
        raise ValueError("--lr_schedule_steps must be positive")
    args.warmup_steps = int(args.warmup_ratio * args.lr_schedule_steps)
    if args.eval_interval is None:
        args.eval_interval = args.log_interval
    if args.log_interval < 0:
        raise ValueError("--log_interval must be non-negative")
    if args.eval_interval < 0:
        raise ValueError("--eval_interval must be non-negative")
    if args.min_lr_factor < 0:
        raise ValueError("--min_lr_factor must be non-negative")
    if args.spectral_lr_scaling_offset < 0:
        raise ValueError("--spectral_lr_scaling_offset must be non-negative")
    if not 0.0 <= args.warmup_start_factor <= 1.0:
        raise ValueError("--warmup_start_factor must be between 0 and 1")
    if args.checkpoint_interval_steps < 0:
        raise ValueError("--checkpoint_interval_steps must be non-negative")

    # Validate Muon optimizer availability
    if args.optimizer == 'muon' and not MUON_AVAILABLE:
        raise ImportError("Muon optimizer requested but not available. Install with: pip install muon")

    # Validate spectral_lr_scaling requires low_rank
    if args.spectral_lr_scaling and not args.low_rank:
        raise ValueError("--spectral_lr_scaling requires --low_rank to be enabled")

    # Validate self-guided and low-rank are mutually exclusive
    if args.self_guided and args.low_rank:
        raise ValueError("--self_guided and --low_rank are mutually exclusive. Choose one baseline.")

    # Calculate guided_steps if self-guided is enabled
    if args.self_guided:
        if args.guided_steps is None:
            args.guided_steps = int(args.guided_steps_ratio * args.total_steps)

    if args.rescale_lr and args.stochastic_depth:
        scaling_factor = args.num_overlaps / args.num_layers
        original_lr = args.max_lr
        args.max_lr *= scaling_factor
        print(f"Rescaling LR. Original: {original_lr:.6f}, New: {args.max_lr:.6f}, Factor: {scaling_factor:.2f}")

    # Initialize distributed training (torchrun handles the environment variables)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    print(f"Rank {rank}/{world_size} (local rank {local_rank}) on device {device}")

    # Seed everything for reproducibility BEFORE model creation
    seed_everything(args.seed, rank)

    # Virtual world setup
    virtual_workers_per_gpu = max(1, int(getattr(args, 'virtual_workers_per_gpu', 1)))
    virtual_world_size = world_size * virtual_workers_per_gpu
    first_virtual_worker_index = rank * virtual_workers_per_gpu
    local_virtual_worker_indices = list(range(first_virtual_worker_index, first_virtual_worker_index + virtual_workers_per_gpu))
    if rank == 0:
        print(f"Using virtual workers per GPU: {virtual_workers_per_gpu}. Virtual world size: {virtual_world_size}")

    # Create model and criterion
    model_args = TitanModelArgs(
        vocab_size=args.vocab_size,
        n_layers=args.num_layers,
        n_heads=args.num_heads,
        dim=args.hidden_size,
        max_seq_len=args.max_position_embeddings,
        norm_eps=args.layer_norm_epsilon,
        use_flex_attn=args.use_flex_attn,
        n_kv_heads=args.n_kv_heads,
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        rope_theta=args.rope_theta,
        depth_init=args.tt_style_init,
    )
    model = TitanGPT(model_args).to(device)
    if rank == 0 and args.tt_style_init:
        print("Using TorchTitan Llama3-style initialization")

    # Apply low-rank decomposition if enabled
    if args.low_rank:
        if rank == 0:
            print(f"Applying low-rank decomposition with ratio={args.low_rank_ratio}")
            print(f"Original model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        model = replace_linear_with_lowrank(
            model,
            rank_ratio=args.low_rank_ratio,
            method="svd",
            exclude_modules=args.exclude_modules,
            max_layers=args.max_layers,
            layer_indices=args.layer_indices,
            disable_c=args.disable_c,
            sanity_check=args.sanity_check_lowrank
        )
        model = model.to(device)
        if rank == 0:
            low_rank_params = sum(p.numel() for p in model.parameters())
            reduction = (1 - low_rank_params / (sum(p.numel() for p in TitanGPT(model_args).parameters()))) * 100

            low_rank_non_embedding_params = sum(p.numel() for n, p in model.named_parameters() if n not in ['tok_embeddings.weight', 'output.weight', 'output.bias'])
            full_rank_non_embedding_params = sum(p.numel() for n, p in TitanGPT(model_args).named_parameters() if n not in ['tok_embeddings.weight', 'output.weight', 'output.bias'])
            reduction_non_embedding = (1 - low_rank_non_embedding_params / full_rank_non_embedding_params) * 100

            print(f"Low-rank non-embedding parameters: {low_rank_non_embedding_params/1e6:.2f}M")
            print(f"Low-rank model parameters: {low_rank_params/1e6:.2f}M")
            print(f"Parameter reduction: {reduction:.1f}%")
            print(f"Non-embedding parameter reduction: {reduction_non_embedding:.1f}%")

    # Apply self-guided decomposition if enabled
    elif args.self_guided:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Applying self-guided training with rank_ratio={args.low_rank_ratio}")
            print(f"Phase 1 (guided): steps 0 to {args.guided_steps} ({args.guided_steps/args.total_steps*100:.1f}%)")
            print(f"Phase 2 (low-rank): steps {args.guided_steps} to {args.total_steps}")
            print(f"Reduce FLOPs: {args.reduce_flop}")
            print(f"Warmup steps: {args.sg_warmup_steps}")
            print(f"{'='*60}\n")
            original_params = sum(p.numel() for p in model.parameters())
            print(f"Original model parameters: {original_params/1e6:.2f}M")

        model = replace_linear_with_selfguided(
            model,
            rank_ratio=args.low_rank_ratio,
            method="svd",
            exclude_modules=args.exclude_modules,
            reduce_flop=args.reduce_flop
        )
        model = model.to(device)

        if rank == 0:
            sg_params = sum(p.numel() for p in model.parameters())
            guide_params = sum(p.numel() for n, p in model.named_parameters() if 'guide_linear' in n)
            lowrank_params = sum(p.numel() for n, p in model.named_parameters() if 'lr1' in n or 'lr2' in n)

            print(f"\nSelf-guided model parameters: {sg_params/1e6:.2f}M")
            print(f"  Guide layer: {guide_params/1e6:.2f}M")
            print(f"  Low-rank (U+V): {lowrank_params/1e6:.2f}M")
            print(f"  Overhead: {(sg_params - original_params)/1e6:.2f}M (+{(sg_params/original_params - 1)*100:.1f}%)\n")

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        total_non_embedding_parameters = sum(p.numel() for n, p in model.named_parameters() if n not in ['tok_embeddings.weight', 'output.weight', 'output.bias'])
        print(f"Final model parameters: {total_params/1e6:.2f}M")
        print(f"Final non-embedding parameters: {total_non_embedding_parameters/1e6:.2f}M")

    # Broadcast model parameters and buffers from rank 0 to all workers
    broadcast_model_parameters(model, rank, src=0)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Generate block selection matrix for stochastic depth
    block_selection_matrix = param_sharing = None

    if args.stochastic_depth:
        if rank == 0:
            # Generate block selection matrix across VIRTUAL workers only on rank 0
            block_selection_matrix = generate_spread_with_min_col_constraint(
                virtual_world_size, args.num_layers, args.num_overlaps, args.block_selection_seed
            ).to(device)

            print(f"Block selection matrix:\n{block_selection_matrix}")

        # Broadcast block selection matrix to all ranks
        if rank != 0:
            block_selection_matrix = torch.zeros((virtual_world_size, args.num_layers), dtype=torch.int, device=device)

        dist.broadcast(block_selection_matrix, src=0)

        # Prepare a mutable mask tensor that hooks will read; we will overwrite it per virtual worker
        # Initialize with the first local virtual worker's mask
        worker_mask = block_selection_matrix[first_virtual_worker_index].clone()
        print(f"Rank {rank}: initial virtual worker {first_virtual_worker_index} mask: {worker_mask}")

        if args.stochastic_depth_mode == 'backward_hook':
            # Register backward hooks for stochastic depth
            register_stochastic_depth_hooks(model, worker_mask)
        elif args.stochastic_depth_mode == 'backward_nograd':
            # Apply mask with backward no-grad behavior (identity gradients)
            model.set_mask_with_mode(worker_mask, 'backward_nograd')
        else:
            # Apply mask for forward skipping
            model.set_mask_from_vector(worker_mask)

        # Store original mask for rank 0 (for evaluation with forward_skip)
        original_mask = None
        if rank == 0 and args.stochastic_depth_mode in ['forward_skip', 'backward_nograd']:
            original_mask = worker_mask.clone()

        # Debug: Print which layers each worker has
        if rank == 0:
            print(f"\n=== Worker Layer Assignment (Mode: {args.stochastic_depth_mode}) ===")
            for worker_idx in range(virtual_world_size):
                worker_layers = torch.where(block_selection_matrix[worker_idx] == 1)[0].tolist()
                print(f"Worker {worker_idx}: layers {worker_layers} (count: {len(worker_layers)})")
            print("===============================\n")

        # Get parameter sharing information (only once, based on VIRTUAL workers)
        param_sharing = get_parameter_sharing_info(model, block_selection_matrix, device)

        # Build private parameter ownership per VIRTUAL worker (params not in shared map)
        all_param_names = [name for name, _ in model.named_parameters()]
        shared_param_names = set(param_sharing.keys())

        def extract_layer_index_from_param_name(param_name: str):
            if '.layers.' in param_name:
                try:
                    return int(param_name.split('.layers.')[1].split('.')[0])
                except Exception:
                    return None
            return None

        private_params_by_virtual_worker = {vw: [] for vw in range(virtual_world_size)}
        for pn in all_param_names:
            if pn in shared_param_names:
                continue
            layer_idx = extract_layer_index_from_param_name(pn)
            if layer_idx is None:
                # Non-layer params not marked as shared are rare; default to shared
                shared_param_names.add(pn)
                continue
            owners = torch.where(block_selection_matrix[:, layer_idx] == 1)[0].tolist()
            if len(owners) == 1:
                private_params_by_virtual_worker[owners[0]].append(pn)
            else:
                # If multiple owners, treat as shared
                shared_param_names.add(pn)
    else:
        # No stochastic depth: all parameters are shared; no private params
        all_param_names = [name for name, _ in model.named_parameters()]
        shared_param_names = set(all_param_names)
        private_params_by_virtual_worker = {vw: [] for vw in range(virtual_world_size)}

    # Create per-virtual-worker dataloaders for training; validation stays on rank 0
    # To avoid GPU OOM when simulating multiple virtual workers on one GPU, keep datasets on CPU
    dataset_device = 'cpu' if virtual_workers_per_gpu > 1 else f"cuda:{local_rank}"
    dataset_device = 'cpu' # Always use CPU for datasets to minimize GPU memory usage

    virtual_train_iterators = {}
    validation_loader = None
    for vw in local_virtual_worker_indices:
        train_loader_vw, val_loader_vw = create_dataloaders(args, vw, virtual_world_size, device=dataset_device)
        virtual_train_iterators[vw] = iter(train_loader_vw)
        # Capture validation_loader from the first virtual worker (only created on rank 0)
        if validation_loader is None and val_loader_vw is not None:
            validation_loader = val_loader_vw

    # Create optimizers: one for shared params, and one per local VIRTUAL worker for private params
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    def build_adamw_optim_groups(param_list, param_names=None):
        decay_params = [p for p in param_list if p.dim() >= 2]
        nodecay_params = [p for p in param_list if p.dim() < 2]
        return [
            {'params': decay_params, 'weight_decay': args.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

    def build_spectral_lr_muon_groups(param_list, param_names):
        """Build Muon optimizer groups with separate groups for each A/B parameter for spectral LR scaling."""
        # Separate A/B parameters from other parameters
        lowrank_params = {}  # param_name -> param
        hidden_weights = []  # Non-lowrank hidden layer 2D+ params
        hidden_gains_biases = []
        nonhidden_params = []

        for name, param in zip(param_names, param_list):
            if name.endswith('.A') or name.endswith('.B'):
                lowrank_params[name] = param
            else:
                # Check if parameter is from transformer layers
                is_layer_param = 'layers.' in name
                if is_layer_param:
                    if param.dim() >= 2:
                        hidden_weights.append(param)
                    else:
                        hidden_gains_biases.append(param)
                else:
                    nonhidden_params.append(param)

        # Create parameter groups
        param_groups = []

        # Group for non-lowrank hidden weights (use Muon)
        if hidden_weights:
            param_groups.append({
                'params': hidden_weights,
                'use_muon': True,
                'lr': args.max_lr,
                'weight_decay': args.weight_decay,
                'adjust_lr_fn': args.adjust_muon_lr,
                'is_lowrank': False
            })

        # Group for gains/biases and non-hidden params (use Adam)
        if hidden_gains_biases or nonhidden_params:
            param_groups.append({
                'params': hidden_gains_biases + nonhidden_params,
                'use_muon': False,
                'lr': args.max_lr,
                'betas': (args.adam_beta1, args.adam_beta2),
                'weight_decay': args.nh_weight_decay,
                'is_lowrank': False
            })

        # Create individual parameter groups for each A/B parameter (use Muon since they're 2D)
        # This allows us to set individual learning rates
        for name, param in lowrank_params.items():
            param_groups.append({
                'params': [param],
                'use_muon': True,
                'lr': args.max_lr,  # Will be scaled dynamically
                'weight_decay': 0.0,
                'adjust_lr_fn': args.adjust_muon_lr,
                'is_lowrank': True,
                'param_name': name
            })

        return param_groups

    def build_muon_optim_groups(param_list, param_names):
        """Build Muon optimizer groups: hidden weights (2D+) use Muon, others use Adam"""
        # Separate parameters from transformer layers vs embeddings/output
        hidden_weights = []
        hidden_gains_biases = []
        nonhidden_params = []
        for name, param in zip(param_names, param_list):
            # Check if parameter is from transformer layers
            is_layer_param = 'layers.' in name

            if is_layer_param:
                if param.dim() >= 2:
                    hidden_weights.append(param)
                else:
                    hidden_gains_biases.append(param)
            else:
                # Embeddings, norm, and output head parameters
                nonhidden_params.append(param)
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=args.max_lr, weight_decay=args.weight_decay,
                 adjust_lr_fn=args.adjust_muon_lr),
            dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
                 lr=args.max_lr, betas=(args.adam_beta1, args.adam_beta2),
                 weight_decay=args.weight_decay),
        ]
        return param_groups

    def create_optimizer(param_list, param_names=None, use_spectral_lr=False):
        """Create optimizer based on args.optimizer"""
        if use_spectral_lr and param_names is not None:
            # Use Muon with spectral LR scaling groups
            if not MUON_AVAILABLE:
                raise ImportError("Spectral LR scaling requires Muon optimizer. Install with: pip install muon")
            return MuonWithAuxAdam(build_spectral_lr_muon_groups(param_list, param_names))
        elif args.optimizer == 'muon':
            if param_names is None:
                raise ValueError("param_names required for Muon optimizer")
            return MuonWithAuxAdam(build_muon_optim_groups(param_list, param_names))
        else:  # adamw
            return torch.optim.AdamW(
                build_adamw_optim_groups(param_list, param_names),
                lr=args.max_lr,
                betas=(args.adam_beta1, args.adam_beta2)
            )

    # Shared params list
    shared_params_list = [param_dict[n] for n in param_dict.keys() if n in shared_param_names]
    shared_param_names_list = [n for n in param_dict.keys() if n in shared_param_names]
    shared_optimizer = create_optimizer(shared_params_list, shared_param_names_list, use_spectral_lr=args.spectral_lr_scaling)

    if rank == 0:
        print(f"\nOptimizer: {args.optimizer}")
        if args.optimizer == 'muon':
            print(f"  Learning rate: {args.max_lr}")
            print(f"  Aux Adam LR (gains/biases/embeddings): {args.max_lr}")
            print(f"  Adam betas: ({args.adam_beta1}, {args.adam_beta2})")
        else:
            print(f"  Learning rate: {args.max_lr}")
            print(f"  Adam betas: ({args.adam_beta1}, {args.adam_beta2})")
        print(f"  Weight decay: {args.weight_decay}")
        if args.spectral_lr_scaling:
            print(f"  Spectral LR scaling: ENABLED (A/B lr = base_lr / (spectral_norm(A) + spectral_norm(B)))")
        print()

    # Private params storage and per-virtual optimizers
    private_param_store = {}
    private_optimizers = {}
    for vw in local_virtual_worker_indices:
        private_names = private_params_by_virtual_worker.get(vw, [])
        private_params_list = [param_dict[n] for n in private_names]
        # Initialize storage with current weights
        private_param_store[vw] = {n: param_dict[n].data.clone() for n in private_names}
        if private_params_list:
            private_optimizers[vw] = create_optimizer(private_params_list, private_names)
        else:
            private_optimizers[vw] = None

    # Set up scheduler with warmup and decay
    if rank == 0:
        print(f"\nSetting up {args.scheduler} scheduler...")

    if args.scheduler == 'wsd':
        # WSD (Warmup-Stable-Decay) scheduler
        lr_lambda = wsd_schedule(
            n_iterations=args.lr_schedule_steps,
            final_lr_factor=0.001,  # Decay to 0.1 * max_lr
            n_warmup=args.warmup_steps,
            init_div_factor=10,  # Start from 0.1 * max_lr during warmup
            fract_decay=0.1,  # Use 10% of iterations for decay
            decay_type="linear",
            cooldown_start_lr_factor=1.0
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            shared_optimizer,
            lr_lambda=[lr_lambda] * len(shared_optimizer.param_groups)
        )
    elif args.scheduler == 'cosine':
        if args.optimizer == 'muon':
            # For Muon, we use LambdaLR to schedule both parameter groups independently
            def lr_lambda(step):
                if step < args.warmup_steps:
                    # Warmup: linear from 0.0 to 1.0
                    return step / args.warmup_steps
                else:
                    # Cosine decay from 1.0 to 0.0
                    progress = (step - args.warmup_steps) / (args.lr_schedule_steps - args.warmup_steps)
                    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                shared_optimizer,
                lr_lambda=[lr_lambda] * len(shared_optimizer.param_groups)
            )
        else:
            # AdamW scheduler
            def adamw_warmup_lambda(step):
                if args.warmup_steps == 0:
                    return 1.0
                progress = min(step / args.warmup_steps, 1.0)
                return args.warmup_start_factor + (
                    1.0 - args.warmup_start_factor
                ) * progress

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                shared_optimizer,
                lr_lambda=[adamw_warmup_lambda] * len(shared_optimizer.param_groups),
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                shared_optimizer,
                T_max=args.lr_schedule_steps - args.warmup_steps,  # Cosine decay for remaining steps
                eta_min=args.max_lr * args.min_lr_factor,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                shared_optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[args.warmup_steps],
            )

    # Initialize self-guided alpha scheduler if enabled
    alpha_scheduler = None
    if args.self_guided:
        alpha_scheduler = CosineTempDecay(
            guided_steps=args.guided_steps,
            warmup_steps=args.sg_warmup_steps,
            device=device
        )
        if rank == 0:
            print(f"\nSelf-guided alpha scheduler initialized:")
            print(f"  Guided steps: {args.guided_steps}")
            print(f"  Warmup steps: {args.sg_warmup_steps}")
            print(f"  Decay schedule: cosine 1.0 → 0.0\n")

    # Resume from checkpoint if specified
    start_step = 0
    checkpoint_metadata = None
    if args.resume_from:
        # All ranks must load the checkpoint to get model weights and optimizer state
        try:
            checkpoint_metadata = load_checkpoint(
                args.resume_from,
                model,
                shared_optimizer,
                private_optimizers,
                private_param_store,
                scheduler,
                device
            )
            if checkpoint_metadata is not None:
                start_step = checkpoint_metadata['start_step']
                if rank == 0:
                    print(f"✓ Successfully loaded checkpoint, resuming from step {start_step}")
            else:
                if rank == 0:
                    print(f"WARNING: Failed to load checkpoint from {args.resume_from}")
                    print("Starting training from scratch instead")
        except Exception as e:
            if rank == 0:
                print(f"ERROR: Exception while loading checkpoint: {e}")
                print("Starting training from scratch instead")
            checkpoint_metadata = None

        # Synchronize start_step across all ranks (in case of load failures)
        start_step_tensor = torch.tensor(start_step, device=device)
        dist.all_reduce(start_step_tensor, op=dist.ReduceOp.MAX)
        start_step = int(start_step_tensor.item())

        if rank == 0:
            if start_step > 0:
                print(f"All ranks resuming from step {start_step}")
            else:
                print(f"All ranks starting from scratch (checkpoint loading failed)")

    # Initialize wandb (only on rank 0)
    if rank == 0:
        run_name = args.run_name if args.run_name else f"gpt_subnet_overlap_{args.num_overlaps}"

        # Check if we should resume a specific wandb run from checkpoint
        wandb_run_id = None
        resume_mode = None
        if checkpoint_metadata is not None and checkpoint_metadata.get('wandb_run_id'):
            wandb_run_id = checkpoint_metadata['wandb_run_id']
            resume_mode = "must"
            print(f"Resuming wandb run: {wandb_run_id}")
        elif args.resume_from:
            resume_mode = "allow"

        # Initialize wandb with appropriate resume settings
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            group=run_name,
            id=wandb_run_id,
            resume=resume_mode
        )
        try:
            wandb.summary["total_params"] = total_params
            wandb.summary["non_embedding_params"] = total_non_embedding_parameters
        except Exception as e:
            print(f"Error updating wandb summary: {e}")

    # Calculate FLOPs analytically (works with flex_attention and low-rank layers)
    if rank == 0:
        print("\nCalculating FLOPs for forward pass...")

        # Calculate FFN hidden dimension (same logic as FeedForward class)
        ffn_hidden_dim = int(2 * args.hidden_size / 3) * 4
        if args.multiple_of:
            ffn_hidden_dim = args.multiple_of * ((ffn_hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # Calculate forward pass FLOPs
        flops_per_step_forward = calculate_transformer_flops(
            batch_size=args.micro_batch_size,
            seq_len=args.train_seq_len,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_hidden_size=ffn_hidden_dim,
            use_low_rank=args.low_rank,
            rank_ratio=args.low_rank_ratio if args.low_rank else 0.25,
            exclude_modules=args.exclude_modules if args.low_rank else [],
            disable_c=args.disable_c if args.low_rank else False,
            n_kv_heads=args.n_kv_heads,
        )

        # Estimate backward pass as 2x forward pass
        flops_per_step_backward = 2 * flops_per_step_forward
        flops_per_step_total = flops_per_step_forward + flops_per_step_backward

        # Convert to TFLOPs for readable output
        tflops_forward = flops_per_step_forward / 1e12
        tflops_backward = flops_per_step_backward / 1e12
        tflops_total = flops_per_step_total / 1e12

        print(f"Forward FLOPs per micro-batch: {tflops_forward:.3f} TFLOPs ({flops_per_step_forward:,})")
        print(f"Estimated Backward FLOPs per micro-batch: {tflops_backward:.3f} TFLOPs ({flops_per_step_backward:,})")
        print(f"Total FLOPs per micro-batch: {tflops_total:.3f} TFLOPs ({flops_per_step_total:,})")

        print()

    else:
        flops_per_step_forward = 0
        flops_per_step_backward = 0
        flops_per_step_total = 0

    # Broadcast FLOP measurements to all ranks
    flops_tensor = torch.tensor([flops_per_step_forward, flops_per_step_backward, flops_per_step_total], dtype=torch.float64, device=device)
    dist.broadcast(flops_tensor, src=0)
    flops_per_step_forward = int(flops_tensor[0].item())
    flops_per_step_backward = int(flops_tensor[1].item())
    flops_per_step_total = int(flops_tensor[2].item())

    # Training loop
    model.train()
    # Not used with virtual workers; we keep per-virtual iterators
    step = start_step

    # Token counting - restore from checkpoint if resuming
    if checkpoint_metadata:
        total_tokens_rank0 = checkpoint_metadata['total_tokens_rank0']
        total_tokens_world = checkpoint_metadata['total_tokens_world']
        total_flops = checkpoint_metadata['total_flops']
        total_flops_forward = checkpoint_metadata['total_flops_forward']
        total_flops_backward = checkpoint_metadata['total_flops_backward']
        best_val_loss = checkpoint_metadata.get('best_val_loss', float('inf'))
    else:
        total_tokens_rank0 = 0
        total_tokens_world = 0
        total_flops = 0
        total_flops_forward = 0
        total_flops_backward = 0
        best_val_loss = float('inf')

    last_log_time = time.time()
    last_log_tokens = total_tokens_world

    # Checkpoint timing
    last_checkpoint_time = time.time()
    checkpoint_interval_seconds = args.checkpoint_interval_hours * 3600  # Convert hours to seconds

    # Gradient accumulation setup (distribute global batch across VIRTUAL world)
    if args.micro_batch_size is None:
        args.micro_batch_size = max(1, args.batch_size // virtual_world_size)
    effective_per_step = args.micro_batch_size * virtual_world_size
    accumulation_steps = max(1, args.batch_size // effective_per_step)
    if rank == 0:
        if args.batch_size % effective_per_step != 0:
            print(f"[Warn] Global batch {args.batch_size} not divisible by micro_batch_size*virtual_world_size ({effective_per_step}). Using floor division: accumulation_steps={accumulation_steps}")
        print(f"Gradient accumulation: {accumulation_steps} micro-batches per update; micro_batch_size={args.micro_batch_size}, virtual_world_size={virtual_world_size}")

    print(f"Worker {rank} starting training...")
    if args.scheduler == 'wsd':
        print(f"Learning rate schedule: WSD (run for {args.total_steps} steps; schedule horizon {args.lr_schedule_steps}, warmup for {args.warmup_steps} steps [{args.warmup_ratio*100:.1f}% of schedule], stable phase, then decay to {0.1 * args.max_lr:.6f})")
    else:
        print(f"Learning rate schedule: {args.scheduler} (run for {args.total_steps} steps; schedule horizon {args.lr_schedule_steps}, warmup for {args.warmup_steps} steps [{args.warmup_ratio*100:.1f}% of schedule], then decay to {args.min_lr_factor * args.max_lr:.6f})")

    if rank == 0:
        print(f"\nCheckpointing configuration:")
        print(f"  Checkpoint directory: {args.checkpoint_dir}")
        print(f"  Checkpoint interval: {args.checkpoint_interval_hours} hours ({checkpoint_interval_seconds/60:.1f} minutes)")
        print(f"  Periodic step checkpoint interval: {args.checkpoint_interval_steps if args.checkpoint_interval_steps else 'disabled'}")
        print(f"  Keep latest periodic step checkpoints: {args.checkpoint_keep_latest_k if args.checkpoint_keep_latest_k >= 0 else 'all'}")
        print(f"  Save final checkpoint: {args.save_final_checkpoint}")
        print(f"  Save best checkpoint: {args.save_best_checkpoint}")
        if args.resume_from:
            print(f"  Resuming from: {args.resume_from}")
        print()

    while step < args.total_steps:
        # Reset shared grads only at the beginning of the global step
        for p in shared_params_list:
            if p.grad is not None:
                p.grad.zero_()

        accumulated_loss = 0

        # Iterate over local VIRTUAL workers sequentially
        for vw in local_virtual_worker_indices:
            # Load private params for this virtual worker into the live model
            private_names = private_params_by_virtual_worker.get(vw, [])
            for pn in private_names:
                param_dict[pn].data.copy_(private_param_store[vw][pn])

            # Set lr for private optimizer to match shared scheduler's current LR
            if private_optimizers[vw] is not None:
                current_lr = shared_optimizer.param_groups[0]['lr']
                for pg in private_optimizers[vw].param_groups:
                    pg['lr'] = current_lr

            # Zero grads for private params
            if private_optimizers[vw] is not None:
                private_optimizers[vw].zero_grad(set_to_none=False)
            for pn in private_names:
                if param_dict[pn].grad is not None:
                    param_dict[pn].grad.zero_()

            # Micro-batch accumulation for this virtual worker
            for micro_step in range(accumulation_steps):
                # Get batch for this virtual worker
                try:
                    batch = next(virtual_train_iterators[vw])
                except StopIteration:
                    # Recreate iterator if exhausted
                    train_loader_vw, _ = create_dataloaders(
                        args,
                        vw,
                        virtual_world_size,
                        device=dataset_device,
                    )
                    virtual_train_iterators[vw] = iter(train_loader_vw)
                    batch = next(virtual_train_iterators[vw])

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # Count tokens for logging (rank 0 only)
                if rank == 0:
                    batch_tokens = labels.numel()
                    total_tokens_rank0 += batch_tokens
                    total_tokens_world = total_tokens_rank0 * virtual_world_size

                # Activate the correct mask for this virtual worker
                if args.stochastic_depth:
                    current_mask = block_selection_matrix[vw]
                    if args.stochastic_depth_mode == 'backward_hook':
                        worker_mask.copy_(current_mask)
                    elif args.stochastic_depth_mode == 'backward_nograd':
                        model.set_mask_with_mode(current_mask, 'backward_nograd')
                    else:
                        model.set_mask_from_vector(current_mask)

                # Forward pass with optional bf16 autocast
                if args.bf16:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = model(input_ids, input_batch=input_ids)
                        unscaled_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    logits = model(input_ids, input_batch=input_ids)
                    unscaled_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                loss = unscaled_loss / accumulation_steps
                accumulated_loss += loss.item()
                loss.backward()

            # Accumulate FLOPs for this virtual worker (only on rank 0)
            if rank == 0:
                total_flops_forward += flops_per_step_forward * accumulation_steps
                total_flops_backward += flops_per_step_backward * accumulation_steps
                total_flops += flops_per_step_total * accumulation_steps

            # Step private optimizer for this virtual worker only (no communication)
            if private_optimizers[vw] is not None:
                torch.nn.utils.clip_grad_norm_([param_dict[n] for n in private_names], max_norm=1.0)
                private_optimizers[vw].step()
                # Save updated private params back to the per-virtual store
                for pn in private_names:
                    private_param_store[vw][pn].copy_(param_dict[pn].data)
                # Clear grads for private params to avoid interfering with shared step
                for pn in private_names:
                    if param_dict[pn].grad is not None:
                        param_dict[pn].grad.zero_()

        # Communication step for shared params across GPUs
        if args.stochastic_depth and param_sharing is not None:
            # Selective all-reduce for shared parameters only, based on VIRTUAL sharing groups
            for name, param in model.named_parameters():
                if name in param_sharing and name in param_dict:
                    sharing_info = param_sharing[name]
                    sharing_workers = set(sharing_info['sharing_workers'])
                    sharing_count = sharing_info['sharing_count']

                    # Determine if this rank owns any virtual worker that participates
                    participates = any(vw in sharing_workers for vw in local_virtual_worker_indices)

                    if param.grad is None:
                        grad_to_reduce = torch.zeros_like(param)
                    else:
                        grad_to_reduce = param.grad.clone()

                    if not participates:
                        grad_to_reduce.zero_()

                    dist.all_reduce(grad_to_reduce, op=dist.ReduceOp.SUM)

                    if participates:
                        param.grad = grad_to_reduce / sharing_count
                    else:
                        if param.grad is not None:
                            param.grad.zero_()
        else:
            # Normal DDP: all-reduce all parameters across VIRTUAL world (sum locally across virtuals, then across GPUs)
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= virtual_world_size
                else:
                    grad = torch.zeros_like(param)
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM)

        # Low-rank communication for A/B parameters if enabled
        if args.low_rank and args.communicate_lowrank_freq is not None:
            if step % args.communicate_lowrank_freq == 0:
                communicate_lowrank_gradients(model, virtual_world_size)

        # Calculate gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(shared_params_list, max_norm=1.0)

        # Apply spectral LR scaling and/or spectral weight decay if enabled
        if args.spectral_lr_scaling or args.spectral_weight_decay > 0:
            # Get current base LR from scheduler (only needed if spectral_lr_scaling)
            if args.spectral_lr_scaling:
                base_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else args.max_lr

            # Compute spectral norm scaling factors and regularization gradients
            spectral_scaling, regularization_grads = get_lowrank_spectral_norm_scaling(
                model,
                spectral_weight_decay=args.spectral_weight_decay,
                swd_type=args.swd_type,
                spectral_lr_scaling_offset=args.spectral_lr_scaling_offset,
            )

            # Update LR for each low-rank parameter group (only if spectral_lr_scaling)
            if args.spectral_lr_scaling:
                for pg in shared_optimizer.param_groups:
                    if pg.get('is_lowrank', False) and 'param_name' in pg:
                        param_name = pg['param_name']
                        if param_name in spectral_scaling:
                            # Scale LR by 1/(spectral_norm(A) + spectral_norm(B))
                            pg['lr'] = base_lr * spectral_scaling[param_name]

            # Apply decoupled spectral weight decay if enabled
            with torch.no_grad():
                if args.spectral_weight_decay > 0 and regularization_grads:
                    for pg in shared_optimizer.param_groups:
                        if pg.get('is_lowrank', False) and 'param_name' in pg:
                            param_name = pg['param_name']
                            if param_name in regularization_grads:
                                # Apply decoupled regularization: param -= lr * regularization_grad
                                for param in pg['params']:
                                    param.sub_(base_lr * regularization_grads[param_name])

        # Apply decoupled Frobenius decay if enabled
        if args.low_rank and args.frobenius_coef > 0:
            frob_regularization_grads = get_frobenius_regularization_grads(
                model,
                coef=args.frobenius_coef
            )
            with torch.no_grad():
                for pg in shared_optimizer.param_groups:
                    if pg.get('is_lowrank', False) and 'param_name' in pg:
                        param_name = pg['param_name']
                        if param_name in frob_regularization_grads:
                            # Apply decoupled Frobenius regularization: param -= lr * frob_grad
                            for param in pg['params']:
                                param.sub_(
                                    base_lr * frob_regularization_grads[param_name]
                                )

        # Do weight decay with base lr for low-rank params if using spectral LR scaling
        if args.spectral_lr_scaling and args.weight_decay > 0:
            with torch.no_grad():
                for pg in shared_optimizer.param_groups:
                    if pg.get('is_lowrank', False) and 'param_name' in pg:
                        for param in pg['params']:
                            param.mul_(1 - args.weight_decay * base_lr)

        # Step shared optimizer once per global step
        shared_optimizer.step()

        scheduler.step()

        # Update self-guided alpha and handle phase transition
        if args.self_guided and alpha_scheduler is not None:
            current_alpha = alpha_scheduler.get_alpha(step)

            # Update alpha for all self-guided layers
            for module in model.modules():
                if isinstance(module, LowRankWithSelfGuided):
                    module.update_alpha(current_alpha)

            # Check for phase transition
            if step == args.guided_steps:
                if rank == 0:
                    print(f"\n{'='*80}")
                    print(f"PHASE TRANSITION at step {step}")
                    print(f"Disabling self-guided mode (α → 0)")
                    print(f"Switching to pure low-rank training")
                    print(f"{'='*80}\n")

                # Disable self-guided for all layers
                for module in model.modules():
                    if isinstance(module, LowRankWithSelfGuided):
                        module.disable_self_guided()

                # Synchronize phase transition across all ranks
                dist.barrier()

        # Track stable rank, spectral norm, and condition number if enabled (only on rank 0)
        if args.track_stable_rank and rank == 0:
            target_layers = [4, 7]  # 0-indexed: layer 4 and layer 7

            for layer_idx in target_layers:
                if layer_idx < args.num_layers:
                    layer_name = str(layer_idx)
                    layer = model.layers[layer_name]

                    # Track Attention layer weights (wq, wk, wv, wo)
                    for attn_weight_name in ['wq', 'wk', 'wv', 'wo']:
                        if hasattr(layer.attention, attn_weight_name):
                            attn_module = getattr(layer.attention, attn_weight_name)

                            # Check if this is a low-rank module
                            if args.low_rank and hasattr(attn_module, 'A'):
                                # Track low-rank components
                                with torch.no_grad():
                                    # Spectral norm of A (convert to float for computation)
                                    A_spectral = torch.linalg.matrix_norm(attn_module.A.float(), ord=2).item()
                                    # Spectral norm of B (convert to float for computation)
                                    B_spectral = torch.linalg.matrix_norm(attn_module.B.float(), ord=2).item()

                                    log_dict = {
                                        f'lowrank_spectral_norm/layer_{layer_idx}_attention_{attn_weight_name}_A': A_spectral,
                                        f'lowrank_spectral_norm/layer_{layer_idx}_attention_{attn_weight_name}_B': B_spectral,
                                        'step': step
                                    }

                                    # Spectral norm of C (if not disabled)
                                    if not args.disable_c and hasattr(attn_module.C, 'requires_grad') and attn_module.C.requires_grad:
                                        C_spectral = torch.linalg.matrix_norm(attn_module.C.float(), ord=2).item()
                                        log_dict[f'lowrank_spectral_norm/layer_{layer_idx}_attention_{attn_weight_name}_C'] = C_spectral

                                    wandb.log(log_dict)

                            # Track full weight matrix metrics (works for both regular and low-rank)
                            weight = attn_module.weight
                            if weight.dim() == 2:
                                metrics = compute_matrix_metrics(weight)
                                wandb.log({
                                    f'stable_rank/layer_{layer_idx}_attention_{attn_weight_name}': metrics['stable_rank'],
                                    f'spectral_norm/layer_{layer_idx}_attention_{attn_weight_name}': metrics['spectral_norm'],
                                    f'condition_number/layer_{layer_idx}_attention_{attn_weight_name}': metrics['condition_number'],
                                    'step': step
                                })

                    # Track FeedForward layer weights (w1, w2, w3)
                    for ff_weight_name in ['w1', 'w2', 'w3']:
                        if hasattr(layer.feed_forward, ff_weight_name):
                            ff_module = getattr(layer.feed_forward, ff_weight_name)

                            # Check if this is a low-rank module
                            if args.low_rank and hasattr(ff_module, 'A'):
                                # Track low-rank components
                                with torch.no_grad():
                                    # Spectral norm of A (convert to float for computation)
                                    A_spectral = torch.linalg.matrix_norm(ff_module.A.float(), ord=2).item()
                                    # Spectral norm of B (convert to float for computation)
                                    B_spectral = torch.linalg.matrix_norm(ff_module.B.float(), ord=2).item()

                                    log_dict = {
                                        f'lowrank_spectral_norm/layer_{layer_idx}_ff_{ff_weight_name}_A': A_spectral,
                                        f'lowrank_spectral_norm/layer_{layer_idx}_ff_{ff_weight_name}_B': B_spectral,
                                        'step': step
                                    }

                                    # Spectral norm of C (if not disabled)
                                    if not args.disable_c and hasattr(ff_module.C, 'requires_grad') and ff_module.C.requires_grad:
                                        C_spectral = torch.linalg.matrix_norm(ff_module.C.float(), ord=2).item()
                                        log_dict[f'lowrank_spectral_norm/layer_{layer_idx}_ff_{ff_weight_name}_C'] = C_spectral

                                    wandb.log(log_dict)

                            # Track full weight matrix metrics (works for both regular and low-rank)
                            weight = ff_module.weight
                            if weight.dim() == 2:
                                metrics = compute_matrix_metrics(weight)
                                wandb.log({
                                    f'stable_rank/layer_{layer_idx}_ff_{ff_weight_name}': metrics['stable_rank'],
                                    f'spectral_norm/layer_{layer_idx}_ff_{ff_weight_name}': metrics['spectral_norm'],
                                    f'condition_number/layer_{layer_idx}_ff_{ff_weight_name}': metrics['condition_number'],
                                    'step': step
                                })

        # Log training metrics every step (only on rank 0)
        if rank == 0:
            lr = shared_optimizer.param_groups[0]['lr']
            # Calculate the average loss for all ranks before printing
            avg_train_loss = accumulated_loss
            log_dict = {
                'step': step,
                'train_loss': avg_train_loss,
                'learning_rate': lr,
                'grad_norm': grad_norm.item(),
                'total_tokens_rank0': total_tokens_rank0,
                'total_tokens_world': total_tokens_world,
                'total_tflops': total_flops / 1e12,
                'total_tflops_forward': total_flops_forward / 1e12,
                'total_tflops_backward': total_flops_backward / 1e12,
            }

            # Add self-guided metrics if enabled
            if args.self_guided and alpha_scheduler is not None:
                log_dict['self_guided_alpha'] = alpha_scheduler.get_alpha(step).item()
                log_dict['self_guided_phase'] = 1 if step < args.guided_steps else 2

            wandb.log(log_dict)

        # Logging and evaluation are separate: train loss can be printed every
        # step without forcing validation.
        should_log = args.log_interval > 0 and step % args.log_interval == 0
        should_eval = args.eval_interval > 0 and step > 0 and step % args.eval_interval == 0
        if should_log or should_eval:
            lr = shared_optimizer.param_groups[0]['lr']

            # Calculate the average loss for all ranks before printing
            avg_train_loss = accumulated_loss 

            # Calculate tokens/sec (global, all workers)
            current_time = time.time()
            elapsed_time = current_time - last_log_time
            tokens_since_last = total_tokens_world - last_log_tokens
            tokens_per_sec = tokens_since_last / elapsed_time if elapsed_time > 0 else 0.0
            last_log_time = current_time
            last_log_tokens = total_tokens_world

            eval_results = None
            if rank == 0:
                if should_eval:
                    if args.stochastic_depth and args.stochastic_depth_mode in ['forward_skip', 'backward_nograd']:
                        # Temporarily unmask rank 0 for evaluation (all layers active)
                        model.set_mask_from_vector(torch.ones(args.num_layers, dtype=torch.int, device=device))

                        eval_results = evaluate_model(model, validation_loader, criterion, device, args)

                        # Restore original mask
                        if args.stochastic_depth_mode == 'backward_nograd':
                            model.set_mask_with_mode(original_mask, 'backward_nograd')
                        else:
                            model.set_mask_from_vector(original_mask)
                    else:
                        # For 'backward_hook', no change is needed as hooks are not called in no_grad()
                        eval_results = evaluate_model(model, validation_loader, criterion, device, args)

                    # Log validation metrics
                    wandb.log({
                        'step': step,
                        'val_loss': eval_results['val_loss'],
                        'val_perplexity': eval_results['val_perplexity'],
                        'val_token_accuracy': eval_results['val_token_accuracy'],
                        'tokens_per_sec': tokens_per_sec,
                    })

                    # Update best validation loss
                    if eval_results['val_loss'] < best_val_loss:
                        best_val_loss = eval_results['val_loss']
                        try:
                            wandb.summary['best_val_loss'] = best_val_loss
                        except Exception as e:
                            print(f"Warning: Failed to update wandb.summary['best_val_loss']: {e}")
                        wandb.log({'best_val_loss': best_val_loss, 'step': step})

                        # Save best checkpoint (only if enabled)
                        if args.save_best_checkpoint:
                            print(f"\n{'='*80}")
                            print(f"New best validation loss: {best_val_loss:.4f} at step {step}")
                            print(f"Saving best checkpoint...")
                            print(f"{'='*80}\n")

                            best_checkpoint_path = save_checkpoint(
                                checkpoint_dir=args.checkpoint_dir,
                                model=model,
                                model_args=model_args,
                                shared_optimizer=shared_optimizer,
                                private_optimizers=private_optimizers,
                                private_param_store=private_param_store,
                                scheduler=scheduler,
                                step=step,
                                total_tokens_rank0=total_tokens_rank0,
                                total_tokens_world=total_tokens_world,
                                total_flops=total_flops,
                                total_flops_forward=total_flops_forward,
                                total_flops_backward=total_flops_backward,
                                args=args,
                                best_val_loss=best_val_loss,
                                is_final=False,
                                is_best=True,
                                alpha_scheduler=alpha_scheduler
                            )
                        else:
                            print(
                                f"New best validation loss: {best_val_loss:.4f} at step {step} (checkpoint saving disabled)"
                            )

                if should_log:
                    if eval_results is None:
                        print(
                            f"Step {step}, Rank {rank}: Loss={avg_train_loss:.4f}, "
                            f"LR={lr:.6f}, Tokens/sec={tokens_per_sec:.2f}"
                        )
                    else:
                        print(f"Step {step}, Rank {rank}: Loss={avg_train_loss:.4f}, LR={lr:.6f}, "
                            f"Val Loss={eval_results['val_loss']:.4f}, "
                            f"Perplexity={eval_results['val_perplexity']:.2f}, "
                            f"Token Acc={eval_results['val_token_accuracy']:.4f}, "
                            f"Tokens/sec={tokens_per_sec:.2f}")
                    print(f"  Total TFLOPs: {total_flops/1e12:.2f} (Forward: {total_flops_forward/1e12:.2f}, Backward: {total_flops_backward/1e12:.2f})")
            else:
                if should_log:
                    print(f"Step {step}, Rank {rank}: Loss={avg_train_loss:.4f}, LR={lr:.6f}")

            model.train()
            if should_eval:
                dist.barrier()

        if args.checkpoint_interval_steps > 0 and (step + 1) % args.checkpoint_interval_steps == 0:
            if rank == 0:
                print(f"\n{'='*80}")
                print(
                    "Saving periodic step checkpoint "
                    f"after {step + 1} completed optimizer steps"
                )
                print(f"{'='*80}\n")

                save_checkpoint(
                    checkpoint_dir=args.checkpoint_dir,
                    model=model,
                    model_args=model_args,
                    shared_optimizer=shared_optimizer,
                    private_optimizers=private_optimizers,
                    private_param_store=private_param_store,
                    scheduler=scheduler,
                    step=step,
                    total_tokens_rank0=total_tokens_rank0,
                    total_tokens_world=total_tokens_world,
                    total_flops=total_flops,
                    total_flops_forward=total_flops_forward,
                    total_flops_backward=total_flops_backward,
                    args=args,
                    best_val_loss=best_val_loss,
                    is_final=False,
                    alpha_scheduler=alpha_scheduler
                )
                prune_step_checkpoints(args.checkpoint_dir, args.checkpoint_keep_latest_k)

            dist.barrier()

        # Check if it's time to save a checkpoint (every N hours)
        current_time = time.time()
        time_since_last_checkpoint = current_time - last_checkpoint_time

        if time_since_last_checkpoint >= checkpoint_interval_seconds:
            if rank == 0:
                print(f"\n{'='*80}")
                print(f"Saving checkpoint (elapsed time: {time_since_last_checkpoint/3600:.2f} hours)")
                print(f"{'='*80}\n")

                save_checkpoint(
                    checkpoint_dir=args.checkpoint_dir,
                    model=model,
                    model_args=model_args,
                    shared_optimizer=shared_optimizer,
                    private_optimizers=private_optimizers,
                    private_param_store=private_param_store,
                    scheduler=scheduler,
                    step=step,
                    total_tokens_rank0=total_tokens_rank0,
                    total_tokens_world=total_tokens_world,
                    total_flops=total_flops,
                    total_flops_forward=total_flops_forward,
                    total_flops_backward=total_flops_backward,
                    args=args,
                    best_val_loss=best_val_loss,
                    is_final=False,
                    alpha_scheduler=alpha_scheduler
                )

                print(f"\n{'='*80}")
                print(f"Checkpoint saved. Exiting job for scheduled chaining.")
                print(f"Resume from this checkpoint with: --resume_from <checkpoint_path>")
                print(f"{'='*80}\n")

            # Synchronize all ranks before exiting
            dist.barrier()

            if rank == 0:
                print(f"Rank {rank}: Destroying process group and exiting at step {step}")

            # Destroy process group and exit immediately (no final evaluation or checkpoint)
            dist.destroy_process_group()
            return

        step += 1

    # Final evaluation and checkpoint saving (only when training completes normally)
    # Note: If checkpoint interval is reached, we exit early above (no final eval/checkpoint)
    if args.skip_final_eval:
        if rank == 0:
            print("Skipping final validation evaluation")
            print(f"Total tokens processed - Rank 0: {total_tokens_rank0:,}, World: {total_tokens_world:,}")
            print(f"Total TFLOPs - Total: {total_flops/1e12:.2f}, Forward: {total_flops_forward/1e12:.2f}, Backward: {total_flops_backward/1e12:.2f}")
    else:
        # Final evaluation (only on rank 0)
        if args.stochastic_depth:
            if rank == 0:
                if args.stochastic_depth_mode == 'forward_skip':
                    # Temporarily unmask rank 0 for final evaluation
                    model.set_mask_from_vector(torch.ones(args.num_layers, dtype=torch.int, device=device))

                final_eval = evaluate_model(model, validation_loader, criterion, device, args)

                if args.stochastic_depth_mode == 'forward_skip':
                    # Restore original mask
                    model.set_mask_from_vector(original_mask)

                wandb.log({
                    'final_val_loss': final_eval['val_loss'],
                    'final_val_perplexity': final_eval['val_perplexity'],
                    'final_val_token_accuracy': final_eval['val_token_accuracy']
                })
                print(f"Final results: {final_eval}")
                print(f"Total tokens processed - Rank 0: {total_tokens_rank0:,}, World: {total_tokens_world:,}")
                print(f"Total TFLOPs - Total: {total_flops/1e12:.2f}, Forward: {total_flops_forward/1e12:.2f}, Backward: {total_flops_backward/1e12:.2f}")

            # Synchronize: all workers wait for rank 0 to finish final evaluation
            dist.barrier()
        else:
            if rank == 0:
                final_eval = evaluate_model(model, validation_loader, criterion, device, args)
                wandb.log({
                    'final_val_loss': final_eval['val_loss'],
                    'final_val_perplexity': final_eval['val_perplexity'],
                    'final_val_token_accuracy': final_eval['val_token_accuracy']
                })
                print(f"Final results: {final_eval}")
                print(f"Total tokens processed - Rank 0: {total_tokens_rank0:,}, World: {total_tokens_world:,}")
                print(f"Total TFLOPs - Total: {total_flops/1e12:.2f}, Forward: {total_flops_forward/1e12:.2f}, Backward: {total_flops_backward/1e12:.2f}")

    # Save final checkpoint (only on rank 0)
    if rank == 0 and args.save_final_checkpoint:
        print(f"\n{'='*80}")
        print("Saving final checkpoint...")
        print(f"{'='*80}\n")

        save_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            model=model,
            model_args=model_args,
            shared_optimizer=shared_optimizer,
            private_optimizers=private_optimizers,
            private_param_store=private_param_store,
            scheduler=scheduler,
            step=step - 1,  # Use last completed step
            total_tokens_rank0=total_tokens_rank0,
            total_tokens_world=total_tokens_world,
            total_flops=total_flops,
            total_flops_forward=total_flops_forward,
            total_flops_backward=total_flops_backward,
            args=args,
            best_val_loss=best_val_loss,
            is_final=True,
            alpha_scheduler=alpha_scheduler
        )

        print("\nTraining complete! Final checkpoint saved.")

    # Synchronize all ranks before cleanup
    dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    main() 
