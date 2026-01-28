#!/usr/bin/env python3
"""
Simple example script to run the simplified GPT subnet training using torchrun.

This script demonstrates how to use the simplified GPT training code
with basic configuration for subnet training.
"""

import subprocess
import sys
import os
import traceback
import argparse
import copy

def excepthook(type, value, tb):
    traceback.print_exception(type, value, tb)
    sys.__excepthook__(type, value, tb)
sys.excepthook = excepthook

def get_base_config(num_gpus: int = 7, virtual_workers_per_gpu: int = 1):
    """Returns the base configuration for a single training run."""
    return [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "simple_gpt_training.py",
        
        # Training configuration
        "--batch_size", "32",
        "--micro_batch_size", "16",
        "--max_lr", "0.0008",
        "--weight_decay", "0.1",
        "--total_steps", "3750",
        "--warmup_steps", "370",
        "--log_interval", "200",
        "--max_val_samples", "100",
        
        # Model configuration
        "--hidden_size", "768",
        "--num_heads", "12", 
        "--num_layers", "12",
        "--vocab_size", "32000",
        "--max_position_embeddings", "2048",
        "--train_seq_len", "2048",
        "--val_seq_len", "2048",
        "--embed_dropout", "0.2",
        "--resid_dropout", "0.2",
        "--use_flex_attn",
        "--bf16",
        "--virtual_workers_per_gpu", str(virtual_workers_per_gpu),
       # "--rescale_lr",
        "--n_kv_heads", "12",
        "--multiple_of", "256",
        
        # Data paths
        "--train_files", "fineweb10B/fineweb_train_*.bin",
        "--val_files", "fineweb10B/fineweb_val_*.bin",
        
        # Wandb configuration
        "--wandb_project", "gpt_subnet_training",
        "--wandb_entity", "your_username",
    ]

def main():
    parser = argparse.ArgumentParser(description="Run sequential training experiments.")
    parser.add_argument(
        "--no-flop-matched", 
        dest="flop_matched",
        action="store_false",
        help="Disable the default FLOP-matched training schedule adjustment for stochastic depth runs."
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=7,
        help="Number of GPUs to use for torchrun (nproc_per_node)."
    )
    parser.add_argument(
        "--virtual_workers_per_gpu",
        type=int,
        default=2,
        help="Number of virtual workers to simulate per GPU."
    )
    args = parser.parse_args()

    # --- Define Your Experiments Here ---
    experiments = [
        # Baseline (no stochastic depth)
       # {"stochastic_depth": False},

        # Stochastic depth with forward_skip mode
       # {"stochastic_depth": True, "num_overlaps": 10, "mode": "backward_skip"},
       # {"stochastic_depth": True, "num_overlaps": 8, "mode": "backward_hook"},
    #   {"stochastic_depth": True, "num_overlaps": 20, "mode": "forward_skip"},
        #{"stochastic_depth": True, "num_overlaps": 12, "mode": "forward_skip"},

        {"stochastic_depth": True, "num_overlaps": 10, "mode": "backward_nograd"},
        {"stochastic_depth": True, "num_overlaps": 8, "mode": "backward_nograd"},
        {"stochastic_depth": True, "num_overlaps": 6, "mode": "backward_nograd"},
        #{"stochastic_depth": True, "num_overlaps": 6, "mode": "forward_skip"},

        #{"stochastic_depth": True, "num_overlaps": 10, "mode": "forward_skip"},
        #{"stochastic_depth": True, "num_overlaps": 8, "mode": "forward_skip"},
        # Stochastic depth with backward_hook mode
        #{"stochastic_depth": False, "num_layers": 8, "num_overlaps": 8, "total_steps": 5625, "warmup_steps": 562},

        #{"stochastic_depth": False, "num_layers": 10, "num_overlaps": 10, "total_steps": 4500, "warmup_steps": 450},

        # Low-rank training experiments
        # {"low_rank": True, "low_rank_ratio": 0.25, "frobenius_coef": 0.01},
        # {"low_rank": True, "low_rank_ratio": 0.5, "disable_c": True},
        # {"stochastic_depth": True, "num_overlaps": 8, "mode": "backward_nograd", "low_rank": True, "low_rank_ratio": 0.25},
    ]
    
    for i, exp_config in enumerate(experiments):
        print(f"\n--- Running Experiment {i+1}/{len(experiments)} ---")
        print(f"Config: {exp_config}")

        cmd = get_base_config(num_gpus=args.gpus, virtual_workers_per_gpu=args.virtual_workers_per_gpu)
        run_name_parts = []

        # --- Update parameters from experiment config ---
        if "num_layers" in exp_config:
            cmd[cmd.index("--num_layers") + 1] = str(exp_config["num_layers"])
        if "total_steps" in exp_config:
            cmd[cmd.index("--total_steps") + 1] = str(exp_config["total_steps"])
        if "warmup_steps" in exp_config:
            cmd[cmd.index("--warmup_steps") + 1] = str(exp_config["warmup_steps"])

        # --- Base model parameters ---
        num_layers = int(cmd[cmd.index("--num_layers") + 1])
        base_total_steps = int(cmd[cmd.index("--total_steps") + 1])
        base_warmup_steps = int(cmd[cmd.index("--warmup_steps") + 1])
        
        # --- Configure Stochastic Depth ---
        if exp_config.get("stochastic_depth"):
            num_overlaps = exp_config["num_overlaps"]
            mode = exp_config["mode"]
            
            cmd.extend([
                "--stochastic_depth",
                "--stochastic_depth_mode", mode,
                "--num_overlaps", str(num_overlaps),
                "--block_selection_seed", "42",
            ])
            
            run_name_parts.append(f"new_{num_overlaps}of{num_layers}_sd_{mode}")

            # --- Handle FLOP-Matched Training ---
            if args.flop_matched:
                multiplier = num_layers / num_overlaps
                new_total_steps = int(base_total_steps * multiplier)
                new_warmup_steps = int(base_warmup_steps * multiplier)

                # Update steps in the command
                cmd[cmd.index("--total_steps") + 1] = str(new_total_steps)
                cmd[cmd.index("--warmup_steps") + 1] = str(new_warmup_steps)
                
                run_name_parts.append("flop_matched")
                print(f"FLOP Matched: Steps updated to {new_total_steps}, Warmup to {new_warmup_steps}")

        else:
            # Dense run with custom parameters
            run_name_parts.append(f"dense_L{num_layers}")

        # --- Configure Low-Rank Training ---
        if exp_config.get("low_rank"):
            cmd.append("--low_rank")

            if "low_rank_ratio" in exp_config:
                cmd.extend(["--low_rank_ratio", str(exp_config["low_rank_ratio"])])
                run_name_parts.append(f"lr{exp_config['low_rank_ratio']}")

            if "frobenius_coef" in exp_config:
                cmd.extend(["--frobenius_coef", str(exp_config["frobenius_coef"])])
                run_name_parts.append(f"fb{exp_config['frobenius_coef']}")

            if exp_config.get("disable_c"):
                cmd.append("--disable_c")
                run_name_parts.append("noC")

            if "communicate_lowrank_freq" in exp_config:
                cmd.extend(["--communicate_lowrank_freq", str(exp_config["communicate_lowrank_freq"])])
                run_name_parts.append(f"clf{exp_config['communicate_lowrank_freq']}")

            if exp_config.get("sanity_check_lowrank"):
                cmd.append("--sanity_check_lowrank")
                run_name_parts.append("sanity")

        # --- Set Run Name ---
        run_name = "_".join(run_name_parts)
        cmd.extend(["--run_name", run_name])

        # --- Run Training ---
        print("Running simplified GPT training with torchrun:")
        print(" ".join(cmd))
    
        subprocess.run(cmd)

if __name__ == "__main__":
    main() 