#!/usr/bin/env python3
"""
SLURM Job Submitter for SimpleSubnet Training

Supports multiple clusters and parameter grid searches.
"""

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class ClusterConfig:
    """Configuration for different SLURM clusters"""

    @staticmethod
    def get_config(cluster_name: str) -> Dict[str, Any]:
        """Get cluster-specific configuration"""
        configs = {
            "your_cluster": {
                "account": "your_account",
                "gpu_type": "h100",
                "num_gpus": 4,
                "nodes": 1,
                "mem": "1125G",
                "cpus_per_task": 48,
                "ntasks": 1,
                "default_time": "3:00:00",
                "log_dir": "/tmp/logs",
                "work_dir": "$SLURM_TMPDIR/code/",
                "venv_activate": ".venv/bin/activate",
                "data_path": "$SLURM_TMPDIR/data/fineweb",
                "omp_threads": 12,
                "mkl_threads": 12,
                # Mount configuration
                "code_source": "$HOME/spectron",
                "data_source": "$HOME/data/fineweb",
                "code_dest": "$SLURM_TMPDIR/code",
                "data_dest": "$SLURM_TMPDIR/data/fineweb",
                "home_dir": "$HOME",
            },
        }

        if cluster_name not in configs:
            raise ValueError(
                f"Unknown cluster: {cluster_name}. Available: {list(configs.keys())}"
            )

        return configs[cluster_name]


class SlurmSubmitter:
    """SLURM job submission handler"""

    def __init__(
        self,
        cluster: str,
        dry_run: bool = False,
        mount: bool = True,
        multi_node: bool = False,
        flavor: Optional[str] = None,
    ):
        self.cluster = cluster
        self.config = ClusterConfig.get_config(cluster)
        self.dry_run = dry_run
        self.mount = mount
        self.multi_node = multi_node
        self.flavor = flavor
        self.script_dir = Path("/tmp/slurm_scripts")
        self.script_dir.mkdir(exist_ok=True)

    def generate_sbatch_header(self, job_name: str, time: str = None, **kwargs) -> str:
        """Generate SBATCH header for the job"""
        time = time or self.config["default_time"]

        # Override nodes and ntasks for multi-node mode
        nodes = 2 if self.multi_node else self.config["nodes"]
        ntasks = 2 if self.multi_node else self.config["ntasks"]

        header = f"""#!/bin/bash
"""
        # Add account only if present (not needed for some clusters)
        if "account" in self.config:
            header += f"#SBATCH --account={self.config['account']}\n"

        if "partition" in self.config:
            header += f"#SBATCH --partition={self.config['partition']}\n"

        header += f"""#SBATCH --gres=gpu:{self.config['gpu_type']}:{self.config['num_gpus']}
#SBATCH --nodes={nodes}
#SBATCH --mem={self.config['mem']}
#SBATCH --cpus-per-task={self.config['cpus_per_task']}
#SBATCH --ntasks={ntasks}
#SBATCH --time={time}
#SBATCH --job-name={job_name}
#SBATCH --output={self.config['log_dir']}/%j.out
#SBATCH --error={self.config['log_dir']}/%j.err

"""
        return header

    def generate_mount_section(self) -> str:
        """Generate mount section dynamically"""
        if not self.mount:
            return ""

        mount_script = ""

        # Add module load if present (for some clusters)
        if "module_load" in self.config:
            mount_script += f"""
{self.config['module_load']}
"""

        if self.multi_node:
            # Multi-node mount: use srun to execute on all nodes
            mount_script += f"""
# Multi-node setup: distribute code and data to all nodes
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c '
mkdir -p {self.config['code_dest']}
mkdir -p {self.config['data_dest']}
'

# 1) Parallel copy of Spectron sources to all nodes
echo "Copying language/Spectron in parallel to all nodes..."
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c '
cd {self.config['home_dir']}
find spectron -mindepth 1 -maxdepth 1 -print0 | \\
  parallel -0 -j $SLURM_CPUS_PER_TASK rsync -a --inplace {{}} {self.config['code_dest']}/
'

# 2) Parallel copy of fineweb dataset to all nodes
echo "Copying fineweb in parallel to all nodes..."
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c '
cd {self.config['home_dir']}/data/
find fineweb -mindepth 1 -maxdepth 1 -print0 | \\
  parallel -0 -j $SLURM_CPUS_PER_TASK rsync -a --inplace {{}} {self.config['data_dest']}/
'

# 3) Copy config files to all nodes
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c '
cp {self.config['home_dir']}/spectron/pyproject.toml {self.config['code_dest']}/
cp {self.config['home_dir']}/spectron/uv.lock {self.config['code_dest']}/
'

# 4) Set up virtual environment on all nodes
echo "Setting up virtual environment on all nodes..."
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c '
cd {self.config['code_dest']}
uv venv
source .venv/bin/activate
uv sync
'

echo "Multi-node setup complete!"
"""
        else:
            # Single-node mount (original behavior)
            mount_script += f"""
mkdir -p {self.config['code_dest']}
mkdir -p {self.config['data_dest']}


# 1) Parallel copy of Spectron sources
echo "Copying language/Spectron in parallel..."
cd {self.config['home_dir']}
find spectron -mindepth 1 -maxdepth 1 -print0 | \\
  parallel -0 -j $SLURM_CPUS_PER_TASK rsync -a --inplace {{}} {self.config['code_dest']}/

# 2) Parallel copy of fineweb dataset
echo "Copying fineweb in parallel..."
cd {self.config['home_dir']}/data/
find fineweb -mindepth 1 -maxdepth 1 -print0 | \\
  parallel -0 -j $SLURM_CPUS_PER_TASK rsync -a --inplace {{}} {self.config['data_dest']}/

cp {self.config['home_dir']}/spectron/pyproject.toml {self.config['code_dest']}/
cp {self.config['home_dir']}/spectron/uv.lock {self.config['code_dest']}/

"""
        return mount_script

    def generate_setup_section(
        self, job_name: str, job_name_without_suffix: str
    ) -> str:
        """Generate setup section (cd, activate venv, set env vars)"""
        setup = f"""
# Change to work directory
cd {self.config['work_dir']}

"""
        # For single-node, create and activate venv here
        # For multi-node, venv is already created in mount section
        if not self.multi_node:
            setup += f"""# Create and activate virtual environment
uv venv
source {self.config['venv_activate']}
uv sync

"""
        else:
            setup += f"""# Activate virtual environment (already created on all nodes)
source {self.config['venv_activate']}

"""

        setup += f"""# Set data path
export DATA_PATH="{self.config['data_path']}"

# Set environment variables for multi-GPU training
export OMP_NUM_THREADS={self.config['omp_threads']}
export MKL_NUM_THREADS={self.config['mkl_threads']}

# Set wandb tags
export WANDB_TAGS="{job_name_without_suffix}"

echo "Launching LLM training with torchrun"
echo "Working directory: $(pwd)"
echo "Data path: $DATA_PATH"
echo "Wandb tags: $WANDB_TAGS"

"""
        return setup

    def generate_distributed_env(self) -> str:
        """Generate distributed training environment variables for multi-node"""
        if not self.multi_node:
            return ""

        return f"""
# Set up distributed training environment for multi-node
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * 4))

echo "Multi-node distributed training configuration:"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"

"""

    def generate_checkpoint_detection(
        self, job_name: str, base_job_name: Optional[str] = None
    ) -> str:
        """Generate checkpoint detection section to auto-resume from existing checkpoints

        Args:
            job_name: Full job name (including chain suffix if applicable)
            base_job_name: Base job name without chain suffix (for shared checkpoint dir)
        """
        # Use base_job_name for checkpoint dir so all jobs in chain share the same dir
        checkpoint_base_name = base_job_name if base_job_name else job_name
        checkpoint_dir = f"$HOME/checkpoints/low_rank/{checkpoint_base_name}"

        return f"""
# Check for existing checkpoints and auto-resume if found
CHECKPOINT_DIR="{checkpoint_dir}"
RESUME_FROM=""

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Checking for existing checkpoints in $CHECKPOINT_DIR..."

    # Find all checkpoint files sorted by modification time (newest first)
    CHECKPOINTS=$(find "$CHECKPOINT_DIR" -name "*.pt" -type f -printf '%T@ %p\\n' 2>/dev/null | sort -rn | cut -d' ' -f2-)

    if [ -n "$CHECKPOINTS" ]; then
        # Try to find a valid checkpoint by testing them with Python
        echo "Found checkpoint files, validating..."
        for CKPT in $CHECKPOINTS; do
            # Quick validation test: try to load the checkpoint with torch
            if python3 -c "import torch; torch.load('$CKPT', map_location='cpu'); print('✓ Valid')" 2>/dev/null; then
                export RESUME_FROM="$CKPT"
                echo "Found valid checkpoint: $RESUME_FROM"
                echo "Will resume training from this checkpoint"
                break
            else
                echo "  Skipping corrupted checkpoint: $CKPT"
            fi
        done

        if [ -z "$RESUME_FROM" ]; then
            echo "WARNING: No valid checkpoints found (all appear corrupted)"
            echo "Starting fresh training"
        fi
    else
        echo "No checkpoint files found in $CHECKPOINT_DIR"
    fi
else
    echo "No checkpoint directory found at $CHECKPOINT_DIR"
    echo "Starting fresh training"
fi

"""

    def get_default_params(self, flavor: Optional[str] = None) -> Dict[str, Any]:
        """Get default training parameters from run.sh

        Args:
            flavor: Model size flavor ('150M', '500M', or '1B') to override architecture params
        """
        params = {
            "batch_size": 512,
            "micro_batch_size": 16,
            "max_lr": 0.004,
            "weight_decay": 0,
            "total_steps": 3380,
            "warmup_ratio": 0.05,
            "log_interval": 10,
            "max_val_samples": 100,
            "hidden_size": 768,
            "num_heads": 12,
            "num_layers": 12,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,
            "train_seq_len": 2048,
            "val_seq_len": 2048,
            "embed_dropout": 0.2,
            "resid_dropout": 0.2,
            "use_flex_attn": True,
            "bf16": True,
            "virtual_workers_per_gpu": 1,
            "n_kv_heads": 12,
            "multiple_of": 256,
            "low_rank": True,
            "low_rank_ratio": 0.25,
            "frobenius_coef": 1.0e-9,
            "disable_c": True,
            "checkpoint_dir": "$HOME/checkpoints",
            "checkpoint_interval_hours": 2.8,
            "train_files": "$DATA_PATH/fineweb_train_*.bin",
            "val_files": "$DATA_PATH/fineweb_val_*.bin",
            "wandb_project": "your_project",
            "wandb_entity": "your_entity",
        }

        # Apply flavor-specific overrides
        if flavor:
            flavor_configs = {
                "60M": {
                    "hidden_size": 512,
                    "num_layers": 8,
                    "num_heads": 8,
                    "n_kv_heads": 8,
                    "low_rank": False,
                },
                "92M": {
                    "hidden_size": 640,
                    "num_layers": 10,
                    "num_heads": 10,
                    "n_kv_heads": 10,
                    "low_rank": False,
                },
                "134M": {
                    "hidden_size": 768,
                    "num_layers": 12,
                    "num_heads": 12,
                    "n_kv_heads": 12,
                    "low_rank": False,
                },
                "150M": {
                    "hidden_size": 768,
                    "num_layers": 14,
                    "num_heads": 12,
                    "n_kv_heads": 12,
                    "low_rank": False,
                },
                "220M": {
                    "hidden_size": 896,
                    "num_layers": 16,
                    "num_heads": 14,
                    "n_kv_heads": 14,
                    "low_rank": False,
                },
                "325M": {
                    "hidden_size": 1024,
                    "num_layers": 20,
                    "num_heads": 16,
                    "n_kv_heads": 16,
                    "low_rank": False,
                },
                "500M": {
                    "hidden_size": 1280,
                    "num_layers": 20,
                    "num_heads": 20,
                    "n_kv_heads": 20,
                    "low_rank": False,
                },
                "780M": {
                    "hidden_size": 1536,
                    "num_layers": 24,
                    "num_heads": 24,
                    "n_kv_heads": 24,
                    "low_rank": False,
                },
                "835M": {
                    "hidden_size": 1536,
                    "num_layers": 26,
                    "num_heads": 24,
                    "n_kv_heads": 24,
                    "low_rank": False,
                },
                "1.4B": {
                    "hidden_size": 2048,
                    "num_layers": 24,
                    "num_heads": 16,
                    "n_kv_heads": 16,
                    "low_rank": False,
                },
                "1.7B": {
                    "hidden_size": 2048,
                    "num_layers": 30,
                    "num_heads": 16,
                    "n_kv_heads": 16,
                    "low_rank": False,
                },
                "2.7B": {
                    "hidden_size": 2560,
                    "num_layers": 32,
                    "num_heads": 20,
                    "n_kv_heads": 20,
                    "low_rank": False,
                },
            }

            if flavor in flavor_configs:
                params.update(flavor_configs[flavor])
            else:
                raise ValueError(
                    f"Unknown flavor: {flavor}. Available: {list(flavor_configs.keys())}"
                )

        return params

    def generate_training_command(
        self, params: Dict[str, Any], job_name: str, base_job_name: Optional[str] = None
    ) -> str:
        """Generate torchrun training command with parameters

        Args:
            params: Training parameters
            job_name: Full job name (including chain suffix if applicable)
            base_job_name: Base job name without chain suffix (for shared checkpoint dir)
        """
        # Start with defaults and override with provided params
        default_params = self.get_default_params(flavor=self.flavor)
        merged_params = {**default_params, **params}

        # Set run_name to job_name if not already set
        if "run_name" not in merged_params:
            merged_params["run_name"] = job_name
            # Use base_job_name for checkpoint_dir so all jobs in chain share the same dir
            checkpoint_base_name = base_job_name if base_job_name else job_name
            merged_params["checkpoint_dir"] = (
                f"$HOME/checkpoints/low_rank/{checkpoint_base_name}"
            )

        # Extract special parameters
        nproc_per_node = merged_params.pop("nproc_per_node", self.config["num_gpus"])

        # Build torchrun command
        if self.multi_node:
            # Multi-node: wrap in srun and add distributed parameters
            torchrun_cmd = f"torchrun --nproc_per_node={nproc_per_node} \\\n"
            torchrun_cmd += "    --nnodes=$SLURM_JOB_NUM_NODES \\\n"
            torchrun_cmd += "    --node_rank=$SLURM_NODEID \\\n"
            torchrun_cmd += "    --master_addr=$MASTER_ADDR \\\n"
            torchrun_cmd += "    --master_port=$MASTER_PORT \\\n"
            torchrun_cmd += "    simple_gpt_training.py"
        else:
            # Single-node: standard torchrun
            torchrun_cmd = f"torchrun --nproc_per_node={nproc_per_node} \\\n"
            torchrun_cmd += "    simple_gpt_training.py"

        # Add all training parameters
        for key, value in sorted(merged_params.items()):
            if isinstance(value, bool):
                if value:  # Only add flag if True
                    torchrun_cmd += f" \\\n    --{key}"
            elif isinstance(value, (list, tuple)):
                # Handle list arguments
                torchrun_cmd += f" \\\n    --{key} {' '.join(map(str, value))}"
            else:
                # Handle string values that might contain variables
                if isinstance(value, str) and "$" in value:
                    torchrun_cmd += f' \\\n    --{key} "{value}"'
                else:
                    torchrun_cmd += f" \\\n    --{key} {value}"

        # Add resume_from if checkpoint was found
        resume_cmd = """
# Add resume_from parameter if checkpoint was found
if [ -n "$RESUME_FROM" ]; then
    export RESUME_ARG="--resume_from $RESUME_FROM"
else
    RESUME_ARG=""
fi

"""

        # Wrap in srun for multi-node execution
        if self.multi_node:
            cmd = resume_cmd
            cmd += "srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c '\n"
            cmd += torchrun_cmd
            cmd += " $RESUME_ARG\n"
            cmd += "'\n"
        else:
            cmd = resume_cmd
            cmd += torchrun_cmd
            cmd += " $RESUME_ARG\n"

        return cmd

    def create_job_script(
        self,
        job_name: str,
        params: Dict[str, Any],
        time: str = None,
        job_name_without_suffix: str = None,
        base_job_name: str = None,
    ) -> Path:
        """Create complete job script

        Args:
            job_name: Full job name (including chain suffix if applicable)
            params: Training parameters
            time: Wall time for the job
            job_name_without_suffix: Base job name without parameter suffix (for wandb tags)
            base_job_name: Base job name for checkpoint directory (shared across chain)
        """
        # Generate all sections
        script = self.generate_sbatch_header(job_name, time)
        script += self.generate_mount_section()
        script += self.generate_setup_section(job_name, job_name_without_suffix)
        script += self.generate_distributed_env()  # Add distributed env for multi-node
        script += self.generate_checkpoint_detection(
            job_name, base_job_name
        )  # Check for existing checkpoints
        script += self.generate_training_command(params, job_name, base_job_name)

        # Save script
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_name = f"{job_name}_{timestamp}.sh"
        script_path = self.script_dir / script_name

        with open(script_path, "w") as f:
            f.write(script)

        # Make executable
        script_path.chmod(0o755)

        return script_path

    def submit_job(
        self, script_path: Path, dependency_job_id: Optional[str] = None
    ) -> Optional[str]:
        """Submit job to SLURM with optional dependency

        Args:
            script_path: Path to the job script
            dependency_job_id: Optional job ID to depend on (using --afterany)

        Returns:
            Job ID if submission successful, None otherwise
        """
        if self.dry_run:
            print(f"\n{'='*80}")
            if dependency_job_id:
                print(
                    f"DRY RUN: Would submit {script_path} with dependency on job {dependency_job_id}"
                )
            else:
                print(f"DRY RUN: Would submit {script_path}")
            print(f"{'='*80}")
            with open(script_path, "r") as f:
                print(f.read())
            print(f"{'='*80}\n")
            return f"DRYRUN_{script_path.stem}"  # Return fake job ID for dry run
        else:
            try:
                # Build sbatch command
                cmd = ["sbatch"]
                if dependency_job_id:
                    cmd.extend(["--dependency", f"afterany:{dependency_job_id}"])
                cmd.append(str(script_path))

                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                # Extract job ID from output
                job_id = result.stdout.strip().split()[-1]
                if dependency_job_id:
                    print(
                        f"Submitted job {job_id}: {script_path.name} (depends on {dependency_job_id})"
                    )
                else:
                    print(f"Submitted job {job_id}: {script_path.name}")
                return job_id
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job: {e.stderr}")
                return None


def parse_grid_param(value: str) -> List[Any]:
    """Parse grid parameter from command line"""
    # Check if it's a range specification
    if "," in value:
        parts = value.split(",")
        result = []
        for part in parts:
            try:
                result.append(float(part) if "." in part else int(part))
            except ValueError:
                result.append(part)  # Keep as string
        return result
    else:
        # Single valuess
        try:
            return [float(value) if "." in value else int(value)]
        except ValueError:
            return [value]


def abbreviate_param_name(key: str, threshold: int = 5) -> str:
    """
    Abbreviate parameter names that are longer than threshold characters.

    For long names, splits by underscore and takes the first character of each part.
    For short names, returns the key as-is.

    Args:
        key: Parameter name to abbreviate
        threshold: Minimum length to trigger abbreviation (default: 5)

    Returns:
        Abbreviated parameter name

    Examples:
        >>> abbreviate_param_name("spectral_weight_decay")
        'swd'
        >>> abbreviate_param_name("lr")
        'lr'
        >>> abbreviate_param_name("max_lr")
        'ml'
    """
    if len(key) >= threshold:
        # Split by underscore and take first character of each part
        parts = key.split("_")
        return "".join(part[0] for part in parts if part)
    else:
        return key


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs for SimpleSubnet training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single job with default parameters
  python slurm_submitter.py --cluster your_cluster--job-name test_run

  # Use a specific model flavor (150M, 500M, or 1B)
  python slurm_submitter.py --cluster your_cluster--job-name test_500M --flavor 500M

  # Submit a chain of 5 jobs (each job resumes from previous checkpoint)
  python slurm_submitter.py --cluster your_cluster--job-name long_training --chain-length 5

  # Grid search over learning rates and weight decay
  python slurm_submitter.py --cluster your_cluster--job-name lr_sweep \\
      --grid max_lr=0.001,0.002,0.004 --grid weight_decay=0.01,0.1

  # Grid search with job chaining (creates chains for each parameter combination)
  python slurm_submitter.py --cluster your_cluster--job-name lr_sweep \\
      --chain-length 3 --grid max_lr=0.001,0.002 --grid weight_decay=0.01,0.1

  # Dry run to see generated scripts
  python slurm_submitter.py --cluster your_cluster--job-name test --dry-run

  # Custom parameters with a flavor
  python slurm_submitter.py --cluster your_cluster--job-name muon_test \\
      --flavor 1B --param optimizer=muon --param max_lr=0.02
        """,
    )

    parser.add_argument(
        "--cluster", type=str, default="your_cluster", help="Cluster name (default: your_cluster)"
    )
    parser.add_argument(
        "--job-name", type=str, required=True, help="Base name for the job(s)"
    )
    parser.add_argument(
        "--time", type=str, default=None, help="Wall time (default: cluster default)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate scripts but do not submit"
    )
    parser.add_argument(
        "--no-mount",
        action="store_true",
        help="Skip mounting step (data already on node)",
    )
    parser.add_argument(
        "--multi-node",
        action="store_true",
        help="Enable multi-node distributed training (2 nodes, 2 tasks)",
    )
    parser.add_argument(
        "--flavor",
        type=str,
        help="Model size flavor (150M, 500M, or 1B) to use predefined architecture settings",
    )
    parser.add_argument(
        "--chain-length",
        type=int,
        default=1,
        help="Number of chained jobs to submit (default: 1). Each job will depend on the previous one using --afterany",
    )

    # Parameter specification
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Set a single parameter (can be used multiple times)",
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="KEY=VAL1,VAL2,...",
        help="Grid search parameter (can be used multiple times)",
    )

    # Common training parameters
    parser.add_argument(
        "--config",
        type=str,
        help="Load base config from Python file (not implemented yet)",
    )

    args = parser.parse_args()

    # Initialize submitter
    submitter = SlurmSubmitter(
        cluster=args.cluster,
        dry_run=args.dry_run,
        mount=not args.no_mount,
        multi_node=args.multi_node,
        flavor=args.flavor,
    )

    # Parse single parameters
    base_params = {}
    for param_str in args.param:
        if "=" not in param_str:
            print(f"Error: Invalid parameter format '{param_str}'. Use KEY=VALUE")
            sys.exit(1)
        key, value = param_str.split("=", 1)
        # Try to convert to appropriate type
        if value.lower() == "true":
            base_params[key] = True
        elif value.lower() == "false":
            base_params[key] = False
        else:
            try:
                base_params[key] = float(value) if "." in value else int(value)
            except ValueError:
                base_params[key] = value

    # Parse grid parameters
    grid_params = {}
    for grid_str in args.grid:
        if "=" not in grid_str:
            print(f"Error: Invalid grid format '{grid_str}'. Use KEY=VAL1,VAL2,...")
            sys.exit(1)
        key, values_str = grid_str.split("=", 1)
        grid_params[key] = parse_grid_param(values_str)

    # Generate all parameter combinations
    if grid_params:
        # Create grid
        keys = list(grid_params.keys())
        values = [grid_params[k] for k in keys]
        combinations = list(itertools.product(*values))

        print(f"Generating {len(combinations)} jobs from grid search:")
        for k, v in grid_params.items():
            print(f"  {k}: {v}")
        print()

        job_ids = []
        for i, combo in enumerate(combinations):
            # Merge base params with this combination
            params = base_params.copy()
            params.update(dict(zip(keys, combo)))

            # Create job name with parameter values (abbreviate long param names)
            param_suffix = "_".join(
                f"{abbreviate_param_name(k)}{v}".replace(".", "p")
                for k, v in zip(keys, combo)
            )
            job_name = f"{args.job_name}_{param_suffix}"
            job_name_without_suffix = args.job_name  # Base job name without suffix

            # Submit chain of jobs for this parameter combination
            if args.chain_length > 1:
                print(
                    f"\nSubmitting chain of {args.chain_length} jobs for {job_name}..."
                )
                chain_job_ids = []
                prev_job_id = None

                for chain_idx in range(args.chain_length):
                    # Create job name with chain index
                    chained_job_name = f"{job_name}"

                    # Create and submit job with dependency on previous job
                    # Pass job_name as base_job_name so all chain jobs share the same checkpoint dir
                    script_path = submitter.create_job_script(
                        chained_job_name,
                        params,
                        args.time,
                        job_name_without_suffix=job_name_without_suffix,
                        base_job_name=job_name,
                    )
                    job_id = submitter.submit_job(
                        script_path, dependency_job_id=prev_job_id
                    )

                    if job_id:
                        chain_job_ids.append(job_id)
                        prev_job_id = job_id
                    else:
                        print(
                            f"Failed to submit chain job {chain_idx+1} for {job_name}"
                        )
                        break

                job_ids.extend(chain_job_ids)
                if chain_job_ids:
                    print(f"  Submitted chain: {' -> '.join(chain_job_ids)}")
            else:
                # Single job (no chaining)
                script_path = submitter.create_job_script(
                    job_name, params, args.time, job_name_without_suffix
                )
                job_id = submitter.submit_job(script_path)
                if job_id:
                    job_ids.append(job_id)

        if not args.dry_run and job_ids:
            print(f"\nSubmitted {len(job_ids)} total jobs")
    else:
        # Single job or chain of jobs
        job_name_without_suffix = args.job_name.split("_")[0]

        if args.chain_length > 1:
            print(
                f"\nSubmitting chain of {args.chain_length} jobs for {args.job_name}..."
            )
            job_ids = []
            prev_job_id = None

            for chain_idx in range(args.chain_length):
                # Create job name with chain index
                chained_job_name = f"{args.job_name}"

                # Create and submit job with dependency on previous job
                # Pass args.job_name as base_job_name so all chain jobs share the same checkpoint dir
                script_path = submitter.create_job_script(
                    chained_job_name,
                    base_params,
                    args.time,
                    job_name_without_suffix=job_name_without_suffix,
                    base_job_name=args.job_name,
                )
                job_id = submitter.submit_job(
                    script_path, dependency_job_id=prev_job_id
                )

                if job_id:
                    job_ids.append(job_id)
                    prev_job_id = job_id
                else:
                    print(f"Failed to submit chain job {chain_idx+1}")
                    break

            if job_ids and not args.dry_run:
                print(f"\nSubmitted chain: {' -> '.join(job_ids)}")
        else:
            # Single job (no chaining)
            script_path = submitter.create_job_script(
                args.job_name, base_params, args.time, job_name_without_suffix
            )
            job_id = submitter.submit_job(script_path)
            if job_id and not args.dry_run:
                print(f"\nSubmitted job: {job_id}")

    print(f"\nScripts saved in: {submitter.script_dir}")


if __name__ == "__main__":
    main()
