import torch
from torch.utils.data import DataLoader, Dataset
import json
import os
import glob
from pathlib import Path
from tqdm import tqdm
from itertools import chain
from datasets import load_from_disk
from dataclasses import dataclass


def _load_data_shard(file: Path):
    """Load a binary data shard with the modded-nanogpt format."""
    header = torch.from_file(
        str(file), False, 256, dtype=torch.int32
    )  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(
    filename_pattern: str, batch_size: int, rank: int, world_size: int, seq_len: int
):
    """
    Generate batches of data in a distributed fashion, matching modded-nanogpt style.

    Args:
        filename_pattern: Glob pattern for data files
        batch_size: Total batch size across all processes
        rank: Current process rank
        world_size: Total number of processes
        seq_len: Sequence length for training

    Yields:
        (inputs, targets) tensors
    """
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size

    file_iter = iter(
        files
    )  # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0

    while True:
        if pos + batch_size + seq_len >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0

        # Get local batch with sequence length
        start_pos = pos + rank * local_batch_size
        buf = tokens[start_pos : start_pos + local_batch_size + seq_len]

        # Create input and target sequences
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)

        pos += batch_size
        yield inputs, targets


class BinaryDataset(Dataset):
    """Dataset for binary token files, optimized for preloading to GPU."""

    def __init__(
        self,
        filename_pattern: str,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        device="cuda",
        max_samples: int = -1,
    ):
        """
        Args:
            filename_pattern: Glob pattern for data files
            seq_len: Sequence length for training
            rank: Current process rank
            world_size: Total number of processes
            device: The device to load the data onto ('cuda' or 'cpu')
            max_samples: Maximum number of samples to load (-1 for all)
        """
        self.files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.device = device

        # Assign subset of files to this process
        self.file_paths = [
            f for i, f in enumerate(self.files) if i % world_size == rank
        ]
        print(
            f"🔍 Rank {rank}: Found {len(self.files)} files with pattern '{filename_pattern}'. Assigned {len(self.file_paths)} files."
        )

        # Load tokens shard by shard until max_samples is reached
        all_tokens = []
        print(f"🔍 Rank {rank}: loading {len(self.file_paths)} shards to {device}...")

        total_loaded_tokens = 0
        max_tokens_to_load = -1
        if max_samples > 0:
            # Add a buffer to account for the last partial sequence
            max_tokens_to_load = max_samples * self.seq_len + self.seq_len + 1

        for path in tqdm(self.file_paths, desc=f"Loading data (rank {rank})"):
            tokens = _load_data_shard(path)
            all_tokens.append(tokens)
            total_loaded_tokens += len(tokens)
            if max_tokens_to_load != -1 and total_loaded_tokens >= max_tokens_to_load:
                print(f"🔍 Rank {rank}: reached max_samples limit, stopping loading.")
                break

        # Concatenate all tokens into a single tensor and move to device
        if all_tokens:
            self.data = torch.cat(all_tokens).to(device=self.device)
        else:
            self.data = torch.empty(0, dtype=torch.uint16, device=self.device)

        # Trim data if it exceeds max_tokens_to_load
        if max_tokens_to_load != -1 and len(self.data) > max_tokens_to_load:
            self.data = self.data[:max_tokens_to_load]

        self.total_tokens = len(self.data)

        # Calculate number of sequences
        self.num_sequences = (
            (self.total_tokens - self.seq_len) // self.seq_len
            if self.total_tokens > self.seq_len
            else 0
        )

        print(
            f"🔍 Rank {rank}: total tokens on {device}: {self.total_tokens:,}, sequences: {self.num_sequences:,}"
        )
        if "val" in filename_pattern or "valid" in filename_pattern:
            print(
                f"[Validation Set] Rank {rank}: total tokens = {self.total_tokens}, seq_len = {self.seq_len}, num_sequences = {self.num_sequences}"
            )

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Calculate global position in this rank's data
        start_pos = idx * self.seq_len

        # Extract sequence directly from the GPU tensor
        sequence = self.data[start_pos : start_pos + self.seq_len + 1]

        # Convert to proper types, data is already on GPU
        inputs = sequence[:-1].to(dtype=torch.int32)
        targets = sequence[1:].to(dtype=torch.int64)

        return {"input_ids": inputs, "labels": targets}


def create_dataloaders(args, rank, world_size, device=None):
    """
    Create training and validation dataloaders for GPT training.
    This version preloads the entire dataset to the GPU for faster training.
    """

    def collate_fn(batch):
        # The tensors in the batch are already on the GPU
        input_ids = torch.stack([x["input_ids"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])
        return {"input_ids": input_ids, "labels": labels}

    # Create training dataset. Data is loaded to GPU during initialization.
    # Resolve device explicitly when provided (e.g., for virtual workers running on one GPU)
    resolved_device = device if device is not None else f"cuda:{rank}"

    train_dataset = BinaryDataset(
        args.train_files,
        args.train_seq_len,
        rank,
        world_size,
        device=resolved_device,  # Explicit device allows multiple virtual workers per GPU
    )

    # Use micro_batch_size if specified, otherwise use batch_size
    effective_batch_size = (
        args.micro_batch_size
        if hasattr(args, "micro_batch_size") and args.micro_batch_size is not None
        else args.batch_size
    )

    # Data is on GPU, so no need for workers, pinning, or prefetching.
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=False,  # Shuffling is often handled at the dataset level if needed
        collate_fn=collate_fn,
    )

    # Only rank 0 loads the validation set
    validation_loader = None
    if rank == 0:
        max_val_samples = getattr(args, "max_val_samples", -1)
        if max_val_samples == 0:
            max_val_samples = -1  # if 0, load all

        val_dataset = BinaryDataset(
            args.val_files,
            args.val_seq_len,
            rank=0,
            world_size=1,  # Val set is not distributed
            device=resolved_device,
            max_samples=max_val_samples,
        )

        val_loader_batch_size = 16

        # The subsetting is now handled inside the dataset, but we still need to handle the case
        # where the user might want to validate on a smaller subset than what's loaded.
        # However, with the current logic, val_dataset is already limited.
        # So, we just create the loader from the dataset.

        validation_loader = DataLoader(
            val_dataset,
            batch_size=val_loader_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    return train_loader, validation_loader


# Legacy support for JSONL format
class LazyJsonlDataset(Dataset):
    def __init__(self, directory, rank=0, world_size=1, split="train"):
        """
        directory: path to directory containing .jsonl shards
        rank: rank of the current process
        world_size: total number of processes
        """
        all_files = sorted(
            [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.endswith(".jsonl")
            ]
        )
        # Assign subset of shards to this process
        self.file_paths = [f for i, f in enumerate(all_files) if i % world_size == rank]

        # Step 1: Build raw line index with offset
        raw_line_index = []
        print(f"🔍 Rank {rank} on {split}: indexing {len(self.file_paths)} shards...")
        for path in tqdm(self.file_paths, desc=f"Indexing (rank {rank})"):
            with open(path, "r") as f:
                offset = 0
                for line in f:
                    raw_line_index.append((path, offset))
                    offset += len(line)

        # Step 2: Shuffle while preserving original index
        rng = torch.Generator().manual_seed(123)
        shuffled_indices = torch.randperm(len(raw_line_index), generator=rng).tolist()
        # Each entry is now: (path, offset, original_data_index)
        self.line_index = [
            (raw_line_index[i][0], raw_line_index[i][1], i) for i in shuffled_indices
        ]

        if rank == 0:
            print()
            print("file paths", self.file_paths)
            print("sample raw line index", raw_line_index[:2])
            print("sample shuffled line index", self.line_index[:2])
            print()

        print(
            f"🔍 Rank {rank} on {split}: total lines indexed after shuffle: {len(self.line_index)}"
        )

    def __len__(self):
        return len(self.line_index)

    def __getitem__(self, idx):
        path, offset, original_index = self.line_index[idx]
        with open(path, "r") as f:
            f.seek(offset)
            item = json.loads(f.readline())
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "data_index": original_index,  # ✅ correct global shuffled index
        }


@dataclass
class TestArgs:
    train_files: str = "/path/to/train/*.bin"
    val_files: str = "/path/to/val/*.bin"
    train_seq_len: int = 1024
    val_seq_len: int = 1024
    batch_size: int = 8
    micro_batch_size: int = None
    max_val_samples: int = -1


if __name__ == "__main__":
    args = TestArgs()

    # Set some example paths (update these to actual data paths)
    home_dir = os.environ.get("HOME", ".")
    args.train_files = f"{home_dir}/data/fineweb/fineweb_train_*.bin"
    args.val_files = f"{home_dir}/data/fineweb/fineweb_val_*.bin"

    rank = 0
    world_size = 1

    print("Testing create_dataloaders function...")
    print(f"Args: {args}")

    # --- Count total tokens available in the datasets (load on CPU to avoid CUDA) ---
    try:
        # Create CPU-only datasets that load the same shards assigned to rank 0
        train_dataset_cpu = BinaryDataset(
            args.train_files,
            args.train_seq_len,
            rank=0,
            world_size=1,
            device="cpu",
            max_samples=-1,
        )
        print(
            f"Total train tokens (rank 0, assigned shards): {train_dataset_cpu.total_tokens:,}"
        )
        print(
            f"Train sequences (seq_len={args.train_seq_len}): {train_dataset_cpu.num_sequences:,}"
        )
    except Exception as e:
        print(f"⚠️ Could not load train dataset on CPU to count tokens: {e}")

    # Only attempt to count validation tokens on rank 0
    if rank == 0:
        try:
            val_max_samples = getattr(args, "max_val_samples", -1)
            val_dataset_cpu = BinaryDataset(
                args.val_files,
                args.val_seq_len,
                rank=0,
                world_size=1,
                device="cpu",
                max_samples=val_max_samples if val_max_samples != 0 else -1,
            )
            print(
                f"Total val tokens (rank 0, assigned shards): {val_dataset_cpu.total_tokens:,}"
            )
            print(
                f"Val sequences (seq_len={args.val_seq_len}): {val_dataset_cpu.num_sequences:,}"
            )
        except Exception as e:
            print(f"⚠️ Could not load val dataset on CPU to count tokens: {e}")

    try:
        train_loader, val_loader = create_dataloaders(args, rank, world_size)
        print(f"✅ Successfully created dataloaders!")
        print(f"Train loader: {len(train_loader)} batches")
        if val_loader:
            print(f"Val loader: {len(val_loader)} batches")
        else:
            print("Val loader: None (only rank 0 loads validation)")

        # Test getting a batch
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            print(f"Sample batch input_ids shape: {batch['input_ids'].shape}")
            print(f"Sample batch labels shape: {batch['labels'].shape}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the data paths exist and contain .bin files")
