# Memory-Mapped Dataset Implementation - Summary of Changes

## Problem

The original `BinaryDataset` class loaded all data shards into memory at once using `torch.cat(all_tokens)`, causing out-of-memory (OOM) errors with large datasets.

## Solution

Replaced in-memory loading with memory-mapped (mmap) loading that reads data on-demand from disk.

## Key Changes in `simple_gpt_dataloader.py`

### 1. **Indexing Instead of Loading (Lines 87-100)**

**Before:**

- Loaded all shards into memory: `all_tokens.append(tokens)`
- Concatenated everything: `self.data = torch.cat(all_tokens).to(device=self.device)`

**After:**

- Only reads file headers to get token counts
- Builds an index: `self.shard_info` and `self.cumulative_tokens`
- No data loaded during initialization

### 2. **On-Demand Data Loading in `__getitem__` (Lines 127-168)**

**Before:**

- Retrieved data from pre-loaded tensor: `sequence = self.data[start_pos : start_pos + self.seq_len + 1]`

**After:**

- Calculates which shard(s) contain the requested sequence
- Opens file and reads only the required tokens using `f.seek()` and `f.readinto()`
- Handles edge cases where sequences span multiple shards
- Loads data to device only when needed

### 3. **Memory Efficiency**

**Before:**

- Memory usage: O(total_dataset_size)
- All data resident in GPU/CPU memory

**After:**

- Memory usage: O(batch_size × seq_len)
- Only current batch data in memory
- Rest stays on disk

## Technical Details

### File Format Support

- Maintains compatibility with modded-nanogpt binary format:
  - 256 int32 header (1024 bytes)
  - Token data as uint16 (2 bytes per token)

### Features Preserved

✅ `max_samples` parameter still works
✅ Distributed training (rank/world_size) still works
✅ Same API - no changes needed in training code
✅ Support for both CPU and CUDA devices

### Performance Characteristics

- **Pros:**
  - No OOM errors regardless of dataset size
  - Minimal memory footprint
  - Fast initialization (only reads headers)

- **Cons:**
  - Slightly slower per-batch loading (disk I/O overhead)
  - Mitigated by DataLoader prefetching and disk caching

## No Changes Required in Training Code

The `simple_gpt_training.py` file requires **no modifications** because:

- Same class name: `BinaryDataset`
- Same constructor signature
- Same `__len__` and `__getitem__` methods
- Compatible with existing `create_dataloaders` function

## Testing

Test the changes by running:

```bash
python simple_gpt_dataloader.py
```

## Backward Compatibility

If you need the old behavior (full in-memory loading), you can:

1. Keep a copy of the old implementation
2. Add a parameter like `use_mmap=True` to switch between modes

## Expected Results

- ✅ No OOM errors even with very large datasets
- ✅ Fast startup time (indexing is fast)
- ✅ Training throughput should be siyour_clusterr (disk caching helps)
- ✅ Works seamlessly with existing training scripts
