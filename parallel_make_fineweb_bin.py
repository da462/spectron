"""
make_fineweb_bin_parallel.py
Download and preprocess the FineWeb dataset into .bin files compatible with modded-nanogpt.
Parallel processing version with 48 CPUs, each processing 3 chunks of 100M tokens.

Usage:
  pip install datasets tiktoken tqdm numpy transformers torch sentencepiece
  python make_fineweb_bin_parallel.py --tokenizer llama --version 10B --num_cpus 48 --out_dir data/fineweb10B_llama
  python make_fineweb_bin_parallel.py --tokenizer gpt2 --version 10B --num_cpus 48 --out_dir data/fineweb10B_gpt2

Arguments:
  --tokenizer    Which tokenizer to use: llama or gpt2 (default: llama)
  --version      Which FineWeb version to use: 10B or 100B (default: 10B)
  --num_cpus     Number of CPUs to use for parallel processing (default: 48)
  --shard_size   Number of tokens per .bin file (default: 100_000_000)
  --out_dir      Output directory for .bin files (default: ./fineweb10B)

Each CPU processes 3 chunks in loops. Total shards = 48 * 3 = 144 shards.
The first shard (chunk 0) is used for validation, the rest for training.
"""

import os
import argparse
import numpy as np
from datasets import load_dataset, get_dataset_config_names
import tiktoken
from tqdm import tqdm
from transformers import AutoTokenizer
import multiprocessing as mp
from functools import partial
import math

def write_datafile(filename, toks):
    """Write tokens to .bin file with modded-nanogpt header."""
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = len(toks) # number of tokens
    toks_np = np.array(toks, dtype=np.uint16)
    print(f"Writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def get_tokenizer_func(tokenizer_type):
    """Get the appropriate tokenizer function."""
    if tokenizer_type == "gpt2":
        enc = tiktoken.get_encoding("gpt2")
        def tokenize_doc(doc):
            tokens = [enc._special_tokens['<|endoftext|>']]
            tokens.extend(enc.encode_ordinary(doc["text"]))
            return np.array(tokens, dtype=np.uint16)
    else:  # llama
        enc = AutoTokenizer.from_pretrained("togethercomputer/Llama-2-7B-32K")
        def tokenize_doc(doc):
            tokens = [enc.eos_token_id]
            tokens.extend(enc.encode(doc["text"], add_special_tokens=False))
            return np.array(tokens, dtype=np.uint16)
    
    return tokenize_doc

def process_worker_chunks(worker_id, dataset_slice, tokenizer_type, shard_size, out_dir, total_workers):
    """Process 3 chunks for a single worker (CPU)."""
    try:
        print(f"Worker {worker_id} starting processing...")
        
        tokenize_func = get_tokenizer_func(tokenizer_type)
        
        # Each worker processes 3 chunks
        chunks_per_worker = 21
        results = []
        
        for chunk_idx_in_worker in range(chunks_per_worker):
            # Calculate global chunk index
            global_chunk_idx = worker_id * chunks_per_worker + chunk_idx_in_worker
            
            print(f"Worker {worker_id} processing chunk {chunk_idx_in_worker + 1}/21 (global chunk {global_chunk_idx})")
            
            # Initialize arrays for this chunk
            all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
            token_count = 0
            doc_start_idx = chunk_idx_in_worker * (len(dataset_slice) // chunks_per_worker)
            doc_end_idx = min((chunk_idx_in_worker + 1) * (len(dataset_slice) // chunks_per_worker), len(dataset_slice))
            
            # If this is the last chunk for this worker, include any remaining documents
            if chunk_idx_in_worker == chunks_per_worker - 1:
                doc_end_idx = len(dataset_slice)
            
            # Process documents for this chunk
            doc_count = 0
            for doc_idx in range(doc_start_idx, doc_end_idx):
                if doc_idx >= len(dataset_slice):
                    break
                    
                doc = dataset_slice[doc_idx]
                tokens = tokenize_func(doc)
                doc_count += 1
                
                # Check if we can fit these tokens in the current shard
                if token_count + len(tokens) <= shard_size:
                    all_tokens_np[token_count:token_count+len(tokens)] = tokens
                    token_count += len(tokens)
                else:
                    # Current shard is full, write it and start a new one if needed
                    break
            
            # Write the shard if we have tokens
            if token_count > 0:
                split = "val" if global_chunk_idx == 0 else "train"
                filename = os.path.join(out_dir, f"fineweb_{split}_{global_chunk_idx:06d}.bin")
                write_datafile(filename, all_tokens_np[:token_count])
                results.append((global_chunk_idx, token_count, doc_count))
                print(f"Worker {worker_id} completed chunk {global_chunk_idx}: {token_count:,} tokens from {doc_count} docs")
            else:
                print(f"Worker {worker_id} chunk {global_chunk_idx}: No tokens to write")
                results.append((global_chunk_idx, 0, 0))
        
        print(f"Worker {worker_id} completed all chunks")
        return results
        
    except Exception as e:
        print(f"Error in worker {worker_id}: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="llama", choices=["llama", "gpt2"])
    parser.add_argument("--version", type=str, default="100B", choices=["10B", "100B"])
    parser.add_argument("--num_cpus", type=int, default=48)
    parser.add_argument("--shard_size", type=int, default=100_000_000)
    parser.add_argument("--out_dir", type=str, default="fineweb10B")
    args = parser.parse_args()

    if args.version == "10B":
        remote_name = "sample-10BT"
    else:
        remote_name = "sample-100BT"

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Downloading FineWeb {args.version} from HuggingFace...")
    
    fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")
    
    total_docs = len(fw)
    total_shards = args.num_cpus * 21  # 48 CPUs * 21 chunks each = 1008 shards
    
    print(f"Total documents: {total_docs:,}")
    print(f"Using {args.num_cpus} CPUs, each processing 21 chunks")
    print(f"Shard size: {args.shard_size:,} tokens per shard")
    print(f"Total shards to create: {total_shards}")
    
    # Calculate documents per worker
    docs_per_worker = math.ceil(total_docs / args.num_cpus)
    
    print(f"Documents per worker: ~{docs_per_worker:,}")
    
    # Create dataset slices for each worker
    worker_slices = []
    for worker_id in range(args.num_cpus):
        start_idx = worker_id * docs_per_worker
        end_idx = min((worker_id + 1) * docs_per_worker, total_docs)
        
        if start_idx < total_docs:
            worker_slice = fw.select(range(start_idx, end_idx))
            worker_slices.append((worker_id, worker_slice))
    
    print(f"Created {len(worker_slices)} worker slices")
    
    # Process in parallel
    print("Starting parallel processing...")
    
    # Create partial function with fixed arguments
    worker_func = partial(
        process_worker_chunks,
        tokenizer_type=args.tokenizer,
        shard_size=args.shard_size,
        out_dir=args.out_dir,
        total_workers=args.num_cpus
    )
    
    # Process with multiprocessing
    with mp.Pool(processes=args.num_cpus) as pool:
        # Prepare arguments for each worker
        worker_args = [(worker_id, worker_slice) for worker_id, worker_slice in worker_slices]
        
        # Start all workers
        results = []
        for worker_id, worker_slice in worker_args:
            result = pool.apply_async(worker_func, (worker_id, worker_slice))
            results.append((worker_id, result))
        
        # Collect results
        all_chunk_results = []
        completed_workers = 0
        total_tokens_written = 0
        
        for worker_id, result in results:
            try:
                worker_results = result.get(timeout=7200)  # 2 hour timeout per worker
                all_chunk_results.extend(worker_results)
                completed_workers += 1
                
                worker_tokens = sum(r[1] for r in worker_results)
                total_tokens_written += worker_tokens
                print(f"Worker {worker_id} completed: {worker_tokens:,} tokens across {len(worker_results)} chunks")
                
            except Exception as e:
                print(f"Error getting results from worker {worker_id}: {e}")
    
    # Summary
    print(f"\nProcessing complete!")
    print(f"Workers completed: {completed_workers}/{args.num_cpus}")
    print(f"Total chunks created: {len(all_chunk_results)}")
    print(f"Total tokens written: {total_tokens_written:,}")
    print(f"Average tokens per shard: {total_tokens_written / max(1, len(all_chunk_results)):,.0f}")
    
    # Show chunk distribution
    val_chunks = [r for r in all_chunk_results if r[0] == 0]
    train_chunks = [r for r in all_chunk_results if r[0] != 0]
    
    print(f"Validation chunks: {len(val_chunks)}")
    print(f"Training chunks: {len(train_chunks)}")
    print(f"Shards written to {args.out_dir}")

if __name__ == "__main__":
    main()