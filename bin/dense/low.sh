

#!/bin/bash

# Array of model flavors
#flavors=(60M 92M 134M 150M 220M 325M 500M 780M 835M 1.4B 1.7B 2.7B)
# Array of model flavors
# flavors=(60M 92M 134M 150M 220M 325M 500M 780M 835M 1.4B 1.7B)
flavors=(1.4B)
# flavors=(780M)

# Corresponding number of steps for each flavor
num_step=(26703)

chain_lengths=(13)


# Fixed learning rate
# max_lr=0.05


# Loop through all configurations
for i in "${!flavors[@]}"; do
    flavor="${flavors[$i]}"
    steps="${num_step[$i]}"
    chain_length="${chain_lengths[$i]}"
    
    echo "=========================================="
    echo "Running configuration for flavor: $flavor"
    echo "Total steps: $steps"
    echo "=========================================="
    
    python bin/slurm_submitter.py \
        --cluster your_cluster \
        --job-name "fm-low_${flavor}" \
        --flavor "$flavor" \
        --chain-length "$chain_length" \
        --multi-node \
        --time 3:00:00 \
        --param optimizer=muon \
        --param low_rank=true \
        --param low_rank_ratio=0.25 \
        --grid max_lr=0.05,0.07,0.01 \
        --grid frobenius_coef=0.0 \
        --grid spectral_weight_decay=0.0 \
        --param spectral_lr_scaling=true \
        --param track_stable_rank=false \
        --param checkpoint_interval_hours=2.5 \
        --param save_best_checkpoint=true \
        --param total_steps="$steps" \
        --param micro_batch_size=4 \
        --grid weight_decay=1e-2 \
        --grid nh_weight_decay=1e-2 
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully submitted job for $flavor"
    else
        echo "✗ Failed to submit job for $flavor"
    fi
    
    echo ""
done

echo "All jobs submitted!"
