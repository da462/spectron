#!/bin/bash


# Array of model flavors
# flavors=(134M 150M 220M 325M 500M 780M 835M 1.4B 1.7B 2.7B)
flavors=(2.7B)

# Updated number of steps for each flavor
# num_steps=(60624 56176 39617 28243 19110 12503 11737 7268 6016 3795)
num_steps=(3795)

# Fixed learning rate
max_lr=0.07
chain_length=4


# Loop through all configurations
for i in "${!flavors[@]}"; do
    flavor="${flavors[$i]}"
    steps="${num_steps[$i]}"


    echo "=========================================="
    echo "Running configuration for flavor: $flavor"
    echo "Total steps: $steps"
    echo "=========================================="


    python bin/slurm_submitter.py \
        --cluster your_cluster\
        --job-name "${flavor}-sl-4bud" \
        --flavor "$flavor" \
        --chain-length "$chain_length" \
        --multi-node \
        --time 3:00:00 \
        --param optimizer=muon \
        --param low_rank=true \
        --param low_rank_ratio=0.25 \
        --grid max_lr="$max_lr" \
        --param frobenius_coef=0.0 \
        --param spectral_weight_decay=0.0 \
        --param spectral_lr_scaling=true \
        --param track_stable_rank=false \
        --param checkpoint_interval_hours=2.5 \
        --grid total_steps="$steps" \
        --param micro_batch_size=4 \
        --param weight_decay=1e-2 \
        --param nh_weight_decay=1e-2


    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully submitted job for $flavor"
    else
        echo "✗ Failed to submit job for $flavor"
    fi


    echo ""
done


echo "All jobs submitted!"


