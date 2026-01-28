#!/bin/bash

# Define directories and tasks
DIRS=(

)

TASKS=("hellaswag" "piqa" "arc_easy")
# CSV output file
CSV_FILE="evaluation_results.csv"

# Remove existing CSV file to start fresh
if [ -f "$CSV_FILE" ]; then
    echo "Removing existing $CSV_FILE..."
    rm "$CSV_FILE"
fi

# Loop over directories
for dir in "${DIRS[@]}"; do
    checkpoint_path="$HOME/checkpoints/low_rank/$dir/checkpoint_final.pt"

    # Extract a clean checkpoint name from the directory
    checkpoint_name="${dir%/}"  # Remove trailing slash

    # Create task list as comma-separated string
    task_string=$(IFS=,; echo "${TASKS[*]}")

    # Determine if the model is low-rank or dense
    if [[ $dir == *"low"* ]]; then
        echo "Evaluating low-rank model: $checkpoint_name..."
        # Include --low_rank and --low_rank_ratio in the command
        python evaluation/evaluate_lm_harness.py \
            --checkpoint "$checkpoint_path" \
            --model_type titan \
            --tasks "$task_string" \
            --tokenizer llama \
            --low_rank \
            --low_rank_ratio 0.25 \
            --checkpoint_name "$checkpoint_name" \
            --csv_output "$CSV_FILE"
    else
        echo "Evaluating dense model: $checkpoint_name..."
        # Omit --low_rank and --low_rank_ratio
        python evaluation/evaluate_lm_harness.py \
            --checkpoint "$checkpoint_path" \
            --model_type titan \
            --tasks "$task_string" \
            --tokenizer llama \
            --checkpoint_name "$checkpoint_name" \
            --csv_output "$CSV_FILE"
    fi

    echo "Completed evaluation for $checkpoint_name"
    echo "----------------------------------------"
done

echo ""
echo "All evaluations complete!"
echo "Results saved to $CSV_FILE"
echo ""
echo "Preview of results:"
head -20 "$CSV_FILE"