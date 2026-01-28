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

    # Determine if the model is self-guided or regular low-rank
    if [[ "$checkpoint_name" == *"sg"* || "$checkpoint_name" == *"self-guided"* ]]; then
        echo "Evaluating self-guided model: $checkpoint_name..."
        python evaluation/evaluate_lm_harness.py \
            --checkpoint "$checkpoint_path" \
            --model_type titan \
            --tasks "$task_string" \
            --tokenizer llama \
            --self_guided \
            --low_rank_ratio 0.25 \
            --checkpoint_name "$checkpoint_name" \
            --csv_output "$CSV_FILE"
    else
        echo "Evaluating low-rank model: $checkpoint_name..."
        python evaluation/evaluate_lm_harness.py \
            --checkpoint "$checkpoint_path" \
            --model_type titan \
            --tasks "$task_string" \
            --tokenizer llama \
            --low_rank \
            --low_rank_ratio 0.25 \
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
