#!/bin/bash

# Run all combinations of model, use_demos, and filter_skin_tone

models=("OpenFlamingo-4B" "OpenFlamingo-3B-Instruct")
skin_tones=("All" "12" "56")

for model in "${models[@]}"; do
    for skin_tone in "${skin_tones[@]}"; do
        # Run without demos
        echo "================================================"
        echo "Running: $model, skin_tone=$skin_tone, no ICL"
        echo "================================================"
        python generate_ddi_explanations_with_pvals.py \
            --model "$model" \
            --filter_skin_tone "$skin_tone"
        
        # Run with demos
        echo "================================================"
        echo "Running: $model, skin_tone=$skin_tone, with ICL"
        echo "================================================"
        python generate_ddi_explanations_with_pvals.py \
            --model "$model" \
            --filter_skin_tone "$skin_tone" \
            --use_demos
    done
done

echo "================================================"
echo "All experiments complete!"
echo "================================================"