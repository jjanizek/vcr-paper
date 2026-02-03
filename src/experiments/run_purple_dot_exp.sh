#!/bin/bash

python purple_dot_experiment.py \
    --fst12_folder your/path/to/synthetic_skin/fst12 \
    --fst56_folder your/path/to/synthetic_skin/fst56 \
    --num_malignant 4 \
    --benign_multiplier 3 \
    --demo_skin_type both \
    --num_augments 10 \
    --output purple_dot_results.png \
    --example_output purple_dot_examples.png \
    --seed 42