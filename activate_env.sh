#!/bin/bash

# Default environment name
DEFAULT_ENV="finance"

# Use parameter if provided, otherwise default
ENV_NAME="${1:-$DEFAULT_ENV}"

# Load Conda into the shell
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda not found!"
    return 1  # use return instead of exit for source-safety
fi

# Activate the environment
conda activate "$ENV_NAME"

# Optional: print info
echo "Activated Conda environment: $ENV_NAME"
