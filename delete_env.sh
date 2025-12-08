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
    return 1  # source-safe
fi

# Check if environment exists
if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Deleting Conda environment: $ENV_NAME"
    conda env remove -n "$ENV_NAME" -y
    echo "Environment '$ENV_NAME' deleted."
else
    echo "Environment '$ENV_NAME' does not exist."
fi
