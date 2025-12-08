#!/usr/bin/env bash

# Default environment and target directory
DEFAULT_ENV="finance"
ENV_NAME="${1:-$DEFAULT_ENV}"
TARGET_DIR="${2:-.}"

# Load Conda into the shell
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda not found!"
    return 1
fi

# Activate environment
echo "Activating Conda environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Check if black is installed
if ! command -v black >/dev/null 2>&1; then
    echo "ERROR: 'black' is not installed in '$ENV_NAME'."
    return 1
fi

# Run black
echo "Running Black code formatter on '$TARGET_DIR'..."
black "$TARGET_DIR"

# Stage reformatted files
echo "Staging reformatted files..."
git add "$TARGET_DIR"

echo "Formatting complete. Changes have been staged for commit."
