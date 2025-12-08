#!/usr/bin/env bash

# Default environment name and YAML file
DEFAULT_ENV="finance"
DEFAULT_ENV_FILE="environment.yml"

# Optional parameters
ENV_NAME="${1:-$DEFAULT_ENV}"
ENV_FILE="${2:-$DEFAULT_ENV_FILE}"

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
if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Activating existing environment '$ENV_NAME'..."
    conda activate "$ENV_NAME"
else
    echo "Environment '$ENV_NAME' does not exist. Creating..."
    conda env create -f "$ENV_FILE" -y
    conda activate "$ENV_NAME"
fi

# Update environment from YAML file
if [[ -f "$ENV_FILE" ]]; then
    echo "Updating environment '$ENV_NAME' from '$ENV_FILE'..."
    conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    echo "Environment file '$ENV_FILE' not found. Skipping YAML update."
fi

# Optional: update pip requirements if present
for req_file in requirements.txt requirements_jupyter.txt requirements_test.txt; do
    if [[ -f "$req_file" ]]; then
        echo "Installing/updating packages from $req_file..."
        pip install --upgrade -r "$req_file"
    fi
done

echo "Environment '$ENV_NAME' has been updated."