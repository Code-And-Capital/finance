#!/usr/bin/env bash

# Default values
ENV_NAME="finance"
ENV_FILE="environment.yml"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            ENV_NAME="$2"
            shift 2
            ;;
        -f|--file)
            ENV_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            return 1  # source-safe
            ;;
    esac
done

# Check YAML file exists
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: Environment file '$ENV_FILE' not found."
    return 1
fi

# Find conda or micromamba
if command -v conda >/dev/null 2>&1; then
    CONDA_CMD="conda"
elif command -v micromamba >/dev/null 2>&1; then
    CONDA_CMD="micromamba"
else
    echo "Error: conda or micromamba not found."
    return 1
fi

echo "Using: $CONDA_CMD"

# Initialize conda in this shell
if [[ "$CONDA_CMD" == "conda" ]]; then
    eval "$($CONDA_CMD shell.bash hook)"
fi

# Check if environment exists
ENV_EXISTS=false
if $CONDA_CMD info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    ENV_EXISTS=true
fi

# Prompt user if env exists
if $ENV_EXISTS; then
    echo -n "Environment '$ENV_NAME' already exists. Overwrite it? (y/n): "
    read RESP
    if [[ "$RESP" == "y" ]]; then
        echo "Removing existing environment '$ENV_NAME'..."
        $CONDA_CMD env remove -n "$ENV_NAME" -y
    else
        echo "Keeping existing environment. Activating..."
        $CONDA_CMD activate "$ENV_NAME"
        return 0
    fi
fi

# Create environment from YAML
echo "Creating environment '$ENV_NAME' from '$ENV_FILE'..."
$CONDA_CMD env create -f "$ENV_FILE" -y

# Activate the environment
echo "Activating environment '$ENV_NAME'..."
$CONDA_CMD activate "$ENV_NAME"

echo "Environment '$ENV_NAME' is active."