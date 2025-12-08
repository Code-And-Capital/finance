#!/usr/bin/env bash

# Default environment and test requirements
DEFAULT_ENV="finance"
TEST_REQUIREMENTS="requirements_test.txt"

# Use parameter if provided
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

# Activate environment
echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# Install test requirements if the file exists
if [[ -f "$TEST_REQUIREMENTS" ]]; then
    echo "Installing test requirements from $TEST_REQUIREMENTS..."
    pip install -r "$TEST_REQUIREMENTS"
fi

# Run all tests and generate coverage
echo "Running tests..."
if command -v pytest >/dev/null 2>&1; then
    pytest --cov=.
else
    echo "pytest not found. Please make sure pytest is installed in '$ENV_NAME'."
    return 1
fi

echo "Tests completed."
