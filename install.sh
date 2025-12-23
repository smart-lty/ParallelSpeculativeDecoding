#!/bin/bash
# Install dependencies for Parallel Speculative Decoding using uv
# Updated: 2025-12-23

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment
uv venv
source .venv/bin/activate

# Install PyTorch (CUDA 12.1) - latest stable
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
uv pip install \
    transformers==4.57.3 \
    accelerate>=1.0.0 \
    fschat==0.2.36 \
    numpy<2.0 \
    tqdm \
    ipdb \
    shortuuid \
    gradio

echo "Installation complete! Activate with: source .venv/bin/activate"
