#!/bin/bash
set -e

# Ensure common local bin directories are in PATH
export PATH="$HOME/.local/bin:$PATH"

echo "=== Starting Superset Workspace Setup ==="
echo "Workspace Path: $SUPERSET_WORKSPACE_PATH"
echo "Root Path: $SUPERSET_ROOT_PATH"

if command -v uv >/dev/null 2>&1; then
    echo "Found 'uv', using it for ultra-fast setup..."
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment with uv..."
        uv venv .venv
    fi
    echo "Installing dependencies with uv..."
    uv pip install --index-url https://pypi.org/simple -e '.[dev]'
else
    echo "'uv' not found, falling back to standard python venv..."
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi
    echo "Upgrading pip and setuptools..."
    .venv/bin/pip install --index-url https://pypi.org/simple --upgrade pip setuptools wheel
    echo "Installing dependencies..."
    .venv/bin/pip install --index-url https://pypi.org/simple -e '.[dev]'
fi

echo "=== Setup completed successfully! ==="
