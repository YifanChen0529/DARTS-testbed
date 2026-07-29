#!/usr/bin/env bash
# DARTS-Simulation: create virtual environment and install dependencies.
# Run from project root: ./setup_env.sh   or   bash setup_env.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_NAME="darts_venv"

echo "Creating virtual environment: $VENV_NAME"
python3 -m venv "$VENV_NAME"

echo "Upgrading pip..."
"$VENV_NAME/bin/pip" install --upgrade pip

echo "Installing dependencies from requirements.txt..."
"$VENV_NAME/bin/pip" install -r requirements.txt

echo "Done. Activate with: source $VENV_NAME/bin/activate"
