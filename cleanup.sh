#!/usr/bin/env bash
# cleanup.sh — removes all installed environments and dummy files, reverting the repo to its clean state.

echo "Cleaning up environment..."

# Deactivate virtual environment if it is currently active
deactivate 2>/dev/null || true

# Remove generated directories and files
rm -rf venv __pycache__ final_model.pkl data/*.csv

echo "Cleanup complete! Virtual environment and temporary files have been removed."
