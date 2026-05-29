#!/usr/bin/env bash
# train.sh — Runs the training step (falls back to generating dummy model if datasets are empty)

set -e
if [[ ! -d "venv" ]]; then
  echo "[ERROR] Virtual environment 'venv' not found. Please run ./install.sh first."
  exit 1
fi

source venv/bin/activate
python train.py
