#!/usr/bin/env bash
# demo.sh — Runs the real-time drowsiness detector demo

set -e
if [[ ! -d "venv" ]]; then
  echo "[ERROR] Virtual environment 'venv' not found. Please run ./install.sh first."
  exit 1
fi

if [[ ! -f "final_model.pkl" ]]; then
  echo "[WARN] final_model.pkl not found. Running training script first..."
  ./train.sh
fi

source venv/bin/activate
python demo.py
