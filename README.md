# Drowsiness / Yawn Detector

Real-time yawning detection from a webcam using MediaPipe FaceMesh and Logistic Regression.

## Run It

```bash
chmod +x install.sh && ./install.sh
./train.sh
./demo.sh
```

Press **ESC** to quit the demo window.

No dataset? `./train.sh` still works — it creates a dummy model so the demo can run.

`./demo.sh` also runs `./train.sh` automatically if `final_model.pkl` is missing.

## Requirements

- Python **3.9–3.12**
- A working **webcam**
- **macOS:** allow camera access for Terminal — *System Settings → Privacy & Security → Camera*
- **Linux:** if the window does not open, run `sudo apt-get install -y libgl1 libglib2.0-0`

## Train With Real Data (Optional)

Put images in folders created by `install.sh`:

```
data/raw/YawDD/yawn/       # training
data/raw/YawDD/no_yawn/    # training
data/raw/ISDDS/yawn/       # testing
data/raw/ISDDS/no_yawn/    # testing
```

`./train.sh` trains on **YawDD** and evaluates on **ISDDS**.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'mediapipe.python'` | Run `./install.sh` again |
| `Virtual environment 'venv' not found` | Run `./install.sh` first |
| `No webcam ... found` | Enable webcam / grant camera permission |
| OpenCV window missing (Linux) | Install `libgl1` and `libglib2.0-0` |

## Clean Up

```bash
chmod +x cleanup.sh && ./cleanup.sh
```

Removes `venv/`, `final_model.pkl`, and cache files.

## Project Files

```
install.sh   → setup environment
train.sh     → train model (or create dummy model)
demo.sh      → run webcam detector
cleanup.sh   → reset workspace
train.py     → training logic
demo.py      → demo logic
logreg.py    → logistic regression
mypipe.py    → preprocessing pipeline
notebooks/   → exploratory notebooks
data/raw/    → dataset folder
```