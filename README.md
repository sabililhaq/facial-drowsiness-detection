# Drowsiness / Yawn Detector

Real-time yawning detection using MediaPipe FaceMesh and a custom Logistic Regression classifier.

This repository is optimized for quick, out-of-the-box execution with 3 simple interfaces.

---

## ⚡ Quick Start (3 Steps)

### 1. Install
Sets up the Python 3.9–3.12 virtual environment (`venv`), installs all dependencies, and creates raw data directories.
```bash
chmod +x install.sh && ./install.sh
```

### 2. Train
Fits the Logistic Regression pipeline and saves it to `final_model.pkl`. 
* **If datasets are missing:** It automatically generates a dummy model so the demo can run immediately.
* **If datasets are present:** Place your training images under `data/raw/YawDD/yawn/` (and other folders created by `install.sh`) to train a real model from scratch.
```bash
./train.sh
```

### 3. Run Demo
Launches the real-time webcam detector using the generated model.
```bash
./demo.sh
```
*(Press `ESC` to quit the detector window).*

---

## 🧹 Clean Up (Optional)
To revert the workspace back to its original clean state and delete the virtual environment and generated models/caches:
```bash
chmod +x cleanup.sh && ./cleanup.sh
```

---

## 📁 Repository Structure

```
├── install.sh                     # [Interface 1] Setup environment & directories
├── train.sh                       # [Interface 2] Script runner for training/model generation
├── demo.sh                        # [Interface 3] Script runner for real-time webcam detector
├── cleanup.sh                     # [Extension] Revert virtualenv and generated files
├── train.py                       # Python script managing model training and dummy model fallback
├── demo.py                        # Entry point for webcam detector
├── logreg.py                      # Custom NumPy implementation of Logistic Regression
├── mypipe.py                      # Feature normalization and PCA pipeline
├── notebooks/                     # Exploratory notebooks for feature extraction/training
└── data/
    └── raw/                       # Place image datasets here
        ├── YawDD/                 # Yawning Detection Dataset (yawn/ / no_yawn/)
        └── ISDDS/                 # In-car Subjective Drowsiness Dataset
```

---

## ⚙️ Platform Notes

* **Python:** 3.9–3.12 required. MediaPipe is pinned to `0.10.14` in `requirements.txt` (newer 0.10.x releases removed the legacy FaceMesh API used by this project).
* **Webcam:** Required for the demo. `demo.sh` auto-runs `train.sh` if `final_model.pkl` is missing.
* **No dataset needed:** If `data/raw/` is empty, `train.sh` creates a dummy model so the demo still runs.
* **macOS:** Grant Camera access to Terminal / VS Code: *System Settings → Privacy & Security → Camera*.
* **Linux/Ubuntu:** If the OpenCV window fails to open, run:
  ```bash
  sudo apt-get install -y libgl1 libglib2.0-0
  ```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'mediapipe.python'` | Re-run `./install.sh` (ensures `mediapipe==0.10.14`). |
| `TypeError: ... not 'Series'` | Use the latest `demo.py` from this repo. |
| `Virtual environment 'venv' not found` | Run `./install.sh` first. |
| `No webcam and no dummy.mp4 found` | Plug in / enable a webcam, or grant camera permission. |
| OpenCV window does not appear (Linux) | Install `libgl1` and `libglib2.0-0` (see Platform Notes). |
