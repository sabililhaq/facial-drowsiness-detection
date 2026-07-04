#!/usr/bin/env bash
# install.sh — sets up the drowsiness detection project
# Supports: Ubuntu 20.04+, macOS 12+, Python 3.9–3.12

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err_exit(){ echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

for py_cmd in python3.11 python3.12 python3.10 python3.9 python3 python; do
  if command -v "$py_cmd" >/dev/null 2>&1; then
    PY_CANDIDATE=$(command -v "$py_cmd")
    PY_MAJOR=$("$PY_CANDIDATE" -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$("$PY_CANDIDATE" -c "import sys; print(sys.version_info.minor)")
    if [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -ge 9 && "$PY_MINOR" -le 12 ]]; then
      PYTHON="$PY_CANDIDATE"
      PY_VER="$PY_MAJOR.$PY_MINOR"
      break
    fi
  fi
done

if [[ -z "$PYTHON" ]]; then
  PYTHON=$(command -v python3 || command -v python || err_exit "Python not found.")
  PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
  PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")
  PY_VER="$PY_MAJOR.$PY_MINOR"
fi

info "Detected Python $PY_VER at $PYTHON"
[[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -ge 9 && "$PY_MINOR" -le 12 ]] \
  || err_exit "Python 3.9–3.12 required (got $PY_VER). mediapipe does not support 3.13+."

VENV_DIR="venv"
if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating virtual environment in ./$VENV_DIR ..."
  "$PYTHON" -m venv "$VENV_DIR"
else
  warn "Virtual environment ./$VENV_DIR already exists — skipping creation."
fi

source "$VENV_DIR/bin/activate"
info "Activated: $(which python)"

info "Creating raw dataset directories..."
mkdir -p data/raw/YawDD/yawn data/raw/YawDD/no_yawn data/raw/ISDDS/yawn data/raw/ISDDS/no_yawn

info "Upgrading pip..."
pip install --quiet --upgrade pip

info "Installing dependencies from requirements.txt..."
pip install --quiet -r requirements.txt

info "Installing Jupyter for notebooks..."
pip install --quiet notebook

info "Verifying imports..."
python - <<'PYEOF'
import cv2, mediapipe, numpy, pandas, sklearn
print(f"  opencv-python : {cv2.__version__}")
print(f"  mediapipe     : {mediapipe.__version__}")
print(f"  numpy         : {numpy.__version__}")
print(f"  pandas        : {pandas.__version__}")
print(f"  scikit-learn  : {sklearn.__version__}")
PYEOF

info "Checking webcam access..."
python - <<'PYEOF'
import cv2, sys
cap = cv2.VideoCapture(0)
ok = cap.isOpened(); cap.release()
print("  Webcam: OK" if ok else "  Webcam: NOT FOUND (demo.py needs a working camera)")
PYEOF

[[ -f "final_model.pkl" ]] && info "final_model.pkl found." \
  || warn "final_model.pkl NOT found — run ./train.sh next (or ./demo.sh, which runs training automatically)."

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN} Installation complete.${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Run the training step (generates dummy model if dataset is empty):"
echo "    ./train.sh"
echo ""
echo "  Run the real-time detector:"
echo "    ./demo.sh"
echo ""
echo "  Press ESC inside the detector window to quit."
