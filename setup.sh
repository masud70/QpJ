#!/usr/bin/env bash
# Sets up QpJ v2 on a Linux GPU machine: venv, packages, sensor access, dataset, caches and backend checks.
# Safe to re-run. Sensor permissions last until the next reboot; nothing else outside this folder is changed.
set -euo pipefail
cd "$(dirname "$0")"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu118}"

step() { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*"; }

step "Checking system"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "Python >= 3.10 is required"; exit 1; }
if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    warn "nvidia-smi not found"
fi
free_gb=$(df -Pk . | awk 'NR == 2 {print int($4 / 1048576)}')
[ "$free_gb" -ge 15 ] || warn "only ${free_gb} GB free; about 15 GB is needed"

step "Python environment (.venv)"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
python -c "import torch" 2> /dev/null || pip install torch torchvision --index-url "$TORCH_INDEX"
pip install -q -r requirements.txt
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

step "Energy sensor access (until next reboot; asks for sudo password)"
sudo chmod -R a+r /sys/class/powercap/intel-rapl* || warn "could not make all RAPL counters readable"

step "Dataset (about 10.6 GB download) and caches"
python run_all.py prepare

step "Energy backend checks"
mkdir -p results/setup_checks
for backend in native codecarbon codegreen; do
    log="results/setup_checks/$backend.txt"
    if python run_all.py check --backend "$backend" > "$log" 2>&1; then
        echo "$backend: ok"
    else
        warn "$backend check failed (see $log)"
    fi
    grep -E "^(idle|load) " "$log" || true
done

step "Done"
echo "Activate the environment with: source .venv/bin/activate"
echo "Then start, for example: python run_all.py train --backend native --stopping fixed_budget"
