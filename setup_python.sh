#!/usr/bin/env bash
set -euo pipefail

# In-project venv with PyTorch XPU + build tooling — safe to re-run, no root.
# Requires setup_system.sh to have been run once on this machine.
# Override the location with:  VENV_DIR=/some/path ./setup_python.sh

if [[ $EUID -eq 0 ]]; then
    echo "Do not run this as root — the venv must be owned by you." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

if [[ ! -d /opt/intel/oneapi ]]; then
    echo "/opt/intel/oneapi not found — run ./setup_system.sh first." >&2
    exit 1
fi

echo "=== Step 1/3: Python venv at $VENV_DIR ==="
if [[ -d "$VENV_DIR" ]]; then
    echo "  Reusing existing venv."
else
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "=== Step 2/3: PyTorch XPU + build tooling ==="
# torch 2.12.0+xpu requires setuptools<82; an unpinned upgrade breaks SyclExtension builds.
pip install --upgrade pip wheel 'setuptools<82'
pip install torch==2.12.0+xpu torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/xpu
pip install ninja numpy sympy packaging

echo "=== Step 3/3: Verify ==="
# oneAPI vars.sh scripts read unset variables (OCL_ICD_FILENAMES et al); -u would abort.
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
python3 -c "import torch; print(f'PyTorch {torch.__version__}, XPU available: {torch.xpu.is_available()}')"
python3 -c "import setuptools; print(f'setuptools {setuptools.__version__} (must be <82)')"

echo ""
echo "=== Python setup complete ==="
echo "Activate for future sessions:"
echo "  source $VENV_DIR/bin/activate"
echo "  source /opt/intel/oneapi/setvars.sh --force"
echo ""
echo "Build the ESIMD kernels (from the repo root):"
echo "  cd vllm/custom-esimd-kernels-vllm && TORCH_XPU_ARCH_LIST=bmg-g21 pip install -e ."
