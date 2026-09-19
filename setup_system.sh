#!/usr/bin/env bash
set -euo pipefail

# System-level setup for llm-scaler on Ubuntu 26.04 — RUN ONCE PER MACHINE.
# Installs Intel oneAPI, Level Zero / compute runtime, and build dependencies.
#
# Run as an ordinary user: it calls sudo only where root is needed, so that the
# venv setup_python.sh creates afterwards stays owned by you.

if [[ $EUID -eq 0 ]]; then
    echo "Run this as a normal user, not root — it calls sudo where needed." >&2
    exit 1
fi

echo "=== Step 1/4: Intel APT repositories ==="
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor | sudo tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | sudo tee /etc/apt/sources.list.d/intel-oneapi.list

wget -qO- https://repositories.intel.com/gpu/intel-graphics.key \
    | gpg --dearmor | sudo tee /usr/share/keyrings/intel-graphics-archive-keyring.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics-archive-keyring.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" \
    | sudo tee /etc/apt/sources.list.d/intel-gpu.list

sudo apt-get update

echo "=== Step 2/4: Intel oneAPI toolkit (DPC++, MKL, CCL, PTI) ==="
# The custom all-reduce binds sycl_ext_oneapi_inter_process_communication, which
# no compiler before 2026.1 provides. Pinned so a host built here and the
# container image (vllm/docker/Dockerfile, ONEAPI_COMPILER_VERSION) agree.
ONEAPI_COMPILER_VERSION="${ONEAPI_COMPILER_VERSION:-2026.1.1-325}"
sudo apt-get install -y --no-install-recommends \
    "intel-oneapi-compiler-dpcpp-cpp=${ONEAPI_COMPILER_VERSION}" \
    intel-oneapi-mkl-devel \
    intel-oneapi-ccl-devel \
    intel-pti

echo "=== Step 3/4: Level Zero + compute runtime (GPU driver) ==="
sudo apt-get install -y --no-install-recommends \
    intel-level-zero-gpu \
    libze1 \
    libze-dev \
    intel-opencl-icd

echo "=== Step 4/4: System build dependencies ==="
sudo apt-get install -y --no-install-recommends \
    cmake \
    ninja-build \
    git \
    curl \
    ca-certificates \
    python3-dev \
    python3-venv \
    protobuf-compiler

echo ""
echo "=== Verifying compiler ==="
# oneAPI vars.sh scripts read unset variables (OCL_ICD_FILENAMES et al); -u would abort.
set +u
source /opt/intel/oneapi/setvars.sh --force
set -u
icpx --version

# The all-reduce binding fails to compile without this header, well after setup.
if [[ ! -f /opt/intel/oneapi/compiler/latest/include/sycl/ext/oneapi/experimental/ipc_memory.hpp ]]; then
    echo "  ERROR: sycl_ext_oneapi_inter_process_communication is missing from" >&2
    echo "  $(readlink -f /opt/intel/oneapi/compiler/latest) — the custom all-reduce cannot build." >&2
    exit 1
fi
echo "  sycl_ext_oneapi_inter_process_communication: present"

echo ""
echo "=== Checking for Intel GPUs ==="
if [[ -d /dev/dri ]]; then
    sycl-ls 2>/dev/null | grep -i level_zero || echo "  No Level Zero devices reported by sycl-ls."
else
    echo "  /dev/dri does not exist — no GPU driver device nodes."
    echo "  Expected on a build-only host. Kernels still compile AOT (-device bmg),"
    echo "  but nothing can be executed or measured here."
fi

echo ""
echo "=== System setup complete ==="
echo "Next, as your normal user:  ./setup_python.sh"
