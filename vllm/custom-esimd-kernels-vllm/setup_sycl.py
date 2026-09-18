import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension
from esimd_build_extention import BuildExtension

root = Path(__file__).parent.resolve()

import torch

# Both Battlemage dies get their own device binary: B60 is BMG-G21 and B70 is
# BMG-G31. The family target `bmg` builds one binary that runs on both, so a
# G31-specific specialization would silently not be selected. ocloc accepts the
# pair and emits an archive holding both IP versions.
BMG_DEVICES = "bmg-g21,bmg-g31"

torch_include = str(Path(torch.__file__).parent / "include")

ext_modules = [
    SyclExtension(
        name="custom_esimd_kernels_vllm_ar",
        sources=[
            "csrc/xpu/torch_extension_ar.cc",
            "csrc/xpu/ipc_allreduce.sycl",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel",
                     "-fsycl-targets=spir64_gen",
                     "-Xs", f"-device {BMG_DEVICES} -options -cl-intel-enable-auto-fma",
                     "-funroll-loops",
                     "-fno-sycl-early-optimizations"],
        }
    ),
    SyclExtension(
        name="custom_esimd_kernels_vllm.custom_esimd_kernels",
        sources=[
            "csrc/xpu/esimd_kernel.sycl",
            "csrc/xpu/torch_extension.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl-targets=spir64_gen", "-funroll-loops", "-Xs", f"-device {BMG_DEVICES} -options -cl-intel-enable-auto-fma", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
]

### for lgrf esimd kernels (GDN conv fused — separate module, doubleGRF)
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.custom_esimd_kernels_lgrf",
        sources=[
            "csrc/xpu/esimd_kernel_lgrf.sycl",
            "csrc/xpu/torch_extension_lgrf.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl-targets=spir64_gen", "-funroll-loops", "-Xs", f"-device {BMG_DEVICES} -options -cl-intel-enable-auto-fma", "-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### for lgrf esimd kernels

### The FP8 block-scale prefill kernel holds ~5.5 KB of live DPAS state and
### requests the large register file itself, via grf_size<256> on the kernel.
### The module also carries the light topk/scatter/silu/gather kernels, whose
### live state is under 300 B, so a module-wide -doubleGRF would halve their
### occupancy to buy nothing. per_kernel code split keeps the two apart.
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.custom_esimd_kernels_moe",
        sources=[
            "csrc/xpu/esimd_kernel_moe.sycl",
            "csrc/xpu/torch_extension_moe.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl-targets=spir64_gen", "-funroll-loops", "-Xs", f"-device {BMG_DEVICES} -options -cl-intel-enable-auto-fma", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### MoE auxiliary kernels

### FP8 GEMM (M>1) — uses DPAS, compile with JIT only (no AOT to avoid device mismatch)
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.custom_esimd_kernels_gemm",
        sources=[
            "csrc/xpu/esimd_kernel_gemm.sycl",
            "csrc/xpu/torch_extension_gemm.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl-targets=spir64_gen", "-funroll-loops", "-Xs", f"-device {BMG_DEVICES} -options -cl-intel-enable-auto-fma", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### FP8 GEMM kernels

### TopK V2 — vectorized softmax+topk for 512 experts (AOT for BMG)
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.esimd_topk_v2",
        sources=[
            "csrc/xpu/esimd_kernel_topk_v2.sycl",
            "csrc/xpu/torch_extension_topk_v2.cc",
        ],
        include_dirs=[
            root / "include",
            root / "csrc",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl-targets=spir64_gen", "-funroll-loops", "-Xs", f"-device {BMG_DEVICES} -options -cl-intel-enable-auto-fma", "-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### TopK V2 kernels

### Eagle kernels (GDN + Page Attention) — from custom-esimd-kernels-vllm-eagle
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.eagle_ops",
        sources=[
            "csrc/eagle/eagle.sycl",
        ],
        include_dirs=[
            root / "csrc" / "eagle",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-fsycl-targets=spir64_gen", "-funroll-loops", "-Xs", f"-device {BMG_DEVICES} -options -cl-intel-enable-auto-fma", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### Eagle kernels

### MoE Batch kernels (Router, TopK, Up/Down, Accumulate) — from custom-esimd-kernels-vllm-moe-batch-test
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.moe_ops",
        sources=[
            "csrc/moe_batch/moe.sycl",
        ],
        include_dirs=[],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20"],
            "sycl": ["-fsycl-targets=spir64_gen", "-funroll-loops", "-Xs", f"-device {BMG_DEVICES} -options -cl-intel-enable-auto-fma", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)
### MoE Batch kernels

### DeepSeek V4.1: FP4 GEMM and the group-limited noaux_tc router. Built here
### as well as in setup.py so both build paths produce the same set of modules;
### the package import of deepseek_v41 is otherwise absent from sycl builds.
ext_modules.append(
    SyclExtension(
        name="custom_esimd_kernels_vllm.deepseek_v41",
        sources=[
            "csrc/xpu/deepseek_kernels.sycl",
            "csrc/xpu/torch_extension_deepseek.cc",
        ],
        include_dirs=[
            root.joinpath("csrc"),
            root.joinpath("csrc/xpu"),
            root.joinpath("csrc/deepseek_v41"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "sycl": ["-fsycl-targets=spir64_gen", "-funroll-loops", "-Xs", f"-device {BMG_DEVICES} -options -cl-intel-enable-auto-fma", "-ffast-math", "-fsycl-device-code-split=per_kernel",
                     f"-I{torch_include}"],
        },
        extra_link_args=["-Wl,-rpath,$ORIGIN/../../torch/lib"],
        py_limited_api=False,
    )
)

setup(
    name="custom-esimd-kernels-vllm",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
