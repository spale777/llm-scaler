#include <tuple>
// DeepSeek V4.1 Flash -- PyTorch bindings for the FP4 and TopK routines.

#include <torch/extension.h>
#include <c10/core/Device.h>
#include <c10/xpu/XPUStream.h>

void launch_fp4_gemm(
    torch::Tensor& a, 
    torch::Tensor& b_fp4, 
    torch::Tensor& b_scales, 
    torch::Tensor& c);

void launch_noaux_tc_topk(
    torch::Tensor& logits, 
    torch::Tensor& bias, 
    torch::Tensor& out_indices, 
    torch::Tensor& out_weights,
    int64_t top_k);

#define CHECK_XPU(x) TORCH_CHECK(x.device().is_xpu(), #x " must be a XPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_XPU(x); CHECK_CONTIGUOUS(x)

torch::Tensor deepseek_v41_fp4_gemm(
    torch::Tensor a,
    torch::Tensor b_fp4,
    torch::Tensor b_scales) 
{
    CHECK_INPUT(a);
    CHECK_INPUT(b_fp4);
    CHECK_INPUT(b_scales);
    // Xe2 XMX has no FP4 and no FP8 matrix arithmetic, so the weights are
    // unpacked to FP16 and fed to the FP16 DPAS. The activation is FP16 for
    // the same reason: quantising it would only add a dequant on the hot path.
    TORCH_CHECK(a.scalar_type() == torch::kFloat16, "A must be Float16");
    TORCH_CHECK(b_fp4.scalar_type() == torch::kUInt8, "b_fp4 must be packed UInt8");
    TORCH_CHECK(b_scales.scalar_type() == torch::kUInt8,
                "b_scales must be UE8M0 bytes (UInt8)");
    TORCH_CHECK(a.dim() == 2 && b_fp4.dim() == 2 && b_scales.dim() == 2,
                "deepseek_v41_fp4_gemm: expected A [M, K], b_fp4 [N, K/2], "
                "b_scales [N, K/32]");
    TORCH_CHECK(a.size(1) == b_fp4.size(1) * 2,
                "FP4 K dimension mismatch: b_fp4 must be packed K/2, got K=",
                a.size(1), " and b_fp4.size(1)=", b_fp4.size(1));

    const int64_t K = a.size(1);
    constexpr int64_t GROUP = 32;
    TORCH_CHECK(K % GROUP == 0,
                "deepseek_v41_fp4_gemm: K must be a multiple of the ", GROUP,
                "-element scale group, got K=", K);
    TORCH_CHECK(b_scales.size(0) == b_fp4.size(0) && b_scales.size(1) == K / GROUP,
                "deepseek_v41_fp4_gemm: b_scales must be [N, K/", GROUP, "] = [",
                b_fp4.size(0), ", ", K / GROUP, "], got [", b_scales.size(0), ", ",
                b_scales.size(1), "]");

    auto c = torch::empty({a.size(0), b_fp4.size(0)}, a.options());
    launch_fp4_gemm(a, b_fp4, b_scales, c);
    return c;
}

std::tuple<torch::Tensor, torch::Tensor> deepseek_v41_noaux_tc_topk(
    torch::Tensor logits,
    torch::Tensor bias,
    int64_t top_k)
{
    CHECK_INPUT(logits);
    CHECK_INPUT(bias);
    TORCH_CHECK(logits.scalar_type() == torch::kFloat16, "logits must be Float16");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat16, "bias must be Float16");
    TORCH_CHECK(logits.dim() == 2 && bias.dim() == 1,
                "deepseek_v41_noaux_tc_topk: expected logits [T, E] and bias [E], got ",
                logits.dim(), "D and ", bias.dim(), "D");
    TORCH_CHECK(logits.size(1) == 384 && bias.size(0) == 384,
                "deepseek_v41_noaux_tc_topk: the kernel is instantiated for 384 "
                "experts, got logits.size(1)=", logits.size(1),
                " bias.size(0)=", bias.size(0));
    // The launcher instantiates these three only; anything else would fall
    // through the switch and throw from device code.
    TORCH_CHECK(top_k == 4 || top_k == 6 || top_k == 8,
                "deepseek_v41_noaux_tc_topk: top_k must be 4, 6 or 8, got ", top_k);
    auto out_indices = torch::empty({logits.size(0), top_k}, logits.options().dtype(torch::kInt32));
    auto out_weights = torch::empty({logits.size(0), top_k}, logits.options().dtype(torch::kFloat16));
    
    launch_noaux_tc_topk(logits, bias, out_indices, out_weights, top_k);
    
    return std::make_tuple(out_weights, out_indices);
}

TORCH_LIBRARY_FRAGMENT(custom_esimd_kernels_vllm, m) {
    m.def("deepseek_v41_fp4_gemm", &deepseek_v41_fp4_gemm);
    m.def("deepseek_v41_noaux_tc_topk", &deepseek_v41_noaux_tc_topk);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
