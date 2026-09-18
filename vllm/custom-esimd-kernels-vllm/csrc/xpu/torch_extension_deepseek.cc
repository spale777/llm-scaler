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
    TORCH_CHECK(false,
                "deepseek_v41_fp4_gemm is not implemented: the kernel loads "
                "neither operand and stores no result. Xe2 XMX takes FP16, "
                "BF16, INT8, INT4 and INT2 only, so an FP4 path must "
                "dequantise before the DPAS.");

    CHECK_INPUT(a);
    CHECK_INPUT(b_fp4);
    CHECK_INPUT(b_scales);
    TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn || a.scalar_type() == torch::kFloat8_e5m2, "A must be FP8");
    TORCH_CHECK(b_fp4.scalar_type() == torch::kUInt8, "b_fp4 must be packed UInt8");
    TORCH_CHECK(a.size(1) == b_fp4.size(1) * 2, "FP4 K dimension mismatch: b_fp4 must be packed K/2");
    auto c = torch::empty({a.size(0), b_fp4.size(0)}, a.options().dtype(torch::kBFloat16));
    launch_fp4_gemm(a, b_fp4, b_scales, c);
    return c;
}

std::tuple<torch::Tensor, torch::Tensor> deepseek_v41_noaux_tc_topk(
    torch::Tensor logits,
    torch::Tensor bias,
    int64_t top_k)
{
    TORCH_CHECK(false,
                "deepseek_v41_noaux_tc_topk is not implemented: the kernel "
                "neither loads its inputs nor stores its outputs, and the "
                "group-limited stage of noaux_tc routing is absent.");

    CHECK_INPUT(logits);
    CHECK_INPUT(bias);
    TORCH_CHECK(logits.scalar_type() == torch::kFloat16, "logits must be Float16");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat16, "bias must be Float16");
    TORCH_CHECK(logits.size(1) == 384 && bias.size(0) == 384,
                "deepseek_v41_noaux_tc_topk: the kernel is instantiated for 384 "
                "experts, got logits.size(1)=", logits.size(1),
                " bias.size(0)=", bias.size(0));
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
