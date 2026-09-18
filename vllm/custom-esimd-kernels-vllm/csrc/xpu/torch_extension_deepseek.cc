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

void launch_lightning_indexer(
    torch::Tensor& q,
    torch::Tensor& index_k,
    torch::Tensor& weights,
    torch::Tensor& scores,
    int64_t compress_len);

void launch_candidate_blocks(
    torch::Tensor& scores,
    torch::Tensor& keep,
    int64_t block_size,
    int64_t topk_blocks,
    int64_t compress_len);

void launch_sparse_attn(
    torch::Tensor& q,
    torch::Tensor& kv,
    torch::Tensor& attn_sink,
    torch::Tensor& topk_idxs,
    torch::Tensor& out,
    double scale);

void launch_engram_gate(
    torch::Tensor& h,
    torch::Tensor& key,
    torch::Tensor& weight,
    torch::Tensor& value,
    torch::Tensor& mask,
    torch::Tensor& out,
    double eps,
    double clamp_value);

void launch_act_quant(
    torch::Tensor& x,
    torch::Tensor& y,
    torch::Tensor& s,
    int64_t group_size,
    bool round_scale,
    double qmax,
    double amax_floor);

void launch_o_group_proj(
    torch::Tensor& x,
    torch::Tensor& w,
    torch::Tensor& out);

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

torch::Tensor deepseek_v41_lightning_indexer(
    torch::Tensor q,
    torch::Tensor index_k,
    torch::Tensor weights,
    int64_t compress_len)
{
    CHECK_INPUT(q);
    CHECK_INPUT(index_k);
    CHECK_INPUT(weights);
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 &&
                index_k.scalar_type() == torch::kFloat16 &&
                weights.scalar_type() == torch::kFloat16,
                "deepseek_v41_lightning_indexer: q, index_k and weights must be Float16");
    TORCH_CHECK(q.dim() == 3, "expected q [S, H, D], got ", q.dim(), "D");
    TORCH_CHECK(index_k.dim() == 2, "expected index_k [T, D]");
    TORCH_CHECK(weights.dim() == 2, "expected weights [S, H]");
    // The kernel is instantiated for the model's index geometry; a mismatch
    // would read past the query rows rather than fail.
    TORCH_CHECK(q.size(1) == 32 && q.size(2) == 128,
                "deepseek_v41_lightning_indexer: kernel is built for "
                "index_n_heads=32 index_head_dim=128, got H=", q.size(1),
                " D=", q.size(2));
    TORCH_CHECK(index_k.size(1) == q.size(2),
                "index_k head dim ", index_k.size(1), " != q head dim ", q.size(2));
    TORCH_CHECK(weights.size(0) == q.size(0) && weights.size(1) == q.size(1),
                "weights must be [S, H] matching q");
    TORCH_CHECK(compress_len >= 0 && compress_len <= index_k.size(0),
                "compress_len ", compress_len, " outside [0, ", index_k.size(0), "]");

    auto scores = torch::empty({q.size(0), index_k.size(0)},
                               q.options().dtype(torch::kFloat32));
    launch_lightning_indexer(q, index_k, weights, scores, compress_len);
    return scores;
}

torch::Tensor deepseek_v41_candidate_blocks(
    torch::Tensor scores,
    int64_t block_size,
    int64_t topk_blocks,
    int64_t compress_len)
{
    CHECK_INPUT(scores);
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32,
                "deepseek_v41_candidate_blocks: scores must be Float32");
    TORCH_CHECK(scores.dim() == 2, "expected scores [S, T]");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(topk_blocks > 0, "topk_blocks must be positive");
    TORCH_CHECK(compress_len >= 0 && compress_len <= scores.size(1),
                "compress_len ", compress_len, " outside [0, ", scores.size(1), "]");

    const int64_t T = scores.size(1);
    const int64_t num_blocks = (T + block_size - 1) / block_size;
    auto keep = torch::zeros({scores.size(0), num_blocks},
                             scores.options().dtype(torch::kUInt8));
    launch_candidate_blocks(scores, keep, block_size, topk_blocks, compress_len);
    return keep;
}

torch::Tensor deepseek_v41_sparse_attn(
    torch::Tensor q,
    torch::Tensor kv,
    torch::Tensor attn_sink,
    torch::Tensor topk_idxs,
    double scale)
{
    CHECK_INPUT(q);
    CHECK_INPUT(kv);
    CHECK_INPUT(topk_idxs);
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 &&
                kv.scalar_type() == torch::kFloat16,
                "deepseek_v41_sparse_attn: q and kv must be Float16");
    TORCH_CHECK(topk_idxs.scalar_type() == torch::kInt32,
                "deepseek_v41_sparse_attn: topk_idxs must be Int32");
    TORCH_CHECK(q.dim() == 3, "expected q [S, H, D], got ", q.dim(), "D");
    // One KV row per position, shared by every head: num_key_value_heads is 1.
    TORCH_CHECK(kv.dim() == 2, "expected kv [N, D] shared across heads");
    TORCH_CHECK(kv.size(1) == q.size(2),
                "kv head dim ", kv.size(1), " != q head dim ", q.size(2));
    TORCH_CHECK(topk_idxs.dim() == 2 && topk_idxs.size(0) == q.size(0),
                "expected topk_idxs [S, TOPK] matching q");
    if (attn_sink.defined() && attn_sink.numel() > 0) {
        CHECK_INPUT(attn_sink);
        TORCH_CHECK(attn_sink.scalar_type() == torch::kFloat32,
                    "deepseek_v41_sparse_attn: attn_sink must be Float32");
        TORCH_CHECK(attn_sink.numel() == q.size(1),
                    "attn_sink has ", attn_sink.numel(), " entries for ",
                    q.size(1), " heads");
    }

    auto out = torch::empty_like(q);
    launch_sparse_attn(q, kv, attn_sink, topk_idxs, out, scale);
    return out;
}

torch::Tensor deepseek_v41_engram_gate(
    torch::Tensor h,
    torch::Tensor key,
    torch::Tensor weight,
    torch::Tensor value,
    torch::Tensor mask,
    double eps,
    double clamp_value)
{
    CHECK_INPUT(h);
    CHECK_INPUT(key);
    CHECK_INPUT(weight);
    CHECK_INPUT(value);
    TORCH_CHECK(h.scalar_type() == torch::kFloat32 &&
                key.scalar_type() == torch::kFloat32 &&
                weight.scalar_type() == torch::kFloat32 &&
                value.scalar_type() == torch::kFloat32,
                "deepseek_v41_engram_gate: h, key, weight and value must be Float32");
    TORCH_CHECK(h.dim() == 3, "expected h [T, HC, D], got ", h.dim(), "D");
    TORCH_CHECK(key.sizes() == h.sizes(), "key must match h");
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == h.size(1) &&
                weight.size(1) == h.size(2),
                "weight must be [HC, D]; it is q_weight * k_weight");
    // One lookup written into every hc copy, each scaled by its own gate.
    TORCH_CHECK(value.dim() == 2 && value.size(0) == h.size(0) &&
                value.size(1) == h.size(2),
                "value must be [T, D], shared across the hc copies");
    if (mask.defined() && mask.numel() > 0) {
        CHECK_INPUT(mask);
        TORCH_CHECK(mask.scalar_type() == torch::kUInt8,
                    "deepseek_v41_engram_gate: mask must be UInt8");
        TORCH_CHECK(mask.numel() == h.size(0),
                    "mask has ", mask.numel(), " entries for ", h.size(0), " tokens");
    }

    auto out = torch::empty_like(h);
    launch_engram_gate(h, key, weight, value, mask, out, eps, clamp_value);
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> deepseek_v41_act_quant(
    torch::Tensor x,
    int64_t group_size,
    bool fp4,
    bool round_scale)
{
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == torch::kFloat32,
                "deepseek_v41_act_quant: x must be Float32");
    TORCH_CHECK(x.dim() == 2, "expected x [M, N], got ", x.dim(), "D");
    TORCH_CHECK(group_size == 16 || group_size == 32 || group_size == 64 ||
                group_size == 128,
                "deepseek_v41_act_quant: group_size must be 16, 32, 64 or 128, "
                "got ", group_size);

    // E2M1 tops out at 6, E4M3 at 448. The amax floor keeps the rounded scale
    // normal: without it an all-zero group divides by zero or a subnormal.
    const double qmax = fp4 ? 6.0 : 448.0;
    const double amax_floor = fp4 ? (6.0 * 1.1754943508222875e-38) : 1e-4;

    const int64_t groups = (x.size(1) + group_size - 1) / group_size;
    auto y = torch::empty({x.size(0), x.size(1)}, x.options().dtype(torch::kUInt8));
    auto s = torch::empty({x.size(0), groups}, x.options());
    launch_act_quant(x, y, s, group_size, round_scale, qmax, amax_floor);
    return std::make_tuple(y, s);
}

torch::Tensor deepseek_v41_o_group_proj(
    torch::Tensor x,
    torch::Tensor w)
{
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                w.scalar_type() == torch::kFloat32,
                "deepseek_v41_o_group_proj: x and w must be Float32");
    TORCH_CHECK(x.dim() == 3, "expected x [T, G, D], got ", x.dim(), "D");
    // Block diagonal: one [D, R] weight per group, not one dense
    // [G*D, G*R]. Passing a dense weight here would mix the groups.
    TORCH_CHECK(w.dim() == 3, "expected w [G, R, D], got ", w.dim(), "D");
    TORCH_CHECK(w.size(0) == x.size(1),
                "w has ", w.size(0), " groups for x's ", x.size(1));
    TORCH_CHECK(w.size(2) == x.size(2),
                "w per-group input dim ", w.size(2), " != x's ", x.size(2));

    auto out = torch::empty({x.size(0), x.size(1), w.size(1)}, x.options());
    launch_o_group_proj(x, w, out);
    return out;
}

TORCH_LIBRARY_FRAGMENT(custom_esimd_kernels_vllm, m) {
    m.def("deepseek_v41_fp4_gemm", &deepseek_v41_fp4_gemm);
    m.def("deepseek_v41_noaux_tc_topk", &deepseek_v41_noaux_tc_topk);
    m.def("deepseek_v41_lightning_indexer", &deepseek_v41_lightning_indexer);
    m.def("deepseek_v41_candidate_blocks", &deepseek_v41_candidate_blocks);
    m.def("deepseek_v41_sparse_attn", &deepseek_v41_sparse_attn);
    m.def("deepseek_v41_engram_gate", &deepseek_v41_engram_gate);
    m.def("deepseek_v41_act_quant", &deepseek_v41_act_quant);
    m.def("deepseek_v41_o_group_proj", &deepseek_v41_o_group_proj);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
