// DeepSeek V4.1 engram gate: write an n-gram lookup into the residual stream,
// scaled by how well it matches that stream.
//
// Follows inference/model.py::Engram.forward. The embedding lookup and the wkv
// projection are ordinary GEMMs and stay on the existing paths; what is
// specific here is the gate:
//
//   rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)
//   dot  = sum(h * weight * key) * rstd * dim^-0.5
//   gate = sigmoid(copysign(sqrt(clamp_min(|dot|, 1e-6)), dot))
//   out  = h + gate * value
//
// Four details decide the result:
//
//   1. The normalisation is per (token, hc copy) over `dim`, not jointly over
//      the copies. Reducing across copies couples them and changes every gate.
//
//   2. `weight` is q_weight * k_weight. The reference only ever uses the
//      product, so the host passes one vector; multiplying by each separately
//      is the same value but two passes over `dim`.
//
//   3. The square root is applied to the magnitude and the sign restored, so a
//      negative dot stays negative. Taking sqrt of the raw dot would be NaN
//      for half the inputs, and sqrt then negate is a different function.
//
//   4. The clamp is a floor on the magnitude before the root, which keeps the
//      derivative finite at zero. Clamping after the root, or clamping the
//      signed value, both move the gate.
//
// `value` is shared across the hc copies: one lookup is written into all of
// them, each scaled by its own gate.
//
// Layouts:
//   h       [T, HC, D]   float, the residual stream
//   key     [T, HC, D]   float
//   weight  [HC, D]      float, q_weight * k_weight
//   value   [T, D]       float, shared across copies
//   mask    [T]          uint8, 0 shuts the gate for that token
//   out     [T, HC, D]   float

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdint>

namespace dsv41_engram {
using namespace sycl;
using namespace sycl::ext::intel::esimd;

// The scalar sycl:: math functions are rejected inside an ESIMD kernel, and
// the rejection only appears at device codegen -- a syntax-only check passes.
// ESIMD's own overloads take vectors, so a one-wide simd is the scalar form.
ESIMD_INLINE float ds_rsqrt(float x) {
  simd<float, 1> v = x;
  return rsqrt(v)[0];
}
ESIMD_INLINE float ds_sqrt(float x) {
  simd<float, 1> v = x;
  return sqrt(v)[0];
}
ESIMD_INLINE float ds_exp(float x) {
  simd<float, 1> v = x;
  return exp(v)[0];
}

// One work-item per (token, hc copy): that is exactly the granularity the
// normalisation is defined at, so no cross-item reduction is needed.
template <int D>
struct EngramGateKernel {
  const float* h;       // [T, HC, D]
  const float* key;     // [T, HC, D]
  const float* weight;  // [HC, D]
  const float* value;   // [T, D]
  const uint8_t* mask;  // [T] or null
  float* out;           // [T, HC, D]
  int T, HC;
  float eps;
  float clamp_value;

  void operator()(nd_item<1> item) const SYCL_ESIMD_KERNEL {
    const int gid = (int)item.get_global_id(0);
    if (gid >= T * HC) return;
    const int c = gid % HC;
    const int t = gid / HC;

    const float* h_row = h + (size_t)gid * D;
    const float* k_row = key + (size_t)gid * D;
    const float* w_row = weight + (size_t)c * D;
    const float* v_row = value + (size_t)t * D;
    float* o_row = out + (size_t)gid * D;

    simd<float, D> hv, kv, wv;
#pragma unroll
    for (int i = 0; i < D; i += 64) {
      hv.template select<64, 1>(i) = block_load<float, 64>(h_row + i);
      kv.template select<64, 1>(i) = block_load<float, 64>(k_row + i);
      wv.template select<64, 1>(i) = block_load<float, 64>(w_row + i);
    }

    // Means over D, per copy. Accumulating in float matches the reference,
    // which casts to float before the reduction.
    const float h_ms = reduce<float>(hv * hv, std::plus<>()) / (float)D;
    const float k_ms = reduce<float>(kv * kv, std::plus<>()) / (float)D;
    // The scalar sycl math functions are not callable from an ESIMD kernel, so
    // every one of these goes through a one-wide simd. This is invisible to
    // -fsyntax-only and only fails at device codegen.
    const float rstd = ds_rsqrt(h_ms + eps) * ds_rsqrt(k_ms + eps);

    const float dot =
        reduce<float>(hv * wv * kv, std::plus<>()) * rstd * ds_rsqrt((float)D);

    // Floor the magnitude, take the root, restore the sign.
    float a = dot < 0.0f ? -dot : dot;
    if (a < clamp_value) a = clamp_value;
    float g = ds_sqrt(a);
    if (dot < 0.0f) g = -g;
    float gate = 1.0f / (1.0f + ds_exp(-g));

    if (mask != nullptr && mask[t] == 0) gate = 0.0f;

    simd<float, D> vv;
#pragma unroll
    for (int i = 0; i < D; i += 64)
      vv.template select<64, 1>(i) = block_load<float, 64>(v_row + i);

    simd<float, D> res = hv + vv * gate;
#pragma unroll
    for (int i = 0; i < D; i += 64)
      block_store<float, 64>(o_row + i, res.template select<64, 1>(i));
  }
};

template <int D>
inline void launch_engram_gate(queue& q, const float* h, const float* key,
                               const float* weight, const float* value,
                               const uint8_t* mask, float* out, int T, int HC,
                               float eps, float clamp_value) {
  constexpr int WG = 16;
  const size_t rows = (size_t)T * HC;
  const size_t global = ((rows + WG - 1) / WG) * WG;
  EngramGateKernel<D> kern{h,   key, weight, value, mask,
                           out, T,   HC,     eps,   clamp_value};
  q.submit([&](handler& cgh) {
    cgh.parallel_for(nd_range<1>(range<1>(global), range<1>(WG)), kern);
  });
}

}  // namespace dsv41_engram
