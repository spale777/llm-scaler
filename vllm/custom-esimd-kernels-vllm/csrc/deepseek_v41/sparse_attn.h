// DeepSeek V4.1 sparse attention: gather-by-index + online softmax + sink.
//
// Follows inference/kernel.py::sparse_attn_kernel. Each query attends to a
// per-query list of KV positions produced by the indexer, not to a contiguous
// range. The KV is shared across heads -- kv is [n, d], one row per position,
// which is what num_key_value_heads=1 means -- so every head reads the same
// gathered row and only the query differs.
//
// Three details decide correctness:
//
//   1. The running max starts at a finite -1e30, not -inf. A query whose whole
//      index list is -1 would otherwise evaluate exp(-inf - -inf) = NaN; with a
//      finite bound it yields an all-zero output, which is the convention the
//      training kernel uses.
//
//   2. A -1 index contributes a score of -inf for that lane, so it is excluded
//      from both the max and the sum, and its gathered KV row is zeroed so it
//      cannot reach the accumulator either.
//
//   3. The attention sink is a per-head learned logit folded into the
//      denominator ONCE, after the loop, against the FINAL running max:
//      sum_exp += exp(attn_sink[h] - scores_max[h]). It never joins the
//      running max and never rescales the accumulator. Treating it as an extra
//      score would rescale acc by a factor the reference does not apply.
//
// Layouts:
//   q          [S, H, D]     fp16
//   kv         [N, D]        fp16, shared across heads
//   attn_sink  [H]           float
//   topk_idxs  [S, TOPK]     int32, -1 marks an unused slot
//   out        [S, H, D]     fp16

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <cstdint>

namespace dsv41_sparse {
using namespace sycl;
using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;

// Finite, matching the reference: an all-invalid row must give zeros, not NaN.
#define DS_SCORE_FLOOR (-1e30f)

template <typename T, int N, int CHUNK = 64>
ESIMD_INLINE simd<float, N> dsLoadVec(const T* p) {
  simd<float, N> v;
#pragma unroll
  for (int i = 0; i < N; i += CHUNK)
    v.template select<CHUNK, 1>(i) = block_load<T, CHUNK>(p + i);
  return v;
}

ESIMD_INLINE float dsExp(float x) {
  simd<float, 1> v = x;
  simd<float, 1> r = exp(v);
  return r[0];
}

template <int D>
ESIMD_INLINE float dsDot(const simd<float, D>& a, const simd<float, D>& b) {
  return reduce<float>(a * b, std::plus<>());
}

// One work-item per (query token, head). The KV row is shared across heads, so
// a head-major grid would reload it H times; the loop is short enough that the
// per-head accumulator stays in registers at D=512 only for modest D, hence
// the template.
template <typename T, int D>
struct SparseAttnKernel {
  const T* q;               // [S, H, D]
  const T* kv;              // [N, D] shared across heads
  const float* attn_sink;   // [H]
  const int32_t* topk_idxs; // [S, TOPK]
  T* out;                   // [S, H, D]
  int S, H, N, TOPK;
  float scale;

  void operator()(nd_item<1> item) const SYCL_ESIMD_KERNEL {
    const int gid = (int)item.get_global_id(0);
    if (gid >= S * H) return;
    const int h = gid % H;
    const int s = gid / H;

    simd<float, D> qv = dsLoadVec<T, D>(q + (size_t)(s * H + h) * D);

    simd<float, D> acc = 0.0f;
    float m = DS_SCORE_FLOOR;
    float sum_exp = 0.0f;

    const int32_t* idx_row = topk_idxs + (size_t)s * TOPK;

    for (int c = 0; c < TOPK; c++) {
      const int32_t j = idx_row[c];
      // An unused slot scores -inf, so it leaves both the max and the sum
      // untouched; skipping it outright is the same thing and avoids the load.
      if (j < 0 || j >= N) continue;

      simd<float, D> kvv = dsLoadVec<T, D>(kv + (size_t)j * D);
      const float score = dsDot<D>(qv, kvv) * scale;

      const float mNew = m > score ? m : score;
      const float corr = dsExp(m - mNew);
      const float p = dsExp(score - mNew);
      sum_exp = sum_exp * corr + p;
      acc = acc * corr + kvv * p;
      m = mNew;
    }

    // The sink joins the denominator once, against the final max, and carries
    // no value vector. Rescaling the accumulator by exp(m - max(m, sink)) as
    // well would be algebraically identical -- the factor cancels in the ratio
    // -- but it is not what the reference does, and the two forms part company
    // exactly at the floor below.
    //
    // When no index was valid, m is still the floor and exp(sink - floor)
    // overflows to +inf. That is harmless here and only because acc is written
    // on valid indices alone: m == floor implies acc == 0, so the epilogue
    // computes 0 * (1/inf) == 0 rather than 0 * inf == NaN. Any future change
    // that seeds acc before the loop breaks that.
    if (attn_sink != nullptr) sum_exp += dsExp(attn_sink[h] - m);

    simd<T, D> outT;
    if (sum_exp > 0.0f) {
      simd<float, D> o = acc * (1.0f / sum_exp);
      outT = o;
    } else {
      outT = simd<T, D>(T(0));
    }
#pragma unroll
    for (int i = 0; i < D; i += 64)
      block_store<T, 64>(out + (size_t)(s * H + h) * D + i,
                         outT.template select<64, 1>(i));
  }
};

template <typename T, int D>
inline void launch_sparse_attn(queue& q_, const T* q, const T* kv,
                               const float* attn_sink,
                               const int32_t* topk_idxs, T* out, int S, int H,
                               int N, int TOPK, float scale) {
  constexpr int WG = 16;
  const size_t rows = (size_t)S * H;
  const size_t global = ((rows + WG - 1) / WG) * WG;
  SparseAttnKernel<T, D> kern{q, kv, attn_sink, topk_idxs, out,
                              S, H,  N,         TOPK,      scale};
  q_.submit([&](handler& h) {
    h.parallel_for(nd_range<1>(range<1>(global), range<1>(WG)), kern);
  });
}

}  // namespace dsv41_sparse
