// DeepSeek V4.1 sparse attention: gather-by-index + online softmax + sink.
//
// Each query head attends to a per-head candidate list produced by the indexer,
// not to a contiguous KV range. The list holds TOPN token positions into the
// paged KV cache; positions are gathered, scored, and reduced with the same
// flash/online-softmax recurrence a dense decode uses. Two differences carry
// all the risk:
//
//   1. The KV index is read from `indices` rather than derived from the loop
//      counter, so a slot can repeat or be out of range. A negative entry is
//      the "unused slot" sentinel and must contribute nothing -- masking it to
//      -inf before the max, not skipping it after, keeps the running max from
//      seeing a value that never enters the sum.
//
//   2. The attention sink is a per-head learned logit that joins the softmax
//      denominator while contributing no value vector. It is folded in once,
//      after the candidate loop, using the same rescale the loop uses; adding
//      it to `l` without rescaling `acc` by the same factor silently biases
//      every output toward the sink.
//
// Layouts:
//   query    [B, HQ, HEAD_DIM]                fp16/bf16
//   k/vCache [pages, PAGE, HKV, HEAD_DIM]     fp16/bf16, paged
//   indices  [B, HQ, TOPN]                    int32, -1 = unused slot
//   sinks    [HQ]                             float, one logit per query head
//   out      [B, HQ, HEAD_DIM]                fp16/bf16

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <cstdint>

namespace dsv41_sparse {
using namespace sycl;
using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;

#define DS_FP32_MIN (-1e30f)

template <typename T, uint32_t N, uint32_t CHUNK = 128>
ESIMD_INLINE simd<T, N> dsLoadVec(const T* p) {
  simd<T, N> v;
#pragma unroll
  for (uint32_t i = 0; i < N; i += CHUNK)
    v.template select<CHUNK, 1>(i) = block_load<T, CHUNK>(p + i);
  return v;
}

ESIMD_INLINE float dsExp(float x) {
  simd<float, 1> v = x;
  simd<float, 1> r = exp(v);
  return r[0];
}

template <uint32_t N>
ESIMD_INLINE float dsReduceSum(simd<float, N> v) {
  if constexpr (N == 1) return v[0];
  else {
    simd<float, N / 2> h =
        v.template select<N / 2, 1>(0) + v.template select<N / 2, 1>(N / 2);
    return dsReduceSum<N / 2>(h);
  }
}

// One work-item per (batch, query head). The candidate list is short (TOPN is
// 2048 at most in V4.1), so the head's whole reduction stays in registers and
// no cross-thread combine is needed.
template <typename T, uint32_t HEAD_DIM>
struct SparseAttnKernel {
  const T* query;
  const T* kCache;
  const T* vCache;
  const int32_t* indices;   // [B, HQ, TOPN]
  const uint32_t* pageTable;
  const uint32_t* seqLens;
  const float* sinks;       // [HQ], may be null
  T* out;
  uint32_t B, HQ, HKV, TOPN;
  uint32_t pageSize, pageSizeLog2, pageTableStride;
  float scale;

  void operator()(nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
    const uint32_t t = (uint32_t)ndi.get_global_id(0);
    if (t >= B * HQ) return;
    const uint32_t h = t % HQ;
    const uint32_t b = t / HQ;

    const uint32_t gqaRatio = HQ / HKV;
    const uint32_t kvHead = h / gqaRatio;
    const uint32_t kvSeqLen = seqLens[b];
    const uint32_t pageMask = pageSize - 1;

    simd<float, HEAD_DIM> qF =
        dsLoadVec<T, HEAD_DIM>(query + (uint64_t)t * HEAD_DIM);

    simd<float, HEAD_DIM> acc = 0.0f;
    float m = DS_FP32_MIN, l = 0.0f;

    const int32_t* idx_row = indices + (uint64_t)t * TOPN;

    for (uint32_t c = 0; c < TOPN; c++) {
      const int32_t j = idx_row[c];
      // The sentinel and any position past this sequence contribute nothing.
      // Skipping before the max is what keeps them out of the running max as
      // well as the sum; masking to -inf afterwards would already have moved m.
      if (j < 0 || (uint32_t)j >= kvSeqLen) continue;

      const uint32_t jj = (uint32_t)j;
      const uint32_t physPage =
          pageTable[b * pageTableStride + (jj >> pageSizeLog2)];
      const uint64_t kvIdx =
          ((uint64_t)physPage * pageSize + (jj & pageMask)) * HKV + kvHead;

      const T* kptr = kCache + kvIdx * HEAD_DIM;
      const T* vptr = vCache + kvIdx * HEAD_DIM;
      // The next candidate is an unpredictable address, so the prefetch is
      // issued as soon as its index is known rather than one iteration late.
      if (c + 1 < TOPN) {
        const int32_t jn = idx_row[c + 1];
        if (jn >= 0 && (uint32_t)jn < kvSeqLen) {
          const uint32_t jnu = (uint32_t)jn;
          const uint32_t pn =
              pageTable[b * pageTableStride + (jnu >> pageSizeLog2)];
          const uint64_t kn =
              ((uint64_t)pn * pageSize + (jnu & pageMask)) * HKV + kvHead;
          xesimd::lsc_prefetch<T, 32, xesimd::lsc_data_size::default_size,
                               xesimd::cache_hint::cached,
                               xesimd::cache_hint::cached>(
              kCache + kn * HEAD_DIM);
        }
      }

      simd<float, HEAD_DIM> kF = dsLoadVec<T, HEAD_DIM>(kptr);
      const float score = dsReduceSum<HEAD_DIM>(qF * kF) * scale;

      const float mNew = m > score ? m : score;
      const float corr = dsExp(m - mNew), p = dsExp(score - mNew);
      l = l * corr + p;
      acc = acc * corr + dsLoadVec<T, HEAD_DIM>(vptr) * p;
      m = mNew;
    }

    // The sink is a logit with no value vector: it enters the denominator and
    // rescales the accumulator, which is what makes it able to pull the whole
    // output toward zero when every real score is small.
    if (sinks != nullptr) {
      const float s = sinks[h];
      const float mNew = m > s ? m : s;
      const float corr = dsExp(m - mNew);
      l = l * corr + dsExp(s - mNew);
      acc = acc * corr;
      m = mNew;
    }

    simd<T, HEAD_DIM> outT;
    if (l > 0.0f) {
      simd<float, HEAD_DIM> o = acc * (1.0f / l);
      outT = o;
    } else {
      // No candidate survived and no sink: an all-masked row is zero rather
      // than a division by zero.
      outT = simd<T, HEAD_DIM>(T(0));
    }
#pragma unroll
    for (uint32_t i = 0; i < HEAD_DIM; i += 128)
      block_store<T, 128>(out + (uint64_t)t * HEAD_DIM + i,
                          outT.template select<128, 1>(i));
  }
};

template <typename T, uint32_t HEAD_DIM>
inline void launch_sparse_attn(queue& q, const T* query, const T* kCache,
                               const T* vCache, const int32_t* indices,
                               const uint32_t* pageTable,
                               const uint32_t* seqLens, const float* sinks,
                               T* out, uint32_t B, uint32_t HQ, uint32_t HKV,
                               uint32_t TOPN, uint32_t pageSize,
                               uint32_t pageSizeLog2, uint32_t pageTableStride,
                               float scale) {
  // Row-parallel with no SLM and no barrier, so the group width is free.
  constexpr uint32_t WG = 16;
  const size_t rows = (size_t)B * HQ;
  const size_t global = ((rows + WG - 1) / WG) * WG;
  SparseAttnKernel<T, HEAD_DIM> kern{query,   kCache,       vCache,
                                     indices, pageTable,    seqLens,
                                     sinks,   out,          B,
                                     HQ,      HKV,          TOPN,
                                     pageSize, pageSizeLog2, pageTableStride,
                                     scale};
  q.submit([&](handler& h) {
    h.parallel_for(nd_range<1>(range<1>(global), range<1>(WG)), kern);
  });
}

}  // namespace dsv41_sparse
