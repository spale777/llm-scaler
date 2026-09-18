// DeepSeek V4.1 lightning indexer: per-query scores over compressed positions.
//
// A small side attention that decides which compressed KV positions the main
// attention will read. Semantics follow inference/model.py::Indexer.forward:
//
//   index_score = einsum("bshd,btd->bsht", q, index_k)   one shared key per
//                                                        position (MQA)
//   index_score = (relu(index_score) * weights[..., h]).sum(over h)
//
// Three details carry the behaviour:
//
//   1. The ReLU comes before the head-weighted sum, so a head that dislikes a
//      position contributes zero rather than a negative that another head has
//      to overcome. Summing first and rectifying after changes the ranking.
//
//   2. `weights` already carries softmax_scale * n_heads^-0.5 from the host,
//      so the kernel applies no scale of its own; multiplying again here is
//      the natural mistake and rescales every score uniformly, which is
//      invisible in a top-k but not in the candidate mask's -inf comparisons.
//
//   3. A position is reachable only once the query has passed its compressed
//      group's last token. Unreachable positions must score -inf before any
//      selection: a block whose best position is -inf is what "not reachable
//      yet" means to the candidate stage.
//
// Layouts:
//   q        [S, H, D]     fp16, per query token, H = index_n_heads (32)
//   index_k  [T, D]        fp16, one key per compressed position, D = 128
//   weights  [S, H]        fp16, pre-scaled by the host
//   scores   [S, T]        float, -inf where unreachable

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <cstdint>
#include <limits>

namespace dsv41_indexer {
using namespace sycl;
using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;

#define DSI_NEG_INF (-std::numeric_limits<float>::infinity())

template <int D>
ESIMD_INLINE float dsi_dot(const simd<float, D>& a, const simd<float, D>& b) {
  return reduce<float>(a * b, std::plus<>());
}

// One work-item per (query token, compressed position). H is small (32) and D
// is 128, so the whole per-position reduction stays in registers; the query
// rows are reloaded per position but they are the small operand, and the key
// stream -- which is the long one -- is read exactly once.
template <int H, int D>
struct LightningIndexerKernel {
  const sycl::half* q;        // [S, H, D]
  const sycl::half* index_k;  // [T, D]
  const sycl::half* weights;  // [S, H]
  float* scores;              // [S, T]
  int S, T;
  // Compressed positions reachable by query s. Prefill passes the per-query
  // array; decode passes a single value broadcast to every query.
  const int32_t* compress_lens;
  int compress_len_scalar;

  void operator()(nd_item<2> item) const SYCL_ESIMD_KERNEL {
    const int s = (int)item.get_global_id(0);
    const int t = (int)item.get_global_id(1);
    if (s >= S || t >= T) return;

    const int reach =
        (compress_lens != nullptr) ? compress_lens[s] : compress_len_scalar;
    if (t >= reach) {
      // Not reachable yet. This must be -inf and not a small number: the
      // candidate stage reads a block's max and treats -inf as unreachable.
      scores[(size_t)s * T + t] = DSI_NEG_INF;
      return;
    }

    simd<float, D> kv;
#pragma unroll
    for (int i = 0; i < D; i += 64)
      kv.template select<64, 1>(i) =
          block_load<sycl::half, 64>(index_k + (size_t)t * D + i);

    const sycl::half* q_row = q + (size_t)s * H * D;
    const sycl::half* w_row = weights + (size_t)s * H;

    float acc = 0.0f;
#pragma unroll
    for (int h = 0; h < H; ++h) {
      simd<float, D> qv;
#pragma unroll
      for (int i = 0; i < D; i += 64)
        qv.template select<64, 1>(i) =
            block_load<sycl::half, 64>(q_row + (size_t)h * D + i);
      const float dot = dsi_dot<D>(qv, kv);
      // Rectify per head, then weight: a head that scores this position
      // negatively contributes nothing rather than cancelling another head.
      const float r = dot > 0.0f ? dot : 0.0f;
      acc += r * (float)w_row[h];
    }

    scores[(size_t)s * T + t] = acc;
  }
};

template <int H, int D>
inline void launch_lightning_indexer(queue& q_, const sycl::half* q,
                                     const sycl::half* index_k,
                                     const sycl::half* weights, float* scores,
                                     int S, int T, const int32_t* compress_lens,
                                     int compress_len_scalar) {
  // Position-major so consecutive work-items read consecutive key rows.
  constexpr int WG_T = 16;
  const size_t gt = ((size_t)(T + WG_T - 1) / WG_T) * WG_T;
  LightningIndexerKernel<H, D> kern{q,  index_k, weights, scores,
                                    S,  T,       compress_lens,
                                    compress_len_scalar};
  q_.submit([&](handler& h) {
    h.parallel_for(nd_range<2>(range<2>((size_t)S, gt), range<2>(1, WG_T)),
                   kern);
  });
}

// Level one of the two-level selection: keep the topk_blocks highest-scoring
// blocks per query, where a block scores as the best position inside it.
//
// The block holding the query's newest position is pinned in regardless of
// score: it carries the most recent tokens but is only partly filled, so a
// full older block can outscore it. Blocks whose best position is -inf are
// unreachable and must not be kept even when fewer than topk_blocks survive --
// otherwise the mask admits positions the query cannot see.
struct CandidateBlockKernel {
  const float* scores;   // [S, T]
  uint8_t* keep;         // [S, num_blocks] bool mask
  int S, T, num_blocks, block_size, topk_blocks;
  const int32_t* compress_lens;
  int compress_len_scalar;

  void operator()(nd_item<1> item) const SYCL_ESIMD_KERNEL {
    const int s = (int)item.get_global_id(0);
    if (s >= S) return;

    const int reach =
        (compress_lens != nullptr) ? compress_lens[s] : compress_len_scalar;
    const float* row = scores + (size_t)s * T;
    uint8_t* keep_row = keep + (size_t)s * num_blocks;

    const int last_block = (reach - 1) / block_size;

    for (int b = 0; b < num_blocks; ++b) keep_row[b] = 0;

    // Selection is a repeated max rather than a sort: topk_blocks is 2048 and
    // num_blocks is bounded by the context, so a full sort would cost more
    // than the scan it replaces.
    for (int r = 0; r < topk_blocks && r < num_blocks; ++r) {
      float best = DSI_NEG_INF;
      int best_b = -1;
      for (int b = 0; b < num_blocks; ++b) {
        if (keep_row[b]) continue;
        // The newest block is pinned: treat it as unbeatable on the first
        // round it is still available.
        const float bscore = (b == last_block) ? std::numeric_limits<float>::infinity()
                                               : block_max(row, b);
        if (bscore > best) { best = bscore; best_b = b; }
      }
      // Every remaining block is unreachable, so stop rather than padding the
      // selection with positions the query cannot see.
      if (best_b < 0 || best == DSI_NEG_INF) break;
      keep_row[best_b] = 1;
    }
  }

  ESIMD_INLINE float block_max(const float* row, int b) const {
    const int start = b * block_size;
    int end = start + block_size;
    if (end > T) end = T;
    float m = DSI_NEG_INF;
    for (int i = start; i < end; ++i) {
      const float v = row[i];
      if (v > m) m = v;
    }
    return m;
  }
};

inline void launch_candidate_blocks(queue& q_, const float* scores,
                                    uint8_t* keep, int S, int T,
                                    int block_size, int topk_blocks,
                                    const int32_t* compress_lens,
                                    int compress_len_scalar) {
  const int num_blocks = (T + block_size - 1) / block_size;
  constexpr int WG = 16;
  const size_t global = ((size_t)(S + WG - 1) / WG) * WG;
  CandidateBlockKernel kern{scores,      keep,        S,
                            T,           num_blocks,  block_size,
                            topk_blocks, compress_lens, compress_len_scalar};
  q_.submit([&](handler& h) {
    h.parallel_for(nd_range<1>(range<1>(global), range<1>(WG)), kern);
  });
}

}  // namespace dsv41_indexer
