// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/grouped_flash_attention.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/common/float16.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {
namespace {

constexpr std::ptrdiff_t kKvTile = 128;             // cached rows per tile
constexpr std::ptrdiff_t kSubBlockRows = 128;       // query rows, (token, head) pairs, per GEMM
constexpr std::ptrdiff_t kMaxSubBlocksPerTask = 8;  // GEMM row blocks sharing each loaded tile, at most
constexpr std::ptrdiff_t kMinKvChunk = 512;         // cached rows per split of the cache, at least
constexpr float kLowest = std::numeric_limits<float>::lowest();

inline std::ptrdiff_t CeilDiv(std::ptrdiff_t a, std::ptrdiff_t b) { return (a + b - 1) / b; }

// Lengths come in as int32, but the plan rounds them up and multiplies them, so it is computed in std::ptrdiff_t.
struct WorkPlan {
  std::ptrdiff_t group_size;    // query heads per kv head
  std::ptrdiff_t sub_tokens;    // query tokens per GEMM row block
  std::ptrdiff_t block_tokens;  // query tokens per task, a multiple of sub_tokens or all of them
  std::ptrdiff_t row_block;     // rows per task, block_tokens * group_size
  std::ptrdiff_t token_blocks;  // tasks per (batch entry, kv head) and split of the cache
  std::ptrdiff_t row_tasks;     // batch_size * kv_num_heads * token_blocks
  std::ptrdiff_t kv_chunk;      // cached rows per split, a multiple of kKvTile
  std::ptrdiff_t kv_splits;     // splits of the cache per row block
  std::ptrdiff_t task_count;    // row_tasks * kv_splits
};

WorkPlan MakeWorkPlan(int batch_size, int sequence_length, int num_heads, int kv_num_heads,
                      std::ptrdiff_t max_total_length, int thread_count) {
  WorkPlan plan{};
  plan.group_size = num_heads / kv_num_heads;
  plan.sub_tokens = std::clamp<std::ptrdiff_t>(kSubBlockRows / plan.group_size, 1, sequence_length);

  // A task loads each tile once for all of its GEMM row blocks, so tasks take as many row blocks as they can while
  // there are still at least as many tasks as threads.
  const std::ptrdiff_t heads = static_cast<std::ptrdiff_t>(batch_size) * kv_num_heads;
  const std::ptrdiff_t sub_blocks = CeilDiv(sequence_length, plan.sub_tokens);
  const std::ptrdiff_t per_task = std::clamp<std::ptrdiff_t>(heads * sub_blocks / thread_count, 1,
                                                             kMaxSubBlocksPerTask);
  plan.block_tokens = std::min<std::ptrdiff_t>(plan.sub_tokens * per_task, sequence_length);
  plan.row_block = plan.block_tokens * plan.group_size;
  plan.token_blocks = CeilDiv(sequence_length, plan.block_tokens);
  plan.row_tasks = heads * plan.token_blocks;

  // Split the cache when there are too few tasks to balance the threads: decode, few heads or many threads. Small
  // (decode) tasks aim at four tasks per thread, large ones at two, which also bounds the split state's memory.
  const std::ptrdiff_t wanted_tasks = (plan.row_block <= kSubBlockRows ? 4 : 2) * static_cast<std::ptrdiff_t>(thread_count);
  std::ptrdiff_t kv_splits = 1;
  if (thread_count > 1 && plan.row_tasks < wanted_tasks && max_total_length > kMinKvChunk) {
    kv_splits = std::min(CeilDiv(wanted_tasks, plan.row_tasks), CeilDiv(max_total_length, kMinKvChunk));
  }
  plan.kv_chunk = std::max(kKvTile, CeilDiv(CeilDiv(max_total_length, kv_splits), kKvTile) * kKvTile);
  plan.kv_splits = std::max<std::ptrdiff_t>(1, CeilDiv(max_total_length, plan.kv_chunk));
  plan.task_count = plan.row_tasks * plan.kv_splits;
  return plan;
}

// Both reductions keep independent lanes so they vectorize without -ffast-math.
float RowMax(const float* x, std::ptrdiff_t n) {
  constexpr std::ptrdiff_t kLanes = 8;
  float lanes[kLanes] = {kLowest, kLowest, kLowest, kLowest, kLowest, kLowest, kLowest, kLowest};
  std::ptrdiff_t i = 0;
  for (; i + kLanes <= n; i += kLanes) {
    for (std::ptrdiff_t j = 0; j < kLanes; ++j) {
      lanes[j] = x[i + j] > lanes[j] ? x[i + j] : lanes[j];
    }
  }
  float result = std::max(std::max(std::max(lanes[0], lanes[1]), std::max(lanes[2], lanes[3])),
                          std::max(std::max(lanes[4], lanes[5]), std::max(lanes[6], lanes[7])));
  for (; i < n; ++i) {
    result = std::max(result, x[i]);
  }
  return result;
}

float RowSum(const float* x, std::ptrdiff_t n) {
  constexpr std::ptrdiff_t kLanes = 8;
  float lanes[kLanes] = {};
  std::ptrdiff_t i = 0;
  for (; i + kLanes <= n; i += kLanes) {
    for (std::ptrdiff_t j = 0; j < kLanes; ++j) {
      lanes[j] += x[i + j];
    }
  }
  float result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
  for (; i < n; ++i) {
    result += x[i];
  }
  return result;
}

// Copies `rows` rows of `count` elements each, `src_stride` apart, to a dense fp32 buffer.
template <typename T>
void LoadRows(const T* src, size_t src_stride, std::ptrdiff_t rows, size_t count, float* dst) {
  if (rows <= 0) {
    return;
  }
  if (src_stride == count) {
    const size_t total = static_cast<size_t>(rows) * count;
    if constexpr (std::is_same_v<T, float>) {
      std::memcpy(dst, src, total * sizeof(float));
    } else {
      MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(src), dst, total);
    }
    return;
  }
  for (std::ptrdiff_t r = 0; r < rows; ++r) {
    const T* s = src + static_cast<size_t>(r) * src_stride;
    float* d = dst + static_cast<size_t>(r) * count;
    if constexpr (std::is_same_v<T, float>) {
      std::memcpy(d, s, count * sizeof(float));
    } else {
      MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(s), d, count);
    }
  }
}

// Per-worker scratch.
struct WorkerBuffers {
  float* q;        // the task's query rows, fp32
  float* k;        // fp32 copy of a key tile
  float* v;        // fp32 copy of a value tile
  void* packed_k;  // key tile packed as the GEMM's B operand, or nullptr when packing is unavailable
  void* packed_v;  // value tile likewise
  float* scores;   // one GEMM row block x one tile
  float* bias;     // fp32 copy of a bias segment (fp16 bias)
  float* state;    // m, l and acc of the task's rows, unless they go to the split buffer
};

template <typename T>
class GroupedAttention {
 public:
  GroupedAttention(const GroupedFlashAttentionArgs<T>& args, const WorkPlan& plan, bool pack_k, bool pack_v)
      : args_(args), plan_(plan), pack_k_(pack_k), pack_v_(pack_v) {}

  // Cached rows [Low(High(b, s)), High(b, s)) are visible to query token s of batch entry b. Both bounds are
  // non-decreasing in s.
  std::ptrdiff_t High(int b, std::ptrdiff_t s) const {
    const std::ptrdiff_t total = args_.total_lengths[b];
    return args_.causal ? std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(args_.causal_offsets[b]) + s + 1, total)
                        : total;
  }
  std::ptrdiff_t Low(std::ptrdiff_t high) const {
    return args_.local_window_size > 0 && high > args_.local_window_size ? high - args_.local_window_size : 0;
  }

  // Valid query tokens [s_begin, s_end) of token block `block`; empty for a block past query_lengths[b].
  void BlockTokens(int b, std::ptrdiff_t block, std::ptrdiff_t& s_begin, std::ptrdiff_t& s_end) const {
    const std::ptrdiff_t valid = args_.query_lengths != nullptr
                                     ? std::min<std::ptrdiff_t>(args_.query_lengths[b], args_.sequence_length)
                                     : args_.sequence_length;
    s_begin = block * plan_.block_tokens;
    s_end = std::min(s_begin + plan_.block_tokens, valid);
  }

  // Runs the online softmax of the rows of tokens [s_begin, s_end) of (b, kv head) over cached rows
  // [kv_begin, kv_end). Row r is token s_begin + r / group_size and query head kv * group_size + r % group_size. Each
  // tile is loaded and packed once and then multiplied against every GEMM row block that sees it. On return m, l and
  // acc hold each row's running max, running sum and unnormalized sum of exp(score - max) * V; l is 0 for a row that
  // saw no cached row. With a sink, the running max starts at the sink logit, as the softmax's max includes it, so
  // acc stays bounded when the sink outweighs the scores; the sink's own term joins the sum only in WriteRows, once
  // per row however the cache is split.
  void AttendRows(int b, int kv, std::ptrdiff_t s_begin, std::ptrdiff_t s_end, std::ptrdiff_t kv_begin,
                  std::ptrdiff_t kv_end, const WorkerBuffers& w, float* m, float* l, float* acc) const {
    const std::ptrdiff_t group = plan_.group_size;
    const std::ptrdiff_t rows = (s_end - s_begin) * group;
    const size_t head_size = static_cast<size_t>(args_.head_size);
    const size_t v_head_size = static_cast<size_t>(args_.v_head_size);

    if (args_.use_sink) {
      for (std::ptrdiff_t r = 0; r < rows; ++r) {
        m[r] = Sink(kv * group + r % group);
      }
    } else {
      std::fill_n(m, rows, kLowest);
    }
    std::fill_n(l, rows, 0.0f);
    std::fill_n(acc, static_cast<size_t>(rows) * v_head_size, 0.0f);

    kv_begin = std::max(kv_begin, Low(High(b, s_begin)));
    kv_end = std::min(kv_end, High(b, s_end - 1));
    if (kv_begin >= kv_end) {
      return;
    }

    for (std::ptrdiff_t s = s_begin; s < s_end; ++s) {
      const T* q = args_.query + static_cast<size_t>(b) * args_.q_batch_stride +
                   static_cast<size_t>(kv * group) * args_.q_head_stride + static_cast<size_t>(s) * args_.q_seq_stride;
      LoadRows(q, args_.q_head_stride, group, head_size,
               w.q + static_cast<size_t>((s - s_begin) * group) * head_size);
    }
    if (!pack_k_) {
      // The unpacked Q*K' GEMM takes alpha = 1, so a one-row block reaches MLAS's single-row kernel.
      const size_t q_count = static_cast<size_t>(rows) * head_size;
      for (size_t i = 0; i < q_count; ++i) w.q[i] *= args_.scale;
    }

    const T* key = args_.key + static_cast<size_t>(b) * args_.k_batch_stride +
                   static_cast<size_t>(kv) * args_.k_head_stride;
    const T* value = args_.value + static_cast<size_t>(b) * args_.v_batch_stride +
                     static_cast<size_t>(kv) * args_.v_head_stride;

    for (std::ptrdiff_t tile_begin = kv_begin; tile_begin < kv_end; tile_begin += kKvTile) {
      const std::ptrdiff_t n = std::min(kKvTile, kv_end - tile_begin);
      const std::ptrdiff_t tile_end = tile_begin + n;

      // GEMM row blocks [first_sub, last_sub) see part of this tile; with a local window some may not.
      std::ptrdiff_t first_sub = -1;
      std::ptrdiff_t last_sub = -1;
      for (std::ptrdiff_t t0 = s_begin, sub = 0; t0 < s_end; t0 += plan_.sub_tokens, ++sub) {
        const std::ptrdiff_t t1 = std::min(t0 + plan_.sub_tokens, s_end);
        if (Low(High(b, t0)) < tile_end && High(b, t1 - 1) > tile_begin) {
          first_sub = first_sub < 0 ? sub : first_sub;
          last_sub = sub + 1;
        }
      }
      if (first_sub < 0) {
        continue;
      }

      // Copying (or converting) the tile first reads the cache sequentially. Packing straight from the cache reads it
      // in strided column panels, which is several times slower on some ARM64 cores.
      LoadRows(key + static_cast<size_t>(tile_begin) * args_.k_row_stride, args_.k_row_stride, n, head_size, w.k);
      LoadRows(value + static_cast<size_t>(tile_begin) * args_.v_row_stride, args_.v_row_stride, n, v_head_size, w.v);
      if (pack_k_) {
        MlasGemmPackB(CblasNoTrans, CblasTrans, static_cast<size_t>(n), head_size, w.k, head_size, w.packed_k,
                      args_.mlas_config);
      }
      if (pack_v_) {
        MlasGemmPackB(CblasNoTrans, CblasNoTrans, v_head_size, static_cast<size_t>(n), w.v, v_head_size, w.packed_v,
                      args_.mlas_config);
      }

      for (std::ptrdiff_t sub = first_sub; sub < last_sub; ++sub) {
        const std::ptrdiff_t t0 = s_begin + sub * plan_.sub_tokens;
        const std::ptrdiff_t t1 = std::min(t0 + plan_.sub_tokens, s_end);
        const std::ptrdiff_t r0 = (t0 - s_begin) * group;
        const std::ptrdiff_t sub_rows = (t1 - t0) * group;
        const float* q = w.q + static_cast<size_t>(r0) * head_size;
        if (pack_k_) {
          MlasGemm(CblasNoTrans, static_cast<size_t>(sub_rows), static_cast<size_t>(n), head_size, args_.scale, q,
                   head_size, w.packed_k, 0.0f, w.scores, static_cast<size_t>(n), nullptr, args_.mlas_config);
        } else {
          MlasGemm(CblasNoTrans, CblasTrans, static_cast<size_t>(sub_rows), static_cast<size_t>(n), head_size,
                   1.0f, q, head_size, w.k, head_size, 0.0f, w.scores, static_cast<size_t>(n), nullptr,
                   args_.mlas_config);
        }

        for (std::ptrdiff_t i = 0; i < sub_rows; ++i) {
          const std::ptrdiff_t r = r0 + i;
          const std::ptrdiff_t s = t0 + i / group;
          const std::ptrdiff_t high = High(b, s);
          const std::ptrdiff_t begin = std::clamp<std::ptrdiff_t>(Low(high) - tile_begin, 0, n);
          const std::ptrdiff_t end = std::clamp<std::ptrdiff_t>(high - tile_begin, 0, n);
          float* score = w.scores + static_cast<size_t>(i) * n;
          if (begin >= end) {
            std::fill_n(score, n, 0.0f);
            continue;
          }
          float* x = score + begin;
          const std::ptrdiff_t count = end - begin;
          if (args_.softcap > 0.0f) {
            MlasComputeSoftcap(x, x, static_cast<size_t>(count), args_.softcap);
          }
          if (args_.attention_bias != nullptr) {
            AddBias(b, kv * group + i % group, s, tile_begin + begin, count, w.bias, x);
          }

          const float m_new = std::max(m[r], RowMax(x, count));
          for (std::ptrdiff_t j = 0; j < count; ++j) {
            x[j] -= m_new;
          }
          MlasComputeExp(x, x, static_cast<size_t>(count));
          const float sum = RowSum(x, count);
          std::fill(score, x, 0.0f);
          std::fill(score + end, score + n, 0.0f);
          if (m_new > m[r]) {
            if (l[r] > 0.0f) {
              const float alpha = std::exp(m[r] - m_new);
              l[r] *= alpha;
              float* a = acc + static_cast<size_t>(r) * v_head_size;
              for (size_t c = 0; c < v_head_size; ++c) {
                a[c] *= alpha;
              }
            }
            m[r] = m_new;
          }
          l[r] += sum;
        }

        float* out = acc + static_cast<size_t>(r0) * v_head_size;
        if (pack_v_) {
          MlasGemm(CblasNoTrans, static_cast<size_t>(sub_rows), v_head_size, static_cast<size_t>(n), 1.0f, w.scores,
                   static_cast<size_t>(n), w.packed_v, 1.0f, out, v_head_size, nullptr, args_.mlas_config);
        } else {
          MlasGemm(CblasNoTrans, CblasNoTrans, static_cast<size_t>(sub_rows), v_head_size, static_cast<size_t>(n),
                   1.0f, w.scores, static_cast<size_t>(n), w.v, v_head_size, 1.0f, out, v_head_size, nullptr,
                   args_.mlas_config);
        }
      }
    }
  }

  // Normalizes the rows of acc in place, folding in the head sink, and writes them to the output.
  void WriteRows(int b, int kv, std::ptrdiff_t s_begin, std::ptrdiff_t rows, const float* m, const float* l,
                 float* acc) const {
    const std::ptrdiff_t group = plan_.group_size;
    const size_t v_head_size = static_cast<size_t>(args_.v_head_size);
    for (std::ptrdiff_t r = 0; r < rows; ++r) {
      const std::ptrdiff_t h = kv * group + r % group;
      const std::ptrdiff_t s = s_begin + r / group;
      float factor = 0.0f;
      if (args_.use_sink) {
        const float sink = Sink(h);
        const float row_max = std::max(m[r], sink);
        const float kept = l[r] > 0.0f ? std::exp(m[r] - row_max) : 0.0f;
        factor = kept / (l[r] * kept + std::exp(sink - row_max));
      } else if (l[r] > 0.0f) {
        factor = 1.0f / l[r];
      }
      float* a = acc + static_cast<size_t>(r) * v_head_size;
      for (size_t c = 0; c < v_head_size; ++c) {
        a[c] *= factor;
      }
      T* out = args_.output + static_cast<size_t>(b) * args_.o_batch_stride +
               static_cast<size_t>(s) * args_.o_seq_stride + static_cast<size_t>(h) * args_.o_head_stride;
      if constexpr (std::is_same_v<T, float>) {
        std::memcpy(out, a, v_head_size * sizeof(float));
      } else {
        MlasConvertFloatToHalfBuffer(a, reinterpret_cast<MLAS_FP16*>(out), v_head_size);
      }
    }
  }

 private:
  // The extra softmax logit of query head h when use_sink is set.
  float Sink(std::ptrdiff_t h) const {
    return args_.head_sink != nullptr ? static_cast<float>(args_.head_sink[h]) : 0.0f;
  }

  // Adds the bias of (b, head h, token s) at cached rows [column, column + count) to x.
  void AddBias(int b, std::ptrdiff_t h, std::ptrdiff_t s, std::ptrdiff_t column, std::ptrdiff_t count,
               float* bias_buffer, float* x) const {
    const std::ptrdiff_t offset = args_.bias_offsets != nullptr ? args_.bias_offsets[b] : 0;
    const T* bias = args_.attention_bias + static_cast<size_t>(b) * args_.bias_batch_stride +
                    static_cast<size_t>(h) * args_.bias_head_stride + static_cast<size_t>(s) * args_.bias_seq_stride +
                    static_cast<size_t>(offset + column);
    const float* bias_fp32 = nullptr;
    if constexpr (std::is_same_v<T, float>) {
      bias_fp32 = bias;
    } else {
      MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(bias), bias_buffer, static_cast<size_t>(count));
      bias_fp32 = bias_buffer;
    }
    for (std::ptrdiff_t j = 0; j < count; ++j) {
      x[j] += bias_fp32[j];
    }
  }

  const GroupedFlashAttentionArgs<T>& args_;
  const WorkPlan& plan_;
  const bool pack_k_;
  const bool pack_v_;
};

// Scratch sizes are rounded up to whole cache lines so every buffer keeps the allocator's alignment, which the packed
// GEMM operands need.
size_t AlignFloats(size_t count) { return (count + 15) / 16 * 16; }

}  // namespace

template <typename T>
Status RunGroupedFlashAttention(const GroupedFlashAttentionArgs<T>& args, AllocatorPtr allocator,
                                ThreadPool* thread_pool) {
  ORT_RETURN_IF(args.num_heads <= 0 || args.kv_num_heads <= 0 || args.num_heads % args.kv_num_heads != 0,
                "num_heads must be a positive multiple of kv_num_heads.");
  ORT_RETURN_IF(args.head_size <= 0 || args.v_head_size <= 0, "head sizes must be positive.");
  if (args.batch_size <= 0 || args.sequence_length <= 0) {
    return Status::OK();
  }

  std::ptrdiff_t max_total_length = 0;
  for (int b = 0; b < args.batch_size; ++b) {
    ORT_RETURN_IF(args.total_lengths[b] < 0, "total length must not be negative.");
    max_total_length = std::max<std::ptrdiff_t>(max_total_length, args.total_lengths[b]);
  }
  const int thread_count = std::max(1, ThreadPool::DegreeOfParallelism(thread_pool));
  const WorkPlan plan = MakeWorkPlan(args.batch_size, args.sequence_length, args.num_heads, args.kv_num_heads,
                                     std::max<std::ptrdiff_t>(max_total_length, 1), thread_count);

  const size_t head_size = static_cast<size_t>(args.head_size);
  const size_t v_head_size = static_cast<size_t>(args.v_head_size);
  const size_t row_block = static_cast<size_t>(plan.row_block);
  const size_t sub_rows = static_cast<size_t>(std::min(plan.sub_tokens, plan.block_tokens) * plan.group_size);
  const size_t tile = static_cast<size_t>(kKvTile);
  const bool split = plan.kv_splits > 1;

  // Each tile is packed once as the B operand of the GEMMs of all row blocks that use it. Packing is skipped (MlasGemm
  // then packs per call) if the backend reports no packed size.
  const size_t packed_k_bytes = MlasGemmPackBSize(CblasNoTrans, CblasTrans, tile, head_size, args.mlas_config);
  const size_t packed_v_bytes = MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, v_head_size, tile, args.mlas_config);
  // Packing pays off when several query rows share a tile. One-row blocks (decode without grouping) use MLAS's
  // single-row kernels on the unpacked tile instead: P*V's on most platforms, Q*K''s (transposed B) only on x64, not
  // ARM64EC.
#if defined(MLAS_TARGET_AMD64)
  constexpr bool kSingleRowTransposedKernel = true;
#else
  constexpr bool kSingleRowTransposedKernel = false;
#endif
  const bool can_pack = packed_k_bytes > 0 && packed_v_bytes > 0;
  const bool pack_k = can_pack && (sub_rows > 1 || !kSingleRowTransposedKernel);
  const bool pack_v = can_pack && sub_rows > 1;

  constexpr bool kConvert = !std::is_same_v<T, float>;
  const size_t q_size = AlignFloats(SafeInt<size_t>(row_block) * head_size);
  const size_t k_size = AlignFloats(SafeInt<size_t>(tile) * head_size);
  const size_t v_size = AlignFloats(SafeInt<size_t>(tile) * v_head_size);
  const size_t packed_k_size = pack_k ? AlignFloats(packed_k_bytes / sizeof(float) + 1) : 0;
  const size_t packed_v_size = pack_v ? AlignFloats(packed_v_bytes / sizeof(float) + 1) : 0;
  const size_t score_size = AlignFloats(SafeInt<size_t>(sub_rows) * tile);
  const size_t bias_size = kConvert && args.attention_bias != nullptr ? AlignFloats(tile) : 0;
  // Softmax state of one task: m[row_block], l[row_block], acc[row_block][v_head_size].
  const size_t state_size = AlignFloats(SafeInt<size_t>(row_block) * (SafeInt<size_t>(v_head_size) + 2));
  const size_t worker_size = SafeInt<size_t>(q_size) + k_size + v_size + packed_k_size + packed_v_size + score_size +
                             bias_size + (split ? 0 : state_size);
  const std::ptrdiff_t worker_count = std::min<std::ptrdiff_t>(thread_count, plan.task_count);

  auto scratch = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(worker_size) * worker_count);
  IAllocatorUniquePtr<float> partial;
  if (split) {
    partial = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(state_size) * plan.task_count);
  }

  const GroupedAttention<T> attention(args, plan, pack_k, pack_v);
  auto decode_row_task = [&](std::ptrdiff_t row_task, int& b, int& kv, std::ptrdiff_t& block) {
    // Later token blocks attend to more cached rows, so they come first.
    block = plan.token_blocks - 1 - row_task % plan.token_blocks;
    const std::ptrdiff_t head_task = row_task / plan.token_blocks;
    b = static_cast<int>(head_task / args.kv_num_heads);
    kv = static_cast<int>(head_task % args.kv_num_heads);
  };

  std::atomic<std::ptrdiff_t> next_task{0};
  ThreadPool::TrySimpleParallelFor(thread_pool, worker_count, [&](std::ptrdiff_t worker) {
    WorkerBuffers w{};
    w.q = scratch.get() + static_cast<size_t>(worker) * worker_size;
    w.k = w.q + q_size;
    w.v = w.k + k_size;
    float* packed_k = w.v + v_size;
    float* packed_v = packed_k + packed_k_size;
    w.packed_k = pack_k ? packed_k : nullptr;
    w.packed_v = pack_v ? packed_v : nullptr;
    w.scores = packed_v + packed_v_size;
    w.bias = w.scores + score_size;
    w.state = w.bias + bias_size;
    for (std::ptrdiff_t task = next_task++; task < plan.task_count; task = next_task++) {
      const std::ptrdiff_t split_index = task % plan.kv_splits;
      const std::ptrdiff_t row_task = task / plan.kv_splits;
      int b = 0;
      int kv = 0;
      std::ptrdiff_t block = 0;
      decode_row_task(row_task, b, kv, block);
      std::ptrdiff_t s_begin = 0;
      std::ptrdiff_t s_end = 0;
      attention.BlockTokens(b, block, s_begin, s_end);

      float* m = split ? partial.get() + static_cast<size_t>(task) * state_size : w.state;
      float* l = m + row_block;
      float* acc = l + row_block;
      if (s_begin >= s_end) {
        continue;
      }
      const std::ptrdiff_t kv_begin = split_index * plan.kv_chunk;
      attention.AttendRows(b, kv, s_begin, s_end, kv_begin, kv_begin + plan.kv_chunk, w, m, l, acc);
      if (!split) {
        attention.WriteRows(b, kv, s_begin, (s_end - s_begin) * plan.group_size, m, l, acc);
      }
    }
  });

  if (!split) {
    return Status::OK();
  }

  // Merges the split states of each row: out = sum_c exp(m_c - M) acc_c / sum_c exp(m_c - M) l_c.
  const std::ptrdiff_t merge_workers = std::min<std::ptrdiff_t>(thread_count, plan.row_tasks);
  auto merged = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(state_size) * merge_workers);
  std::atomic<std::ptrdiff_t> next_merge{0};
  ThreadPool::TrySimpleParallelFor(thread_pool, merge_workers, [&](std::ptrdiff_t worker) {
    float* m = merged.get() + static_cast<size_t>(worker) * state_size;
    float* l = m + row_block;
    float* acc = l + row_block;
    for (std::ptrdiff_t row_task = next_merge++; row_task < plan.row_tasks; row_task = next_merge++) {
      int b = 0;
      int kv = 0;
      std::ptrdiff_t block = 0;
      decode_row_task(row_task, b, kv, block);
      std::ptrdiff_t s_begin = 0;
      std::ptrdiff_t s_end = 0;
      attention.BlockTokens(b, block, s_begin, s_end);
      if (s_begin >= s_end) {
        continue;
      }
      const std::ptrdiff_t rows = (s_end - s_begin) * plan.group_size;
      const float* states = partial.get() + static_cast<size_t>(row_task * plan.kv_splits) * state_size;
      for (std::ptrdiff_t r = 0; r < rows; ++r) {
        float row_max = kLowest;
        for (std::ptrdiff_t c = 0; c < plan.kv_splits; ++c) {
          const float* state = states + static_cast<size_t>(c) * state_size;
          if (state[row_block + r] != 0.0f) {  // NaN states count, so a NaN propagates as in an unsplit row
            row_max = std::max(row_max, state[r]);
          }
        }
        float* out = acc + static_cast<size_t>(r) * v_head_size;
        std::fill_n(out, v_head_size, 0.0f);
        float sum = 0.0f;
        for (std::ptrdiff_t c = 0; c < plan.kv_splits; ++c) {
          const float* state = states + static_cast<size_t>(c) * state_size;
          const float split_sum = state[row_block + r];
          if (split_sum == 0.0f) {
            continue;  // this split saw no cached row of this query row
          }
          const float weight = std::exp(state[r] - row_max);
          sum += weight * split_sum;
          const float* a = state + 2 * row_block + static_cast<size_t>(r) * v_head_size;
          for (size_t d = 0; d < v_head_size; ++d) {
            out[d] += weight * a[d];
          }
        }
        m[r] = row_max;
        l[r] = sum;
      }
      attention.WriteRows(b, kv, s_begin, rows, m, l, acc);
    }
  });
  return Status::OK();
}

template Status RunGroupedFlashAttention<float>(const GroupedFlashAttentionArgs<float>&, AllocatorPtr, ThreadPool*);
template Status RunGroupedFlashAttention<MLFloat16>(const GroupedFlashAttentionArgs<MLFloat16>&, AllocatorPtr,
                                                    ThreadPool*);

}  // namespace contrib
}  // namespace onnxruntime
