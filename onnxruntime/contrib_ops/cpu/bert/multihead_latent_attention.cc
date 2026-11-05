// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/multihead_latent_attention.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#include "contrib_ops/cpu/bert/multihead_latent_attention_helper.h"
#include "core/common/inlined_containers.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/mlas_backend_kernel_selector_config_utils.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      MultiHeadLatentAttention,                                         \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCpuExecutionProvider,                                            \
      KernelDefBuilder()                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()), \
      MultiHeadLatentAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

namespace {

// The query rows of a batch entry, (token, head) pairs in BSNH order, form one (S * num_heads) x head_size matrix.
// With a single KV head every row attends over the same cache, so each tile of cached rows is read (and converted to
// fp32) once and multiplied against a whole block of query rows. V is the column window
// [v_offset, v_offset + v_head_size) of the same tile, used in place. The softmax is computed online in fp32 tile by
// tile, so no score matrix over the whole cache is materialized. Tiles past a row block's causal limit are skipped.
// When there are fewer row blocks than threads (decode), the cache is also split into chunks that run in parallel,
// and their partial softmax states are merged with a log-sum-exp reduction.
constexpr int kKvTile = 128;       // cached rows per tile
constexpr int kMaxRowBlock = 128;  // query rows per task
constexpr int kMinKvChunk = 512;   // cached rows per split of the cache, at least
constexpr int kMergeRows = 8;      // query rows per merge task
constexpr int kCopyRows = 4096;    // past rows per copy task
constexpr float kLowest = std::numeric_limits<float>::lowest();

template <typename T>
struct LatentAttentionArgs {
  const T* query;  // (B, S, num_heads, head_size)
  const T* cache;  // rows of batch entry b start at cache + b * cache_batch_stride
  size_t cache_batch_stride;
  T* output;              // (B, S, num_heads, v_head_size)
  const int* past_rows;   // per batch entry: rows cached before this step
  const int* total_rows;  // per batch entry: valid rows after this step
  int batch_size;
  int sequence_length;
  int num_heads;
  int head_size;
  int v_head_size;
  int v_offset;
  float scale;
  const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* mlas_config;
};

// Row counts and positions are int32 at the op's interface, but the work plan rounds them up and adds them, which can
// pass INT32_MAX. All of that arithmetic is done in std::ptrdiff_t.
struct WorkPlan {
  std::ptrdiff_t rows_per_batch;  // sequence_length * num_heads
  std::ptrdiff_t row_block;       // query rows per task
  std::ptrdiff_t row_blocks;      // row blocks per batch entry
  std::ptrdiff_t kv_chunk;        // cached rows per split, a multiple of kKvTile
  std::ptrdiff_t kv_splits;       // splits of the cache per row block
  std::ptrdiff_t task_count;
};

inline std::ptrdiff_t CeilDiv(std::ptrdiff_t a, std::ptrdiff_t b) { return (a + b - 1) / b; }

// Number of cached rows that query row `row` (token row / num_heads) of a batch entry attends to.
inline std::ptrdiff_t CausalLimit(std::ptrdiff_t row, int num_heads, int past_rows, int total_rows) {
  return std::min<std::ptrdiff_t>(past_rows + row / num_heads + 1, total_rows);
}

WorkPlan MakeWorkPlan(int batch_size, int sequence_length, int num_heads, int max_total_rows, int thread_count) {
  WorkPlan plan{};
  plan.rows_per_batch = static_cast<std::ptrdiff_t>(sequence_length) * num_heads;
  // Whole tokens per block when they fit, so a block's causal limits span as few tiles as possible.
  const std::ptrdiff_t block = num_heads <= kMaxRowBlock ? (kMaxRowBlock / num_heads) * num_heads : kMaxRowBlock;
  plan.row_block = std::min(plan.rows_per_batch, block);
  plan.row_blocks = CeilDiv(plan.rows_per_batch, plan.row_block);

  // Split the cache when there are fewer than two row tasks per thread, aiming at about four tasks per thread.
  const std::ptrdiff_t total = max_total_rows;
  const std::ptrdiff_t row_tasks = static_cast<std::ptrdiff_t>(batch_size) * plan.row_blocks;
  std::ptrdiff_t kv_splits = 1;
  if (thread_count > 1 && row_tasks < 2 * static_cast<std::ptrdiff_t>(thread_count) && total > kMinKvChunk) {
    const std::ptrdiff_t wanted = CeilDiv(4 * static_cast<std::ptrdiff_t>(thread_count), row_tasks);
    kv_splits = std::min(wanted, CeilDiv(total, kMinKvChunk));
  }
  plan.kv_chunk = std::max<std::ptrdiff_t>(kKvTile, CeilDiv(CeilDiv(total, kv_splits), kKvTile) * kKvTile);
  plan.kv_splits = std::max<std::ptrdiff_t>(1, CeilDiv(total, plan.kv_chunk));
  plan.task_count = row_tasks * plan.kv_splits;
  return plan;
}

// Runs the online softmax for query rows [row_begin, row_begin + row_count) of batch entry b over cached rows
// [kv_begin, kv_end). On return m, l and acc hold each row's running max, running sum and sum of
// exp(score - max) * V, not yet normalized; l is 0 for a row that saw no cached row.
template <typename T>
void AttendRowBlock(const LatentAttentionArgs<T>& args, const WorkPlan& plan, int b, std::ptrdiff_t row_begin,
                    int row_count, std::ptrdiff_t kv_begin, std::ptrdiff_t kv_end, [[maybe_unused]] float* q_buffer, [[maybe_unused]] float* k_buffer,
                    float* scores, float* m, float* l, float* acc) {
  const size_t head_size = static_cast<size_t>(args.head_size);
  const size_t v_head_size = static_cast<size_t>(args.v_head_size);
  const int past_rows = args.past_rows[b];
  const int total_rows = args.total_rows[b];

  std::fill_n(m, row_count, kLowest);
  std::fill_n(l, row_count, 0.0f);
  std::fill_n(acc, static_cast<size_t>(row_count) * v_head_size, 0.0f);

  // The first row of the block sees the fewest cached rows and the last row the most.
  const std::ptrdiff_t first_limit = CausalLimit(row_begin, args.num_heads, past_rows, total_rows);
  kv_end = std::min(kv_end, CausalLimit(row_begin + row_count - 1, args.num_heads, past_rows, total_rows));
  if (kv_begin >= kv_end) {
    return;
  }

  const size_t q_offset = (static_cast<size_t>(b) * plan.rows_per_batch + row_begin) * head_size;
  const float* q = nullptr;
  if constexpr (std::is_same_v<T, float>) {
    q = args.query + q_offset;
  } else {
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(args.query + q_offset), q_buffer,
                                 static_cast<size_t>(row_count) * head_size);
    q = q_buffer;
  }
  const T* cache = args.cache + static_cast<size_t>(b) * args.cache_batch_stride;

  for (std::ptrdiff_t tile_begin = kv_begin; tile_begin < kv_end; tile_begin += kKvTile) {
    const int n = static_cast<int>(std::min<std::ptrdiff_t>(kKvTile, kv_end - tile_begin));
    const float* k = nullptr;
    if constexpr (std::is_same_v<T, float>) {
      k = cache + static_cast<size_t>(tile_begin) * head_size;
    } else {
      MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(cache + static_cast<size_t>(tile_begin) *
                                                                                  head_size),
                                   k_buffer, static_cast<size_t>(n) * head_size);
      k = k_buffer;
    }

    MlasGemm(CblasNoTrans, CblasTrans, static_cast<size_t>(row_count), static_cast<size_t>(n), head_size, args.scale,
             q, head_size, k, head_size, 0.0f, scores, static_cast<size_t>(n), nullptr, args.mlas_config);

    const bool masked = tile_begin + n > first_limit;
    int used = 0;  // tile rows that at least one query row attends to
    for (int i = 0; i < row_count; ++i) {
      float* s = scores + static_cast<size_t>(i) * n;
      int valid = n;
      if (masked) {
        const std::ptrdiff_t limit = CausalLimit(row_begin + i, args.num_heads, past_rows, total_rows);
        valid = static_cast<int>(std::clamp<std::ptrdiff_t>(limit - tile_begin, 0, n));
      }
      if (valid == 0) {
        std::fill_n(s, n, 0.0f);
        continue;
      }
      const float m_new = std::max(m[i], *std::max_element(s, s + valid));
      for (int j = 0; j < valid; ++j) {
        s[j] -= m_new;
      }
      MlasComputeExp(s, s, static_cast<size_t>(valid));
      float sum = 0.0f;
      for (int j = 0; j < valid; ++j) {
        sum += s[j];
      }
      std::fill(s + valid, s + n, 0.0f);
      if (m_new > m[i]) {
        if (l[i] > 0.0f) {
          const float alpha = std::exp(m[i] - m_new);
          l[i] *= alpha;
          float* a = acc + static_cast<size_t>(i) * v_head_size;
          for (size_t c = 0; c < v_head_size; ++c) {
            a[c] *= alpha;
          }
        }
        m[i] = m_new;
      }
      l[i] += sum;
      used = std::max(used, valid);
    }

    MlasGemm(CblasNoTrans, CblasNoTrans, static_cast<size_t>(row_count), v_head_size, static_cast<size_t>(used), 1.0f,
             scores, static_cast<size_t>(n), k + args.v_offset, head_size, 1.0f, acc, v_head_size, nullptr,
             args.mlas_config);
  }
}

// Normalizes rows of acc in place and writes them to out.
template <typename T>
void WriteOutputRows(float* acc, const float* l, int row_count, size_t v_head_size, T* out) {
  for (int i = 0; i < row_count; ++i) {
    const float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
    float* a = acc + static_cast<size_t>(i) * v_head_size;
    for (size_t c = 0; c < v_head_size; ++c) {
      a[c] *= inv;
    }
  }
  const size_t count = static_cast<size_t>(row_count) * v_head_size;
  if constexpr (std::is_same_v<T, float>) {
    std::memcpy(out, acc, count * sizeof(float));
  } else {
    MlasConvertFloatToHalfBuffer(acc, reinterpret_cast<MLAS_FP16*>(out), count);
  }
}

template <typename T>
Status RunLatentAttention(const LatentAttentionArgs<T>& args, AllocatorPtr allocator, ThreadPool* tp) {
  int max_total_rows = 0;
  for (int b = 0; b < args.batch_size; ++b) {
    max_total_rows = std::max(max_total_rows, args.total_rows[b]);
  }
  const int thread_count = std::max(1, ThreadPool::DegreeOfParallelism(tp));
  const WorkPlan plan = MakeWorkPlan(args.batch_size, args.sequence_length, args.num_heads, max_total_rows,
                                     thread_count);
  const size_t head_size = static_cast<size_t>(args.head_size);
  const size_t v_head_size = static_cast<size_t>(args.v_head_size);
  const size_t row_block = static_cast<size_t>(plan.row_block);
  const bool split = plan.kv_splits > 1;

  // Per worker: fp32 copies of the query block and of one cache tile (fp16 only), the score tile, and the softmax
  // state of a row block unless it goes to the split buffer.
  constexpr bool kConvert = !std::is_same_v<T, float>;
  const size_t q_size = kConvert ? row_block * head_size : 0;
  const size_t k_size = kConvert ? static_cast<size_t>(kKvTile) * head_size : 0;
  const size_t score_size = row_block * kKvTile;
  const size_t state_size = row_block * (v_head_size + 2);
  const size_t worker_size = SafeInt<size_t>(q_size) + k_size + score_size + (split ? 0 : state_size);
  const std::ptrdiff_t worker_count = std::min<std::ptrdiff_t>(thread_count, plan.task_count);

  auto scratch = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(worker_size) * worker_count);
  IAllocatorUniquePtr<float> partial;
  if (split) {
    partial = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(state_size) * plan.task_count);
  }
  // Softmax state of a row block over one split of the cache: m[row_block], l[row_block], acc[row_block][Dv].
  auto split_state = [&](int b, std::ptrdiff_t block, std::ptrdiff_t split_index) {
    const size_t index = (static_cast<size_t>(b) * plan.row_blocks + block) * plan.kv_splits + split_index;
    return partial.get() + index * state_size;
  };

  // Tasks are handed out heaviest first: causally later row blocks attend to more cached rows.
  std::atomic<std::ptrdiff_t> next_task{0};
  ThreadPool::TrySimpleParallelFor(tp, worker_count, [&](std::ptrdiff_t worker) {
    float* q_buffer = scratch.get() + static_cast<size_t>(worker) * worker_size;
    float* k_buffer = q_buffer + q_size;
    float* scores = k_buffer + k_size;
    for (std::ptrdiff_t task = next_task++; task < plan.task_count; task = next_task++) {
      const std::ptrdiff_t split_index = task % plan.kv_splits;
      const std::ptrdiff_t row_task = task / plan.kv_splits;
      const std::ptrdiff_t block = plan.row_blocks - 1 - row_task % plan.row_blocks;
      const int b = static_cast<int>(row_task / plan.row_blocks);
      const std::ptrdiff_t row_begin = block * plan.row_block;
      const int row_count = static_cast<int>(std::min(plan.row_block, plan.rows_per_batch - row_begin));
      const std::ptrdiff_t kv_begin = split_index * plan.kv_chunk;
      const std::ptrdiff_t kv_end = kv_begin + plan.kv_chunk;

      float* m = split ? split_state(b, block, split_index) : scores + score_size;
      float* l = m + row_block;
      float* acc = l + row_block;
      AttendRowBlock(args, plan, b, row_begin, row_count, kv_begin, kv_end, q_buffer, k_buffer, scores, m, l, acc);
      if (!split) {
        WriteOutputRows(acc, l, row_count, v_head_size,
                        args.output + (static_cast<size_t>(b) * plan.rows_per_batch + row_begin) * v_head_size);
      }
    }
  });

  if (!split) {
    return Status::OK();
  }

  // Merges the split states of each query row: out = sum_c exp(m_c - M) acc_c / sum_c exp(m_c - M) l_c.
  const std::ptrdiff_t merge_groups = CeilDiv(plan.rows_per_batch, kMergeRows);
  const std::ptrdiff_t merge_tasks = static_cast<std::ptrdiff_t>(args.batch_size) * merge_groups;
  auto merged = IAllocator::MakeUniquePtr<float>(allocator, SafeInt<size_t>(merge_tasks) * kMergeRows * v_head_size);
  ThreadPool::TrySimpleParallelFor(tp, merge_tasks, [&](std::ptrdiff_t task) {
    const int b = static_cast<int>(task / merge_groups);
    const std::ptrdiff_t row_begin = task % merge_groups * kMergeRows;
    const int row_count = static_cast<int>(std::min<std::ptrdiff_t>(kMergeRows, plan.rows_per_batch - row_begin));
    float* out = merged.get() + static_cast<size_t>(task) * kMergeRows * v_head_size;
    float sums[kMergeRows];
    for (int i = 0; i < row_count; ++i) {
      const std::ptrdiff_t row = row_begin + i;
      const std::ptrdiff_t block = row / plan.row_block;
      const size_t r = static_cast<size_t>(row % plan.row_block);
      float max_m = kLowest;
      for (std::ptrdiff_t c = 0; c < plan.kv_splits; ++c) {
        const float* state = split_state(b, block, c);
        if (state[row_block + r] > 0.0f) {
          max_m = std::max(max_m, state[r]);
        }
      }
      float* o = out + static_cast<size_t>(i) * v_head_size;
      std::fill_n(o, v_head_size, 0.0f);
      float sum = 0.0f;
      for (std::ptrdiff_t c = 0; c < plan.kv_splits; ++c) {
        const float* state = split_state(b, block, c);
        const float l = state[row_block + r];
        if (!(l > 0.0f)) {
          continue;  // this split saw no cached row of this query row
        }
        const float w = std::exp(state[r] - max_m);
        sum += w * l;
        const float* a = state + 2 * row_block + r * v_head_size;
        for (size_t d = 0; d < v_head_size; ++d) {
          o[d] += w * a[d];
        }
      }
      sums[i] = sum;
    }
    WriteOutputRows(out, sums, row_count, v_head_size,
                    args.output + (static_cast<size_t>(b) * plan.rows_per_batch + row_begin) * v_head_size);
  });
  return Status::OK();
}

// Copies the first past_rows[b] rows of each batch entry from past to cache unless they share a buffer, then writes
// the sequence_length new rows of `key` at rows [past_rows[b], past_rows[b] + sequence_length).
template <typename T>
void UpdateCache(const T* past, size_t past_batch_stride, const T* key, T* cache, size_t cache_batch_stride,
                 gsl::span<const int> past_rows, int sequence_length, size_t head_size, ThreadPool* tp) {
  const bool copy_past = past != nullptr && past != cache;
  int max_past_rows = 0;
  if (copy_past) {
    for (int rows : past_rows) {
      max_past_rows = std::max(max_past_rows, rows);
    }
  }
  const std::ptrdiff_t chunks = 1 + CeilDiv(max_past_rows, kCopyRows);
  const std::ptrdiff_t batch_size = static_cast<std::ptrdiff_t>(past_rows.size());
  const size_t new_rows_size = static_cast<size_t>(sequence_length) * head_size;
  ThreadPool::TrySimpleParallelFor(tp, batch_size * chunks, [&](std::ptrdiff_t unit) {
    const size_t b = static_cast<size_t>(unit / chunks);
    const std::ptrdiff_t chunk = unit % chunks;
    T* dst = cache + b * cache_batch_stride;
    if (chunk == 0) {
      std::memcpy(dst + static_cast<size_t>(past_rows[b]) * head_size, key + b * new_rows_size,
                  new_rows_size * sizeof(T));
      return;
    }
    const std::ptrdiff_t row_begin = (chunk - 1) * kCopyRows;
    const std::ptrdiff_t row_end = std::min<std::ptrdiff_t>(row_begin + kCopyRows, past_rows[b]);
    if (row_begin < row_end) {
      const size_t offset = static_cast<size_t>(row_begin) * head_size;
      std::memcpy(dst + offset, past + b * past_batch_stride + offset,
                  static_cast<size_t>(row_end - row_begin) * head_size * sizeof(T));
    }
  });
}

}  // namespace

template <typename T>
MultiHeadLatentAttention<T>::MultiHeadLatentAttention(const OpKernelInfo& info) : OpKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0, "num_heads must be positive.");
  int64_t v_head_size = 0;
  ORT_ENFORCE(info.GetAttr("v_head_size", &v_head_size).IsOK() && v_head_size > 0, "v_head_size must be positive.");
  const int64_t v_offset = info.GetAttrOrDefault<int64_t>("v_offset", 0);
  ORT_ENFORCE(v_offset >= 0, "v_offset must be non-negative.");
  ORT_ENFORCE(info.GetAttr("scale", &scale_).IsOK() && scale_ > 0.0f, "scale must be positive.");
  ORT_ENFORCE(num_heads <= std::numeric_limits<int>::max() && v_head_size <= std::numeric_limits<int>::max() &&
                  v_offset <= std::numeric_limits<int>::max(),
              "num_heads, v_head_size and v_offset must fit in int32.");
  num_heads_ = static_cast<int>(num_heads);
  v_head_size_ = static_cast<int>(v_head_size);
  v_offset_ = static_cast<int>(v_offset);
  SetupMlasBackendKernelSelectorFromConfigOptions(mlas_backend_kernel_selector_config_, info.GetConfigOptions());
}

template <typename T>
Status MultiHeadLatentAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* past_kv = context->Input<Tensor>(2);
  const Tensor* seqlens_k = context->Input<Tensor>(3);
  const Tensor* total_seqlen = context->Input<Tensor>(4);

  MultiHeadLatentAttentionParameters parameters;
  ORT_RETURN_IF_ERROR(multihead_latent_attention_helper::CheckInputs(query, key, past_kv, seqlens_k, total_seqlen,
                                                                     num_heads_, v_head_size_, v_offset_, scale_,
                                                                     parameters));
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const size_t head_size = static_cast<size_t>(parameters.head_size);

  InlinedVector<int> past_rows(batch_size);
  InlinedVector<int> total_rows(batch_size);
  const int32_t* seqlens_k_data = seqlens_k->Data<int32_t>();
  for (int b = 0; b < batch_size; ++b) {
    ORT_RETURN_IF_ERROR(multihead_latent_attention_helper::GetBatchRows(seqlens_k_data[b], b, parameters,
                                                                        past_rows[b], total_rows[b]));
  }

  Tensor* output = context->Output(0, {static_cast<int64_t>(batch_size), static_cast<int64_t>(sequence_length),
                                       static_cast<int64_t>(parameters.v_hidden_size)});
  Tensor* present_kv = context->Output(1, {static_cast<int64_t>(batch_size), int64_t{1},
                                           static_cast<int64_t>(parameters.present_sequence_length),
                                           static_cast<int64_t>(head_size)});
  if (batch_size == 0) {
    return Status::OK();
  }

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
  ThreadPool* tp = context->GetOperatorThreadPool();

  const T* past_data = past_kv != nullptr ? past_kv->Data<T>() : nullptr;
  const size_t past_batch_stride = static_cast<size_t>(parameters.past_sequence_length) * head_size;
  const T* key_data = key->Data<T>();

  // The cache the attention reads: present_kv when it is an output, the new rows alone when there is no cache at
  // all (always a first prompt), and otherwise a scratch copy of the rows the step needs.
  const T* cache = nullptr;
  size_t cache_batch_stride = 0;
  IAllocatorUniquePtr<T> staged_cache;
  if (present_kv != nullptr) {
    T* present_data = present_kv->MutableData<T>();
    cache_batch_stride = static_cast<size_t>(parameters.present_sequence_length) * head_size;
    UpdateCache(past_data, past_batch_stride, key_data, present_data, cache_batch_stride,
                gsl::span<const int>(past_rows.data(), past_rows.size()), sequence_length, head_size, tp);
    cache = present_data;
  } else if (past_kv == nullptr) {
    cache = key_data;
    cache_batch_stride = static_cast<size_t>(sequence_length) * head_size;
  } else {
    int rows = 0;
    for (int b = 0; b < batch_size; ++b) {
      rows = std::max(rows, past_rows[b] + sequence_length);
    }
    cache_batch_stride = static_cast<size_t>(rows) * head_size;
    staged_cache = IAllocator::MakeUniquePtr<T>(allocator, SafeInt<size_t>(cache_batch_stride) * batch_size);
    UpdateCache(past_data, past_batch_stride, key_data, staged_cache.get(), cache_batch_stride,
                gsl::span<const int>(past_rows.data(), past_rows.size()), sequence_length, head_size, tp);
    cache = staged_cache.get();
  }

  LatentAttentionArgs<T> args{};
  args.query = query->Data<T>();
  args.cache = cache;
  args.cache_batch_stride = cache_batch_stride;
  args.output = output->MutableData<T>();
  args.past_rows = past_rows.data();
  args.total_rows = total_rows.data();
  args.batch_size = batch_size;
  args.sequence_length = sequence_length;
  args.num_heads = num_heads_;
  args.head_size = parameters.head_size;
  args.v_head_size = v_head_size_;
  args.v_offset = v_offset_;
  args.scale = scale_;
  args.mlas_config = &mlas_backend_kernel_selector_config_;
  return RunLatentAttention(args, allocator, tp);
}

}  // namespace contrib
}  // namespace onnxruntime
