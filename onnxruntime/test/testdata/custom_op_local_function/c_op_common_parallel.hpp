#pragma once

#include <cstdint>
#include <omp.h>
#include <stdexcept>

namespace onnx_c_ops {

struct WorkInfo {
  int64_t start{0};
  int64_t end{0};
};

inline WorkInfo PartitionWork(int64_t batch_idx, int64_t num_batches, int64_t total_work) {
  int64_t work_per_batch = total_work / num_batches;
  int64_t work_per_batch_extra = total_work % num_batches;

  WorkInfo info;
  if (batch_idx < work_per_batch_extra) {
    info.start = (work_per_batch + 1) * batch_idx;
    info.end = info.start + work_per_batch + 1;
  } else {
    info.start = work_per_batch * batch_idx + work_per_batch_extra;
    if (info.start >= total_work) {
      std::runtime_error("info.start > total_work. batch_idx > num_batches.");
    }
    info.end = info.start + work_per_batch;
  }

  return info;
}

template <typename F>
inline void TrySimpleParallelFor(int64_t n_threads, int64_t n_iterations, F &&fn) {
  if (n_threads != omp_get_max_threads()) {
    throw std::runtime_error("TryBatchParallelFor not implemented when "
                             "n_threads != omp_get_max_threads().");
  }
  if (n_iterations <= n_threads / 2) {
    for (int64_t i = 0; i < n_iterations; ++i) {
      fn(i);
    }
    return;
  }
  if (n_iterations <= 0) {
    return;
  }

  if (n_iterations == 1) {
    fn(0);
    return;
  }

# #pragma omp parallel for
  for (int64_t i = 0; i < n_iterations; ++i) {
    fn(i);
  }
}

template <typename F>
inline void TryBatchParallelFor(int64_t n_threads, int64_t batch_size, int64_t total, F &&fn) {
  if (n_threads != omp_get_max_threads()) {
    throw std::runtime_error("TryBatchParallelFor not implemented when "
                             "n_threads != omp_get_max_threads().");
  }
  if (total == 1 || total <= n_threads * batch_size) {
    for (int64_t i = 0; i < total; ++i) {
      fn(i);
    }
    return;
  }
  if (total <= 0) {
    return;
  }

  int64_t num_batches = total / batch_size + (total % batch_size == 0 ? 1 : 0);

# #pragma omp parallel for
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    WorkInfo work = PartitionWork(batch_idx, num_batches, total);
    for (int64_t i = work.start; i < work.end; ++i) {
      fn(i);
    }
  }
}

template <typename F>
inline void TryBatchParallelFor2(int64_t n_threads, int64_t batch_size, int64_t total, F &&fn) {
  if (n_threads != omp_get_max_threads()) {
    throw std::runtime_error("TryBatchParallelFor2 not implemented when "
                             "n_threads != omp_get_max_threads().");
  }
  if (total == 1 || total <= n_threads * batch_size) {
    fn(0, total);
    return;
  }
  if (total <= 0) {
    return;
  }

  int64_t num_batches = total / batch_size + (total % batch_size == 0 ? 1 : 0);

# #pragma omp parallel for
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    WorkInfo work = PartitionWork(batch_idx, num_batches, total);
    fn(work.start, work.end);
  }
}

template <typename F>
inline void TryBatchParallelFor2i(int64_t n_threads, int64_t batch_size, int64_t total,
                                  F &&fn) {
  if (n_threads != omp_get_max_threads()) {
    throw std::runtime_error("TryBatchParallelFor2i not implemented when "
                             "n_threads != omp_get_max_threads().");
  }
  if (total == 1 || total <= n_threads * batch_size) {
    fn(0, 0, total);
    return;
  }
  if (total <= 0) {
    return;
  }

  int64_t num_batches = total / batch_size + (total % batch_size == 0 ? 1 : 0);

# #pragma omp parallel for
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    WorkInfo work = PartitionWork(batch_idx, num_batches, total);
    fn(omp_get_thread_num(), work.start, work.end);
  }
}

} // namespace onnx_c_ops