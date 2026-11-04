// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <gsl/gsl>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/external_data_loader.h"
#include "core/platform/env.h"

namespace onnxruntime {

// Loader-private workers: Run calls are serialized by the loader's mutex.
class ExternalDataLoaderThreadPool {
 public:
  explicit ExternalDataLoaderThreadPool(size_t reader_count) : results_(reader_count) {
    ORT_ENFORCE(reader_count > 1, "A parallel external-data reader pool requires at least two readers.");
    workers_.reserve(reader_count);
    ORT_TRY {
      for (size_t reader = 0; reader < reader_count; ++reader) {
        workers_.emplace_back([this, reader]() { Worker(reader); });
      }
    }
    ORT_CATCH(...) {
      Shutdown();
      ORT_RETHROW;
    }
  }

  ~ExternalDataLoaderThreadPool() {
    Shutdown();
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ExternalDataLoaderThreadPool);

  size_t Size() const noexcept { return workers_.size(); }

  common::Status Run(const std::function<common::Status(size_t)>& read) {
    std::unique_lock<std::mutex> lock(mutex_);
    read_ = &read;
    pending_ = workers_.size();
    ++generation_;
    work_available_.notify_all();
    work_complete_.wait(lock, [this] { return pending_ == 0; });
    read_ = nullptr;

    // Join every read before returning an error so staging buffers can be safely reused.
    for (const auto& result : results_) {
      ORT_RETURN_IF_ERROR(result);
    }
    return common::Status::OK();
  }

 private:
  void Shutdown() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    work_available_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  void Worker(size_t reader) {
    uint64_t generation = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      work_available_.wait(lock, [&] { return stop_ || generation_ != generation; });
      if (stop_) {
        return;
      }
      generation = generation_;
      const auto* read = read_;
      lock.unlock();

      common::Status result;
      ORT_TRY {
        result = (*read)(reader);
      }
      ORT_CATCH(const std::exception& ex) {
        ORT_HANDLE_EXCEPTION([&]() {
          result = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to read external data: ", ex.what());
        });
      }
      ORT_CATCH(...) {
        ORT_HANDLE_EXCEPTION([&]() {
          result = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to read external data: unknown exception");
        });
      }

      lock.lock();
      results_[reader] = std::move(result);
      if (--pending_ == 0) {
        work_complete_.notify_one();
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable work_available_;
  std::condition_variable work_complete_;
  const std::function<common::Status(size_t)>* read_{nullptr};
  size_t pending_{0};
  uint64_t generation_{0};
  bool stop_{false};
  InlinedVector<common::Status> results_;
  InlinedVector<std::thread> workers_;
};

inline size_t ResolveExternalDataBatchThreadCount(size_t configured_threads, bool threads_explicit) noexcept {
  if (threads_explicit || configured_threads != ExternalDataLoaderOptions::kDefaultReadingThreads) {
    return configured_threads;
  }
  return ExternalDataLoaderOptions::GetDefaultReadingThreads();
}

inline void SwapExternalDataByteOrderInplace(void* buffer, size_t length, size_t element_size) {
  auto* bytes = static_cast<uint8_t*>(buffer);
  for (size_t offset = 0; offset < length; offset += element_size) {
    std::reverse(bytes + offset, bytes + offset + element_size);
  }
}

inline void TransformExternalDataStagingBuffer(uint8_t* buffer, size_t length, bool is_bool, size_t element_size) {
  if (is_bool) {
    std::transform(buffer, buffer + length, buffer,
                   [](uint8_t value) { return value != 0; });
  }

  if constexpr (std::endian::native != std::endian::little) {
    if (element_size > 1) {
      SwapExternalDataByteOrderInplace(buffer, length, element_size);
    }
  }
}

#if !defined(__wasm__)
// Converts exceptions into a Status so callers can still drain asynchronous copies before unwinding.
template <typename Fn>
common::Status RunExternalDataLoadStep(Fn&& fn) {
  common::Status status;
  ORT_TRY {
    status = fn();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to load external data: ", ex.what());
    });
  }
  ORT_CATCH(...) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to load external data: unknown exception");
    });
  }
  return status;
}

struct ExternalDataLoadTask {
  const RandomAccessFile* file{nullptr};
  FileOffsetType file_offset{0};
  uint8_t* destination{nullptr};
  size_t length{0};
  size_t element_size{1};
  bool is_bool{false};
};

template <typename OnWorkerStartFn, typename SyncStreamFn, typename CopyAsyncFn>
common::Status ExecuteExternalDataLoadTasks(
    gsl::span<const ExternalDataLoadTask> tasks,
    bool allow_small_parallel,
    size_t reading_thread_count,
    size_t batch_reading_thread_count,
    size_t single_buffer_size,
    size_t max_chunk_bytes,
    const std::array<void*, 2>& buffers,
    std::unique_ptr<ExternalDataLoaderThreadPool>& reader_pool,
    const std::function<bool()>& is_canceled,
    OnWorkerStartFn&& on_worker_start,
    SyncStreamFn&& sync_stream,
    CopyAsyncFn&& copy_async) {
  size_t total_bytes = 0;
  for (const auto& task : tasks) {
    total_bytes += task.length;
  }
  if (total_bytes == 0) {
    return common::Status::OK();
  }

  const size_t worker_count =
      (allow_small_parallel && total_bytes >= single_buffer_size)
          ? batch_reading_thread_count
          : reading_thread_count;
  const bool use_parallel =
      (worker_count > 1) &&
      (allow_small_parallel
           ? (total_bytes >= ExternalDataLoaderOptions::kMinChunkBytes &&
              (total_bytes >= kExternalDataLoaderParallelReadThreshold || tasks.size() > 1))
           : (total_bytes >= kExternalDataLoaderParallelReadThreshold));

  if (use_parallel) {
    const size_t slice_size = (single_buffer_size / worker_count) & ~size_t(4095);
    if (slice_size >= ExternalDataLoaderOptions::kMinChunkBytes) {
      const size_t aligned_max_chunk = std::max<size_t>(
          ExternalDataLoaderOptions::kMinChunkBytes, max_chunk_bytes & ~size_t(4095));
      size_t target_chunk_size = std::min<size_t>(slice_size, aligned_max_chunk);
      if (total_bytes < worker_count * target_chunk_size) {
        const size_t balanced = (total_bytes + worker_count - 1) / worker_count;
        target_chunk_size = std::clamp<size_t>(
            (balanced + 4095) & ~size_t(4095),
            ExternalDataLoaderOptions::kMinChunkBytes,
            target_chunk_size);
      }

      // Chunks are generated lazily from their index so descriptor memory is O(tasks), not O(total bytes / chunk).
      std::vector<size_t> first_chunk_of_task;
      first_chunk_of_task.reserve(tasks.size() + 1);
      first_chunk_of_task.push_back(0);
      for (const auto& task : tasks) {
        first_chunk_of_task.push_back(first_chunk_of_task.back() +
                                      (task.length + target_chunk_size - 1) / target_chunk_size);
      }
      const size_t chunk_count = first_chunk_of_task.back();
      const auto get_chunk = [&](size_t chunk_idx) {
        const auto it = std::upper_bound(first_chunk_of_task.begin(), first_chunk_of_task.end(), chunk_idx);
        const auto& task = tasks[static_cast<size_t>(it - first_chunk_of_task.begin()) - 1];
        const size_t offset = (chunk_idx - *(it - 1)) * target_chunk_size;
        return ExternalDataLoadTask{
            task.file,
            task.file_offset + static_cast<FileOffsetType>(offset),
            task.destination + offset,
            std::min(target_chunk_size, task.length - offset),
            task.element_size,
            task.is_bool};
      };

      if (chunk_count > 1) {
        ORT_TRY {
          if (!reader_pool || reader_pool->Size() != worker_count) {
            reader_pool.reset();
            reader_pool = std::make_unique<ExternalDataLoaderThreadPool>(worker_count);
          }
        }
        ORT_CATCH(const std::exception& ex) {
          common::Status pool_status;
          ORT_HANDLE_EXCEPTION([&]() {
            pool_status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create external data reader pool: ", ex.what());
          });
          return pool_status;
        }
        ORT_CATCH(...) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create external data reader pool: unknown exception");
        }

        std::atomic<size_t> next_chunk_idx{0};
        std::atomic<bool> stop_on_error{false};
        return reader_pool->Run([&](size_t worker_id) -> common::Status {
          if (worker_id >= chunk_count) {
            return common::Status::OK();
          }

          ORT_RETURN_IF_ERROR(on_worker_start(worker_id));

          size_t slot = 0;
          std::array<bool, 2> stream_used{false, false};
          auto sync_worker_streams = [&]() -> common::Status {
            common::Status sync_result = common::Status::OK();
            for (size_t s = 0; s < 2; ++s) {
              if (stream_used[s]) {
                const auto st = sync_stream(worker_id * 2 + s);
                if (sync_result.IsOK() && !st.IsOK()) {
                  sync_result = st;
                }
                stream_used[s] = false;
              }
            }
            return sync_result;
          };

          // Exceptions must not skip the stream drain: in-flight copies still read the staging buffers.
          const auto worker_status = RunExternalDataLoadStep([&]() -> common::Status {
            common::Status loop_status = common::Status::OK();
            for (;;) {
              if (stop_on_error.load(std::memory_order_relaxed)) {
                break;
              }
              if (is_canceled && is_canceled()) {
                loop_status = ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                                              "Saving session state weights is canceled due to user request.");
                stop_on_error.store(true, std::memory_order_relaxed);
                break;
              }

              const size_t chunk_idx = next_chunk_idx.fetch_add(1, std::memory_order_relaxed);
              if (chunk_idx >= chunk_count) {
                break;
              }

              const ExternalDataLoadTask chunk = get_chunk(chunk_idx);
              auto* staging = static_cast<uint8_t*>(buffers[slot]) + worker_id * slice_size;
              const size_t stream_idx = worker_id * 2 + slot;

              if (stream_used[slot]) {
                const auto sync_st = sync_stream(stream_idx);
                stream_used[slot] = false;
                if (!sync_st.IsOK()) {
                  loop_status = sync_st;
                  stop_on_error.store(true, std::memory_order_relaxed);
                  break;
                }
              }

              const auto read_st = chunk.file->Read(
                  chunk.file_offset, gsl::span<char>{reinterpret_cast<char*>(staging), chunk.length});
              if (!read_st.IsOK()) {
                loop_status = read_st;
                stop_on_error.store(true, std::memory_order_relaxed);
                break;
              }

              TransformExternalDataStagingBuffer(staging, chunk.length, chunk.is_bool, chunk.element_size);

              // Mark the stream used before submitting so a throwing copy is still drained.
              stream_used[slot] = true;
              const auto copy_st = copy_async(chunk.destination, staging, chunk.length, stream_idx);
              if (!copy_st.IsOK()) {
                loop_status = copy_st;
                stop_on_error.store(true, std::memory_order_relaxed);
                break;
              }

              slot ^= 1;
            }
            return loop_status;
          });
          if (!worker_status.IsOK()) {
            stop_on_error.store(true, std::memory_order_relaxed);
          }

          const auto drain_st = RunExternalDataLoadStep(sync_worker_streams);
          return worker_status.IsOK() ? drain_st : worker_status;
        });
      }
    }
  }

  std::array<bool, 2> stream_used{};
  auto synchronize_streams = [&]() {
    common::Status status = common::Status::OK();
    for (size_t i = 0; i < 2; ++i) {
      if (stream_used[i]) {
        const auto sync_status = sync_stream(i);
        if (status.IsOK() && !sync_status.IsOK()) {
          status = sync_status;
        }
        stream_used[i] = false;
      }
    }
    return status;
  };

  // Exceptions must not skip the stream drain: in-flight copies still read the staging buffers.
  const auto submit_status = RunExternalDataLoadStep([&]() -> common::Status {
    size_t chunk_index = 0;
    for (const auto& task : tasks) {
      for (size_t offset = 0; offset < task.length; ++chunk_index) {
        if (is_canceled && is_canceled()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                                 "Saving session state weights is canceled due to user request.");
        }

        const size_t buffer_index = chunk_index % 2;
        const size_t chunk_size = std::min(single_buffer_size, task.length - offset);

        if (stream_used[buffer_index]) {
          const auto sync_status = sync_stream(buffer_index);
          stream_used[buffer_index] = false;
          ORT_RETURN_IF_ERROR(sync_status);
        }

        auto* staging = static_cast<uint8_t*>(buffers[buffer_index]);
        ORT_RETURN_IF_ERROR(task.file->Read(
            task.file_offset + static_cast<FileOffsetType>(offset),
            gsl::span<char>{reinterpret_cast<char*>(staging), chunk_size}));

        TransformExternalDataStagingBuffer(staging, chunk_size, task.is_bool, task.element_size);

        // Mark the stream used before submitting so a throwing copy is still drained.
        stream_used[buffer_index] = true;
        ORT_RETURN_IF_ERROR(copy_async(task.destination + offset, staging, chunk_size, buffer_index));
        offset += chunk_size;
      }
    }
    return common::Status::OK();
  });

  const auto drain_status = RunExternalDataLoadStep(synchronize_streams);
  return submit_status.IsOK() ? drain_status : submit_status;
}
#endif  // !defined(__wasm__)

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
