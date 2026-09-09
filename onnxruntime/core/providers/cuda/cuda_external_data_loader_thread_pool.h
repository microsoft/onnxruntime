// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace cuda {

// Loader-private workers: Run calls are serialized by the loader's mutex.
class ExternalDataLoaderThreadPool {
 public:
  explicit ExternalDataLoaderThreadPool(size_t reader_count) : results_(reader_count) {
    ORT_ENFORCE(reader_count > 1, "A parallel external-data reader pool requires at least two readers.");
    workers_.reserve(reader_count);
    for (size_t reader = 0; reader < reader_count; ++reader) {
      workers_.emplace_back([this, reader](std::stop_token stop) { Worker(reader, stop); });
    }
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ExternalDataLoaderThreadPool);

  common::Status Run(const std::function<common::Status(size_t)>& read) {
    std::unique_lock lock(mutex_);
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
  void Worker(size_t reader, std::stop_token stop) {
    uint64_t generation = 0;
    std::unique_lock lock(mutex_);
    while (work_available_.wait(lock, stop, [&] { return generation_ != generation; })) {
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
  std::condition_variable_any work_available_;
  std::condition_variable work_complete_;
  const std::function<common::Status(size_t)>* read_{nullptr};
  size_t pending_{0};
  uint64_t generation_{0};
  InlinedVector<common::Status> results_;
  // Destroy workers first. jthread also stops and joins partially constructed pools.
  InlinedVector<std::jthread> workers_;
};

}  // namespace cuda
}  // namespace onnxruntime
