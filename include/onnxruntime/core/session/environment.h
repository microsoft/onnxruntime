// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <memory>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/platform/threadpool.h"
#include "core/common/logging/logging.h"

struct ThreadingOptions;
namespace onnxruntime {
/** TODO: remove this class
   Provides the runtime environment for onnxruntime.
   Create one instance for the duration of execution.
*/
class Environment {
 public:
  /**
     Create and initialize the runtime environment.
  */
  static Status Create(std::unique_ptr<logging::LoggingManager> logging_manager,
                       std::unique_ptr<Environment>& environment,
                       const ThreadingOptions* tp_options = nullptr,
                       bool create_thread_pool = false);

  logging::LoggingManager* GetLoggingManager() const {
    return logging_manager_.get();
  }

  void SetLoggingManager(std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager) {
    logging_manager_ = std::move(logging_manager);
  }

  onnxruntime::concurrency::ThreadPool* GetIntraOpThreadPool() const {
    return intra_op_thread_pool_.get();
  }

  onnxruntime::concurrency::ThreadPool* GetInterOpThreadPool() const {
    return inter_op_thread_pool_.get();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Environment);

  Environment() = default;
  Status Initialize(std::unique_ptr<logging::LoggingManager> logging_manager,
                    const ThreadingOptions* tp_options = nullptr,
                    bool create_global_thread_pool = false);

  std::unique_ptr<logging::LoggingManager> logging_manager_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> intra_op_thread_pool_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> inter_op_thread_pool_;
};
}  // namespace onnxruntime
