// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/platform/threadpool.h"
#include "core/session/onnxruntime_c_api.h"
#include <memory>
#include <string>

struct OrtThreadPoolParams {
  // 0: Use default setting. (All the physical cores or half of the logical cores)
  // 1: Don't create thread pool
  // n: Create a thread pool with n threads.
  int thread_pool_size = 0;

  // If it is true and thread_pool_size = 0, populate the thread affinity information in ThreadOptions.
  // Otherwise if the thread_options has affinity information, we'll use it and set it.
  // In the other case, don't set affinity
  bool auto_set_affinity = false;

  // If it is true, the thread pool will spin a while after the queue became empty.
  bool allow_spinning = true;

  // It it is non-negative, thread pool will split a task by a decreasing block size
  // of remaining_of_total_iterations / (num_of_threads * dynamic_block_base_)
  int dynamic_block_base_ = 0;

  unsigned int stack_size = 0;

  // A utf-8 string of affinity settings, format be like:
  // <1st_thread_affinity_config>;<2nd_thread_affinity_config>;<3rd_thread_affinity_config>...
  // ith_thread_affinity_config could be:
  // 1,2,3
  // meaing ith thread attach to logical processor 1,2,3
  // or
  // 1-8
  // meaning ith thread will be attached to first 8 logical processors
  std::string affinity_str;

  const ORTCHAR_T* name = nullptr;

  // Set or unset denormal as zero
  bool set_denormal_as_zero = false;

  // members to manage custom threads
  OrtCustomCreateThreadFn custom_create_thread_fn = nullptr;
  void* custom_thread_creation_options = nullptr;
  OrtCustomJoinThreadFn custom_join_thread_fn = nullptr;
};

std::ostream& operator<<(std::ostream& os, const OrtThreadPoolParams& params);

struct OrtThreadingOptions {
  // Params for creating the threads that parallelizes execution of an op
  OrtThreadPoolParams intra_op_thread_pool_params;

  // Params for creating the threads that parallelizes execution across ops
  OrtThreadPoolParams inter_op_thread_pool_params;
};

namespace onnxruntime {

namespace concurrency {
enum class ThreadPoolType : uint8_t {
  INTRA_OP,
  INTER_OP
};
std::unique_ptr<ThreadPool> CreateThreadPool(Env* env, OrtThreadPoolParams options,
                                             ThreadPoolType tpool_type);
}  // namespace concurrency
}  // namespace onnxruntime
