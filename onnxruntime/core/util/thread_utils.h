// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/platform/threadpool.h"
#include "core/session/onnxruntime_c_api.h"
#include <memory>
#include <string>

struct OrtThreadPoolParams{
    //0: Use default setting. (All the physical cores or half of the logical cores)
    //1: Don't create thread pool
    //n: Create a thread pool with n threads
    int thread_pool_size = 0;
    //If it is true and thread_pool_size = 0, populate the thread affinity information in ThreadOptions. 
    //Otherwise if the thread_options has affinity information, we'll use it and set it.
    //In the other case, don't set affinity
    bool auto_set_affinity = false;
    bool allow_spinning = true;
    unsigned int stack_size = 0;
    //Index is thread id, value is CPU ID, starting from zero
    //If the vector is empty, no explict affinity binding
    size_t* affinity_vec = nullptr;
    size_t affinity_vec_len = 0;
    const ORTCHAR_T* name = nullptr;
} ;

struct OrtThreadingOptions {
  // threads used to parallelize execution of an op
  OrtThreadPoolParams intra_op_thread_pool_params;  // use 0 if you want onnxruntime to choose a value for you

  // threads used to parallelize execution across ops
  OrtThreadPoolParams inter_op_thread_pool_params;  // use 0 if you want onnxruntime to choose a value for you
} ;

namespace onnxruntime {

namespace concurrency {


std::unique_ptr<ThreadPool> CreateThreadPool(Env* env, OrtThreadPoolParams options,
                                             Eigen::Allocator* allocator = nullptr);
}  // namespace concurrency
}  // namespace onnxruntime