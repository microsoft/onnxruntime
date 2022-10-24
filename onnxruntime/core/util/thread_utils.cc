// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/util/thread_utils.h"

#include <algorithm>

#ifdef _WIN32
#include <Windows.h>
#endif
#include <thread>
#include "core/session/ort_apis.h"

namespace onnxruntime {
namespace concurrency {
static std::unique_ptr<ThreadPool>
CreateThreadPoolHelper(Env* env, OrtThreadPoolParams options) {
  if (options.thread_pool_size == 1)
    return nullptr;
  ThreadOptions to;

  if (options.affinity_vec_len != 0) {
    // Currently, the affinities are passed in as bit masks and they need to be converted to integers.
    // We when create a public API, bit-masks must be done away with because of the following reasons:
    // 1) integers have a limited number of bits
    // 2) bit-masks of integers can only represent numbers 0 -63, but on VMs the actual logical processor numbering
    //    may not start with zero for a given core and may be way beyond 63.
    // 3) Customers would be forced to concoct bit-masks which is far less convenient than simply an array of processor integers. 
    to.affinity.reserve(options.affinity_vec_len);
    std::transform(options.affinity_vec, options.affinity_vec + options.affinity_vec_len, std::back_inserter(to.affinity),
                   [](size_t affinity) {
                     return LogicalProcessors{static_cast<int>(affinity)};
                   });
  }

  if (options.thread_pool_size <= 0) {  // default
    auto cpu_list = Env::Default().GetThreadAffinityMasks();
    if (cpu_list.empty() || cpu_list.size() == 1)
      return nullptr;
    options.thread_pool_size = static_cast<int>(cpu_list.size());
    if (options.auto_set_affinity)
      to.affinity = cpu_list;
  }

  to.set_denormal_as_zero = options.set_denormal_as_zero;

  // set custom thread management members
  to.custom_create_thread_fn = options.custom_create_thread_fn;
  to.custom_thread_creation_options = options.custom_thread_creation_options;
  to.custom_join_thread_fn = options.custom_join_thread_fn;
  to.dynamic_block_base_ = options.dynamic_block_base_;
  if (to.custom_create_thread_fn) {
    ORT_ENFORCE(to.custom_join_thread_fn, "custom join thread function not set");
  }

  return std::make_unique<ThreadPool>(env, to, options.name, options.thread_pool_size,
                                      options.allow_spinning);
}

std::unique_ptr<ThreadPool>
CreateThreadPool(Env* env, OrtThreadPoolParams options, ThreadPoolType tpool_type) {
  // If openmp is enabled we don't want to create any additional threadpools for sequential execution.
  // However, parallel execution relies on the existence of a separate threadpool. Hence we allow eigen threadpools
  // to be created for parallel execution.
  ORT_UNUSED_PARAMETER(tpool_type);
  return CreateThreadPoolHelper(env, options);
}

}  // namespace concurrency
}  // namespace onnxruntime
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26409)
#endif
namespace OrtApis {
ORT_API_STATUS_IMPL(CreateThreadingOptions, _Outptr_ OrtThreadingOptions** out) {
  *out = new OrtThreadingOptions();
  return nullptr;
}

ORT_API(void, ReleaseThreadingOptions, _Frees_ptr_opt_ OrtThreadingOptions* p) {
  delete p;
}

ORT_API_STATUS_IMPL(SetGlobalIntraOpNumThreads, _Inout_ OrtThreadingOptions* tp_options, int intra_op_num_threads) {
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  tp_options->intra_op_thread_pool_params.thread_pool_size = intra_op_num_threads;
  return nullptr;
}
ORT_API_STATUS_IMPL(SetGlobalInterOpNumThreads, _Inout_ OrtThreadingOptions* tp_options, int inter_op_num_threads) {
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  tp_options->inter_op_thread_pool_params.thread_pool_size = inter_op_num_threads;
  return nullptr;
}

ORT_API_STATUS_IMPL(SetGlobalSpinControl, _Inout_ OrtThreadingOptions* tp_options, int allow_spinning) {
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  if (!(allow_spinning == 1 || allow_spinning == 0)) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received invalid value for allow_spinning. Valid values are 0 or 1");
  }
  tp_options->intra_op_thread_pool_params.allow_spinning = (allow_spinning != 0);
  tp_options->inter_op_thread_pool_params.allow_spinning = (allow_spinning != 0);
  return nullptr;
}

ORT_API_STATUS_IMPL(SetGlobalDenormalAsZero, _Inout_ OrtThreadingOptions* tp_options) {
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  tp_options->intra_op_thread_pool_params.set_denormal_as_zero = true;
  tp_options->inter_op_thread_pool_params.set_denormal_as_zero = true;
  return nullptr;
}

ORT_API_STATUS_IMPL(SetGlobalCustomCreateThreadFn, _Inout_ OrtThreadingOptions* tp_options, _In_ OrtCustomCreateThreadFn ort_custom_create_thread_fn) {
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  tp_options->inter_op_thread_pool_params.custom_create_thread_fn = ort_custom_create_thread_fn;
  tp_options->intra_op_thread_pool_params.custom_create_thread_fn = ort_custom_create_thread_fn;
  return nullptr;
}

ORT_API_STATUS_IMPL(SetGlobalCustomThreadCreationOptions, _Inout_ OrtThreadingOptions* tp_options, _In_ void* ort_custom_thread_creation_options) {
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  tp_options->inter_op_thread_pool_params.custom_thread_creation_options = ort_custom_thread_creation_options;
  tp_options->intra_op_thread_pool_params.custom_thread_creation_options = ort_custom_thread_creation_options;
  return nullptr;
}

ORT_API_STATUS_IMPL(SetGlobalCustomJoinThreadFn, _Inout_ OrtThreadingOptions* tp_options, _In_ OrtCustomJoinThreadFn ort_custom_join_thread_fn) {
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  tp_options->inter_op_thread_pool_params.custom_join_thread_fn = ort_custom_join_thread_fn;
  tp_options->intra_op_thread_pool_params.custom_join_thread_fn = ort_custom_join_thread_fn;
  return nullptr;
}

}  // namespace OrtApis
