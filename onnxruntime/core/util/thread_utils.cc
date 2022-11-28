// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/util/thread_utils.h"

#include <algorithm>

#ifdef _WIN32
#include <Windows.h>
#endif
#include <thread>
#include "core/session/ort_apis.h"
#include "core/common/string_utils.h"

namespace onnxruntime {
namespace concurrency {

// extract affinity from affinity string
// processor id in affinity string starts from 1
std::vector<LogicalProcessors> ReadThreadAffinityConfig(const std::string& affinity_str) {
  std::vector<LogicalProcessors> logical_processors_vector;
  auto affinities = utils::SplitString(affinity_str, ";");
  for (const auto& affinity : affinities) {
    LogicalProcessors logical_processors;
    auto processor_interval = utils::SplitString(affinity, "-");
    if (processor_interval.size() == 2) {
      auto processor_from = std::stoi(processor_interval[0].data());
      auto processor_to = std::stoi(processor_interval[1].data());
      ORT_ENFORCE(processor_from > 0 && processor_to > 0,
                  std::string{"Processor id must starts from 1: "} + affinity.data());
      ORT_ENFORCE(processor_from <= processor_to,
                  std::string{"Invalid processor interval: "} + affinity.data());
      logical_processors.resize(static_cast<size_t>(1ULL + processor_to - processor_from));
      std::iota(logical_processors.begin(), logical_processors.end(), processor_from - 1);
    } else {
      for (const auto& processor_str : utils::SplitString(affinity, ",")) {
        auto processor_id = std::stoi(processor_str.data());
        ORT_ENFORCE(processor_id > 0, std::string{"Processor id must starts from 1: "} + affinity.data());
        logical_processors.push_back(processor_id - 1);
      }
    }
    logical_processors_vector.push_back(std::move(logical_processors));
  }
  return logical_processors_vector;
}

static std::unique_ptr<ThreadPool>
CreateThreadPoolHelper(Env* env, OrtThreadPoolParams options) {
  if (options.thread_pool_size == 1) {
    return nullptr;
  }

  ThreadOptions to;
  if (options.thread_pool_size <= 0) {  // default
    to.affinity = Env::Default().GetDefaultThreadAffinities();
    if (to.affinity.size() <= 1) {
      return nullptr;
    }
    options.thread_pool_size = static_cast<int>(to.affinity.size());
  } else if (!options.affinity_str.empty()) {
    to.affinity = ReadThreadAffinityConfig(options.affinity_str);
    ORT_ENFORCE(to.affinity.size() == static_cast<size_t>(options.thread_pool_size) - 1,
                "Number of affinities must equal to thread pool size minus one");
    // prepend an empty affinity as placeholder for the main thread
    to.affinity.insert(to.affinity.begin(), LogicalProcessors{});
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

ORT_API_STATUS_IMPL(SetGlobalIntraOpThreadAffinity, _Inout_ OrtThreadingOptions* tp_options, const char* affinity_string) {
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  if (!affinity_string) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null affinity string");
  }
  tp_options->intra_op_thread_pool_params.affinity_str = affinity_string;
  return nullptr;
}

}  // namespace OrtApis
