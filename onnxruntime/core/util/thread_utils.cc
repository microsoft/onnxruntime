// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/util/thread_utils.h"

#include <algorithm>
#include <iostream>

#ifdef _WIN32
#include <Windows.h>
#endif
#include <thread>
#include "core/session/ort_apis.h"

namespace onnxruntime {

#ifdef _WIN32
GroupAffinities GetGroupAffinities() {
  GroupAffinities group_affinities;
  LOGICAL_PROCESSOR_RELATIONSHIP relation = RelationGroup;
  constexpr static const size_t num_information = 128;
  constexpr static const size_t size_information = sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX);
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX processorInfos[num_information];
  DWORD returnLength = num_information * size_information;
  WORD group_id = 0;
  ORT_ENFORCE(GetLogicalProcessorInformationEx(relation, processorInfos, &returnLength),
              "Failed to fetch processor info, error code: ", GetLastError());
  auto numGroups = returnLength / size_information;
  for (int64_t i = 0; i < static_cast<int64_t>(numGroups); ++i) {
    ORT_ENFORCE(processorInfos[i].Relationship == RelationGroup, "Returned processors not belong to same group");
    for (int64_t j = 0; j < static_cast<int>(processorInfos[i].Group.ActiveGroupCount); ++j) {
      const auto& groupInfo = processorInfos[i].Group.GroupInfo[j];
      KAFFINITY processor_affinity = 1UL;
      for (int64_t k = 0; k < groupInfo.ActiveProcessorCount; ++k) {
        group_affinities.push_back({static_cast<int64_t>(group_id), static_cast<int64_t>(processor_affinity)});
        processor_affinity <<= 1;
      }
      group_id++;
    }
  }
  return std::move(group_affinities);
}
#endif

namespace concurrency {
static std::unique_ptr<ThreadPool>
CreateThreadPoolHelper(Env* env, OrtThreadPoolParams options) {
  if (options.thread_pool_size == 1)
    return nullptr;
  std::vector<size_t> cpu_list;
  ThreadOptions to;
  if (options.affinity_vec_len != 0) {
    to.affinity.assign(options.affinity_vec, options.affinity_vec + options.affinity_vec_len);
  }
  if (options.thread_pool_size <= 0) {  // default
#ifdef _WIN32
    to.group_affinities = GetGroupAffinities();
    options.thread_pool_size = static_cast<int>(to.group_affinities.size());
    std::cout << "thread_pool size: " << options.thread_pool_size << std::endl;
    int tid = 0;
    for (const auto& affinity : to.group_affinities) {
      std::cout << "thread " << tid++ << ", group " << affinity.first << ", core " << affinity.second << std::endl;
    }
#else
    cpu_list = Env::Default().GetThreadAffinityMasks();
    if (cpu_list.empty() || cpu_list.size() == 1)
      return nullptr;
    options.thread_pool_size = static_cast<int>(cpu_list.size());
    if (options.auto_set_affinity)
      to.affinity = cpu_list;
#endif
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
