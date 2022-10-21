#include "thread_utils.h"
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

bool ExtractAffinityFromString(const char* affinity_string, GroupAffinities& group_affinities) {
  group_affinities.clear();
  auto Split = [](const std::string& s, char splitor) {
    std::vector<std::string> ans;
    std::string tmp;
    std::stringstream ss;
    ss << s;
    while (getline(ss, tmp, splitor)) {
      if (!tmp.empty()) {
        ans.push_back(tmp);
      }
    }
    return ans;
  };
  auto ReadGroupAffinity = [&](const std::string& s) {
    auto affinity_strings = Split(s, ',');
    GroupAffinity group_affinity;
    group_affinity.first = std::stoull(affinity_strings[0].c_str());
    group_affinity.second = std::stoull(affinity_strings[1].c_str());
    return std::move(group_affinity);
  };
  auto ReadGroupAffinities = [&](const std::string& s) {
    auto affinity_strings = Split(s, ';');
    GroupAffinities group_affinities;
    for (const auto& iter : affinity_strings) {
      group_affinities.push_back(ReadGroupAffinity(iter));
    }
    return group_affinities;
  };
  try {
    group_affinities = ReadGroupAffinities(affinity_string);
  } catch (...) {
    return false;
  }
  return true;
}

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
    std::cout << "setting default affinity, thread_pool size (including the main thread): " << options.thread_pool_size << std::endl;
    for (int i = 0; i < options.thread_pool_size - 1; ++i) {
      std::cout << "sub-thread " << i + 1 << " affnity set to: group "
                << to.group_affinities[i].first << " with processor bitmask "
                << to.group_affinities[i].second << std::endl;
    }
#else
    cpu_list = Env::Default().GetThreadAffinityMasks();
    if (cpu_list.empty() || cpu_list.size() == 1)
      return nullptr;
    options.thread_pool_size = static_cast<int>(cpu_list.size());
    if (options.auto_set_affinity)
      to.affinity = cpu_list;
#endif
  } else if (!options.group_affinities.empty()) {
    ORT_ENFORCE(static_cast<int>(options.group_affinities.size()) == options.thread_pool_size - 1,
                "Invalid thread options, number of group affinities must equal to options.thread_pool_size - 1");
    to.group_affinities = options.group_affinities;
    std::cout << "applying non-default affinity:" << std::endl;
    for (int i = 0; i < options.thread_pool_size - 1; ++i) {
      std::cout << "sub-thread " << i + 1 << " affnity set to: group "
                << to.group_affinities[i].first << " with processor bitmask "
                << to.group_affinities[i].second << std::endl;
    }
  }
  to.set_denormal_as_zero = options.set_denormal_as_zero;

  return std::make_unique<ThreadPool>(env, to, options.name, options.thread_pool_size,
                                              options.allow_spinning);
}

std::unique_ptr<ThreadPool>
CreateThreadPool(Env* env, OrtThreadPoolParams options, ThreadPoolType tpool_type) {
// If openmp is enabled we don't want to create any additional threadpools for sequential execution.
// However, parallel execution relies on the existence of a separate threadpool. Hence we allow eigen threadpools
// to be created for parallel execution.
#ifdef _OPENMP
  ORT_UNUSED_PARAMETER(env);
  ORT_UNUSED_PARAMETER(options);
  if (tpool_type != ThreadPoolType::INTER_OP) {
    return nullptr;
  } else {
    return CreateThreadPoolHelper(env, options);
  }
#else
  ORT_UNUSED_PARAMETER(tpool_type);
  return CreateThreadPoolHelper(env, options);
#endif
}

}  // namespace concurrency
}  // namespace onnxruntime
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
  tp_options->intra_op_thread_pool_params.allow_spinning = allow_spinning;
  tp_options->inter_op_thread_pool_params.allow_spinning = allow_spinning;
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

ORT_API_STATUS_IMPL(SetGlobalIntraOpThreadAffinity, _Inout_ OrtThreadingOptions* tp_options, const char* affinity_string) {
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  if (!affinity_string) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null affinity string");
  }
  if (!onnxruntime::concurrency::ExtractAffinityFromString(affinity_string, tp_options->intra_op_thread_pool_params.group_affinities)) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Affinity string invalid, failed to set affinity to intra thread option");
  }
  return nullptr;
}

}  // namespace OrtApis
