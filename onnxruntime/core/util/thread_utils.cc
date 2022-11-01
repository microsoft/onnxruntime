#include "thread_utils.h"
#include <algorithm>

#include <core/common/make_unique.h>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <thread>
#include "core/session/ort_apis.h"
#include "core/common/logging/logging.h"


namespace onnxruntime {

#ifdef _WIN32
ThreadAffinities GetDefaultThreadAffinities() {
  DWORD returnLength = 0;
  GetLogicalProcessorInformationEx(RelationGroup, nullptr, &returnLength);
  auto last_error = GetLastError();
  if (last_error != ERROR_INSUFFICIENT_BUFFER) {
    LOGS_DEFAULT(ERROR) << "GetLogicalProcessorInformationEx failed to obtain buffer length. error code: "
                        << last_error
                        << " error msg: " << std::system_category().message(last_error);
    return {};
  }

  std::unique_ptr<char[]> allocation = std::make_unique<char[]>(returnLength);
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* processorInfos = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(allocation.get());

  if (!GetLogicalProcessorInformationEx(RelationGroup, processorInfos, &returnLength)) {
    last_error = GetLastError();
    LOGS_DEFAULT(ERROR) << "GetLogicalProcessorInformationEx failed to obtain processor info. error code: "
                        << last_error
                        << " error msg: " << std::system_category().message(last_error);
    return {};
  }

  WORD group_id = 0;
  static constexpr WORD max_group_id = std::numeric_limits<WORD>::max();
  ThreadAffinities thread_affinities;
  // size of KAFFINITY varies, must not exceed boundary
  int64_t num_bit_in_affinity_ = sizeof(KAFFINITY) << 3;
  // allow only one thread for every two logical processors
  int64_t max_num_thread_per_group = num_bit_in_affinity_ >> 1;

  auto num_processor_info = returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX);
  for (int64_t i = 0; i < static_cast<int64_t>(num_processor_info); ++i) {
    if (processorInfos[i].Relationship != RelationGroup) {
      LOGS_DEFAULT(ERROR) << "Returned processors not belong to same group";
      return {};
    }
    for (int64_t j = 0; j < static_cast<int>(processorInfos[i].Group.ActiveGroupCount); ++j) {
      const auto& groupInfo = processorInfos[i].Group.GroupInfo[j];
      LOGS_DEFAULT(INFO) << "Discovered process group: " << group_id << ", active processor mask: " << groupInfo.ActiveProcessorMask;
      KAFFINITY thread_affinity = 3UL;
      // allow only one thread for every two logical processors
      int64_t num_of_physical_cores = groupInfo.ActiveProcessorCount >> 1;
      int64_t max_num_thread_cur_group = std::min( max_num_thread_per_group, num_of_physical_cores);
      for (int64_t k = 0; k < max_num_thread_cur_group; ++k) {
        thread_affinities.push_back({static_cast<int64_t>(group_id), static_cast<int64_t>(thread_affinity)});
        thread_affinity <<= 2;
      }
      if (group_id == max_group_id) {
        // preventing overflow
        return std::move(thread_affinities);
      }
      group_id++;
    }
  }
  return std::move(thread_affinities);
}
#endif

namespace concurrency {

bool ExtractAffinityFromString(const char* affinity_string, ThreadAffinities& group_affinities) {
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
    ThreadAffinity thread_affinity;
    thread_affinity.first = std::stoull(affinity_strings[0].c_str());
    thread_affinity.second = std::stoull(affinity_strings[1].c_str());
    return std::move(thread_affinity);
  };
  auto ReadGroupAffinities = [&](const std::string& s) {
    auto affinity_strings = Split(s, ';');
    ThreadAffinities thread_affinities;
    for (const auto& iter : affinity_strings) {
      thread_affinities.push_back(ReadGroupAffinity(iter));
    }
    return thread_affinities;
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
    to.thread_affinities = GetDefaultThreadAffinities();
    if (to.thread_affinities.empty()) {
      options.thread_pool_size = std::thread::hardware_concurrency() >> 1;
      LOGS_DEFAULT(WARNING) << "Failed to initialize default thread affinity setting";
    } else {
      options.thread_pool_size = static_cast<int>(to.thread_affinities.size());
      LOGS_DEFAULT(INFO) << "setting default affinity, thread_pool size (including the main thread): " << options.thread_pool_size;
      for (int i = 0; i < options.thread_pool_size - 1; ++i) {
        LOGS_DEFAULT(INFO) << "sub-thread " << i << " affnity set to: group "
                           << to.thread_affinities[i].first << " with processor bitmask "
                           << to.thread_affinities[i].second;
      }
    }
#else
    cpu_list = Env::Default().GetThreadAffinityMasks();
    if (cpu_list.empty() || cpu_list.size() == 1)
      return nullptr;
    options.thread_pool_size = static_cast<int>(cpu_list.size());
    if (options.auto_set_affinity)
      to.affinity = cpu_list;
#endif
  } else if (!options.thread_affinities.empty()) {
#ifdef _WIN32
    ORT_ENFORCE(static_cast<int>(options.thread_affinities.size()) == options.thread_pool_size - 1,
                "Invalid thread options, number of group affinities must equal to options.thread_pool_size - 1");
    to.thread_affinities = options.thread_affinities;
    LOGS_DEFAULT(INFO) << "applying non-default affinity:" << std::endl;
    for (int i = 0; i < options.thread_pool_size - 1; ++i) {
      LOGS_DEFAULT(INFO) << "sub-thread " << i << " affnity set to: group "
                         << to.thread_affinities[i].first << " with processor bitmask "
                         << to.thread_affinities[i].second;
    }
#else
    LOGS_DEFAULT(WARNING) << "Setting thread affinity not implemented for POSIX";
#endif
  }

  to.set_denormal_as_zero = options.set_denormal_as_zero;
  return onnxruntime::make_unique<ThreadPool>(env, to, options.name, options.thread_pool_size,
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
#ifdef _WIN32
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  if (!affinity_string) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null affinity string");
  }
  if (!onnxruntime::concurrency::ExtractAffinityFromString(affinity_string, tp_options->intra_op_thread_pool_params.thread_affinities)) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Affinity string invalid, failed to set affinity to intra thread option");
  }
  return nullptr;
#else
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "SetGlobalIntraOpThreadAffinity not implemented for POSIX");
#endif
}

}  // namespace OrtApis
