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
#include "core/common/logging/logging.h"

std::ostream& operator<<(std::ostream& os, const OrtThreadPoolParams& params) {
    os << "OrtThreadPoolParams {";
    os << " thread_pool_size: " << params.thread_pool_size;
    os << " auto_set_affinity: " << params.auto_set_affinity;
    os << " allow_spinning: " << params.allow_spinning;
    os << " dynamic_block_base_: " << params.dynamic_block_base_;
    os << " stack_size: " << params.stack_size;
    os << " affinity_str: " << params.affinity_str;
    // os << " name: " << (params.name ? params.name : L"nullptr");
    os << " set_denormal_as_zero: " << params.set_denormal_as_zero;
    //os << " custom_create_thread_fn: " << (params.custom_create_thread_fn ? "set" : "nullptr");
    //os << " custom_thread_creation_options: " << (params.custom_thread_creation_options ? "set" : "nullptr");
    //os << " custom_join_thread_fn: " << (params.custom_join_thread_fn ? "set" : "nullptr");
    os << " }";
  return os;
}

namespace onnxruntime {
namespace concurrency {

#if !defined(ORT_MINIMAL_BUILD) && !defined(ORT_EXTENDED_MINIMAL_BUILD)
// Extract affinity from affinity string.
// Processor id from affinity string starts from 1,
// but internally, processor id starts from 0, so here we minus the id by 1
static std::vector<LogicalProcessors> ReadThreadAffinityConfig(const std::string& affinity_str) {
  ORT_TRY {
    std::vector<LogicalProcessors> logical_processors_vector;
    auto affinities = utils::SplitString(affinity_str, ";");

    for (const auto& affinity : affinities) {
      LogicalProcessors logical_processors;
      auto processor_interval = utils::SplitString(affinity, "-");

      if (processor_interval.size() == 2) {
        ORT_ENFORCE(std::all_of(processor_interval[0].begin(), processor_interval[0].end(), ::isdigit) &&
                        std::all_of(processor_interval[1].begin(), processor_interval[1].end(), ::isdigit),
                    std::string{"Processor id must consist of only digits: "} + std::string{affinity});

        auto processor_from = std::stoi(std::string{processor_interval[0]});
        auto processor_to = std::stoi(std::string{processor_interval[1]});

        ORT_ENFORCE(processor_from > 0 && processor_to > 0,
                    std::string{"Processor id must start from 1: "} + std::string{affinity});
        ORT_ENFORCE(processor_from <= processor_to,
                    std::string{"Invalid processor interval: "} + std::string{affinity});

        logical_processors.resize(static_cast<size_t>(1ULL + processor_to - processor_from));
        std::iota(logical_processors.begin(), logical_processors.end(), processor_from - 1);

      } else {
        for (const auto& processor_str : utils::SplitString(affinity, ",")) {
          ORT_ENFORCE(std::all_of(processor_str.begin(), processor_str.end(), ::isdigit),
                      std::string{"Processor id must consist of only digits: "} + std::string{processor_str});

          auto processor_id = std::stoi(std::string{processor_str});
          ORT_ENFORCE(processor_id > 0, std::string{"Processor id must start from 1: "} + std::string{processor_str});
          logical_processors.push_back(processor_id - 1);
        }
      }
      logical_processors_vector.push_back(std::move(logical_processors));
    }
    return logical_processors_vector;
  }
  ORT_CATCH(const std::invalid_argument&) {
    LOGS_DEFAULT(ERROR) << "Found invalid processor id in affinity string: "
                        << affinity_str << ", skip affinity setting";
  }
  ORT_CATCH(const std::out_of_range&) {
    LOGS_DEFAULT(ERROR) << "Found out-of-range processor id in affinity string: "
                        << affinity_str << ", skip affinity setting";
  }
  ORT_THROW("Failed to read affinities from affinity string");
}
#endif

static std::unique_ptr<ThreadPool>
CreateThreadPoolHelper(Env* env, OrtThreadPoolParams options) {
  ThreadOptions to;
  if (options.thread_pool_size <= 0) {  // default
    auto default_affinities = Env::Default().GetDefaultThreadAffinities();
    if (default_affinities.size() <= 1) {
      return nullptr;
    }
    options.thread_pool_size = static_cast<int>(default_affinities.size());
    if (options.auto_set_affinity) {
      to.affinities = std::move(default_affinities);
    }
  }
  if (options.thread_pool_size <= 1) {
    return nullptr;
  }
  // override affinity setting if specified from customer
  if (!options.affinity_str.empty()) {
#if defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
    ORT_THROW("Setting thread affinity is not implemented in this build.");
#else
    to.affinities = ReadThreadAffinityConfig(options.affinity_str);
    // Limiting the number of affinities to be of thread_pool_size - 1,
    // for the fact that the main thread is a special "member" of the threadpool,
    // which onnxruntime has no control.
    auto actual_num_affinities = to.affinities.size();
    ORT_ENFORCE(actual_num_affinities == static_cast<size_t>(options.thread_pool_size) - 1,
                (std::string{"Number of affinities does not equal to thread_pool_size minus one, affinities: "} +
                 std::to_string(actual_num_affinities) +
                 std::string{", thread_pool_size: "} +
                 std::to_string(options.thread_pool_size))
                    .c_str());
    // prepend with an empty affinity as placeholder for the main thread,
    // it will be dropped later during threadpool creation.
    to.affinities.insert(to.affinities.begin(), LogicalProcessors{});
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

ORT_API_STATUS_IMPL(SetGlobalIntraOpThreadAffinity, _Inout_ OrtThreadingOptions* tp_options, const char* affinity_string) {
#if defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  ORT_UNUSED_PARAMETER(tp_options);
  ORT_UNUSED_PARAMETER(affinity_string);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED,
                               "Setting thread affinity is not implemented in this build.");
#else
  if (!tp_options) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null OrtThreadingOptions");
  }
  if (!affinity_string) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Received null affinity_string");
  }
  auto len = strnlen(affinity_string, onnxruntime::kMaxStrLen + 1);
  if (0 == len || len > onnxruntime::kMaxStrLen) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 (std::string{"Size of affinity string must be between 1 and "} +
                                  std::to_string(onnxruntime::kMaxStrLen))
                                     .c_str());
  }
  tp_options->intra_op_thread_pool_params.affinity_str = affinity_string;
  return nullptr;
#endif
}

}  // namespace OrtApis
