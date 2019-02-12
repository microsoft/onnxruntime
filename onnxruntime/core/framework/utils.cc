// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/utils.h"

#include "core/graph/graph_viewer.h"

#include "core/framework/execution_frame.h"
#include "core/framework/execution_providers.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/parallel_executor.h"
#include "core/framework/session_state.h"
#include "core/framework/sequential_executor.h"

namespace onnxruntime {
namespace utils {

const KernelDef* GetKernelDef(const KernelRegistryManager& kernel_registry,
                              const onnxruntime::Node& node) {
  const KernelCreateInfo* kernel_create_info = nullptr;
  const KernelDef* kernel_def = nullptr;

  if (kernel_registry.SearchKernelRegistry(node, &kernel_create_info).IsOK()) {
    kernel_def = kernel_create_info->kernel_def.get();
  }

  return kernel_def;
}

AllocatorPtr GetAllocator(const ExecutionProviders& exec_providers, const OrtAllocatorInfo& allocator_info) {
  auto exec_provider = exec_providers.Get(allocator_info);
  if (exec_provider == nullptr) {
    return nullptr;
  }

  return exec_provider->GetAllocator(allocator_info.id, allocator_info.mem_type);
}

AllocatorPtr GetAllocator(const SessionState& session_state, const OrtAllocatorInfo& allocator_info) {
  return GetAllocator(session_state.GetExecutionProviders(), allocator_info);
}

common::Status AllocateHelper(const IExecutionProvider& execution_provider,
                              int device_id,
                              const Tensor& fetched_tensor,
                              MLValue& output_mlvalue) {
  auto allocator = execution_provider.GetAllocator(device_id, OrtMemTypeDefault);
  if (!allocator) {
    return Status(common::ONNXRUNTIME, common::FAIL, "invalid allocator");
  }

  void* buffer = nullptr;
  if (fetched_tensor.Size() != 0) {
    buffer = allocator->Alloc(fetched_tensor.Size());
    if (!buffer) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate buffer. Execution provider type=",
                             execution_provider.Type());
    }
  }

  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(fetched_tensor.DataType(),
                                                              fetched_tensor.Shape(),
                                                              buffer,
                                                              allocator->Info(),
                                                              allocator);
  output_mlvalue.Init(p_tensor.release(),
                      DataTypeImpl::GetType<Tensor>(),
                      DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  return Status::OK();
}

const std::string& GetNodeInputProviderType(const SessionState::NodeInfo& info) {
  // the input index will be std::numeric_limits<size_t>::max() if it's an implicit input to a control flow node.
  // the input will be processed fully when executing the subgraph that consumes the implicit input.
  bool implicit_input = info.index == std::numeric_limits<size_t>::max();

  // node may declare input_mem_type to be on CPU explicitly
  // skip implicit inputs as they don't have a valid 'index' value
  bool node_input_on_cpu = !implicit_input &&
                           info.kci && MemTypeOnCpuExplicitly(info.kci->kernel_def->InputMemoryType(info.index));

  // need a std::string that doesn't go away for kCpuExecutionProvider so we can return a reference.
  static const std::string cpu_execution_provider{onnxruntime::kCpuExecutionProvider};

  auto& required_provider_type = node_input_on_cpu ? cpu_execution_provider
                                                   : info.p_node->GetExecutionProviderType();

  return required_provider_type;
}

static Status SimpleCopy(const MLValue& orig_value, MLValue& new_value) {
  new_value = orig_value;
  return Status::OK();
}

// TODO should we handle the case of one input name feeding 2 nodes placed on different devices?
common::Status CopyOneInputAcrossDevices(const SessionState& session_state,
                                         const std::string& input_name,
                                         const MLValue& orig_mlvalue,
                                         MLValue& new_mlvalue,
                                         bool& needed_copy,
                                         FeedsFetchesManager::MLValueCopyFunc* copier_out) {
  needed_copy = false;

  //TODO: make it configurable
  // This could be a const, but to make the MacOS compiler and VC++ compiler happy we make it non-const.
  // If it's const AppleClang doesn't want it included in the capture list for copier (it's in reaching scope
  // and should be able to be used implicitly), and VC++ complains it can't be implicitly captured.
  // Seems like the VC++ bug is low priority given it has been around for year:
  // https://developercommunity.visualstudio.com/content/problem/1997/constexpr-not-implicitely-captured-in-lambdas.html
  // https://stackoverflow.com/questions/43119643/lambda-expression-with-empty-capture
  int target_device_id = 0;
  std::vector<SessionState::NodeInfo> node_info_vec;
  ORT_RETURN_IF_ERROR(session_state.GetInputNodeInfo(input_name, node_info_vec));

  auto& exec_providers = session_state.GetExecutionProviders();

  do {
    // currently we only support one device per input. see SessionState::AddInputNameToNodeInfoMapping for more
    // info on the logic to create the node_info_vec.
    // for (auto& node_info : node_info_vec) {
    auto& node_info = node_info_vec.front();

    if (node_info.p_node == nullptr) {
      // dummy entry for an input that we didn't find a use of in the graph.
      // use the input as is given we don't believe it's actually needed.
      new_mlvalue = orig_mlvalue;
      break;
    }

    if (!orig_mlvalue.IsTensor()) {
      // copying not supported for non-tensor types
      new_mlvalue = orig_mlvalue;
      break;
    }

    auto& required_provider_type = GetNodeInputProviderType(node_info);
    auto& input_tensor = orig_mlvalue.Get<Tensor>();
    auto& input_tensor_loc = input_tensor.Location();

    auto* p_input_provider = exec_providers.Get(input_tensor_loc);
    if (!p_input_provider) {
      p_input_provider = exec_providers.Get(onnxruntime::kCpuExecutionProvider);
      ORT_ENFORCE(p_input_provider);
    }

    //no copy for TRT
    if (required_provider_type == onnxruntime::kTRTExecutionProvider) {
      new_mlvalue = orig_mlvalue;
      break;
    }

    auto input_provider_type = p_input_provider->Type();
    if (input_provider_type == required_provider_type && input_tensor_loc.mem_type == OrtMemTypeDefault) {
      new_mlvalue = orig_mlvalue;
      break;
    }

    // If a node requires input on cpu and input tensor is allocated with pinned memory allocator, don't do copy
    if (required_provider_type == onnxruntime::kCpuExecutionProvider &&
        (input_tensor_loc.mem_type == OrtMemTypeCPU ||
         input_tensor_loc.mem_type == OrtMemTypeCPUOutput)) {
      new_mlvalue = orig_mlvalue;
      break;
    }

    auto* required_provider = exec_providers.Get(required_provider_type);
    ORT_ENFORCE(required_provider);

    auto* p_copy_provider = (required_provider_type != onnxruntime::kCpuExecutionProvider)
                                ? required_provider
                                : p_input_provider;

    auto copier = [required_provider,
                   p_copy_provider,
                   target_device_id](const MLValue& feed_value, MLValue& new_value) {
      const auto& feed_tensor = feed_value.Get<Tensor>();
      ORT_RETURN_IF_ERROR(utils::AllocateHelper(*required_provider, target_device_id, feed_tensor, new_value));
      auto* new_tensor = new_value.GetMutable<Tensor>();

      ORT_RETURN_IF_ERROR(p_copy_provider->CopyTensor(feed_tensor, *new_tensor));

      return Status::OK();
    };

    ORT_RETURN_IF_ERROR(copier(orig_mlvalue, new_mlvalue));

    if (copier_out) {
      *copier_out = std::move(copier);
    }

    needed_copy = true;

    // } loop of node_info_vec
  } while (false);

  if (!needed_copy && copier_out)
    *copier_out = SimpleCopy;

  return Status::OK();
}

common::Status CopyOneInputAcrossDevices(const SessionState& session_state,
                                         const std::string& input_name,
                                         const MLValue& orig_mlvalue,
                                         MLValue& new_mlvalue) {
  bool needed_copy;
  return CopyOneInputAcrossDevices(session_state, input_name, orig_mlvalue, new_mlvalue, needed_copy, nullptr);
}

// copies inputs across devices only if required
static common::Status CopyInputsAcrossDevices(const SessionState& session_state,
                                              const std::vector<std::string>& feed_names,
                                              const std::vector<MLValue>& orig_feeds,
                                              std::vector<MLValue>& new_feeds,
                                              bool& needed_copy,
                                              std::vector<FeedsFetchesManager::MLValueCopyFunc>* copiers) {
  bool copied = false;
  size_t num_feeds = orig_feeds.size();
  ORT_ENFORCE(feed_names.size() == num_feeds);

  new_feeds.resize(num_feeds);

  // use cached copy logic if available
  if (copiers && !copiers->empty()) {
    ORT_ENFORCE(num_feeds == copiers->size());
    needed_copy = true;

    for (size_t idx = 0; idx < num_feeds; ++idx) {
      ORT_RETURN_IF_ERROR((*copiers)[idx](orig_feeds[idx], new_feeds[idx]));
    }

  } else {
    if (copiers) {
      copiers->reserve(num_feeds);
    }

    for (size_t idx = 0; idx < num_feeds; ++idx) {
      bool copied_this_input = false;
      std::function<Status(const MLValue&, MLValue&)> copier;
      ORT_RETURN_IF_ERROR(CopyOneInputAcrossDevices(session_state, feed_names[idx], orig_feeds[idx], new_feeds[idx],
                                                    copied_this_input, &copier));
      copied = copied || copied_this_input;

      if (copiers) {
        copiers->push_back(std::move(copier));
      }
    }

    needed_copy = copied;
  }

  return Status::OK();
}

static std::pair<bool, size_t> Contains(const std::vector<std::string>& output_names,
                                        const std::string& name) {
  auto it = std::find(std::begin(output_names), std::end(output_names), name);
  if (it == output_names.end()) {
    return {false, 0};
  }
  return {true, it - output_names.begin()};
}

// Setup fetches for execution. Use any provided fetches directly if the provider matches.
// If the provider doesn't match, we don't know what device the execution output may be on, so can't assume the output
// can be returned to the user directly.
// TODO: We should be able to use the allocation plan to know which device an output will be on.
static common::Status SetupFetchesForExecute(const SessionState& session_state,
                                             const std::vector<std::string>& output_names,
                                             std::vector<MLValue>& fetches,
                                             std::vector<MLValue>& new_fetches,
                                             std::vector<bool>* copy_to_new_fetches_cached_values) {
  ORT_ENFORCE(new_fetches.empty());

  const auto& execution_providers = session_state.GetExecutionProviders();
  auto num_outputs = output_names.size();

  new_fetches.resize(num_outputs);

  if (copy_to_new_fetches_cached_values && !copy_to_new_fetches_cached_values->empty()) {
    // use the cached values
    ORT_ENFORCE(copy_to_new_fetches_cached_values->size() == num_outputs);

    auto& copy = *copy_to_new_fetches_cached_values;
    for (size_t i = 0; i < num_outputs; ++i) {
      if (copy[i]) {
        new_fetches[i] = fetches[i];
      }
    }

    return Status::OK();
  }

  // track which fetches can be copied to new_fetches and used directly in the execution.
  std::vector<bool> local_can_copy_flags(num_outputs, false);

  std::set<std::string> seen_outputs;
  auto p_graph = session_state.GetGraphViewer();
  ORT_ENFORCE(p_graph);

  std::pair<bool, size_t> found;
  for (auto& node : p_graph->Nodes()) {
    if (seen_outputs.size() == num_outputs) {
      break;
    }

    for (auto* arg : node.OutputDefs()) {
      if (!arg->Exists() ||
          !(found = Contains(output_names, arg->Name())).first) {
        continue;
      }

      seen_outputs.insert(arg->Name());
      size_t idx = found.second;
      const MLValue& provided_mlvalue = fetches[idx];

      if (provided_mlvalue.IsAllocated()) {
        if (!provided_mlvalue.IsTensor()) {
          new_fetches[idx] = fetches[idx];
          local_can_copy_flags[idx] = true;
          continue;
        }

        const auto& node_provider_type = node.GetExecutionProviderType();
        const auto& provided_tensor = provided_mlvalue.Get<Tensor>();
        const auto& provided_tensor_loc = provided_tensor.Location();
        const auto* tensor_provider = execution_providers.Get(provided_tensor_loc);
        if (!tensor_provider) {
          tensor_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);
        }

        auto tensor_provider_type = tensor_provider->Type();
        if (node_provider_type == tensor_provider_type) {
          new_fetches[idx] = fetches[idx];
          local_can_copy_flags[idx] = true;
          continue;
        }

        continue;
      }
    }
  }

  if (copy_to_new_fetches_cached_values) {
    *copy_to_new_fetches_cached_values = local_can_copy_flags;
  }

  return Status::OK();
}

// copies outputs across devices only if required
static common::Status CopyOutputsAcrossDevices(const SessionState& session_state,
                                               const std::vector<MLValue>& fetches,
                                               std::vector<MLValue>& user_fetches,
                                               bool& needed_copy,
                                               std::vector<FeedsFetchesManager::MLValueCopyFunc>* copiers) {
  needed_copy = false;
  auto num_outputs = fetches.size();

  // used the cached copy logic if available
  if (copiers && !copiers->empty()) {
    for (size_t idx = 0, end = num_outputs; idx < end; ++idx) {
      ORT_RETURN_IF_ERROR((*copiers)[idx](fetches[idx], user_fetches[idx]));
    }

    return Status::OK();
  }

  if (copiers) {
    copiers->reserve(num_outputs);
  }

  auto& execution_providers = session_state.GetExecutionProviders();

  // CPU execution provider is always registered so this is not null
  const auto* cpu_execution_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);

  // lambda to capture a straight copy from fetches to user_fetches
  auto do_simple_copy = [&user_fetches, &copiers](const MLValue& fetch, size_t idx) {
    user_fetches[idx] = fetch;
    if (copiers) {
      copiers->push_back(SimpleCopy);
    }
  };

  for (size_t idx = 0; idx < num_outputs; ++idx) {
    auto& fetched_mlvalue = fetches[idx];
    if (!fetched_mlvalue.IsTensor()) {
      do_simple_copy(fetched_mlvalue, idx);
      continue;
    }

    auto& fetched_tensor = fetched_mlvalue.Get<Tensor>();
    auto& fetched_tensor_location = fetched_tensor.Location();
    auto* p_fetched_provider = execution_providers.Get(fetched_tensor_location);
    if (!p_fetched_provider) {
      p_fetched_provider = cpu_execution_provider;
    }

    auto fetched_provider_type = p_fetched_provider->Type();
    auto& output_mlvalue = user_fetches[idx];

    const IExecutionProvider* p_output_provider = nullptr;

    if (output_mlvalue.IsAllocated()) {
      Tensor* p_output_tensor = output_mlvalue.GetMutable<Tensor>();
      p_output_provider = execution_providers.Get(p_output_tensor->Location());
    }

    if (!p_output_provider) {
      p_output_provider = cpu_execution_provider;
    }

    auto output_provider_type = p_output_provider->Type();

    if (fetched_provider_type == output_provider_type ||
        (p_output_provider == cpu_execution_provider && fetched_tensor_location.mem_type == OrtMemTypeCPUOutput)) {
      do_simple_copy(fetched_mlvalue, idx);
      continue;
    }

    needed_copy = true;

    auto* p_copy_provider = (fetched_provider_type != onnxruntime::kCpuExecutionProvider)
                                ? p_fetched_provider
                                : p_output_provider;

    auto copy_between_providers = [p_copy_provider,
                                   p_output_provider](const MLValue& fetched_mlvalue, MLValue& output_mlvalue) {
      auto& fetched_tensor = fetched_mlvalue.Get<Tensor>();

      if (!output_mlvalue.IsAllocated()) {
        ORT_RETURN_IF_ERROR(utils::AllocateHelper(*p_output_provider, 0, fetched_tensor, output_mlvalue));
      }

      Tensor* p_output_tensor = output_mlvalue.GetMutable<Tensor>();

      ORT_RETURN_IF_ERROR(p_copy_provider->CopyTensor(fetched_tensor, *p_output_tensor));

      return Status::OK();
    };

    ORT_RETURN_IF_ERROR(copy_between_providers(fetched_mlvalue, output_mlvalue));

    if (copiers) {
      copiers->push_back(std::move(copy_between_providers));
    }
  }

  return Status::OK();
}  // namespace utils

common::Status ExecuteGraph(const SessionState& session_state,
                            FeedsFetchesManager& feeds_fetches_manager,
                            const std::vector<MLValue>& feeds,
                            std::vector<MLValue>& fetches,
                            const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                            bool sequential_execution,
                            const bool& terminate_flag,
                            const logging::Logger& logger,
                            bool cache_copy_info) {
  const auto& feeds_fetches_info = feeds_fetches_manager.GetFeedsFetchesInfo();
  auto device_copy_checks = feeds_fetches_manager.GetDeviceCopyChecks();

  std::unique_ptr<IExecutor> p_exec;

  if (sequential_execution) {
    p_exec = std::unique_ptr<IExecutor>(new SequentialExecutor(terminate_flag));
  } else {
    p_exec = std::unique_ptr<IExecutor>(new ParallelExecutor(session_state, terminate_flag));
  }

  if (device_copy_checks.status == DeviceCopyCheck::Unknown) {
    // see if we can skip copies due to the types of execution providers available
    device_copy_checks.status = feeds_fetches_manager.CheckExecutionProviders(session_state.GetExecutionProviders());
  }

  if (device_copy_checks.status == DeviceCopyCheck::NoCopy) {
    // no device copies are needed so simple execute
    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, fetches, fetch_allocators, logger));
  } else {
    // first execution we check and update. after that we use cached values.
    bool check_all = device_copy_checks.status == DeviceCopyCheck::Unknown;
    bool copy_needed = false;

    const std::vector<MLValue>* p_feeds = &feeds;
    std::vector<MLValue>* p_fetches = &fetches;

    std::vector<MLValue> device_feeds;
    std::vector<MLValue> device_fetches;

    // Copy inputs
    if (check_all || device_copy_checks.input_copy_needed == DeviceCopyCheck::Copy) {
      auto* copiers = cache_copy_info ? &feeds_fetches_manager.GetFeedsDeviceCopiers() : nullptr;
      ORT_RETURN_IF_ERROR(CopyInputsAcrossDevices(session_state,
                                                  feeds_fetches_info.feed_names, feeds, device_feeds,
                                                  copy_needed, copiers));

      if (copy_needed) {
        p_feeds = &device_feeds;
      }

      if (check_all) {
        device_copy_checks.input_copy_needed = copy_needed ? DeviceCopyCheck::Copy
                                                           : DeviceCopyCheck::NoCopy;
      }
    }

    // setup fetches.
    if (fetches.empty()) {
      fetches.resize(feeds_fetches_info.output_names.size());
    }

    // if no output copy is needed, we can just use the fetches directly. otherwise we need to use a temporary set
    // and run CopyOutputsAcrossDevices.
    if (check_all || device_copy_checks.output_copy_needed != DeviceCopyCheck::NoCopy) {
      auto* use_provided_fetch_flags = cache_copy_info ? &feeds_fetches_manager.GetCanUseFetchDuringExecutionFlags()
                                                       : nullptr;

      ORT_RETURN_IF_ERROR(SetupFetchesForExecute(session_state, feeds_fetches_info.output_names,
                                                 fetches, device_fetches,
                                                 use_provided_fetch_flags));
      p_fetches = &device_fetches;
    }

    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, *p_feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, *p_fetches, fetch_allocators,
                                        logger));

    if (check_all || device_copy_checks.output_copy_needed != DeviceCopyCheck::NoCopy) {
      auto* copiers = cache_copy_info ? &feeds_fetches_manager.GetFetchesDeviceCopiers() : nullptr;

      ORT_RETURN_IF_ERROR(CopyOutputsAcrossDevices(session_state, *p_fetches, fetches, copy_needed, copiers));
      if (check_all) {
        device_copy_checks.output_copy_needed = copy_needed ? DeviceCopyCheck::Copy : DeviceCopyCheck::NoCopy;
      }
    }

    // save the result of all the checks and use cached info next time
    if (check_all && cache_copy_info) {
      feeds_fetches_manager.SetDeviceCopyChecks(device_copy_checks);
    }
  }

  return Status::OK();
}

}  // namespace utils
}  // namespace onnxruntime
