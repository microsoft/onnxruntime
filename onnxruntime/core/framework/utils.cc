// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/utils.h"

#include "core/graph/graph_viewer.h"

#include "core/framework/execution_frame.h"
#include "core/framework/execution_providers.h"
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

// TODO should we handle the case of one input name feeding 2 nodes placed on different devices?
common::Status CopyOneInputAcrossDevices(const SessionState& session_state,
                                         const std::string& input_name,
                                         const MLValue& orig_mlvalue,
                                         MLValue& new_mlvalue) {
  //TODO: make it configurable
  const int target_device_id = 0;
  std::vector<SessionState::NodeInfo> node_info_vec;
  ORT_RETURN_IF_ERROR(session_state.GetInputNodeInfo(input_name, node_info_vec));

  auto& exec_providers = session_state.GetExecutionProviders();

  // currently we only support one device per input. see SessionState::AddInputNameToNodeInfoMapping for more
  // info on the logic to create the node_info_vec.
  // for (auto& node_info : node_info_vec) {
  auto& node_info = node_info_vec.front();

  if (node_info.p_node == nullptr) {
    // dummy entry for an input that we didn't find a use of in the graph.
    // use the input as is given we don't believe it's actually needed.
    new_mlvalue = orig_mlvalue;
    return Status::OK();
  }

  if (!orig_mlvalue.IsTensor()) {
    // copying not supported for non-tensor types
    new_mlvalue = orig_mlvalue;
    return Status::OK();
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
     return Status::OK();
  }

  auto input_provider_type = p_input_provider->Type();
  if (input_provider_type == required_provider_type && input_tensor_loc.mem_type == OrtMemTypeDefault) {
    new_mlvalue = orig_mlvalue;
    return Status::OK();
  }

  // If a node requires input on cpu and input tensor is allocated with pinned memory allocator, don't do copy
  if (required_provider_type == onnxruntime::kCpuExecutionProvider &&
      (input_tensor_loc.mem_type == OrtMemTypeCPU ||
       input_tensor_loc.mem_type == OrtMemTypeCPUOutput)) {
    new_mlvalue = orig_mlvalue;
    return Status::OK();
  }

  auto* required_provider = exec_providers.Get(required_provider_type);
  ORT_ENFORCE(required_provider);
  ORT_RETURN_IF_ERROR(utils::AllocateHelper(*required_provider, target_device_id, input_tensor, new_mlvalue));

  auto* new_tensor = new_mlvalue.GetMutable<Tensor>();

  // our CPU exec provider doesn't support copy from GPU->CPU
  if (required_provider_type != onnxruntime::kCpuExecutionProvider) {
    ORT_RETURN_IF_ERROR(required_provider->CopyTensor(input_tensor, *new_tensor));
  } else {
    ORT_RETURN_IF_ERROR(p_input_provider->CopyTensor(input_tensor, *new_tensor));
  }

  // } loop of node_info_vec

  return Status::OK();
}

// copies inputs across devices only if required
common::Status CopyInputsAcrossDevices(const SessionState& session_state,
                                       const NameMLValMap& orig_feeds,
                                       NameMLValMap& new_feeds) {
  for (auto& pair : orig_feeds) {
    MLValue new_mlvalue;
    auto& input_name = pair.first;
    auto& orig_mlvalue = pair.second;
    ORT_RETURN_IF_ERROR(CopyOneInputAcrossDevices(session_state, input_name, orig_mlvalue, new_mlvalue));
    new_feeds[input_name] = new_mlvalue;
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

// ensures pre-allocated outputs match the node providers.
common::Status MatchOutputsWithProviders(const SessionState& session_state,
                                         const std::vector<std::string>& output_names,
                                         std::vector<MLValue>& fetches,
                                         std::vector<MLValue>& new_fetches) {
  const auto& execution_providers = session_state.GetExecutionProviders();

  if (fetches.empty()) {
    fetches.resize(output_names.size());
  }

  new_fetches.resize(output_names.size());

  std::set<std::string> seen_outputs;
  auto p_graph = session_state.GetGraphViewer();
  ORT_ENFORCE(p_graph);

  std::pair<bool, size_t> found;
  for (auto& node : p_graph->Nodes()) {  // TODO optimize this
    if (seen_outputs.size() == fetches.size()) {
      break;
    }

    for (auto* arg : node.OutputDefs()) {
      if (!arg->Exists() ||
          arg->Name().empty() ||
          !(found = Contains(output_names, arg->Name())).first) {
        continue;
      }

      seen_outputs.insert(arg->Name());
      size_t idx = found.second;
      MLValue orig_mlvalue = fetches[idx];

      if (orig_mlvalue.IsAllocated()) {
        if (!orig_mlvalue.IsTensor()) {
          new_fetches[idx] = fetches[idx];
          continue;
        }

        auto& node_provider_type = node.GetExecutionProviderType();
        auto& orig_tensor = orig_mlvalue.Get<Tensor>();
        auto& orig_tensor_loc = orig_tensor.Location();
        auto* tensor_provider = execution_providers.Get(orig_tensor_loc);
        if (!tensor_provider) {
          tensor_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);
        }

        auto tensor_provider_type = tensor_provider->Type();
        if (node_provider_type == tensor_provider_type) {
          new_fetches[idx] = fetches[idx];
          continue;
        }

        // leave the new_fetches[idx] as it is since it'll get allocated on the appropriate
        // provider by the op kernel context when requested.
        continue;

      } else {
        new_fetches[idx] = fetches[idx];
        continue;
      }
    }
  }

  // If we've already seen all the outputs requested just return.
  if (seen_outputs.size() == output_names.size()) {
    return Status::OK();
  }

  // Handle the case when a constant is an output but has been folded into a weight
  // and hence it doesn't show up in any of the OutputDefs before.
  // assume that the weight has already been placed in the appropriate device before
  auto& defs = p_graph->GetOutputs();
  auto& mlvalue_name_idx_map{session_state.GetMLValueNameIdxMap()};
  auto& weights = session_state.GetInitializedTensors();

  for (auto& one_def : defs) {
    if (!one_def->Exists() ||
        one_def->Name().empty() ||
        seen_outputs.count(one_def->Name()) ||
        !(found = Contains(output_names, one_def->Name())).first) {
      continue;
    }

    auto& def_name = one_def->Name();
    size_t idx = found.second;
    int mlvalue_idx;
    ORT_RETURN_IF_ERROR(mlvalue_name_idx_map.GetIdx(def_name, mlvalue_idx));
    if (!weights.count(mlvalue_idx)) {
      LOGS(session_state.Logger(), INFO) << "Output with name " << def_name << " is not a weight.";
      continue;
    }

    seen_outputs.insert(def_name);
    const auto& weight = weights.at(mlvalue_idx);
    new_fetches[idx] = weight;
  }

  if (seen_outputs.size() != output_names.size())  // make sure we've seen all outputs
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "output size mismatch, expected ", output_names.size(),
                           " got ", seen_outputs.size());

  return Status::OK();
}

// copies outputs across devices only if required
common::Status CopyOutputsAcrossDevices(const SessionState& session_state,
                                        std::vector<MLValue>& fetches,
                                        std::vector<MLValue>& user_fetches) {
  auto& execution_providers = session_state.GetExecutionProviders();

  for (size_t idx = 0, end = fetches.size(); idx < end; ++idx) {
    auto& fetched_mlvalue = fetches[idx];
    if (!fetched_mlvalue.IsTensor()) {
      user_fetches[idx] = fetched_mlvalue;
      continue;
    }

    auto& fetched_tensor = fetched_mlvalue.Get<Tensor>();
    auto& fetched_tensor_location = fetched_tensor.Location();
    auto* p_fetched_provider = execution_providers.Get(fetched_tensor_location);
    if (!p_fetched_provider) {
      p_fetched_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);
      ORT_ENFORCE(p_fetched_provider);
    }

    auto fetched_provider_type = p_fetched_provider->Type();
    auto& output_mlvalue = user_fetches[idx];

    if (!output_mlvalue.IsAllocated()) {
      if (fetched_provider_type != onnxruntime::kCpuExecutionProvider) {
        ORT_RETURN_IF_ERROR(utils::AllocateHelper(*execution_providers.Get(onnxruntime::kCpuExecutionProvider),
                                                  0,
                                                  fetched_tensor,
                                                  output_mlvalue));
      } else {
        user_fetches[idx] = fetched_mlvalue;
        continue;
      }
    }

    Tensor* p_output_tensor = output_mlvalue.GetMutable<Tensor>();
    auto& output_tensor_loc = p_output_tensor->Location();
    auto* p_output_provider = execution_providers.Get(output_tensor_loc);
    if (!p_output_provider) {
      p_output_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);
      ORT_ENFORCE(p_output_provider);
    }

    auto output_provider_type = p_output_provider->Type();

    if (output_provider_type == fetched_provider_type || fetched_tensor_location.mem_type == OrtMemTypeCPUOutput) {
      user_fetches[idx] = fetched_mlvalue;
      continue;
    }

    // our CPU exec provider doesn't support copy from GPU->CPU
    if (fetched_provider_type != onnxruntime::kCpuExecutionProvider) {
      ORT_RETURN_IF_ERROR(p_fetched_provider->CopyTensor(fetched_tensor, *p_output_tensor));
    } else {
      ORT_RETURN_IF_ERROR(p_output_provider->CopyTensor(fetched_tensor, *p_output_tensor));
    }
  }

  return Status::OK();
}

common::Status ExecuteGraph(const SessionState& session_state,
                            const NameMLValMap& feeds,
                            const std::vector<std::string>& output_names,
                            std::vector<MLValue>& fetches,
                            bool sequential_execution,
                            const bool& terminate_flag,
                            const logging::Logger& logger) {
  // TODO: Would be better to check upfront whether there was a need to copy inputs/outputs across devices,
  // especially when a subgraph is repeatedly executed in a Scan or Loop node. If we checked once and no copy was
  // needed we can skip everything here apart from the Execute call.

  NameMLValMap device_feeds;
  ORT_RETURN_IF_ERROR(utils::CopyInputsAcrossDevices(session_state, feeds, device_feeds));

  std::vector<MLValue> device_fetches;
  ORT_RETURN_IF_ERROR(utils::MatchOutputsWithProviders(session_state, output_names, fetches, device_fetches));

  std::unique_ptr<IExecutor> p_exec;

  if (sequential_execution) {
    p_exec = std::unique_ptr<IExecutor>(new SequentialExecutor(terminate_flag));
  } else {
    p_exec = std::unique_ptr<IExecutor>(new ParallelExecutor(session_state, terminate_flag));
  }

  ORT_RETURN_IF_ERROR(p_exec->Execute(session_state, device_feeds, output_names, device_fetches, logger));
  ORT_RETURN_IF_ERROR(utils::CopyOutputsAcrossDevices(session_state, device_fetches, fetches));

  return Status::OK();
}

}  // namespace utils
}  // namespace onnxruntime
