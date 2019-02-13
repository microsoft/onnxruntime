// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_state_initializer.h"

#include <functional>
#include <limits>

#include "core/common/common.h"
#include "core/common/logging/logging.h"

#include "core/graph/graph_viewer.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/ml_value.h"
#include "core/framework/ml_value_patterns_planner.h"
#include "core/framework/mlvalue_name_idx_map.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorutils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"

namespace onnxruntime {

static common::Status SaveMLValueNameIndexMapping(const onnxruntime::Graph& graph,
                                                  MLValueNameIdxMap& mlvalue_name_idx_map,
                                                  const logging::Logger& logger);

using SaveTensorFunc = std::function<void(int idx, const onnxruntime::MLValue&)>;

static common::Status SaveInitializedTensors(const onnxruntime::Graph& graph,
                                             bool enable_memory_pattern,
                                             const SequentialExecutionPlan& execution_plan,
                                             const ExecutionProviders& exec_providers,
                                             const MLValueNameIdxMap& mlvalue_name_idx_map,
                                             std::map<OrtAllocatorInfo, BufferUniquePtr>& weights_buffers,
                                             const SaveTensorFunc& save_tensor_func,
                                             const logging::Logger& logger);

static common::Status SaveKernels(const ExecutionProviders& execution_providers,
                                  SessionState& session_state,
                                  const KernelRegistryManager& custom_registry_manager,
                                  const logging::Logger& logger);

static common::Status SaveInputOutputNamesToNodeMapping(const onnxruntime::Graph& graph,
                                                        const KernelRegistryManager& custom_registry_manager,
                                                        SessionState& session_state,
                                                        const std::vector<NodeArg*>* implicit_inputs);

SessionStateInitializer::SessionStateInitializer(onnxruntime::Graph& graph,
                                                 SessionState& session_state,
                                                 const ExecutionProviders& providers,
                                                 KernelRegistryManager& kernel_registry_manager)
    : graph_{graph},
      session_state_{session_state},
      execution_providers_{providers},
      kernel_registry_manager_{kernel_registry_manager},
      logger_{session_state.Logger()} {
}

common::Status SessionStateInitializer::CreatePlan(const std::vector<NodeArg*>& outer_scope_node_args,
                                                   bool enable_sequential_execution) {
  session_state_.SetGraphViewer(std::make_unique<onnxruntime::GraphViewer>(graph_));

  auto& mlvalue_name_idx_map = session_state_.GetMLValueNameIdxMap();
  // populate the SessionState MLValueNameIdxMap
  ORT_RETURN_IF_ERROR(SaveMLValueNameIndexMapping(graph_, mlvalue_name_idx_map, logger_));

  // remove any outer scope args we don't know about. this can happen if a node contains multiple subgraphs.
  std::vector<const NodeArg*> valid_outer_scope_node_args;
  std::for_each(outer_scope_node_args.cbegin(), outer_scope_node_args.cend(),
                [&mlvalue_name_idx_map, &valid_outer_scope_node_args](const NodeArg* node_arg) {
                  int idx;
                  if (mlvalue_name_idx_map.GetIdx(node_arg->Name(), idx).IsOK()) {
                    valid_outer_scope_node_args.push_back(node_arg);
                  };
                });

  std::unique_ptr<SequentialExecutionPlan> exec_plan;
  GraphViewer viewer(graph_);
  if (enable_sequential_execution) {
    // CreatePlan will create a new SequentialExecutionPlan instance that we will
    // save into the session state.
    ORT_RETURN_IF_ERROR(
        SequentialPlanner::CreatePlan(viewer, valid_outer_scope_node_args, execution_providers_,
                                      kernel_registry_manager_, mlvalue_name_idx_map, exec_plan));

    session_state_.SetExecutionPlan(std::move(exec_plan));
  } else {
    // Parallel execution still uses same allocation plan, but has limitation of memory buffer reuse.
    SequentialPlannerContext context(true /* enable parallel execution */);
    ORT_RETURN_IF_ERROR(
        SequentialPlanner::CreatePlan(viewer, valid_outer_scope_node_args, execution_providers_,
                                      kernel_registry_manager_, mlvalue_name_idx_map, context, exec_plan));

    session_state_.SetExecutionPlan(std::move(exec_plan));
  }

  return Status::OK();
}

common::Status SessionStateInitializer::InitializeAndSave(bool enable_memory_pattern,
                                                          const std::vector<NodeArg*>* implicit_inputs) {
  const auto* exec_plan_ptr = session_state_.GetExecutionPlan();
  ORT_ENFORCE(exec_plan_ptr, "Execution plan was not found in SessionState. CreatePlan must be called first.");

  const auto& exec_plan{*exec_plan_ptr};
  const auto& mlvalue_name_idx_map{session_state_.GetMLValueNameIdxMap()};

  // lambda to save initialized tensors into SessionState directly
  auto add_initialized_tensor = [this](int idx, const onnxruntime::MLValue& value) {
    session_state_.AddInitializedTensor(idx, value);
  };

  ORT_RETURN_IF_ERROR(SaveInitializedTensors(graph_, enable_memory_pattern, exec_plan, execution_providers_,
                                             mlvalue_name_idx_map, session_state_.GetMutableWeightsBuffers(),
                                             add_initialized_tensor, logger_));

  graph_.CleanAllInitializedTensors();  // remove weights from the graph now to save memory

  ORT_RETURN_IF_ERROR(SaveKernels(execution_providers_, session_state_, kernel_registry_manager_, logger_));
  ORT_RETURN_IF_ERROR(SaveInputOutputNamesToNodeMapping(graph_, kernel_registry_manager_, session_state_,
                                                        implicit_inputs));

  return Status::OK();
}

// Build the MLValue name->idx mapping
common::Status SaveMLValueNameIndexMapping(const onnxruntime::Graph& graph,
                                           MLValueNameIdxMap& mlvalue_name_idx_map,
                                           const logging::Logger& logger) {
  LOGS(logger, INFO) << "SaveMLValueNameIndexMapping";
  int idx = 0;

  // we keep all graph inputs (including initializers), even if they are unused, so make sure they all have an entry
  for (const auto* input_def : graph.GetInputsIncludingInitializers()) {
    idx = mlvalue_name_idx_map.Add(input_def->Name());
    VLOGS(logger, 1)
        << "Added graph input with name: " << input_def->Name() << " to MLValueIndex with index: " << idx;
  }

  for (auto& node : graph.Nodes()) {
    // build the MLValue->index map
    for (const auto* input_def : node.InputDefs()) {
      if (input_def->Exists()) {
        idx = mlvalue_name_idx_map.Add(input_def->Name());
        VLOGS(logger, 1)
            << "Added input argument with name: " << input_def->Name() << " to MLValueIndex with index: " << idx;
      }
    }

    for (const auto* input_def : node.ImplicitInputDefs()) {
      if (input_def->Exists()) {
        idx = mlvalue_name_idx_map.Add(input_def->Name());
        VLOGS(logger, 1)
            << "Added implicit input argument with name: " << input_def->Name() << " to MLValueIndex with index: " << idx;
      }
    }

    for (const auto* output_def : node.OutputDefs()) {
      if (output_def->Exists()) {
        mlvalue_name_idx_map.Add(output_def->Name());
        VLOGS(logger, 1)
            << "Added output argument with name: " << output_def->Name() << " to MLValueIndex with index: " << idx;
      }
    }
  }

  // allocate MLValue for graph outputs when coming from initializers
  for (const auto& output : graph.GetOutputs()) {
    if (output->Exists()) {
      idx = mlvalue_name_idx_map.Add(output->Name());
      VLOGS(logger, 1)
          << "Added graph output with name: " << output->Name() << " to MLValueIndex with index: " << idx;
    }
  }

  LOGS(logger, INFO) << "Done saving MLValue mappings.";
  return Status::OK();
}

common::Status DeserializeTensorProto(const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                      const OrtAllocatorInfo& alloc_info,
                                      const ExecutionProviders& exec_providers,
                                      MLValue& mlvalue, void* preallocated, size_t preallocated_size) {
  auto alloc_ptr = utils::GetAllocator(exec_providers, alloc_info);
  if (!alloc_ptr) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get allocator for alloc_info: " + alloc_info.ToString());
  }

  if (strcmp(alloc_info.name, CPU) == 0 || alloc_info.mem_type == OrtMemTypeCPUOutput) {
    // deserialize directly to CPU tensor
    return utils::TensorProtoToMLValue(tensor_proto, alloc_ptr, preallocated, preallocated_size, mlvalue);
  }

  std::unique_ptr<Tensor> p_tensor;
  // deserialize to CPU first for non-CPU allocator, then alloc and copy
  AllocatorPtr deserialize_alloc_ptr;
  std::unique_ptr<Tensor> p_deserialize_tensor;
  deserialize_alloc_ptr = exec_providers.Get(kCpuExecutionProvider)->GetAllocator(0, OrtMemTypeDefault);
  ORT_RETURN_IF_ERROR(utils::GetTensorFromTensorProto(tensor_proto, &p_deserialize_tensor,
                                                      deserialize_alloc_ptr));
  const IExecutionProvider* provider = exec_providers.Get(alloc_info);
  ORT_ENFORCE(provider != nullptr);
  p_tensor = std::make_unique<Tensor>(
      p_deserialize_tensor->DataType(),
      p_deserialize_tensor->Shape(),
      preallocated ? preallocated : static_cast<void*>(alloc_ptr->Alloc(p_deserialize_tensor->Size())),
      alloc_info,
      preallocated ? nullptr : alloc_ptr);  // no deleter for preallocated

  Status copy_status = provider->CopyTensor(*p_deserialize_tensor, *p_tensor);
  if (!copy_status.IsOK()) {
    if (copy_status.ErrorMessage().empty()) {
      // The windows execution provider does not return any error message today for CopyTensor since it is
      // not implemented yet. That's the reason we're adding our own error message so that we can debug better.
      return Status(copy_status.Category(),
                    copy_status.Code(),
                    "Failed to copy tensor to execution provider: " + provider->Type());
    }
    return copy_status;
  }
  mlvalue.Init(p_tensor.release(),
               DataTypeImpl::GetType<Tensor>(),
               DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  return common::Status::OK();
}

static common::Status PlanTensor(MLValuePatternPlanner& planner, const MLValueNameIdxMap& mlvalue_name_idx_map, const std::string& name, const ONNX_NAMESPACE::TensorProto& tensor_proto) {
  int mlvalue_index;
  ORT_RETURN_IF_ERROR(mlvalue_name_idx_map.GetIdx(name, mlvalue_index));
  size_t len;
  Status st = utils::GetSizeInBytesFromTensorProto<256>(tensor_proto, &len);
  if (st.Code() == common::NOT_IMPLEMENTED) return Status::OK();
  if (!st.IsOK()) return st;
  return planner.TraceAllocation(mlvalue_index, len);
}

common::Status SaveInitializedTensorsWithMemPattern(const Graph& graph,
                                                    const SequentialExecutionPlan& execution_plan,
                                                    const ExecutionProviders& exec_providers,
                                                    const MLValueNameIdxMap& mlvalue_name_idx_map,
                                                    std::map<OrtAllocatorInfo, BufferUniquePtr>& weights_buffers,
                                                    const SaveTensorFunc& save_tensor_func,
                                                    const logging::Logger& logger) {
  LOGS(logger, INFO) << "Saving initialized tensors.";

  ORT_ENFORCE(mlvalue_name_idx_map.MaxIdx() > 0, "MLValue indexes should have been populated.");

  MLValuePatternPlanner planner(execution_plan);

  //1. first plan the memory
  const onnxruntime::InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
  for (const auto& entry : initialized_tensor_set) {
    //string/complex64/complex128 tensors will be skipped
    ORT_RETURN_IF_ERROR(PlanTensor(planner, mlvalue_name_idx_map, entry.first, *entry.second));
  }

  //2. allocate weight buffer on different locations
  MemoryPatternGroup mem_patterns;
  ORT_RETURN_IF_ERROR(planner.GeneratePatterns(&mem_patterns));
  for (size_t i = 0; i < mem_patterns.locations.size(); i++) {
    auto& location = mem_patterns.locations[i];
    ORT_ENFORCE(weights_buffers.find(location) == weights_buffers.end(),
                "Existing entry in weights buffer for ", location.name);

    auto alloc = utils::GetAllocator(exec_providers, location);
    if (!alloc)
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get allocator for location: " + location.ToString());

    void* buffer = mem_patterns.patterns[i].PeakSize() > 0 ? alloc->Alloc(mem_patterns.patterns[i].PeakSize())
                                                           : nullptr;
    weights_buffers[location] = BufferUniquePtr(buffer, alloc);
  }

  //3. create weight tensors based on weights buffer
  for (const auto& entry : initialized_tensor_set) {
    const std::string& name = entry.first;
    int mlvalue_index;
    ORT_RETURN_IF_ERROR(mlvalue_name_idx_map.GetIdx(name, mlvalue_index));
    const ONNX_NAMESPACE::TensorProto& tensor_proto = *(entry.second);

    auto& location = execution_plan.allocation_plan[mlvalue_index].location;
    auto it = weights_buffers.find(location);
    if (it == weights_buffers.end())
      return Status(common::ONNXRUNTIME, common::FAIL, "Weight buffer not found");

    auto pattern = mem_patterns.GetPatterns(location);
    if (pattern == nullptr)
      return Status(common::ONNXRUNTIME, common::FAIL, "mem pattern not found");
    auto block = pattern->GetBlock(mlvalue_index);
    MLValue mlvalue;
    // if block is not found, means this mlvalue is not traced
    // fall back to allocate separate buffer.

    // if it->second.get() is null, then fall back to the block not found case
    if (it->second == nullptr) {
      block = nullptr;
    }
    Status st;
    if (!block) {
      st = DeserializeTensorProto(tensor_proto, location, exec_providers, mlvalue, nullptr, 0);
    } else {
      st = DeserializeTensorProto(tensor_proto, location, exec_providers, mlvalue,
                                  (uint8_t*)it->second.get() + block->offset_, block->size_);
    }
    if (!st.IsOK()) {
      std::ostringstream oss;
      oss << "Deserialize tensor " << name << " failed." << st.ErrorMessage();
      return Status(st.Category(), st.Code(), oss.str());
    }

    save_tensor_func(mlvalue_index, mlvalue);

    VLOGS(logger, 1) << "Added weight with name : " << name << " with index: " << mlvalue_index;
  }

  LOGS(logger, INFO) << "Done saving initialized tensors";
  return common::Status::OK();
}

common::Status SaveInitializedTensorsWithSeperateBuffer(const onnxruntime::Graph& graph,
                                                        const SequentialExecutionPlan& execution_plan,
                                                        const ExecutionProviders& exec_providers,
                                                        const MLValueNameIdxMap& mlvalue_name_idx_map,
                                                        const SaveTensorFunc& save_tensor_func,
                                                        const logging::Logger& logger) {
  LOGS(logger, INFO) << "Saving initialized tensors.";

  ORT_ENFORCE(mlvalue_name_idx_map.MaxIdx() > 0, "MLValue indexes should have been populated.");

  const onnxruntime::InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
  for (const auto& entry : initialized_tensor_set) {
    const std::string& name = entry.first;
    int mlvalue_index;
    ORT_RETURN_IF_ERROR(mlvalue_name_idx_map.GetIdx(name, mlvalue_index));
    VLOGS(logger, 1) << "About to add weight with name: " << name << " and index: " << mlvalue_index;
    auto& location = execution_plan.allocation_plan[mlvalue_index].location;
    MLValue mlvalue;
    ORT_RETURN_IF_ERROR(DeserializeTensorProto(*(entry.second), location, exec_providers, mlvalue, nullptr, 0));
    save_tensor_func(mlvalue_index, mlvalue);
    VLOGS(logger, 1) << "Added weight with name : " << name << " with index: " << mlvalue_index;
  }

  LOGS(logger, INFO) << "Done saving initialized tensors";
  return common::Status::OK();
}

common::Status SaveInitializedTensors(const onnxruntime::Graph& graph,
                                      bool enable_memory_pattern,
                                      const SequentialExecutionPlan& execution_plan,
                                      const ExecutionProviders& exec_providers,
                                      const MLValueNameIdxMap& mlvalue_name_idx_map,
                                      std::map<OrtAllocatorInfo, BufferUniquePtr>& weights_buffers,
                                      const SaveTensorFunc& save_tensor_func,
                                      const logging::Logger& logger) {
  // if we enable the memory pattern and already have the execution plan
  // go with mem pattern approach, which will allocate a big chunk for all
  // the weights.
  if (enable_memory_pattern) {
    return SaveInitializedTensorsWithMemPattern(graph, execution_plan, exec_providers,
                                                mlvalue_name_idx_map, weights_buffers, save_tensor_func, logger);
  }
  return SaveInitializedTensorsWithSeperateBuffer(graph, execution_plan, exec_providers,
                                                  mlvalue_name_idx_map, save_tensor_func, logger);
}

static common::Status CreateOpKernelInternal(const onnxruntime::Node& node,
                                             const IExecutionProvider& exec_provider,
                                             const SessionState& session_state,
                                             const KernelRegistryManager& custom_registry_manager,
                                             std::unique_ptr<OpKernel>& op_kernel) {
  return custom_registry_manager.CreateKernel(node,
                                              exec_provider,
                                              session_state,
                                              op_kernel);
}

static common::Status CreateOpKernel(const onnxruntime::Node& node,
                                     const ExecutionProviders& execution_providers,
                                     const SessionState& session_state,
                                     const KernelRegistryManager& custom_registry_manager,
                                     std::unique_ptr<OpKernel>& op_kernel,
                                     const logging::Logger& logger) {
  onnxruntime::ProviderType exec_provider_name = node.GetExecutionProviderType();

  const IExecutionProvider* exec_provider = nullptr;
  if (exec_provider_name.empty() || (exec_provider = execution_providers.Get(exec_provider_name)) == nullptr) {
    auto status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Could not create kernel for node: ", node.Name(),
                                  " as there's no execution provider allocated.");
    LOGS(logger, ERROR) << status.ErrorMessage();
  }

  common::Status status = CreateOpKernelInternal(node, *exec_provider, session_state, custom_registry_manager,
                                                 op_kernel);

  if (!status.IsOK()) {
    LOGS(logger, ERROR) << "Kernel creation failed for node: "
                        << node.Name() << " with error: " << status.ErrorMessage();
  }

  return status;
}

common::Status SaveKernels(const ExecutionProviders& execution_providers,
                           SessionState& session_state,
                           const KernelRegistryManager& custom_registry_manager,
                           const logging::Logger& logger) {
  LOGS(logger, INFO) << "Saving kernels.";

  for (auto& node : session_state.GetGraphViewer()->Nodes()) {
    // construct and save the kernels
    std::unique_ptr<OpKernel> op_kernel;
    ORT_RETURN_IF_ERROR(CreateOpKernel(node, execution_providers, session_state, custom_registry_manager, op_kernel, logger));
    session_state.AddKernel(node.Index(), std::move(op_kernel));
  }

  LOGS(logger, INFO) << "Done saving kernels.";

  return Status::OK();
}

template <typename T>  // T is const NodeArg or NodeArg
static bool IsArgNameInInputsOutputs(const std::string& name,
                                     const std::vector<T*>& graph_args) {
  auto it = std::find_if(std::begin(graph_args), std::end(graph_args), [&name](const onnxruntime::NodeArg* arg) {
    return arg->Name() == name;
  });
  return it != graph_args.end();
}

common::Status SaveInputOutputNamesToNodeMapping(const onnxruntime::Graph& graph,
                                                 const KernelRegistryManager& custom_registry_manager,
                                                 SessionState& session_state,
                                                 const std::vector<NodeArg*>* implicit_inputs) {
  auto& graph_inputs = graph.GetInputsIncludingInitializers();
  auto& graph_outputs = graph.GetOutputs();

  if (implicit_inputs && implicit_inputs->empty()) {
    implicit_inputs = nullptr;
  }

  for (auto& node : graph.Nodes()) {
    // note that KernelCreateInfo may not exist for custom kernel
    const KernelCreateInfo* kci = nullptr;
    custom_registry_manager.SearchKernelRegistry(node, &kci);

    ORT_RETURN_IF_ERROR(
        onnxruntime::Node::ForEachWithIndex(
            node.InputDefs(),
            [&](const onnxruntime::NodeArg& arg, size_t index) {
              if (arg.Name().empty()) {
                return Status::OK();
              }

              SessionState::NodeInfo node_info(index, &node, kci);

              if (IsArgNameInInputsOutputs(arg.Name(), graph_inputs)) {
                session_state.AddInputNameToNodeInfoMapping(arg.Name(), node_info);
                return Status::OK();
              }

              if (implicit_inputs) {
                if (IsArgNameInInputsOutputs(arg.Name(), *implicit_inputs)) {
                  session_state.AddInputNameToNodeInfoMapping(arg.Name(), node_info);
                  return Status::OK();
                }
              }

              if (IsArgNameInInputsOutputs(arg.Name(), graph_outputs)) {
                session_state.AddOutputNameToNodeInfoMapping(arg.Name(), node_info);
                return Status::OK();
              }

              return Status::OK();
            }));

    // implicit inputs to a node could come directly from a feed, so we need to make sure they have an entry too
    const auto& node_implicit_inputs = node.ImplicitInputDefs();
    if (!node_implicit_inputs.empty()) {
      // nested subgraph. for now map them to this node (which will be CPU based as all the control flow nodes
      // are currently CPU based and they're the only ones that have implicit inputs) as the inputs will be passed as a
      // feed when executing the subgraph and need to be in the mapping.
      // in the future we want to recurse and find where the implicit input is actually used to try and avoid a
      // copy to/from CPU to go through the control flow nodes where possible/applicable.
      // the processing for the subgraph where the implicit input is consumed will do the real check on whether any
      // copy to a different device is required
      SessionState::NodeInfo node_info(std::numeric_limits<size_t>::max(), &node, kci);
      for (const auto& input_def : node_implicit_inputs) {
        session_state.AddInputNameToNodeInfoMapping(input_def->Name(), node_info);
      }
    }
  }

  // It's possible (although assumably rare) for a graph to have inputs that aren't used. one reasonable occurrence
  // is in the Loop subgraph where the value of the condition used to decide whether to continue looping is passed in.
  // The condition evaluated to 'true' given the subgraph is being executed, so it's of dubious value as an input.
  // Similar is the current iteration number which may or may not be needed by the Loop subgraph.
  // In order to handle those, create a dummy entry in the input name to node info mapping so that
  // utils::CopyOneInputAcrossDevices is happy.

  auto& input_map = session_state.GetInputNodeInfoMap();
  auto end_map = input_map.cend();
  SessionState::NodeInfo empty_node_info(std::numeric_limits<size_t>::max(), nullptr, nullptr);

  for (const auto& graph_input : graph_inputs) {
    const auto& name = graph_input->Name();
    if (input_map.find(name) == end_map) {
      // dummy entry for an input that we didn't find a use of in the graph. warn about it in case that's a bug.
      // utils::CopyOneInputAcrossDevices will use the input MLValue as is given we don't believe it's used anywhere.
      LOGS(session_state.Logger(), WARNING) << "Graph input with name " << name << " is not associated with a node. ";
      session_state.AddInputNameToNodeInfoMapping(name, empty_node_info);
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
