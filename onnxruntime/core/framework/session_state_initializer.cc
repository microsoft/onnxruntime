// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/framework/session_state_initializer.h"

#include <functional>
#include <limits>
#include <core/common/status.h>

#include "core/common/common.h"
#include "core/common/logging/logging.h"

#include "core/graph/graph_viewer.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/ml_value.h"
#include "core/framework/ml_value_patterns_planner.h"
#include "core/framework/mlvalue_name_idx_map.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/framework/mem_buffer.h"

namespace onnxruntime {

static common::Status SaveMLValueNameIndexMapping(const GraphViewer& graph_viewer,
                                                  MLValueNameIdxMap& mlvalue_name_idx_map,
                                                  const logging::Logger& logger);

// T should have signature of '(int idx, const onnxruntime::MLValue& value, const OrtCallback& d) -> Status'
template <typename T>
static common::Status SaveInitializedTensors(const Env& env, const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
                                             const onnxruntime::Graph& graph,
                                             const SequentialExecutionPlan& execution_plan,
                                             const ExecutionProviders& exec_providers,
                                             const MLValueNameIdxMap& mlvalue_name_idx_map,
                                             std::map<OrtAllocatorInfo, BufferUniquePtr>& weights_buffers,
                                             const T& save_tensor_func, const logging::Logger& logger);

static common::Status SaveKernels(const ExecutionProviders& execution_providers,
                                  SessionState& session_state,
                                  const KernelRegistryManager& custom_registry_manager,
                                  const logging::Logger& logger);

static common::Status SaveInputOutputNamesToNodeMapping(const onnxruntime::Graph& graph,
                                                        const KernelRegistryManager& custom_registry_manager,
                                                        SessionState& session_state,
                                                        const std::vector<NodeArg*>* implicit_inputs);

SessionStateInitializer::SessionStateInitializer(const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
                                                 onnxruntime::Graph& graph, SessionState& session_state,
                                                 const ExecutionProviders& providers,
                                                 KernelRegistryManager& kernel_registry_manager)
    : graph_loc_(graph_loc),
      graph_{graph},
      session_state_{session_state},
      execution_providers_{providers},
      kernel_registry_manager_{kernel_registry_manager},
      logger_{session_state.Logger()} {}

common::Status SessionStateInitializer::CreatePlan(const Node* parent_node,
                                                   const std::vector<NodeArg*>& outer_scope_node_args,
                                                   bool enable_sequential_execution) {
  auto graph_viewer = std::make_unique<onnxruntime::GraphViewer>(graph_);

  // populate the SessionState MLValueNameIdxMap
  auto& mlvalue_name_idx_map = session_state_.GetMLValueNameIdxMap();
  ORT_RETURN_IF_ERROR(SaveMLValueNameIndexMapping(*graph_viewer, mlvalue_name_idx_map, logger_));

  // ignore any outer scope args we don't know about. this can happen if a node contains multiple subgraphs.
  std::vector<const NodeArg*> valid_outer_scope_node_args;
  std::for_each(outer_scope_node_args.cbegin(), outer_scope_node_args.cend(),
                [&mlvalue_name_idx_map, &valid_outer_scope_node_args](const NodeArg* node_arg) {
                  int idx;
                  if (mlvalue_name_idx_map.GetIdx(node_arg->Name(), idx).IsOK()) {
                    valid_outer_scope_node_args.push_back(node_arg);
                  };
                });

  std::unique_ptr<SequentialExecutionPlan> exec_plan;

  if (enable_sequential_execution) {
    // CreatePlan will create a new SequentialExecutionPlan instance that we will
    // save into the session state.
    ORT_RETURN_IF_ERROR(
        SequentialPlanner::CreatePlan(parent_node, *graph_viewer, valid_outer_scope_node_args, execution_providers_,
                                      kernel_registry_manager_, mlvalue_name_idx_map, exec_plan));

    session_state_.SetExecutionPlan(std::move(exec_plan));
  } else {
    // Parallel execution still uses same allocation plan, but has limitation of memory buffer reuse.
    SequentialPlannerContext context(true /* enable parallel execution */);
    ORT_RETURN_IF_ERROR(
        SequentialPlanner::CreatePlan(parent_node, *graph_viewer, valid_outer_scope_node_args, execution_providers_,
                                      kernel_registry_manager_, mlvalue_name_idx_map, context, exec_plan));

    session_state_.SetExecutionPlan(std::move(exec_plan));
  }

  session_state_.SetGraphViewer(std::move(graph_viewer));

  return Status::OK();
}

common::Status SessionStateInitializer::InitializeAndSave(const std::vector<NodeArg*>* implicit_inputs) {
  const auto* exec_plan_ptr = session_state_.GetExecutionPlan();
  ORT_ENFORCE(exec_plan_ptr, "Execution plan was not found in SessionState. CreatePlan must be called first.");

  const auto& exec_plan{*exec_plan_ptr};
  const auto& mlvalue_name_idx_map{session_state_.GetMLValueNameIdxMap()};

  // lambda to save initialized tensors into SessionState directly
  const Env& env = Env::Default();
  ORT_RETURN_IF_ERROR(
      SaveInitializedTensors(
          env, graph_loc_, graph_, exec_plan, execution_providers_, mlvalue_name_idx_map,
          session_state_.GetMutableWeightsBuffers(),
          [this](int idx, const onnxruntime::MLValue& value, const OrtCallback& d) -> Status {
            return session_state_.AddInitializedTensor(idx, value, &d);
          },
          logger_));
  // remove weights from the graph now to save memory but in many cases it won't save memory, if the tensor was
  // preallocated with the some other tensors in a single 'allocate' call, which is very common.
  // TODO: make it better
  graph_.CleanAllInitializedTensors();

  ORT_RETURN_IF_ERROR(SaveKernels(execution_providers_, session_state_, kernel_registry_manager_, logger_));
  ORT_RETURN_IF_ERROR(SaveInputOutputNamesToNodeMapping(graph_, kernel_registry_manager_, session_state_,
                                                        implicit_inputs));

  return Status::OK();
}

// Build the MLValue name->idx mapping
common::Status SaveMLValueNameIndexMapping(const GraphViewer& graph_viewer,
                                           MLValueNameIdxMap& mlvalue_name_idx_map,
                                           const logging::Logger& logger) {
  LOGS(logger, INFO) << "SaveMLValueNameIndexMapping";
  int idx = 0;

  // we keep all graph inputs (including initializers), even if they are unused, so make sure they all have an entry
  for (const auto* input_def : graph_viewer.GetInputsIncludingInitializers()) {
    idx = mlvalue_name_idx_map.Add(input_def->Name());
    VLOGS(logger, 1)
        << "Added graph_viewer input with name: " << input_def->Name() << " to MLValueIndex with index: " << idx;
  }

  for (auto& node : graph_viewer.Nodes()) {
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
  for (const auto& output : graph_viewer.GetOutputs()) {
    if (output->Exists()) {
      idx = mlvalue_name_idx_map.Add(output->Name());
      VLOGS(logger, 1)
          << "Added graph output with name: " << output->Name() << " to MLValueIndex with index: " << idx;
    }
  }

  LOGS(logger, INFO) << "Done saving MLValue mappings.";
  return Status::OK();
}

static common::Status DeserializeTensorProto(const Env& env, const std::basic_string<PATH_CHAR_TYPE>& proto_path,
                                             const ONNX_NAMESPACE::TensorProto& tensor_proto, const MemBuffer& m,
                                             const ExecutionProviders& exec_providers, MLValue& mlvalue, OrtCallback& deleter) {
  const OrtAllocatorInfo& alloc_info = m.GetAllocInfo();
  if (strcmp(alloc_info.name, CPU) == 0 || alloc_info.mem_type == OrtMemTypeCPUOutput) {
    // deserialize directly to CPU tensor
    return utils::TensorProtoToMLValue(env, proto_path.c_str(), tensor_proto, m, mlvalue, deleter);
  }
  //alloc_info.name is not 'CPU'
  const IExecutionProvider* provider = exec_providers.Get(alloc_info);
  if (provider == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid allocation info. Provider name = ", alloc_info.name);
  }
  // deserialize and copy. In the copy stage, it won't check if the buffer has enough room.
  // The result tensor won't need a deleter because:
  // 1. It mustn't be a string tensor
  // 2. The memory is not memory-mapped.
  deleter.f = nullptr;
  deleter.param = nullptr;
  if (tensor_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "string tensor is not supported for copying between allocators");
  }
  // deserialize to CPU first for non-CPU allocator, then alloc and copy
  size_t cpu_tensor_length;
  ORT_RETURN_IF_ERROR(utils::GetSizeInBytesFromTensorProto<0>(tensor_proto, &cpu_tensor_length));
  if (m.GetLen() < cpu_tensor_length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Internal error. The preallocated buffer is too small. Requires ", cpu_tensor_length,
                           ", Got ", m.GetLen());
  }
  OrtAllocatorInfo info(CPU, OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::unique_ptr<char[]> data(new char[cpu_tensor_length]);
  std::unique_ptr<Tensor> p_tensor;
  MLValue tmp_mlvalue;
  OrtCallback d;
  ORT_RETURN_IF_ERROR(utils::TensorProtoToMLValue(
      env, proto_path.c_str(), tensor_proto, MemBuffer(data.get(), cpu_tensor_length, info), tmp_mlvalue, d));
  const Tensor& p_deserialize_tensor = tmp_mlvalue.Get<Tensor>();

  p_tensor = std::make_unique<Tensor>(p_deserialize_tensor.DataType(), p_deserialize_tensor.Shape(), m.GetBuffer(),
                                      m.GetAllocInfo());
  // TODO: does this function work for string tensor?
  Status copy_status = provider->CopyTensor(p_deserialize_tensor, *p_tensor);
  if (d.f) d.f(d.param);
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

static common::Status AllocatePlannedBuffers(const MemoryPatternGroup& mem_patterns,
                                             const ExecutionProviders& exec_providers,
                                             std::map<OrtAllocatorInfo, BufferUniquePtr>& weights_buffers) {
  const size_t location_len = mem_patterns.locations.size();
  for (size_t i = 0; i < location_len; ++i) {
    auto& location = mem_patterns.locations[i];
    ORT_ENFORCE(weights_buffers.find(location) == weights_buffers.end(), "Existing entry in weights buffer for ",
                location.name);

    auto alloc = utils::GetAllocator(exec_providers, location);
    if (!alloc)
      return Status(common::ONNXRUNTIME, common::FAIL, "Failed to get allocator for location: " + location.ToString());

    if (mem_patterns.patterns[i].PeakSize() > 0) {
      void* buffer = alloc->Alloc(mem_patterns.patterns[i].PeakSize());
      auto kvp = weights_buffers.insert(std::make_pair(location, BufferUniquePtr(buffer, alloc)));
      if (!kvp.second) {
        alloc->Free(buffer);
        return Status(common::ONNXRUNTIME, common::FAIL, "duplicated location");
      }
    }
  }
  return Status::OK();
}

/**
 * When it succeeded, p could be NULL if the tensor with 'mlvalue_index' will not have any element
 */
static common::Status GetPreallocatedBuffer(const MemoryPatternGroup& mem_patterns, const OrtAllocatorInfo& location,
                                            int mlvalue_index,
                                            const std::map<OrtAllocatorInfo, BufferUniquePtr>& weights_buffers,
                                            const char* name, void*& p, size_t& len) {
  auto pattern = mem_patterns.GetPatterns(location);
  if (pattern == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Mem pattern for initializer ", name, " is not found");
  }
  // if block is not found, means this mlvalue is not traced
  // fall back to allocate separate buffer.
  // if it->second.get() is null, then fall back to the block not found case
  auto block = pattern->GetBlock(mlvalue_index);
  auto it = weights_buffers.find(location);
  if (it == weights_buffers.end()) {
    if (block != nullptr && block->size_ == 0) {
      // Because the size is 0, this miss find is expected. we won't allocate a buffer with size of zero.
      p = nullptr;
      len = 0;
      return Status::OK();
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Weight buffer for initializer '", name, "' is not found");
  }

  if (block == nullptr || it->second == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Get preallocated buffer for initializer '", name, "' failed");
  }

  p = reinterpret_cast<char*>(it->second.get()) + block->offset_;
  len = block->size_;
  return Status::OK();
}

template <typename T>
common::Status SaveInitializedTensors(const Env& env, const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
                                      const Graph& graph, const SequentialExecutionPlan& execution_plan,
                                      const ExecutionProviders& exec_providers,
                                      const MLValueNameIdxMap& mlvalue_name_idx_map,
                                      std::map<OrtAllocatorInfo, BufferUniquePtr>& weights_buffers,
                                      const T& save_tensor_func, const logging::Logger& logger) {
  LOGS(logger, INFO) << "Saving initialized tensors.";
  static constexpr int alignment = 256;
  ORT_ENFORCE(mlvalue_name_idx_map.MaxIdx() > 0, "MLValue indexes should have been populated.");

  MLValuePatternPlanner planner(execution_plan);

  //1. first plan the memory
  const onnxruntime::InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
  std::unordered_map<int, const ONNX_NAMESPACE::TensorProto*> id_to_initialized_tensor;
  for (const auto& entry : initialized_tensor_set) {
    int mlvalue_index;
    ORT_RETURN_IF_ERROR(mlvalue_name_idx_map.GetIdx(entry.first, mlvalue_index));
    id_to_initialized_tensor[mlvalue_index] = entry.second;
  }
  for (const auto& entry : id_to_initialized_tensor) {
    size_t len = 0;
    ORT_RETURN_IF_ERROR(utils::GetSizeInBytesFromTensorProto<alignment>(*entry.second, &len));
    ORT_RETURN_IF_ERROR(planner.TraceAllocation(entry.first, len));
  }

  //2. allocate weight buffer on different locations
  MemoryPatternGroup mem_patterns;
  ORT_RETURN_IF_ERROR(planner.GeneratePatterns(&mem_patterns));
  ORT_RETURN_IF_ERROR(AllocatePlannedBuffers(mem_patterns, exec_providers, weights_buffers));
  OrtCallback deleter;
  //3. create weight tensors based on weights buffer
  for (const auto& entry : id_to_initialized_tensor) {
    int mlvalue_index = entry.first;
    const char* name = entry.second->has_name() ? entry.second->name().c_str() : "";
    const ONNX_NAMESPACE::TensorProto& tensor_proto = *(entry.second);

    auto& location = execution_plan.allocation_plan[mlvalue_index].location;
    void* buffer = nullptr;
    size_t len = 0;
    // TODO: if the tensor need be copied, does it have enough room?
    ORT_RETURN_IF_ERROR(
        GetPreallocatedBuffer(mem_patterns, location, mlvalue_index, weights_buffers, name, buffer, len));
#ifndef NDEBUG
    ORT_ENFORCE(buffer != nullptr || len == 0);
#endif

    MemBuffer m(buffer, len, location);
    MLValue mlvalue;
    Status st = DeserializeTensorProto(env, graph_loc, tensor_proto, m, exec_providers, mlvalue, deleter);
    if (!st.IsOK()) {
      std::ostringstream oss;
      oss << "Deserialize tensor " << name << " failed." << st.ErrorMessage();
      return Status(st.Category(), st.Code(), oss.str());
    }

    ORT_RETURN_IF_ERROR(save_tensor_func(mlvalue_index, mlvalue, deleter));

    VLOGS(logger, 1) << "Added weight with name : " << name << " with index: " << mlvalue_index;
  }

  LOGS(logger, INFO) << "Done saving initialized tensors";
  return common::Status::OK();
}

static common::Status CreateOpKernel(const onnxruntime::Node& node, const ExecutionProviders& execution_providers,
                                     const SessionState& session_state,
                                     const KernelRegistryManager& custom_registry_manager,
                                     std::unique_ptr<OpKernel>& op_kernel) {
  onnxruntime::ProviderType exec_provider_name = node.GetExecutionProviderType();

  const IExecutionProvider* exec_provider = nullptr;
  if (exec_provider_name.empty() || (exec_provider = execution_providers.Get(exec_provider_name)) == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Could not create kernel for node: ", node.Name(),
                           " as there's no execution provider allocated.");
  }

  common::Status status = custom_registry_manager.CreateKernel(node, *exec_provider, session_state, op_kernel);
  if (!status.IsOK()) {
    return common::Status(
        status.Category(), status.Code(),
        MakeString("Kernel creation failed for node: ", node.Name(), " with error: ", status.ErrorMessage()));
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
    ORT_RETURN_IF_ERROR(CreateOpKernel(node, execution_providers, session_state, custom_registry_manager, op_kernel));
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
                ORT_RETURN_IF_ERROR(session_state.AddInputNameToNodeInfoMapping(arg.Name(), node_info));
                return Status::OK();
              }

              if (implicit_inputs) {
                if (IsArgNameInInputsOutputs(arg.Name(), *implicit_inputs)) {
                  ORT_RETURN_IF_ERROR(session_state.AddInputNameToNodeInfoMapping(arg.Name(), node_info));
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
        ORT_RETURN_IF_ERROR(session_state.AddInputNameToNodeInfoMapping(input_def->Name(), node_info));
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
      ORT_RETURN_IF_ERROR(session_state.AddInputNameToNodeInfoMapping(name, empty_node_info));
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
