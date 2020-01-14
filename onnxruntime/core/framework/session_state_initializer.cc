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
#include "core/framework/data_transfer_manager.h"
#include "core/graph/graph_utils.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/ml_value.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/framework/mem_buffer.h"
#include "core/framework/tensor_allocator.h"

namespace onnxruntime {

// T should have signature of '(int idx, const OrtValue& value, const OrtCallback& d) -> Status'
template <typename T>
static common::Status SaveInitializedTensors(const Env& env, const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
                                             const onnxruntime::Graph& graph, const ExecutionProviders& exec_providers,
                                             const OrtValueNameIdxMap& ort_value_name_idx_map,
                                             ITensorAllocator* planner, const T& save_tensor_func,
                                             const logging::Logger& logger,
                                             const DataTransferManager& data_transfer_mgr);

static common::Status SaveInputOutputNamesToNodeMapping(
    const onnxruntime::Graph& graph,
    const KernelRegistryManager& custom_registry_manager,
    SessionState& session_state,
    const ConstPointerContainer<std::vector<NodeArg*>>* implicit_inputs);

SessionStateInitializer::SessionStateInitializer(bool enable_mem_pattern,
                                                 const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
                                                 onnxruntime::Graph& graph, SessionState& session_state,
                                                 const ExecutionProviders& providers,
                                                 KernelRegistryManager& kernel_registry_manager)
    : graph_loc_(graph_loc),
      graph_(graph),
      session_state_(session_state),
      execution_providers_(providers),
      kernel_registry_manager_(kernel_registry_manager),
      logger_(session_state.Logger()),
      enable_mem_pattern_(enable_mem_pattern) {}

common::Status SessionStateInitializer::CreatePlan(
    const Node* parent_node,
    const ConstPointerContainer<std::vector<NodeArg*>>* outer_scope_node_args,
    ExecutionMode execution_mode) {
  session_state_.SetGraph(graph_);
  const GraphViewer* graph_viewer = session_state_.GetGraphViewer();

  // populate the SessionState OrtValueNameIdxMap
  const auto& ort_value_name_idx_map = session_state_.GetOrtValueNameIdxMap();

  // ignore any outer scope args we don't know about. this can happen if a node contains multiple subgraphs.
  std::vector<const NodeArg*> valid_outer_scope_node_args;
  if (outer_scope_node_args) {
    std::for_each(outer_scope_node_args->cbegin(), outer_scope_node_args->cend(),
                  [&ort_value_name_idx_map, &valid_outer_scope_node_args](const NodeArg* node_arg) {
                    int idx;
                    if (ort_value_name_idx_map.GetIdx(node_arg->Name(), idx).IsOK()) {
                      valid_outer_scope_node_args.push_back(node_arg);
                    };
                  });
  }

  std::unique_ptr<SequentialExecutionPlan> exec_plan;
  SequentialPlannerContext context(execution_mode);
  ORT_RETURN_IF_ERROR(SequentialPlanner::CreatePlan(parent_node, *graph_viewer, valid_outer_scope_node_args,
                                                    execution_providers_, kernel_registry_manager_,
                                                    ort_value_name_idx_map, context, exec_plan));
  session_state_.SetExecutionPlan(std::move(exec_plan));

  const auto* exec_plan_ptr = session_state_.GetExecutionPlan();
  ORT_ENFORCE(exec_plan_ptr, "Execution plan was not found in SessionState. CreatePlan must be called first.");

  std::unique_ptr<ITensorAllocator> tensor_allocator_(ITensorAllocator::Create(
      enable_mem_pattern_, *exec_plan_ptr, execution_providers_, session_state_.GetMutableWeightsBuffers()));

  // lambda to save initialized tensors into SessionState directly
  const Env& env = Env::Default();
  ORT_RETURN_IF_ERROR(SaveInitializedTensors(
      env, graph_loc_, graph_, execution_providers_, ort_value_name_idx_map, tensor_allocator_.get(),
      [this](int idx, const OrtValue& value, const OrtCallback& d, bool constant) -> Status {
        return session_state_.AddInitializedTensor(idx, value, &d, constant);
      },
      logger_, session_state_.GetDataTransferMgr()));
  // remove weights from the graph now to save memory but in many cases it won't save memory, if the tensor was
  // preallocated with the some other tensors in a single 'allocate' call, which is very common.
  // TODO: make it better
  graph_.CleanAllInitializedTensors();

  ORT_RETURN_IF_ERROR(session_state_.CreateKernels(kernel_registry_manager_));
  ORT_RETURN_IF_ERROR(
      SaveInputOutputNamesToNodeMapping(graph_, kernel_registry_manager_, session_state_, outer_scope_node_args));
  return Status::OK();
}

static common::Status DeserializeTensorProto(const Env& env, const std::basic_string<PATH_CHAR_TYPE>& proto_path,
                                             const ONNX_NAMESPACE::TensorProto& tensor_proto, const MemBuffer& m,
                                             const ExecutionProviders& exec_providers, OrtValue& ort_value,
                                             OrtCallback& deleter,
                                             const DataTransferManager& data_transfer_mgr) {
  const OrtMemoryInfo& alloc_info = m.GetAllocInfo();
  if (strcmp(alloc_info.name, CPU) == 0 || alloc_info.mem_type == OrtMemTypeCPUOutput) {
    // deserialize directly to CPU tensor
    return utils::TensorProtoToMLValue(env, proto_path.c_str(), tensor_proto, m, ort_value, deleter);
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
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Internal error. The preallocated buffer is too small. Requires ",
                           cpu_tensor_length, ", Got ", m.GetLen());
  }
  OrtMemoryInfo info = exec_providers.GetDefaultCpuMemoryInfo();
  std::unique_ptr<char[]> data(new char[cpu_tensor_length]);
  std::unique_ptr<Tensor> p_tensor;
  OrtValue tmp_ort_value;
  OrtCallback d;
  ORT_RETURN_IF_ERROR(utils::TensorProtoToMLValue(env, proto_path.c_str(), tensor_proto,
                                                  MemBuffer(data.get(), cpu_tensor_length, info), tmp_ort_value, d));
  const Tensor& p_deserialize_tensor = tmp_ort_value.Get<Tensor>();

  p_tensor = onnxruntime::make_unique<Tensor>(p_deserialize_tensor.DataType(), p_deserialize_tensor.Shape(), m.GetBuffer(),
                                      m.GetAllocInfo());
  // TODO: does this function work for string tensor?
  Status copy_status = data_transfer_mgr.CopyTensor(p_deserialize_tensor, *p_tensor);
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

  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
  return common::Status::OK();
}

template <typename T>
common::Status SaveInitializedTensors(const Env& env, const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
                                      const Graph& graph, const ExecutionProviders& exec_providers,
                                      const OrtValueNameIdxMap& ort_value_name_idx_map, ITensorAllocator* planner,
                                      const T& save_tensor_func, const logging::Logger& logger,
                                      const DataTransferManager& data_transfer_mgr) {
  LOGS(logger, INFO) << "Saving initialized tensors.";
  ORT_ENFORCE(ort_value_name_idx_map.MaxIdx() > 0, "OrtValue indexes should have been populated.");

  //1. first plan the memory
  const onnxruntime::InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
  std::unordered_map<int, const ONNX_NAMESPACE::TensorProto*> id_to_initialized_tensor;
  for (const auto& entry : initialized_tensor_set) {
    int ort_value_index;
    ORT_RETURN_IF_ERROR(ort_value_name_idx_map.GetIdx(entry.first, ort_value_index));
    id_to_initialized_tensor[ort_value_index] = entry.second;
  }
  for (const auto& entry : id_to_initialized_tensor) {
    ORT_RETURN_IF_ERROR(planner->Trace(entry.first, entry.second));
  }

  //2. allocate weight buffer on different locations
  ORT_RETURN_IF_ERROR(planner->FinalizePlan());
  OrtCallback deleter;
  //3. create weight tensors based on weights buffer
  for (const auto& entry : id_to_initialized_tensor) {
    int ort_value_index = entry.first;
    const char* name = (entry.second->name().empty()) ? "" : entry.second->name().c_str();
    const ONNX_NAMESPACE::TensorProto& tensor_proto = *(entry.second);

    std::unique_ptr<MemBuffer> m;
    // TODO: if the tensor need be copied, does it have enough room?
    ORT_RETURN_IF_ERROR(planner->GetPreallocatedBuffer(ort_value_index, name, m));
#ifndef NDEBUG
    ORT_ENFORCE(m != nullptr);
    ORT_ENFORCE(m->GetBuffer() != nullptr || m->GetLen() == 0);
#endif
    OrtValue ort_value;
    Status st = DeserializeTensorProto(env, graph_loc, tensor_proto, *m, exec_providers, ort_value, deleter, data_transfer_mgr);
    if (!st.IsOK()) {
      std::ostringstream oss;
      oss << "Deserialize tensor " << name << " failed." << st.ErrorMessage();
      return Status(st.Category(), st.Code(), oss.str());
    }

    bool constant = graph_utils::IsConstantInitializer(graph, name, /* check_outer_scope */ false);
    ORT_RETURN_IF_ERROR(save_tensor_func(ort_value_index, ort_value, deleter, constant));

    VLOGS(logger, 1) << "Added weight with name : " << name << " with index: " << ort_value_index;
  }

  LOGS(logger, INFO) << "Done saving initialized tensors";
  return common::Status::OK();
}

template <typename T>  // T is container of const NodeArg* or NodeArg*
static bool IsArgNameInInputsOutputs(const std::string& name,
                                     const T& graph_args) {
  auto it = std::find_if(graph_args.cbegin(), graph_args.cend(), [&name](const onnxruntime::NodeArg* arg) {
    return arg->Name() == name;
  });
  return it != graph_args.cend();
}

common::Status SaveInputOutputNamesToNodeMapping(const onnxruntime::Graph& graph,
                                                 const KernelRegistryManager& custom_registry_manager,
                                                 SessionState& session_state,
                                                 const ConstPointerContainer<std::vector<NodeArg*>>* implicit_inputs) {
  auto& graph_inputs = graph.GetInputsIncludingInitializers();
  auto& graph_outputs = graph.GetOutputs();

  if (implicit_inputs && implicit_inputs->empty()) {
    implicit_inputs = nullptr;
  }
  const auto* exec_plan = session_state.GetExecutionPlan();
  const auto& name_to_id = session_state.GetOrtValueNameIdxMap();

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

              int arg_index;
              ORT_RETURN_IF_ERROR(name_to_id.GetIdx(arg.Name(), arg_index));
              const auto& device = exec_plan->GetLocation(arg_index).device;

              SessionState::NodeInfo node_info(index, &node, kci, device);

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

              // ??? Why are we checking if an input to a node is also in the graph output
              //if (IsArgNameInInputsOutputs(arg.Name(), graph_outputs)) {
              //  session_state.AddOutputNameToNodeInfoMapping(arg.Name(), node_info);
              //  return Status::OK();
              //}

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
      for (const auto& input_def : node_implicit_inputs) {
        int arg_index;
        ORT_RETURN_IF_ERROR(name_to_id.GetIdx(input_def->Name(), arg_index));
        auto& device = exec_plan->GetLocation(arg_index).device;
        SessionState::NodeInfo node_info(std::numeric_limits<size_t>::max(), &node, kci, device);
        ORT_RETURN_IF_ERROR(session_state.AddInputNameToNodeInfoMapping(input_def->Name(), node_info));
      }
    }

    ORT_RETURN_IF_ERROR(
        onnxruntime::Node::ForEachWithIndex(
            node.OutputDefs(),
            [&](const onnxruntime::NodeArg& arg, size_t index) {
              if (arg.Name().empty()) {
                return Status::OK();
              }

              int arg_index;
              ORT_RETURN_IF_ERROR(name_to_id.GetIdx(arg.Name(), arg_index));
              const auto& device = exec_plan->GetLocation(arg_index).device;

              SessionState::NodeInfo node_info(index, &node, kci, device);

              if (IsArgNameInInputsOutputs(arg.Name(), graph_outputs)) {
                session_state.AddOutputNameToNodeInfoMapping(arg.Name(), node_info);
                return Status::OK();
              }

              return Status::OK();
            }));
  }

  // It's possible (although assumably rare) for a graph to have inputs that aren't used. one reasonable occurrence
  // is in the Loop subgraph where the value of the condition used to decide whether to continue looping is passed in.
  // The condition evaluated to 'true' given the subgraph is being executed, so it's of dubious value as an input.
  // Similar is the current iteration number which may or may not be needed by the Loop subgraph.
  // In order to handle those, create a dummy entry in the input name to node info mapping so that
  // utils::CopyOneInputAcrossDevices is happy.

  auto& input_map = session_state.GetInputNodeInfoMap();
  auto end_map = input_map.cend();

  for (const auto& graph_input : graph_inputs) {
    const auto& name = graph_input->Name();
    if (input_map.find(name) == end_map) {
      // dummy entry for an input that we didn't find a use of in the graph. log it in case that's a bug.
      // utils::CopyOneInputAcrossDevices will use the input OrtValue as is given we don't believe it's used anywhere.
      LOGS(session_state.Logger(), INFO) << (graph.IsSubgraph() ? "Subgraph" : "Graph") << " input with name "
                                         << name << " is not used by any node.";
      int arg_index;
      ORT_RETURN_IF_ERROR(name_to_id.GetIdx(name, arg_index));
      auto& device = exec_plan->GetLocation(arg_index).device;
      SessionState::NodeInfo empty_node_info(std::numeric_limits<size_t>::max(), nullptr, nullptr, device);
      ORT_RETURN_IF_ERROR(session_state.AddInputNameToNodeInfoMapping(name, empty_node_info));
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
