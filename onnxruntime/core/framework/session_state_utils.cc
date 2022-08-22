// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <limits>
#include <utility>

#include <core/common/status.h>

#include "core/framework/ortdevice.h"
#include "core/graph/onnx_protobuf.h"
#include "core/framework/session_state_utils.h"
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/ort_value.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/framework/bfc_arena.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/framework/mem_buffer.h"
#include "core/framework/tensor_allocator.h"
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
#include "core/framework/memory_info.h"
#endif

namespace onnxruntime {
namespace session_state_utils {

// The following method will allocate memory directly using the device allocator.
// It can handle arena-based allocators and non-arena based allocators.
static common::Status AllocateBufferUsingDeviceAllocatorFromShapeAndType(const TensorShape& tensor_shape, const DataTypeImpl* type,
                                                                         const AllocatorPtr& alloc, /*out*/ void*& p_data) {
  int64_t shape_size = tensor_shape.Size();
  if (shape_size < 0)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "shape.Size() must >=0");

  p_data = nullptr;
  if (shape_size > 0) {
    SafeInt<size_t> mem_size = 0;

    if (!IAllocator::CalcMemSizeForArray(SafeInt<size_t>(shape_size), type->Size(), &mem_size)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed memory size calculation");
    }

    p_data = alloc->Reserve(mem_size);
  }

  return Status::OK();
}

// deleter for external data tensors managed by an OrtValue; manages the release of
// the tensor's data buffer (which points to the external data) and the tensor itself
struct ExtDataValueDeleter {
  OrtCallback ext_delete_cb;
  Tensor* p_tensor;
  void operator()(void*) noexcept {
    if (ext_delete_cb.f) {
      ext_delete_cb.f(ext_delete_cb.param);
    }

    delete p_tensor;
  }
};

// given a tensor proto with external data return an OrtValue with a tensor for
// that data; the pointers for the tensor data and the tensor itself are owned
// by the OrtValue's deleter
static inline common::Status ExtDataTensorProtoToTensor(const Env& env,
                                                        const std::basic_string<PATH_CHAR_TYPE>& proto_path,
                                                        const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                                        Tensor& tensor, OrtCallback& ext_data_deleter) {
  ORT_ENFORCE(utils::HasExternalData(tensor_proto));

  void* ext_data_buf = nullptr;
  SafeInt<size_t> ext_data_len = 0;
  ORT_RETURN_IF_ERROR(utils::GetExtDataFromTensorProto(env, proto_path.c_str(), tensor_proto,
                                                       ext_data_buf, ext_data_len, ext_data_deleter));

  // NB: creating a do-nothing allocator per tensor is wasteful; can perhaps be
  // avoided if the Tensor class implements the do-nothing behavior when given a
  // nullptr for the allocator argument
  const DataTypeImpl* const type = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto.data_type())->GetElementType();
  TensorShape tensor_shape = utils::GetTensorShapeFromTensorProto(tensor_proto);
  tensor = Tensor(type, tensor_shape, ext_data_buf, OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator));

  return common::Status::OK();
}

static common::Status DeserializeTensorProto(const Env& env, const std::basic_string<PATH_CHAR_TYPE>& proto_path,
                                             const ONNX_NAMESPACE::TensorProto& tensor_proto, const MemBuffer* m,
                                             const AllocatorPtr& alloc, const AllocatorPtr& default_cpu_alloc,
                                             OrtValue& ort_value, const DataTransferManager& data_transfer_mgr,
                                             bool use_device_allocator_for_initializers = false) {
  if (bool(alloc) == (m != nullptr)) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "DeserializeTensorProto() takes either pre-allocated buffer or an allocator!");
  }

  // Get shape and type of the tensor, and allocate the empty tensor
  TensorShape tensor_shape = utils::GetTensorShapeFromTensorProto(tensor_proto);
  const DataTypeImpl* const type = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto.data_type())->GetElementType();
  std::unique_ptr<Tensor> p_tensor;
  if (m != nullptr) {
    p_tensor = std::make_unique<Tensor>(type, tensor_shape, m->GetBuffer(), m->GetAllocInfo());
    if (m->GetLen() < p_tensor->SizeInBytes()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Internal error. The preallocated buffer is too small. Requires ",
                             p_tensor->SizeInBytes(), ", Got ", m->GetLen());
    }
  } else {
    if (use_device_allocator_for_initializers) {
      void* tensor_buffer = nullptr;
      ORT_RETURN_IF_ERROR(AllocateBufferUsingDeviceAllocatorFromShapeAndType(tensor_shape, type, alloc, tensor_buffer));
      p_tensor = std::make_unique<Tensor>(type, tensor_shape, tensor_buffer, alloc);
    } else {
      // If the provided allocator is an arena-based allocator, the call to Alloc() will tap into memory from the arena
      // (may expand it if there isn't a chunk that can be allotted to the memory request).
      // If the provided allocator is non-arena based, the device specific Alloc() call will be used to allocate the necessary memory.
      p_tensor = std::make_unique<Tensor>(type, tensor_shape, alloc);
    }
  }

  if (p_tensor->Location().device.Type() == OrtDevice::CPU) {
    // deserialize directly to CPU tensor
    if (utils::HasExternalData(tensor_proto)) {
      // NB: The file containing external data for the tensor is mmap'd. If the tensor will be used on CPU we can
      // utilize the mmap'd buffer directly by calling ExtDataTensorProtoToTensor. If we called
      // TensorProtoToTensor it would copy the data, causing unnecessary overhead
      OrtCallback ext_data_deleter;
      ORT_RETURN_IF_ERROR(ExtDataTensorProtoToTensor(env, proto_path, tensor_proto, *p_tensor, ext_data_deleter));

      ExtDataValueDeleter deleter{ext_data_deleter, p_tensor.get()};

      MLDataType ml_tensor_type = DataTypeImpl::GetType<Tensor>();
      ort_value.Init(p_tensor.release(), ml_tensor_type, deleter);
      return common::Status::OK();
    }
    ORT_RETURN_IF_ERROR(utils::TensorProtoToTensor(env, proto_path.c_str(), tensor_proto, *p_tensor));
  } else {  // non-cpu tensor
    if (tensor_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "string tensor is not supported for copying between allocators");
    }

    // deserialize to CPU first for non-CPU allocator, then copy
    std::unique_ptr<Tensor> p_deserialize_tensor;
    if (use_device_allocator_for_initializers) {
      void* tensor_buffer = nullptr;
      ORT_RETURN_IF_ERROR(AllocateBufferUsingDeviceAllocatorFromShapeAndType(tensor_shape, type, default_cpu_alloc, tensor_buffer));
      p_deserialize_tensor = std::make_unique<Tensor>(type, tensor_shape, tensor_buffer, default_cpu_alloc);
    } else {
      // If the provided allocator is an arena-based allocator, the call to Alloc() will tap into memory from the arena
      // (may expand it if there isn't a chunk that can be allotted to the memory request).
      // If the provided allocator is non-arena based, the device specific Alloc() call will be used to allocate the necessary memory.
      p_deserialize_tensor = std::make_unique<Tensor>(type, tensor_shape, default_cpu_alloc);
    }

    OrtCallback ext_data_deleter;
    if (utils::HasExternalData(tensor_proto)) {
      ORT_RETURN_IF_ERROR(ExtDataTensorProtoToTensor(env, proto_path, tensor_proto, *p_deserialize_tensor,
                                                     ext_data_deleter));
    } else {
      ORT_RETURN_IF_ERROR(utils::TensorProtoToTensor(env, proto_path.c_str(), tensor_proto, *p_deserialize_tensor));
    }
    // TODO!! Need a temp buffer allocator for non-escape buffers that maybe too big for stack allocation.

    Status copy_status = data_transfer_mgr.CopyTensor(*p_deserialize_tensor, *p_tensor);
    if (!copy_status.IsOK()) {
      if (copy_status.ErrorMessage().empty()) {
        // The windows execution provider does not return any error message today for CopyTensor since it is
        // not implemented yet. That's the reason we're adding our own error message so that we can debug better.
        return Status(copy_status.Category(), copy_status.Code(),
                      "Failed to copy tensor to " + p_tensor->Location().ToString());
      }
      return copy_status;
    }
  }
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  ort_value.Init(p_tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
  return common::Status::OK();
}

common::Status SaveInitializedTensors(
    const Env& env, const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
    const GraphViewer& graph, const AllocatorPtr& default_cpu_alloc,
    const OrtValueNameIdxMap& ort_value_name_idx_map,
    const std::vector<OrtValueIndex>& initializer_allocation_order,
    ITensorAllocator& planner,
    const SaveTensorFunction& save_tensor_func,
    const logging::Logger& logger, const DataTransferManager& data_transfer_mgr,
    const ExecutionPlanBase& exec_plan,
    const SessionOptions& session_options,
    const MemoryProfileFunction& memory_profile_func) {
  LOGS(logger, INFO) << "Saving initialized tensors.";
  ORT_ENFORCE(ort_value_name_idx_map.MaxIdx() > -1, "OrtValue indexes should have been populated.");

  // Determine if an intializer was supplied by the user for the purpose of sharing and if it requires a cross-device
  // copy. In case a cross-device copy is required, sharing cannot be accomplished since we allocate our own buffer
  // for the destn device which cannot be shared between sessions.
  auto use_user_supplied_initializer =
      [&session_options, &exec_plan, &logger, &ort_value_name_idx_map](const std::string& name) -> bool {
    bool retval = false;
    auto it = session_options.initializers_to_share_map.find(name);
    if (it == session_options.initializers_to_share_map.end()) {
      retval = false;
    } else {
      int ort_value_index = -1;
      if (!ort_value_name_idx_map.GetIdx(name, ort_value_index).IsOK()) {
        retval = false;
      } else {
        const auto& planned_mem_info = exec_plan.GetLocation(ort_value_index);
        const auto& user_mem_info = it->second->Get<Tensor>().Location();
        retval = user_mem_info.device == planned_mem_info.device;
        if (!retval) {
          LOGS(logger, WARNING) << "Cannot use user supplied initializer with name: ("
                                << name << ") because the ORT planned memory location device "
                                << planned_mem_info.ToString()
                                << " ) is different from what is supplied (" << user_mem_info.ToString() << ")";
        }
      }
    }

    return retval;
  };

  // 1. first plan the memory
  const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
  InlinedHashMap<int, const ONNX_NAMESPACE::TensorProto*> id_to_initialized_tensor;
  InlinedHashSet<int> user_supplied_initializer_ids;  // set containing the ort value ids of all user supplied initializers

  id_to_initialized_tensor.reserve(initialized_tensor_set.size());
  user_supplied_initializer_ids.reserve(initialized_tensor_set.size());

  for (const auto& entry : initialized_tensor_set) {
    int ort_value_index;
    ORT_RETURN_IF_ERROR(ort_value_name_idx_map.GetIdx(entry.first, ort_value_index));
    if (use_user_supplied_initializer(entry.first)) {
      user_supplied_initializer_ids.insert(ort_value_index);
    }
    id_to_initialized_tensor[ort_value_index] = entry.second;
  }

  // tensors requiring a specific allocation order are traced first, to ensure they are allocated in order
  // NB1: vector with init allocation order may contain a subset of all tensors (or none at all)
  // NB2: only skip tracing and planning memory when data is external (i.e mmap) and on CPU.
  //    when data is external and on GPU, need to copy first to cpu memory, then to gpu memory.
  auto initialized_tensors_to_allocate = id_to_initialized_tensor;
  for (int ort_value_index : initializer_allocation_order) {
    const auto entry = initialized_tensors_to_allocate.find(ort_value_index);
    if (!(utils::HasExternalData(*entry->second) && exec_plan.GetLocation(ort_value_index).device.Type() == OrtDevice::CPU)) {
      // can not trace string tensor
      ORT_ENFORCE(entry != initialized_tensors_to_allocate.end() &&
                  entry->second->data_type() != ONNX_NAMESPACE::TensorProto_DataType_STRING);
      ORT_RETURN_IF_ERROR(planner.Trace(entry->first, entry->second));
    }
    initialized_tensors_to_allocate.erase(entry);
  }

  for (const auto& entry : initialized_tensors_to_allocate) {
    // We don't want to trace shared initializers since their memory is provided by the user
    if (user_supplied_initializer_ids.find(entry.first) != user_supplied_initializer_ids.end()) {
      continue;
    }
    if (entry.second->data_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
      // do not trace string tensor
      continue;
    }
    ORT_RETURN_IF_ERROR(planner.Trace(entry.first, entry.second));
  }
  // 2. allocate weight buffer on different locations
  //  planned_initializers_memory_size_in_byte is not actual physical size.
  //  It's the virtual size computed by planner.
  InlinedHashMap<std::string, size_t> planned_initializers_memory_sizes_in_byte;
  ORT_RETURN_IF_ERROR(
      planner.FinalizePlan(planned_initializers_memory_sizes_in_byte));

  if (memory_profile_func)
    memory_profile_func(planner);

  for (auto i : planned_initializers_memory_sizes_in_byte) {
    LOGS(logger, INFO) << "[Memory] SessionStateInitializer statically allocates "
                       << i.second << " bytes for " << i.first << std::endl;
  }

  OrtCallback deleter{nullptr, nullptr};

  // 3. create weight tensors based on weights buffer
  for (const auto& entry : id_to_initialized_tensor) {
    int ort_value_index = entry.first;
    const std::string& name = entry.second->name();

    if (name.empty()) {
      LOGS(logger, INFO) << "Skipping entry for missing optional value at idx " << ort_value_index;
      continue;
    }

    OrtValue ort_value;

    if (user_supplied_initializer_ids.find(entry.first) != user_supplied_initializer_ids.end()) {
      ort_value = *(session_options.initializers_to_share_map.at(name));
      LOGS(logger, INFO) << "Using user supplied initializer with name (" << name << ").";
    } else {
      const ONNX_NAMESPACE::TensorProto& tensor_proto = *(entry.second);

      std::optional<MemBuffer> m;
      AllocatorPtr alloc;
      // TODO: if the tensor need be copied, does it have enough room?
      ORT_RETURN_IF_ERROR(planner.GetPreallocatedBuffer(ort_value_index, name, m, alloc));
      bool use_device_allocator_for_initializers =
          session_options.config_options.GetConfigOrDefault(kOrtSessionOptionsUseDeviceAllocatorForInitializers, "0") == "1";

      Status st = DeserializeTensorProto(env, graph_loc, tensor_proto, (m.has_value()) ? &*m : nullptr, alloc,
                                         default_cpu_alloc, ort_value, data_transfer_mgr,
                                         use_device_allocator_for_initializers);
      if (!st.IsOK()) {
        std::ostringstream oss;
        oss << "Deserialize tensor " << name << " failed." << st.ErrorMessage();
        return Status(st.Category(), st.Code(), oss.str());
      }
    }

    // 'name' is a reference to a string within the TensorProto that save_tensor_func may free
    // so we need to output this message prior to calling save_tensor_func
    VLOGS(logger, 1) << "Adding weight with name : " << name << " with index: " << ort_value_index;

    // any outer scope value is shadowed by a local value and can't override it.
    // due to that check_outer_scope is false
    const bool constant = graph.IsConstantInitializer(name, /* check_outer_scope */ false);
#if !defined(DISABLE_SPARSE_TENSORS)
    const bool sparse = graph.GetGraph().IsSparseInitializer(name);
    ORT_RETURN_IF_ERROR(save_tensor_func(name, ort_value_index, ort_value, deleter, constant, sparse));
#else
    ORT_RETURN_IF_ERROR(save_tensor_func(name, ort_value_index, ort_value, deleter, constant, false));
#endif
  }

  LOGS(logger, INFO) << "Done saving initialized tensors";
  return common::Status::OK();
}

template <typename T>  // T is container of const NodeArg* or NodeArg*
static bool IsArgNameInInputsOutputs(const std::string& name,
                                     const T& graph_args) {
  auto it = std::find_if(graph_args.cbegin(), graph_args.cend(),
                         [&name](const onnxruntime::NodeArg* arg) {
                           return arg->Name() == name;
                         });
  return it != graph_args.cend();
}

common::Status SaveInputOutputNamesToNodeMapping(const onnxruntime::GraphViewer& graph,
                                                 SessionState& session_state,
                                                 gsl::span<const NodeArg* const> implicit_inputs) {
  auto& graph_inputs = graph.GetInputsIncludingInitializers();
  auto& graph_outputs = graph.GetOutputs();

  const auto* exec_plan = session_state.GetExecutionPlan();
  const auto& name_to_id = session_state.GetOrtValueNameIdxMap();

  for (auto& node : graph.Nodes()) {
    const KernelCreateInfo& kci = session_state.GetNodeKernelCreateInfo(node.Index());

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

              SessionState::NodeInfo node_info(index, &node, &kci, device);

              if (IsArgNameInInputsOutputs(arg.Name(), graph_inputs)) {
                ORT_RETURN_IF_ERROR(session_state.AddInputNameToNodeInfoMapping(arg.Name(), node_info));
                return Status::OK();
              }

              if (!implicit_inputs.empty()) {
                if (IsArgNameInInputsOutputs(arg.Name(), implicit_inputs)) {
                  ORT_RETURN_IF_ERROR(session_state.AddInputNameToNodeInfoMapping(arg.Name(), node_info));
                  return Status::OK();
                }
              }

              return Status::OK();
            }));

    // implicit inputs to a node could come directly from a feed, so we need to make sure they have an entry too
    const auto& node_implicit_inputs = node.ImplicitInputDefs();
    if (!node_implicit_inputs.empty()) {
      // In the main graph, the location of the implicit input(s) is the location it
      // is consumed in the main graph if there is an explicit consumer.
      // If the only consumer(s) are implicit consumers (i.e.) other control flow nodes and
      // all of them have been partitioned to the same EP, its location is the
      // location of the non-CPU device corresponding to the EP.
      // If multiple EPs are involved, then the planned location for such implicit inputs
      // just default to CPU (as there is ambiguity involved as to which non-CPU device is
      // most optimal)

      // In nested subgraphs, the location of the implicit input(s) is the location it
      // is consumed in the subgraph if there is an explicit consumer.
      // If the only consumer(s) are implicit consumers (i.e.) other control flow nodes, its
      // location is the location of the value in the enclosing outer scope.

      // All this is setup in the planner, we just use the location from the plan here.
      for (const auto& input_def : node_implicit_inputs) {
        int arg_index;
        ORT_RETURN_IF_ERROR(name_to_id.GetIdx(input_def->Name(), arg_index));
        auto& device = exec_plan->GetLocation(arg_index).device;
        SessionState::NodeInfo node_info(std::numeric_limits<size_t>::max(), &node, &kci, device);
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

              SessionState::NodeInfo node_info(index, &node, &kci, device);

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

}  // namespace session_state_utils
}  // namespace onnxruntime
