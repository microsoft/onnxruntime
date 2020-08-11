// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_state.h"

#include <sstream>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/framework/node_index_info.h"
#include "core/framework/op_kernel.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/session_state_utils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/controlflow/utils.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

void SessionState::SetupAllocators() {
  for (const auto& provider : execution_providers_) {
    for (const auto& allocator : provider->GetAllocators()) {
      const OrtMemoryInfo& memory_info = allocator->Info();
      if (allocators_.find(memory_info) != allocators_.end()) {
        // EPs are ordered by priority so ignore the duplicate allocator for this memory location.
        LOGS(logger_, INFO) << "Allocator already registered for " << allocator->Info()
                            << ". Ignoring allocator from " << provider->Type();
      } else {
        // slightly weird indirection to go back to the provider to get the allocator each time it's needed
        // in order to support scenarios such as the CUDA EP's per-thread allocator.
        allocators_[memory_info] = [&provider](int id, OrtMemType mem_type) {
          return provider->GetAllocator(id, mem_type);
        };
      }
    }
  }
}

AllocatorPtr SessionState::GetAllocator(const OrtMemoryInfo& location) const noexcept {
  AllocatorPtr result;
  auto entry = allocators_.find(location);
  if (entry != allocators_.cend()) {
    result = entry->second(location.id, location.mem_type);
  }

  return result;
}

AllocatorPtr SessionState::GetAllocator(OrtDevice device) const noexcept {
  AllocatorPtr result;

  using AllocatorEntry = std::map<OrtMemoryInfo, std::function<AllocatorPtr(int id, OrtMemType mem_type)>,
                                  OrtMemoryInfoLessThanIgnoreAllocType>::const_reference;

  auto entry = std::find_if(allocators_.cbegin(), allocators_.cend(),
                            [device](AllocatorEntry& entry) {
                              return entry.first.device == device &&
                                     entry.first.mem_type == OrtMemTypeDefault;
                            });

  if (entry != allocators_.cend()) {
    result = entry->second(device.Id(), OrtMemTypeDefault);
  }

  return result;
}

void SessionState::CreateGraphInfo() {
  graph_viewer_ = onnxruntime::make_unique<onnxruntime::GraphViewer>(graph_);
  // use graph_viewer_ to initialize ort_value_name_idx_map_
  LOGS(logger_, VERBOSE) << "SaveMLValueNameIndexMapping";
  int idx = 0;

  // we keep all graph inputs (including initializers), even if they are unused, so make sure they all have an entry
  for (const auto* input_def : graph_viewer_->GetInputsIncludingInitializers()) {
    idx = ort_value_name_idx_map_.Add(input_def->Name());
    VLOGS(logger_, 1) << "Added graph_viewer_ input with name: " << input_def->Name()
                      << " to OrtValueIndex with index: " << idx;
  }

  for (auto& node : graph_viewer_->Nodes()) {
    // build the OrtValue->index map
    for (const auto* input_def : node.InputDefs()) {
      if (input_def->Exists()) {
        idx = ort_value_name_idx_map_.Add(input_def->Name());
        VLOGS(logger_, 1) << "Added input argument with name: " << input_def->Name()
                          << " to OrtValueIndex with index: " << idx;
      }
    }

    for (const auto* input_def : node.ImplicitInputDefs()) {
      if (input_def->Exists()) {
        idx = ort_value_name_idx_map_.Add(input_def->Name());
        VLOGS(logger_, 1) << "Added implicit input argument with name: " << input_def->Name()
                          << " to OrtValueIndex with index: " << idx;
      }
    }

    for (const auto* output_def : node.OutputDefs()) {
      if (output_def->Exists()) {
        ort_value_name_idx_map_.Add(output_def->Name());
        VLOGS(logger_, 1) << "Added output argument with name: " << output_def->Name()
                          << " to OrtValueIndex with index: " << idx;
      }
    }
  }

  // allocate OrtValue for graph outputs when coming from initializers
  for (const auto& output : graph_viewer_->GetOutputs()) {
    if (output->Exists()) {
      idx = ort_value_name_idx_map_.Add(output->Name());
      VLOGS(logger_, 1) << "Added graph output with name: " << output->Name() << " to OrtValueIndex with index: " << idx;
    }
  }

  LOGS(logger_, VERBOSE) << "Done saving OrtValue mappings.";
}

Status SessionState::PopulateKernelCreateInfo(KernelRegistryManager& kernel_registry_manager) {
  for (auto& node : graph_.Nodes()) {
    const KernelCreateInfo* kci = nullptr;
    ORT_RETURN_IF_ERROR(kernel_registry_manager.SearchKernelRegistry(node, &kci));
    ORT_IGNORE_RETURN_VALUE(
        kernel_create_info_map_.insert({node.Index(), gsl::not_null<const KernelCreateInfo*>(kci)}));
  }

  for (const auto& entry : subgraph_session_states_) {
    for (const auto& name_to_subgraph_session_state : entry.second) {
      SessionState& subgraph_session_state = *name_to_subgraph_session_state.second;
      ORT_RETURN_IF_ERROR(subgraph_session_state.PopulateKernelCreateInfo(kernel_registry_manager));
    }
  }

  return Status::OK();
}

const KernelCreateInfo& SessionState::GetNodeKernelCreateInfo(NodeIndex node_index) const {
  auto entry = kernel_create_info_map_.find(node_index);
  // invalid node index or FinalizeSessionState should have been called. Either way it's an internal logic error
  ORT_ENFORCE(entry != kernel_create_info_map_.cend());

  return *entry->second;
}

Status SessionState::CreateKernels(const KernelRegistryManager& kernel_registry_manager) {
  const GraphNodes& nodes = graph_viewer_->Nodes();
  if (!nodes.empty()) {
    size_t max_nodeid = 0;
    for (auto& node : graph_viewer_->Nodes()) {
      max_nodeid = std::max(max_nodeid, node.Index());
    }
    session_kernels_.clear();
    session_kernels_.resize(max_nodeid + 1, nullptr);
    for (auto& node : graph_viewer_->Nodes()) {
      // construct and save the kernels
      const KernelCreateInfo& kci = GetNodeKernelCreateInfo(node.Index());

      // the execution provider was required to be valid to find the KernelCreateInfo so we don't need to check it here
      onnxruntime::ProviderType exec_provider_name = node.GetExecutionProviderType();
      const IExecutionProvider& exec_provider = *execution_providers_.Get(exec_provider_name);

      auto op_kernel = kernel_registry_manager.CreateKernel(node, exec_provider, *this, kci);

      // assumes vector is already resize()'ed to the number of nodes in the graph
      session_kernels_[node.Index()] = op_kernel.release();
    }
  }
  node_index_info_ = onnxruntime::make_unique<NodeIndexInfo>(*graph_viewer_, ort_value_name_idx_map_);
  return Status::OK();
}

const SequentialExecutionPlan* SessionState::GetExecutionPlan() const { return p_seq_exec_plan_.get(); }

Status SessionState::AddInitializedTensor(int ort_value_index, const OrtValue& ort_value, const OrtCallback* d,
                                          bool constant) {
  auto p = initialized_tensors_.insert({ort_value_index, ort_value});
  if (!p.second)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "duplicated ort_value index:", ort_value_index,
                           ". Do you have duplicated calls to SessionState::AddInitializedTensor function?");

  if (d != nullptr && d->f != nullptr) {
    deleter_for_initialized_tensors_[ort_value_index] = *d;
  }

  if (constant) {
    constant_initialized_tensors_.insert({ort_value_index, ort_value});
  }

  return Status::OK();
}

const std::unordered_map<int, OrtValue>& SessionState::GetInitializedTensors() const { return initialized_tensors_; }

const std::unordered_map<int, OrtValue>& SessionState::GetConstantInitializedTensors() const {
  return constant_initialized_tensors_;
}

#ifdef ENABLE_TRAINING
Status SessionState::GetInitializedTensors(
    const std::unordered_set<std::string>& interested_weights,
    bool allow_missing_weights, NameMLValMap& retrieved_weights) const {
  NameMLValMap result;
  result.reserve(interested_weights.size());
  for (const auto& weight_name : interested_weights) {
    int idx;
    const auto status = GetOrtValueNameIdxMap().GetIdx(weight_name, idx);
    if (!status.IsOK()) {
      ORT_RETURN_IF_NOT(
          allow_missing_weights,
          "Failed to get OrtValue index from name: ", status.ErrorMessage());
      continue;
    }
    result.emplace(weight_name, initialized_tensors_.at(idx));
  }
  retrieved_weights = std::move(result);
  return Status::OK();
}

NameMLValMap SessionState::GetInitializedTensors(const std::unordered_set<std::string>& interested_weights) const {
  NameMLValMap result;
  const auto status = GetInitializedTensors(interested_weights, true, result);
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
  return result;
}
#endif

void SessionState::CleanInitializedTensorsFromGraph() {
  graph_.CleanAllInitializedTensors();
}

Status SessionState::PrepackInitializedConstantTensors() {
  // calculate the use count of each value
  std::unordered_map<std::string, size_t> node_arg_use_count;
  for (const auto& node : GetGraphViewer().Nodes()) {
    node.ForEachDef([&](const onnxruntime::NodeArg& node_arg, bool is_input) {
      if (is_input) {
        node_arg_use_count[node_arg.Name()]++;
      }
    });
  }

  for (auto& node : GetGraphViewer().Nodes()) {
    auto kernel = GetMutableKernel(node.Index());
    int input_idx = 0;
    for (auto& input_def : node.InputDefs()) {
      if (input_def->Exists()) {
        const std::string& input_name = input_def->Name();
        int ort_value_idx;
        ORT_RETURN_IF_ERROR(ort_value_name_idx_map_.GetIdx(input_name, ort_value_idx));
        if (constant_initialized_tensors_.count(ort_value_idx) &&
            constant_initialized_tensors_[ort_value_idx].IsTensor()) {
          bool is_packed = false;
          const Tensor& const_initialized_tensor = constant_initialized_tensors_[ort_value_idx].Get<Tensor>();
          ORT_RETURN_IF_ERROR(kernel->PrePack(const_initialized_tensor, input_idx, is_packed));
          if (is_packed && node_arg_use_count.count(input_name) && --node_arg_use_count[input_name] == 0) {
            // release the constant intialized tensor
            initialized_tensors_.erase(ort_value_idx);
            constant_initialized_tensors_.erase(ort_value_idx);
          }
        }
      }
      input_idx++;
    }
  }

  return Status::OK();
}

static int64_t CalculateMemoryPatternsKey(const std::vector<std::reference_wrapper<const TensorShape>>& shapes) {
  int64_t key = 0;
  for (auto shape : shapes) {
    for (auto dim : shape.get().GetDims()) key ^= dim;
  }
  return key;
}

#ifdef ENABLE_TRAINING
namespace {
Status ResolveDimParams(const GraphViewer& graph,
                        const std::map<std::string, TensorShape>& feeds,
                        std::unordered_map<std::string, int64_t>& out) {
  for (const auto* input : graph.GetInputs()) {
    auto* shape = input->Shape();
    auto it = feeds.find(input->Name());
    if (it == feeds.end()) {
      return Status(ONNXRUNTIME, FAIL,
                    "Graph input " + input->Name() +
                        " is not found in the feed list, unable to resolve the value for dynamic shape.");
    }
    if (it->second.NumDimensions() == 0 && !shape) {
      // This is a scalar, which has nothing to do with symbolic shapes.
      continue;
    }
    if (!shape || shape->dim_size() != static_cast<int>(it->second.NumDimensions())) {
      return Status(ONNXRUNTIME, FAIL, "Graph input " + input->Name() +
                                           "'s shape is not present or its shape doesn't match feed's shape."
                                           "Unable to resolve the value for dynamic shape");
    }
    for (int k = 0, end = shape->dim_size(); k < end; ++k) {
      if (shape->dim()[k].has_dim_param()) {
        out.insert({shape->dim()[k].dim_param(), it->second.GetDims()[k]});
      }
    }
  }
  return Status::OK();
}

Status ResolveSizeAndShape(
    const NodeArg* arg,
    const std::unordered_map<std::string, int64_t>& symbolic_dimensions,
    size_t& size,  // total number of elements. It's 0 if shape is unknown.
    std::vector<int64_t>& resolved_shape) {
  if (!arg->Shape()) {
    // 0 means no shape information.
    size = 0;
    return Status::OK();
  }

  std::vector<int64_t> shape;

  SafeInt<size_t> safe_size = 1;
  for (auto& dim : arg->Shape()->dim()) {
    if (dim.has_dim_param()) {
      auto it = symbolic_dimensions.find(dim.dim_param());
      if (it == symbolic_dimensions.end()) {
        return Status(ONNXRUNTIME, FAIL, "Unknown symbolic dimension, " + dim.dim_param() + ", found in memory pattern compute.");
      }
      safe_size *= it->second;
      shape.push_back(it->second);
    } else if (dim.has_dim_value() && dim.dim_value() > 0) {
      safe_size *= dim.dim_value();
      shape.push_back(dim.dim_value());
    } else {
      // tensor shape is unknown.
      safe_size = 0;
    }
  }

  size = safe_size;

  // Only assign shape if all symbolic dimensions are resolved.
  if (size != 0) {
    resolved_shape = std::move(shape);
  }

  return Status::OK();
}
}  // namespace

Status SessionState::GeneratePatternGroupCache(const std::vector<std::reference_wrapper<const TensorShape>>& input_shape,
                                               const std::vector<int>& feed_mlvalue_idxs,
                                               MemoryPatternGroup* output,
                                               std::unordered_map<int, TensorShape>& resolved_shapes) const {
  std::map<std::string, TensorShape> feeds;
  for (size_t i = 0, end = feed_mlvalue_idxs.size(); i < end; ++i) {
    std::string name;
    ORT_RETURN_IF_ERROR(this->ort_value_name_idx_map_.GetName(feed_mlvalue_idxs[i], name));
    feeds.insert({name, input_shape[i]});
  }
  std::unordered_map<std::string, int64_t> map;
  ORT_RETURN_IF_ERROR(ResolveDimParams(*graph_viewer_, feeds, map));
  auto* exe_plan = GetExecutionPlan();
  ORT_ENFORCE(exe_plan);
  OrtValuePatternPlanner mem_planner(*exe_plan);
  auto& node_index_info = GetNodeIndexInfo();
  for (auto& node_plan : exe_plan->execution_plan) {
    int node_index = node_index_info.GetNodeOffset(node_plan.node_index);
    auto* node = graph_viewer_->GetNode(node_plan.node_index);
    int output_start = node_index + static_cast<int>(node->InputDefs().size()) + static_cast<int>(node->ImplicitInputDefs().size());
    //allocate output
    for (int i = 0, end = static_cast<int>(node->OutputDefs().size()); i < end; ++i) {
      const auto ml_value_idx = node_index_info.GetMLValueIndex(output_start + i);
      if (ml_value_idx == NodeIndexInfo::kInvalidEntry)
        continue;
      const auto* ml_type = exe_plan->allocation_plan[ml_value_idx].value_type;
      if (!ml_type->IsTensorType())
        continue;
      const auto* ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();

      auto* arg = node->OutputDefs()[i];
      size_t size = 0;
      std::vector<int64_t> resolved_shape;
      ORT_RETURN_IF_ERROR(ResolveSizeAndShape(arg, map, size, resolved_shape));

      // Store all valid resolved shapes. They will be queried in, for example,
      // Recv operator to bypass the dependency of output shapes on inputs.
      if (size != 0) {
        resolved_shapes[ml_value_idx] = resolved_shape;
      }

      // Plan memory if conditions are met.
      if (exe_plan->allocation_plan[ml_value_idx].alloc_kind == AllocKind::kAllocate &&
          ml_data_type != DataTypeImpl::GetType<std::string>() && size != 0) {
        size_t aligned_size = 0;
        if (!IAllocator::CalcMemSizeForArrayWithAlignment<64>(size, ml_data_type->Size(), &aligned_size)) {
          return Status(ONNXRUNTIME, FAIL, "Size overflow");
        }

        mem_planner.TraceAllocation(ml_value_idx, aligned_size);
      }
    }
    //release nodes
    for (int index = node_plan.free_from_index; index <= node_plan.free_to_index; ++index) {
      auto ml_value_idx = exe_plan->to_be_freed[index];
      const auto* ml_type = exe_plan->allocation_plan[ml_value_idx].value_type;
      if (!ml_type->IsTensorType())
        continue;
      const auto* ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
      if (ml_data_type != DataTypeImpl::GetType<std::string>()) {
        mem_planner.TraceFree(ml_value_idx);
      }
    }
  }

  if (!mem_planner.GeneratePatterns(output).IsOK()) {
    return Status(ONNXRUNTIME, FAIL, "Generate Memory Pattern failed");
  }
  return Status::OK();
}
#endif

const MemoryPatternGroup* SessionState::GetMemoryPatternGroup(const std::vector<std::reference_wrapper<const TensorShape>>& input_shapes,
                                                              const std::vector<int>& feed_mlvalue_idxs,
                                                              std::unordered_map<int, TensorShape>& inferred_shapes) const {
  int64_t key = CalculateMemoryPatternsKey(input_shapes);

  std::lock_guard<OrtMutex> lock(mem_patterns_lock_);
  auto it = mem_patterns_.find(key);
  if (it == mem_patterns_.end()) {
#ifdef ENABLE_TRAINING
    auto mem_patterns = onnxruntime::make_unique<MemoryPatternGroup>();
    if (GeneratePatternGroupCache(input_shapes, feed_mlvalue_idxs, mem_patterns.get(), inferred_shapes).IsOK()) {
      key = CalculateMemoryPatternsKey(input_shapes);
      auto ptr = mem_patterns.get();
      mem_patterns_[key] = std::move(mem_patterns);
      shape_patterns_[key] = inferred_shapes;
      return ptr;
    }
    return nullptr;
#else
    ORT_UNUSED_PARAMETER(feed_mlvalue_idxs);
    return nullptr;
#endif
  }

  inferred_shapes = shape_patterns_[key];
  return it->second.get();
}

void SessionState::ResolveMemoryPatternFlag() {
  if (enable_mem_pattern_) {
    for (auto* input : graph_viewer_->GetInputs()) {
      if (!input->HasTensorOrScalarShape()) {
        enable_mem_pattern_ = false;
        break;
      }
    }
  }
}

Status SessionState::UpdateMemoryPatternGroupCache(const std::vector<std::reference_wrapper<const TensorShape>>& input_shapes,
                                                   std::unique_ptr<MemoryPatternGroup> mem_patterns) const {
  int64_t key = CalculateMemoryPatternsKey(input_shapes);

  std::lock_guard<OrtMutex> lock(mem_patterns_lock_);
  auto it = mem_patterns_.find(key);
  if (it == mem_patterns_.end()) {
    mem_patterns_[key] = std::move(mem_patterns);
  }

  return Status::OK();
}

bool SessionState::GetEnableMemoryPattern() const { return enable_mem_pattern_; }

common::Status SessionState::AddInputNameToNodeInfoMapping(const std::string& input_name, const NodeInfo& node_info) {
  // Graph partitioning should ensure an input is only consumed from one device. Copy nodes should have been inserted
  // to handle a scenario where an input is required on different devices by different nodes. Validate that.
  auto& entries = input_names_to_nodeinfo_mapping_[input_name];

  if (entries.empty()) {
    entries.push_back(node_info);
  } else {
    const auto& existing_entry = entries.front();

    // if index == max it's an entry for an implicit input to a subgraph or unused graph input.
    // we want to prefer the entry for explicit usage in this graph, as the implicit usage in a
    // subgraph will be handled by the subgraph's SessionState.
    if (node_info.index == std::numeric_limits<size_t>::max()) {
      // ignore and preserve existing value
    } else if (existing_entry.index == std::numeric_limits<size_t>::max()) {
      // replace existing entry that is for an implicit input with new entry for explicit usage in this graph
      entries[0] = node_info;
    } else {
      // if the devices match we can add the new entry for completeness (it will be ignored in
      // utils::CopyOneInputAcrossDevices though).
      // if they don't, we are broken.
      const auto& current_device = entries[0].device;
      const auto& new_device = node_info.device;

      if (current_device == new_device) {
        entries.push_back(node_info);
      } else {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, NOT_IMPLEMENTED,
            "Using an input in multiple nodes on different devices is not supported currently. Input:", input_name,
            " is used by node ", existing_entry.p_node->Name(), " (", current_device->ToString(), ") and node ",
            node_info.p_node->Name(), " (", new_device->ToString(), ").");
      }
    }
  }

  return Status::OK();
}

common::Status SessionState::GetInputNodeInfo(const std::string& input_name,
                                              std::vector<NodeInfo>& node_info_vec) const {
  auto entry = input_names_to_nodeinfo_mapping_.find(input_name);
  if (entry == input_names_to_nodeinfo_mapping_.cend()) {
    return Status(ONNXRUNTIME, FAIL, "Failed to find input name in the mapping: " + input_name);
  }

  node_info_vec = entry->second;
  return Status::OK();
}

const SessionState::NameNodeInfoMapType& SessionState::GetInputNodeInfoMap() const {
  return input_names_to_nodeinfo_mapping_;
}

void SessionState::AddOutputNameToNodeInfoMapping(const std::string& output_name, const NodeInfo& node_info) {
  auto& output_names_to_nodeinfo = output_names_to_nodeinfo_mapping_[output_name];
  ORT_ENFORCE(output_names_to_nodeinfo.empty(), "Only one node should produce an output. Existing entry for ",
              output_name);

  output_names_to_nodeinfo.push_back(node_info);
}

common::Status SessionState::GetOutputNodeInfo(const std::string& output_name,
                                               std::vector<NodeInfo>& node_info_vec) const {
  auto entry = output_names_to_nodeinfo_mapping_.find(output_name);
  if (entry == output_names_to_nodeinfo_mapping_.cend()) {
    return Status(ONNXRUNTIME, FAIL, "Failed to find output name in the mapping: " + output_name);
  }

  node_info_vec = entry->second;
  return Status::OK();
}

const SessionState::NameNodeInfoMapType& SessionState::GetOutputNodeInfoMap() const {
  return output_names_to_nodeinfo_mapping_;
}

void SessionState::AddSubgraphSessionState(onnxruntime::NodeIndex index, const std::string& attribute_name,
                                           std::unique_ptr<SessionState> session_state) {
  auto entry = subgraph_session_states_.find(index);

  // make sure this is new. internal logic error if it is not so using ORT_ENFORCE.
  if (entry != subgraph_session_states_.cend()) {
    const auto& existing_entries = entry->second;
    ORT_ENFORCE(existing_entries.find(attribute_name) == existing_entries.cend(), "Entry exists in node ", index,
                " for attribute ", attribute_name);
  }
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  session_state->parent_ = this;
  GenerateGraphId();
#endif
  subgraph_session_states_[index].insert(std::make_pair(attribute_name, std::move(session_state)));
}

SessionState* SessionState::GetMutableSubgraphSessionState(onnxruntime::NodeIndex index,
                                                           const std::string& attribute_name) {
  SessionState* session_state = nullptr;

  auto node_entry = subgraph_session_states_.find(index);
  if (node_entry != subgraph_session_states_.cend()) {
    const auto& attribute_state_map = node_entry->second;

    const auto& subgraph_entry = attribute_state_map.find(attribute_name);
    if (subgraph_entry != attribute_state_map.cend()) {
      session_state = subgraph_entry->second.get();
    }
  }

  return session_state;
}

const SessionState* SessionState::GetSubgraphSessionState(onnxruntime::NodeIndex index,
                                                          const std::string& attribute_name) const {
  return const_cast<SessionState*>(this)->GetMutableSubgraphSessionState(index, attribute_name);
}

void SessionState::RemoveSubgraphSessionState(onnxruntime::NodeIndex index) {
  subgraph_session_states_.erase(index);
}

const NodeIndexInfo& SessionState::GetNodeIndexInfo() const {
  ORT_ENFORCE(node_index_info_, "SetGraphAndCreateKernels must be called prior to GetExecutionInfo.");
  return *node_index_info_;
}

void SessionState::UpdateToBeExecutedNodes(const std::vector<int>& fetch_mlvalue_idxs) {
  std::vector<int> sorted_idxs = fetch_mlvalue_idxs;
  std::sort(sorted_idxs.begin(), sorted_idxs.end());
  if (to_be_executed_nodes_.find(sorted_idxs) != to_be_executed_nodes_.end())
    return;

  // Get the nodes generating the fetches.
  std::vector<const Node*> nodes;
  nodes.reserve(fetch_mlvalue_idxs.size());
  std::unordered_set<NodeIndex> reachable_nodes;

  for (auto idx : fetch_mlvalue_idxs) {
    std::string node_arg_name;
    const auto status = this->GetOrtValueNameIdxMap().GetName(idx, node_arg_name);
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    auto ending_node = graph_.GetProducerNode(node_arg_name);
    nodes.push_back(ending_node);
  }

  // Reversely traverse to get reachable nodes.
  graph_.ReverseDFSFrom(
      nodes, {}, [&reachable_nodes](const Node* n) { reachable_nodes.insert(n->Index()); });
  to_be_executed_nodes_.insert(std::make_pair(sorted_idxs, reachable_nodes));
}

const std::unordered_set<NodeIndex>* SessionState::GetToBeExecutedNodes(
    const std::vector<int>& fetch_mlvalue_idxs) const {
  std::vector<int> sorted_idxs = fetch_mlvalue_idxs;
  std::sort(sorted_idxs.begin(), sorted_idxs.end());
  auto it = to_be_executed_nodes_.find(sorted_idxs);
  return (it != to_be_executed_nodes_.end()) ? &it->second : nullptr;
}

Status SessionState::FinalizeSessionState(const std::basic_string<PATH_CHAR_TYPE>& graph_location,
                                          KernelRegistryManager& kernel_registry_manager,
                                          const SessionOptions& session_options,
                                          bool remove_initializers) {
  return FinalizeSessionStateImpl(graph_location, kernel_registry_manager, nullptr, session_options,
                                  remove_initializers);
}

Status SessionState::FinalizeSessionStateImpl(const std::basic_string<PATH_CHAR_TYPE>& graph_location,
                                              KernelRegistryManager& kernel_registry_manager,
                                              _In_opt_ const Node* parent_node,
                                              const SessionOptions& session_options,
                                              bool remove_initializers) {
  // recursively populate the kernel create info.
  // it's simpler to do recursively when deserializing,
  // so also do it recursively when calling PopulateKernelCreateInfo for consistency
  if (parent_node == nullptr) {
    ORT_RETURN_IF_ERROR(PopulateKernelCreateInfo(kernel_registry_manager));
  }

  CreateGraphInfo();

  // ignore any outer scope args we don't know about. this can happen if a node contains multiple subgraphs.
  std::vector<const NodeArg*> valid_outer_scope_node_args;
  if (parent_node) {
    auto outer_scope_node_args = parent_node->ImplicitInputDefs();
    valid_outer_scope_node_args.reserve(outer_scope_node_args.size());

    std::for_each(outer_scope_node_args.cbegin(), outer_scope_node_args.cend(),
                  [this, &valid_outer_scope_node_args](const NodeArg* node_arg) {
                    int idx;
                    if (ort_value_name_idx_map_.GetIdx(node_arg->Name(), idx).IsOK()) {
                      valid_outer_scope_node_args.push_back(node_arg);
                    };
                  });
  }

  SequentialPlannerContext context(session_options.execution_mode);
  ORT_RETURN_IF_ERROR(SequentialPlanner::CreatePlan(parent_node, *graph_viewer_, valid_outer_scope_node_args,
                                                    execution_providers_, kernel_create_info_map_,
                                                    ort_value_name_idx_map_, context, p_seq_exec_plan_));

  // Uncomment the below to dump the allocation plan to std::cout
  // LOGS(logger_, VERBOSE) << std::make_pair(p_seq_exec_plan_.get(), this);

  std::unique_ptr<ITensorAllocator> tensor_allocator_(
      ITensorAllocator::Create(enable_mem_pattern_, *p_seq_exec_plan_, *this, weights_buffers_));

  // move initializers from TensorProto instances in Graph to OrtValue instances in SessionState
  ORT_RETURN_IF_ERROR(
      session_state_utils::SaveInitializedTensors(
          Env::Default(), graph_location, *graph_viewer_,
          execution_providers_.GetDefaultCpuMemoryInfo(),
          ort_value_name_idx_map_, *tensor_allocator_,
          [this](int idx, const OrtValue& value, const OrtCallback& d, bool constant) -> Status {
            return AddInitializedTensor(idx, value, &d, constant);
          },
          logger_, data_transfer_mgr_));

  // remove weights from the graph now to save memory but in many cases it won't save memory, if the tensor was
  // preallocated with the some other tensors in a single 'allocate' call, which is very common.
  // TODO: make it better
  if (remove_initializers) {
    CleanInitializedTensorsFromGraph();
  }

  ORT_RETURN_IF_ERROR(CreateKernels(kernel_registry_manager));

  if (session_options.use_prepacking) {
    ORT_RETURN_IF_ERROR(PrepackInitializedConstantTensors());
  }

  ORT_RETURN_IF_ERROR(
      session_state_utils::SaveInputOutputNamesToNodeMapping(*graph_viewer_, *this, valid_outer_scope_node_args));

  // Need to recurse into subgraph session state to finalize
  // and add the execution info

  // Currently all subgraphs need to be executed using the sequential EP due to potential deadlock with the current
  // parallel executor implementation
  SessionOptions subgraph_session_options(session_options);
  subgraph_session_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;

  for (const auto& node_to_subgraph_ss : subgraph_session_states_) {
    Node& node = *graph_.GetNode(node_to_subgraph_ss.first);

    // We only need subgraph session state for control flow nodes being handled by our CPU or CUDA execution provider.
    // Remove it if it's not needed.
    const auto ep = node.GetExecutionProviderType();
    if (ep != kCpuExecutionProvider && ep != kCudaExecutionProvider) {
      RemoveSubgraphSessionState(node_to_subgraph_ss.first);
      continue;
    }

    for (const auto& attr_subgraph_pair : node.GetAttributeNameToMutableSubgraphMap()) {
      auto& attr_name = attr_subgraph_pair.first;
      auto entry = node_to_subgraph_ss.second.find(attr_name);
      // InferenceSession::CreateSubgraphSessionState should ensure all these entries are created
      ORT_ENFORCE(entry != node_to_subgraph_ss.second.cend(),
                  "Missing session state for subgraph. Node:'", node.Name(),
                  "' OpType:", node.OpType(), " Index:", node.Index(), " Attribute:", attr_name);

      SessionState& subgraph_session_state = *entry->second;

      // recurse
      ORT_RETURN_IF_ERROR(subgraph_session_state.FinalizeSessionStateImpl(
          graph_location, kernel_registry_manager, &node, subgraph_session_options, remove_initializers));

      // setup all the info for handling the feeds and fetches used in subgraph execution
      auto* p_op_kernel = GetMutableKernel(node.Index());
      ORT_ENFORCE(p_op_kernel);

      // Downcast is safe, since only control flow nodes have subgraphs (node.GetAttributeNameToMutableSubgraphMap() is non-empty)
      auto& control_flow_kernel = static_cast<controlflow::IControlFlowKernel&>(*p_op_kernel);
      ORT_RETURN_IF_ERROR(control_flow_kernel.SetupSubgraphExecutionInfo(*this, attr_name, subgraph_session_state));
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
