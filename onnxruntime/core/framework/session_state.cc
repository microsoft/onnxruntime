// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_state.h"

#include <sstream>

#include "core/common/logging/logging.h"
#include "core/framework/node_index_info.h"
#include "core/framework/op_kernel.h"
#include "core/framework/utils.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/allocator.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {

const GraphViewer* SessionState::GetGraphViewer() const { return graph_viewer_.get(); }
Status SessionState::SetGraph(const Graph& graph) {
  graph_viewer_ = onnxruntime::make_unique<onnxruntime::GraphViewer>(graph);
  auto& logger = Logger();
  // use graph_viewer_ to initialize ort_value_name_idx_map_
  LOGS(logger, INFO) << "SaveMLValueNameIndexMapping";
  int idx = 0;

  // we keep all graph inputs (including initializers), even if they are unused, so make sure they all have an entry
  for (const auto* input_def : graph_viewer_->GetInputsIncludingInitializers()) {
    idx = ort_value_name_idx_map_.Add(input_def->Name());
    VLOGS(logger, 1) << "Added graph_viewer_ input with name: " << input_def->Name()
                     << " to OrtValueIndex with index: " << idx;
  }

  for (auto& node : graph_viewer_->Nodes()) {
    // build the OrtValue->index map
    for (const auto* input_def : node.InputDefs()) {
      if (input_def->Exists()) {
        idx = ort_value_name_idx_map_.Add(input_def->Name());
        VLOGS(logger, 1) << "Added input argument with name: " << input_def->Name()
                         << " to OrtValueIndex with index: " << idx;
      }
    }

    for (const auto* input_def : node.ImplicitInputDefs()) {
      if (input_def->Exists()) {
        idx = ort_value_name_idx_map_.Add(input_def->Name());
        VLOGS(logger, 1) << "Added implicit input argument with name: " << input_def->Name()
                         << " to OrtValueIndex with index: " << idx;
      }
    }

    for (const auto* output_def : node.OutputDefs()) {
      if (output_def->Exists()) {
        ort_value_name_idx_map_.Add(output_def->Name());
        VLOGS(logger, 1) << "Added output argument with name: " << output_def->Name()
                         << " to OrtValueIndex with index: " << idx;
      }
    }
  }

  // allocate OrtValue for graph outputs when coming from initializers
  for (const auto& output : graph_viewer_->GetOutputs()) {
    if (output->Exists()) {
      idx = ort_value_name_idx_map_.Add(output->Name());
      VLOGS(logger, 1) << "Added graph output with name: " << output->Name() << " to OrtValueIndex with index: " << idx;
    }
  }

  LOGS(logger, INFO) << "Done saving OrtValue mappings.";
  return Status::OK();
}

Status SessionState::CreateKernels(const KernelRegistryManager& custom_registry_manager) {
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
      std::unique_ptr<OpKernel> op_kernel;
      onnxruntime::ProviderType exec_provider_name = node.GetExecutionProviderType();

      const IExecutionProvider* exec_provider = nullptr;
      if (exec_provider_name.empty() || (exec_provider = execution_providers_.get().Get(exec_provider_name)) == nullptr) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Could not create kernel for node: ", node.Name(),
                               " as there's no execution provider allocated.");
      }

      common::Status status = custom_registry_manager.CreateKernel(node, *exec_provider, *this, op_kernel);
      if (!status.IsOK()) {
        return common::Status(
            status.Category(), status.Code(),
            MakeString("Kernel creation failed for node: ", node.Name(), " with error: ", status.ErrorMessage()));
      }
      assert(session_kernels_[node.Index()] == nullptr);
      // assumes vector is already resize()'ed to the number of nodes in the graph
      session_kernels_[node.Index()] = op_kernel.release();
    }
  }
  node_index_info_ = onnxruntime::make_unique<NodeIndexInfo>(*graph_viewer_, ort_value_name_idx_map_);
  return Status::OK();
}

void SessionState::SetExecutionPlan(std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan) {
  p_seq_exec_plan_ = std::move(p_seq_exec_plan);
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

SessionState& SessionState::SetLogger(const logging::Logger& logger) {
  logger_ = &logger;
  return *this;
}

const logging::Logger& SessionState::Logger() const {
  // DefaultLogger either throws or returns a valid logger.
  const logging::Logger* logger = logger_ != nullptr ? logger_ : &logging::LoggingManager::DefaultLogger();
  return *logger;
}

void SessionState::SetProfiler(profiling::Profiler& profiler) { profiler_ = &profiler; }

::onnxruntime::profiling::Profiler& SessionState::Profiler() const { return *profiler_; }

static int64_t CalculateMemoryPatternsKey(const std::vector<std::reference_wrapper<const TensorShape>>& shapes) {
  int64_t key = 0;
  for (auto shape : shapes) {
    for (auto dim : shape.get().GetDims()) key ^= dim;
  }
  return key;
}

namespace {
Status ResolveDimParams(const GraphViewer& graph, const std::map<std::string, TensorShape>& feeds, std::unordered_map<std::string, int64_t>& out) {
  for (size_t i = 0; i < graph.GetInputs().size(); ++i) {
    auto* input = graph.GetInputs()[i];
    auto* shape = input->Shape();
    auto it = feeds.find(input->Name());
    if (it == feeds.end())
      return Status(ONNXRUNTIME, FAIL, "Graph input " + input->Name() + " is not found in the feed list, unable to resolve the value for dynamic shape.");
    if (!shape || shape->dim_size() != static_cast<int>(it->second.NumDimensions()))
      return Status(ONNXRUNTIME, FAIL, "Graph input " + input->Name() +
                                           "'s shape is not present or its shape doesn't match feed's shape."
                                           "Unable to resolve the value for dynamic shape");
    for (int k = 0; k < shape->dim_size(); ++k) {
      if (shape->dim()[k].has_dim_param()) {
        out.insert({shape->dim()[k].dim_param(), it->second.GetDims()[k]});
      }
    }
  }
  return Status::OK();
}
}  // namespace

Status SessionState::GeneratePatternGroupCache(const std::vector<std::reference_wrapper<const TensorShape>>& input_shape,
                                               const std::vector<int>& feed_mlvalue_idxs,
                                               MemoryPatternGroup* output) const {
  std::map<std::string, TensorShape> feeds;
  for (size_t i = 0; i < feed_mlvalue_idxs.size(); ++i) {
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
    for (int i = 0; i < static_cast<int>(node->OutputDefs().size()); ++i) {
      const auto ml_value_idx = node_index_info.GetMLValueIndex(output_start + i);
      if (ml_value_idx == NodeIndexInfo::kInvalidEntry)
        continue;
      const auto* ml_type = exe_plan->allocation_plan[ml_value_idx].value_type;
      if (!ml_type->IsTensorType())
        continue;
      const auto* ml_data_type = static_cast<const TensorTypeBase*>(ml_type)->GetElementType();
      if (exe_plan->allocation_plan[ml_value_idx].alloc_kind == AllocKind::kAllocate &&
          ml_data_type != DataTypeImpl::GetType<std::string>()) {
        //calculate size
        auto* arg = node->OutputDefs()[i];
        if (!arg->Shape())
          continue;
        size_t size = 0;
        int64_t len = 1;
        for (auto& dim : arg->Shape()->dim()) {
          if (dim.has_dim_param()) {
            auto it = map.find(dim.dim_param());
            if (it == map.end()) {
              return Status(ONNXRUNTIME, FAIL, "Unknow shape found in memory pattern compute");
            }
            len *= it->second;
          } else {
            len *= dim.dim_value();
          }
        }
        if (!IAllocator::CalcMemSizeForArrayWithAlignment<64>(len, ml_data_type->Size(), &size)) {
          return Status(ONNXRUNTIME, FAIL, "Size overflow");
        }
        mem_planner.TraceAllocation(ml_value_idx, size);
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

const MemoryPatternGroup* SessionState::GetMemoryPatternGroup(const std::vector<std::reference_wrapper<const TensorShape>>& input_shapes,
                                                              const std::vector<int>& feed_mlvalue_idxs) const {
  int64_t key = CalculateMemoryPatternsKey(input_shapes);

  std::lock_guard<OrtMutex> lock(mem_patterns_lock_);
  auto it = mem_patterns_.find(key);
  if (it == mem_patterns_.end()) {
#ifdef ENABLE_TRAINING
    auto mem_patterns = onnxruntime::make_unique<MemoryPatternGroup>();
    if (GeneratePatternGroupCache(input_shapes, feed_mlvalue_idxs, mem_patterns.get()).IsOK()) {
      key = CalculateMemoryPatternsKey(input_shapes);
      auto ptr = mem_patterns.get();
      mem_patterns_[key] = std::move(mem_patterns);
      return ptr;
    }
    return nullptr;
#else
    ORT_UNUSED_PARAMETER(feed_mlvalue_idxs);
    return nullptr;
#endif
  }

  return it->second.get();
}

void SessionState::ResolveMemoryPatternFlag() {
  if (enable_mem_pattern_) {
    for (auto* input : graph_viewer_->GetInputs()) {
      if (!input->Shape()) {
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
  if (to_be_executed_nodes_.find(fetch_mlvalue_idxs) != to_be_executed_nodes_.end())
    return;

  const Graph* graph = GetGraphViewer()->GetGraph();

  // Get the nodes generating the fetches.
  std::vector<const Node*> nodes;
  nodes.reserve(fetch_mlvalue_idxs.size());
  std::unordered_set<NodeIndex> reachable_nodes;

  for (auto idx : fetch_mlvalue_idxs) {
    std::string node_arg_name;
    if (!this->GetOrtValueNameIdxMap().GetName(idx, node_arg_name).IsOK()) {
      to_be_executed_nodes_.insert(std::make_pair(fetch_mlvalue_idxs, reachable_nodes));
      return;
    }
    auto ending_node = graph->GetProducerNode(node_arg_name);
    nodes.push_back(ending_node);
  }

  // Reversely traverse to get reachable nodes.
  graph->ReverseDFSFrom(
      nodes, {}, [&reachable_nodes](const Node* n) { reachable_nodes.insert(n->Index()); });
  to_be_executed_nodes_.insert(std::make_pair(fetch_mlvalue_idxs, reachable_nodes));
}

const std::unordered_set<NodeIndex>* SessionState::GetToBeExecutedNodes(
    const std::vector<int>& fetch_mlvalue_idxs) const {
  auto it = to_be_executed_nodes_.find(fetch_mlvalue_idxs);
  return (it != to_be_executed_nodes_.end()) ? &it->second : nullptr;
}

}  // namespace onnxruntime
