// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_state.h"

#include <sstream>

#include "core/common/logging/logging.h"
#include "core/framework/node_index_info.h"
#include "core/framework/op_kernel.h"
#include "core/framework/utils.h"

using namespace ::onnxruntime::common;
namespace onnxruntime {

void SessionState::SetGraphViewer(std::unique_ptr<onnxruntime::GraphViewer> graph_viewer) {
  ORT_ENFORCE(nullptr != graph_viewer);
  graph_viewer_ = std::move(graph_viewer);
}

const GraphViewer* SessionState::GetGraphViewer() const { return graph_viewer_.get(); }

const OpKernel* SessionState::GetKernel(NodeIndex node_id) const {
  if (session_kernels_.count(node_id) == 0) {
    return nullptr;
  }

  return session_kernels_.find(node_id)->second.get();
}

void SessionState::AddKernel(onnxruntime::NodeIndex node_id, std::unique_ptr<OpKernel> p_kernel) {
  // assumes vector is already resize()'ed to the number of nodes in the graph
  session_kernels_[node_id] = std::move(p_kernel);
}

void SessionState::SetExecutionPlan(std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan) {
  p_seq_exec_plan_ = std::move(p_seq_exec_plan);
}

const SequentialExecutionPlan* SessionState::GetExecutionPlan() const { return p_seq_exec_plan_.get(); }

Status SessionState::AddInitializedTensor(int mlvalue_index, const MLValue& mlvalue, const OrtCallback* d) {
  ORT_ENFORCE(mlvalue_index >= 0 && mlvalue_index <= mlvalue_name_idx_map_.MaxIdx());
  auto p = initialized_tensors_.insert({mlvalue_index, mlvalue});
  if (!p.second)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "duplicated mlvalue index:", mlvalue_index,
                           ". Do you have duplicated calls to SessionState::AddInitializedTensor function?");
  if (d != nullptr && d->f != nullptr) deleter_for_initialized_tensors_[mlvalue_index] = *d;
  return Status::OK();
}

const std::unordered_map<int, MLValue>& SessionState::GetInitializedTensors() const { return initialized_tensors_; }

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

static int64_t CalculateMemoryPatternsKey(const std::vector<TensorShape>& shapes) {
  int64_t key = 0;
  for (auto& shape : shapes) {
    for (auto dim : shape.GetDims()) key ^= dim;
  }
  return key;
}

const MemoryPatternGroup* SessionState::GetMemoryPatternGroup(const std::vector<TensorShape>& input_shapes) const {
  std::lock_guard<OrtMutex> lock(mem_patterns_lock_);
  int64_t key = CalculateMemoryPatternsKey(input_shapes);
  auto it = mem_patterns_.find(key);
  if (it == mem_patterns_.end()) return nullptr;

  return it->second.get();
}

Status SessionState::UpdateMemoryPatternGroupCache(const std::vector<TensorShape>& input_shape,
                                                   std::unique_ptr<MemoryPatternGroup> mem_patterns) const {
  int64_t key = CalculateMemoryPatternsKey(input_shape);

  std::lock_guard<OrtMutex> lock(mem_patterns_lock_);
  auto it = mem_patterns_.find(key);
  if (it == mem_patterns_.end()) {
    mem_patterns_[key] = std::move(mem_patterns);
  }

  return Status::OK();
}


common::Status SessionState::AddInputNameToNodeInfoMapping(const std::string& input_name, const NodeInfo& node_info) {
  // in the future we could support multiple nodes on difference devices using an input, however right now
  // the logic in utils::CopyOneInputAcrossDevices only checks the first entry.
  // Instead of failing silently and adding extra entries that will be ignored, check if the required provider
  // is the same for any duplicate entries. If it differs we can't run the model.

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
      // if the providers match we can add the new entry for completeness (it will be ignored in
      // utils::CopyOneInputAcrossDevices though).
      // if they don't, we are broken.
      const auto& current_provider = utils::GetNodeInputProviderType(entries[0]);
      const auto& new_provider = utils::GetNodeInputProviderType(node_info);

      if (current_provider == new_provider) {
        entries.push_back(node_info);
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                               "Using an input in multiple nodes on different devices is not supported currently. Input:",
                               input_name, " is used by node ", existing_entry.p_node->Name(), " (", current_provider,
                               ") and node ", node_info.p_node->Name(), " (", new_provider, ").");
      }
    }
  }

  return Status::OK();
}

common::Status SessionState::GetInputNodeInfo(const std::string& input_name,
                                              std::vector<NodeInfo>& node_info_vec) const {
  if (!input_names_to_nodeinfo_mapping_.count(input_name)) {
    return Status(ONNXRUNTIME, FAIL, "Failed to find input name in the mapping: " + input_name);
  }
  node_info_vec = input_names_to_nodeinfo_mapping_.at(input_name);
  return Status::OK();
}

const SessionState::NameNodeInfoMapType& SessionState::GetInputNodeInfoMap() const {
  return input_names_to_nodeinfo_mapping_;
}

void SessionState::AddOutputNameToNodeInfoMapping(const std::string& output_name, const NodeInfo& node_info) {
  output_names_to_nodeinfo_mapping_[output_name].push_back(node_info);
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

  subgraph_session_states_[index].insert(std::make_pair(attribute_name, std::move(session_state)));
}

SessionState* SessionState::GetMutableSubgraphSessionState(onnxruntime::NodeIndex index,
                                                           const std::string& attribute_name) {
  SessionState* session_state = nullptr;

  auto node_entry = subgraph_session_states_.find(index);
  if (node_entry != subgraph_session_states_.cend()) {
    const auto& attribute_state_map{node_entry->second};

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

void SessionState::CalculateNodeIndexInfo() {
  ORT_ENFORCE(graph_viewer_);
  node_index_info_ = std::make_unique<NodeIndexInfo>(*graph_viewer_, mlvalue_name_idx_map_);

  for (auto& node_to_map_pair : subgraph_session_states_) {
    for (auto& attr_name_to_subgraph : node_to_map_pair.second) {
      attr_name_to_subgraph.second->CalculateNodeIndexInfo();
    }
  }
}

const NodeIndexInfo& SessionState::GetNodeIndexInfo() const {
  ORT_ENFORCE(node_index_info_, "CalculateNodeIndexInfo must be called prior to GetExecutionInfo.");
  return *node_index_info_;
}
}  // namespace onnxruntime
