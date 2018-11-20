// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "gsl/gsl_util"

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/profiler.h"
#include "core/framework/allocation_planner.h"
#include "core/framework/execution_providers.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/mem_pattern.h"
#include "core/framework/ml_value.h"
#include "core/framework/mlvalue_name_idx_map.h"
#include "core/graph/graph.h"

namespace onnxruntime {

class ExecutionProviders;
class KernelDef;
class OpKernel;
class TaskThreadPool;
struct SequentialExecutionPlan;
struct MemoryPatternGroup;

// SessionState should be modified by the inference session class only.
// It is supposed to be passed by const-ref only to all the executors.
class SessionState {
 public:
  SessionState(const ExecutionProviders& execution_providers)
      : execution_providers_{execution_providers} {
  }

  // Graph viewer.
  void SetGraphViewer(std::unique_ptr<onnxruntime::GraphViewer> graph_viewer);
  const onnxruntime::GraphViewer* GetGraphViewer() const;

  // kernels
  // Get kernel for specified node.
  // It should called right before graph execution only.
  const OpKernel* GetKernel(onnxruntime::NodeIndex node_id) const;

  void AddKernel(onnxruntime::NodeIndex node_id, std::unique_ptr<OpKernel> p_kernel);

  const ExecutionProviders& GetExecutionProviders() const noexcept { return execution_providers_; }

  const MLValueNameIdxMap& GetMLValueNameIdxMap() const noexcept { return mlvalue_name_idx_map_; }
  MLValueNameIdxMap& GetMLValueNameIdxMap() noexcept { return mlvalue_name_idx_map_; }

  // initialized tensors
  /**
  * Adds an initialized tensor (weight) so that it can be used by the
  * execution frame to setup the appropriate MLValue vectors.
  */
  void AddInitializedTensor(int mlvalue_index, const MLValue& mlvalue);

  /**
  * Gets the list of all initialized tensors (weights) so that it can be used by the
  * execution frame to setup the appropriate MLValue vectors.
  */
  const std::unordered_map<int, MLValue>& GetInitializedTensors() const;

  // execution plan
  void SetExecutionPlan(std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan);
  const SequentialExecutionPlan* GetExecutionPlan() const;

  /**
  Set the logger to use for this session. 
  */
  SessionState& SetLogger(const logging::Logger& logger);

  /**
  Get the logger for this session. 
  Falls back to returning Logging::LoggingManager::DefaultLogger if SetLogger has not been called.
  */
  const logging::Logger& Logger() const;

  /**
  Set the profiler for this session.
  */
  void SetProfiler(profiling::Profiler& profiler);

  /**
  Get the profiler for this session. It needs to be enabled via the InferenceSession to perform
  profiling actions.
  */
  profiling::Profiler& Profiler() const;

  /**
  Get cached memory pattern based on input shapes
  */
  const MemoryPatternGroup* GetMemoryPatternGroup(const std::vector<TensorShape>& input_shapes) const;

  /**
  Set generated memory pattern with a given input shapes. 
  Const as it's an internal cache update only.
  */
  Status UpdateMemoryPatternGroupCache(const std::vector<TensorShape>& input_shape,
                                       std::unique_ptr<MemoryPatternGroup> mem_patterns) const;

  /**
  Set enable memory pattern flag
  */
  void SetEnableMemoryPattern(bool flag);

  /**
  Get enable memory pattern flag
  */
  bool GetEnableMemoryPattern() const;

  struct NodeInfo {
    NodeInfo(size_t index0, const onnxruntime::Node* p_node0, const KernelCreateInfo* kci0)
        : index(index0),
          p_node(p_node0),
          kci(kci0) {
    }
    NodeInfo() = default;

    size_t index;
    const onnxruntime::Node* p_node = nullptr;
    const KernelCreateInfo* kci = nullptr;
  };

  using NameNodeInfoMapType = std::unordered_map<std::string, std::vector<NodeInfo>>;
  void AddInputNameToNodeInfoMapping(const std::string& input_name, const NodeInfo& node_info);
  common::Status GetInputNodeInfo(const std::string& input_name, std::vector<NodeInfo>& node_info_vec) const;
  const NameNodeInfoMapType& GetInputNodeInfoMap() const;

  void AddOutputNameToNodeInfoMapping(const std::string& output_name, const NodeInfo& node_info);
  const NameNodeInfoMapType& GetOutputNodeInfoMap() const;

  /// Add a SessionState instance for executing a subgraph in a Node
  /// @param index Index of Node containing subgraph
  /// @param attribute_name Name of attribute containing the subgraph GraphProto
  /// @param session_state SessionState for subgraph execution
  void AddSubgraphSessionState(onnxruntime::NodeIndex index, const std::string& attribute_name,
                               const SessionState& session_state);

  /// Return SessionState for the given Node index and attribute name if found.
  const SessionState* GetSubgraphSessionState(onnxruntime::NodeIndex index, const std::string& attribute_name) const;

  TaskThreadPool* GetThreadPool() const { return thread_pool_; }
  void SetThreadPool(TaskThreadPool* p_pool) { thread_pool_ = p_pool; }

 private:
  ONNXRUNTIME_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SessionState);

  // cache of the constructed kernels to avoid spending construction
  // time per executor
  std::unordered_map<onnxruntime::NodeIndex, std::unique_ptr<OpKernel>> session_kernels_;
  std::unique_ptr<onnxruntime::GraphViewer> graph_viewer_;

  const ExecutionProviders& execution_providers_;  // owned by InferenceSession
  MLValueNameIdxMap mlvalue_name_idx_map_;

  // initialized tensorset
  std::unordered_map<int, MLValue> initialized_tensors_;  // key is mlvalue_index
  std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan_ = nullptr;

  const logging::Logger* logger_;
  profiling::Profiler* profiler_;

  // switch for enable memory pattern optimization or not.
  bool enable_mem_pattern_ = true;
  // lock for the mem_patterns_
  mutable std::mutex mem_patterns_lock_;
  // cache for the generated mem_patterns. key is calculated based on input shapes.
  mutable std::map<int64_t, std::unique_ptr<MemoryPatternGroup>> mem_patterns_;

  NameNodeInfoMapType input_names_to_nodeinfo_mapping_;
  NameNodeInfoMapType output_names_to_nodeinfo_mapping_;

  // subgraph SessionState. entry for node containing subgraph, with value containing attribute:SessionState pair
  // as a node may contain multiple subgraphs (e.g. 'If' has one for both the 'then' and 'else' branches).
  using SubgraphSessionStateMap =
      std::unordered_map<onnxruntime::NodeIndex,
                         std::unordered_map<std::string, gsl::not_null<const SessionState*>>>;
  SubgraphSessionStateMap subgraph_session_states_;
  TaskThreadPool* thread_pool_ = nullptr;
};
}  // namespace onnxruntime
