// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <map>
#include <unordered_map>
#include <vector>
#include "gsl/gsl"

#include "core/platform/ort_mutex.h"
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/profiler.h"
#include "core/framework/allocation_planner.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_providers.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/mem_pattern.h"
#include "core/framework/ml_value.h"
#include "core/framework/callback.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/node_index_info.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {

class ExecutionProviders;
class KernelDef;
class OpKernel;
class NodeIndexInfo;
struct SequentialExecutionPlan;
struct MemoryPatternGroup;

/**
 * SessionState should be modified by the inference session class only.
 * It is supposed to be passed by const-ref only to all the executors.
 * This class owns all the initializers.
 * Brief usage:
 * SessionState s(...);
 * for(...) s.AddInitializedTensor(...);
 * s.SetGraphAndCreateKernels(...);
 * Then you can use:
 * s.GetKernel(...);
 */
class SessionState {
 public:
  SessionState(const ExecutionProviders& execution_providers,
               bool enable_mem_pattern,
               concurrency::ThreadPool* thread_pool,
               concurrency::ThreadPool* inter_op_thread_pool)
      : execution_providers_(execution_providers),
        enable_mem_pattern_(enable_mem_pattern),
        thread_pool_(thread_pool),
        inter_op_thread_pool_(inter_op_thread_pool) {
  }

  ~SessionState() {
    for (auto* p : session_kernels_) {
      delete p;
    }
    for (auto& kvp : deleter_for_initialized_tensors_) {
      kvp.second.f(kvp.second.param);
    }
  }

  // Graph viewer.
  const GraphViewer* GetGraphViewer() const;

  // kernels
  // Get kernel for specified node.
  // It should called right before graph execution only.
  const OpKernel* GetKernel(size_t node_id) const {
    return (node_id < session_kernels_.size()) ? session_kernels_[node_id] : nullptr;
  }

  OpKernel* GetMutableKernel(size_t node_id) {
    return (node_id < session_kernels_.size()) ? session_kernels_[node_id] : nullptr;
  }

  const ExecutionProviders& GetExecutionProviders() const noexcept { return execution_providers_; }

  const OrtValueNameIdxMap& GetOrtValueNameIdxMap() const noexcept { return ort_value_name_idx_map_; }

  // initialized tensors
  /**
   * Adds an initialized tensor (weight) so that it can be used by the
   * execution frame to setup the appropriate OrtValue vectors.
   * This function will take a shallow copy of d if d is not NULL.
   * If 'constant' is true the tensor value cannot be overridden by an input at runtime.
   */
  Status AddInitializedTensor(int ort_value_index, const OrtValue& ort_value, const OrtCallback* d, bool constant);

  Status SetGraph(const Graph& graph);
  Status CreateKernels(const KernelRegistryManager& custom_registry_manager);
  Status SetGraphAndCreateKernels(const Graph& graph, const KernelRegistryManager& custom_registry_manager) {
    ORT_RETURN_IF_ERROR(SetGraph(graph));
    return CreateKernels(custom_registry_manager);
  }
  /**
   * Gets the map of ort_value_index to initialized tensors (weights) so that it can be used by the
   * execution frame to setup the appropriate OrtValue vectors.
   * The lifetime of returned OrtValues are limited by this SessionState object.
   */
  const std::unordered_map<int, OrtValue>& GetInitializedTensors() const;

  /**
   * Gets the map of ort_value_index to initialized tensors (e.g. weights) that are constant
   * and cannot be overridden at runtime.
   * The lifetime of returned OrtValues are limited by this SessionState object.
   */
  const std::unordered_map<int, OrtValue>& GetConstantInitializedTensors() const;

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
  const MemoryPatternGroup* GetMemoryPatternGroup(
      const std::vector<std::reference_wrapper<const TensorShape>>& input_shapes) const;

  /**
  Set generated memory pattern with a given input shapes.
  Const as it's an internal cache update only.
  */
  Status UpdateMemoryPatternGroupCache(const std::vector<std::reference_wrapper<const TensorShape>>& input_shape,
                                       std::unique_ptr<MemoryPatternGroup> mem_patterns) const;

  /**
  Get enable memory pattern flag
  */
  bool GetEnableMemoryPattern() const;

  struct NodeInfo {
    /**
     *
     * \param index0
     * \param p_node0 Nullable
     * \param kci0 Nullable
     */
    NodeInfo(size_t index0, const onnxruntime::Node* p_node0, const KernelCreateInfo* kci0, const OrtDevice& device0)
        : index(index0), p_node(p_node0), kci(kci0), device(&device0) {}

    size_t index;
    // Nullable
    const onnxruntime::Node* p_node = nullptr;
    // Nullable
    const KernelCreateInfo* kci = nullptr;
    const OrtDevice* device = nullptr;
  };

  using NameNodeInfoMapType = std::unordered_map<std::string, std::vector<NodeInfo>>;
  common::Status AddInputNameToNodeInfoMapping(const std::string& input_name, const NodeInfo& node_info);
  common::Status GetInputNodeInfo(const std::string& input_name, std::vector<NodeInfo>& node_info_vec) const;
  const NameNodeInfoMapType& GetInputNodeInfoMap() const;

  void AddOutputNameToNodeInfoMapping(const std::string& output_name, const NodeInfo& node_info);
  common::Status GetOutputNodeInfo(const std::string& output_name, std::vector<NodeInfo>& node_info_vec) const;
  const NameNodeInfoMapType& GetOutputNodeInfoMap() const;

  /// Add a SessionState instance for executing a subgraph in a Node
  /// @param index Index of Node containing subgraph
  /// @param attribute_name Name of attribute containing the subgraph GraphProto
  /// @param session_state SessionState for subgraph execution
  void AddSubgraphSessionState(onnxruntime::NodeIndex index, const std::string& attribute_name,
                               std::unique_ptr<SessionState> session_state);

  /// Return SessionState for the given Node index and attribute name if found.
  const SessionState* GetSubgraphSessionState(onnxruntime::NodeIndex index, const std::string& attribute_name) const;

  SessionState* GetMutableSubgraphSessionState(onnxruntime::NodeIndex index, const std::string& attribute_name);

  // Remove the SessionState for a node containing a subgraph.
  // If the node isn't going to be executed by the CPU provider we don't need it.
  void RemoveSubgraphSessionState(onnxruntime::NodeIndex index);

  concurrency::ThreadPool* GetThreadPool() const { return thread_pool_; }
  concurrency::ThreadPool* GetInterOpThreadPool() const { return inter_op_thread_pool_; }

  bool ExportDll() const { return export_fused_dll_; }
  void SetExportDllFlag(bool flag) { export_fused_dll_ = flag; }

  const FuncManager& GetFuncMgr() const { return fused_funcs_mgr_; }
  FuncManager& GetMutableFuncMgr() { return fused_funcs_mgr_; }

  const DataTransferManager& GetDataTransferMgr() const { return *data_transfer_mgr_; }
  void SetDataTransferMgr(const DataTransferManager* data_transfer_mgr) { data_transfer_mgr_ = data_transfer_mgr; }

  std::vector<BufferUniquePtr>& GetMutableWeightsBuffers() { return weights_buffers_; }
  const NodeIndexInfo& GetNodeIndexInfo() const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SessionState);

  // cache of the constructed kernels to avoid spending construction
  // time per executor
  std::vector<OpKernel*> session_kernels_;
  std::unique_ptr<GraphViewer> graph_viewer_;

  std::reference_wrapper<const ExecutionProviders> execution_providers_;  // owned by InferenceSession
  OrtValueNameIdxMap ort_value_name_idx_map_;

  // initialized tensors
  std::unordered_map<int, OrtValue> initialized_tensors_;  // key is ort_value_index
  // subset of initialized_tensors_ that are constant and cannot be overridden at runtime
  std::unordered_map<int, OrtValue> constant_initialized_tensors_;

  // This data structure is for uninitializing string tensors and
  // munmap memory region and close file descriptor
  std::unordered_map<int, OrtCallback> deleter_for_initialized_tensors_;
  std::vector<BufferUniquePtr> weights_buffers_;
  std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan_ = nullptr;

  const logging::Logger* logger_ = nullptr;
  profiling::Profiler* profiler_ = nullptr;

  // switch for enable memory pattern optimization or not.
  const bool enable_mem_pattern_;
  // lock for the mem_patterns_
  mutable OrtMutex mem_patterns_lock_;
  // cache for the generated mem_patterns. key is calculated based on input shapes.
  mutable std::map<int64_t, std::unique_ptr<MemoryPatternGroup>> mem_patterns_;

  NameNodeInfoMapType input_names_to_nodeinfo_mapping_;
  NameNodeInfoMapType output_names_to_nodeinfo_mapping_;

  // subgraph SessionState. entry for node containing subgraph, with value containing attribute:SessionState pair
  // as a node may contain multiple subgraphs (e.g. 'If' has one for both the 'then' and 'else' branches).
  using SubgraphSessionStateMap =
      std::unordered_map<onnxruntime::NodeIndex, std::unordered_map<std::string, std::unique_ptr<SessionState>>>;
  SubgraphSessionStateMap subgraph_session_states_;

  // It could be NULL
  concurrency::ThreadPool* const thread_pool_{};
  concurrency::ThreadPool* const inter_op_thread_pool_{};

  bool export_fused_dll_ = false;
  FuncManager fused_funcs_mgr_;
  const DataTransferManager* data_transfer_mgr_ = nullptr;

  std::unique_ptr<NodeIndexInfo> node_index_info_;
  std::multimap<int, std::unique_ptr<FeedsFetchesManager>> cached_feeds_fetches_managers_;
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  SessionState* parent_ = nullptr;
  //Assign each graph in each session an unique id.
  int graph_id_ = 0;
  int next_graph_id_ = 1;
  
  void GenerateGraphId() {
	SessionState* p = this;
    while (p->parent_ != nullptr) p = p->parent_;
	graph_id_ = p->next_graph_id_ ++;
  }

#endif
};

}  // namespace onnxruntime
