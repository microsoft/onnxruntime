//// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <map>
#include <unordered_map>
#include <vector>

#include "core/common/flatbuffers.h"

#include "core/common/gsl.h"

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/common/profiler.h"
#include "core/framework/allocation_planner.h"
#include "core/framework/callback.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_providers.h"
#include "core/framework/stream_execution_context.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/framework_common.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/mem_pattern.h"
#include "core/framework/ort_value.h"
#include "core/framework/node_index_info.h"
#include "core/framework/op_kernel.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/onnx_protobuf.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/path_lib.h"
#include "core/platform/threadpool.h"
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
#include "core/framework/memory_info.h"
#endif

#include "core/framework/stream_handles.h"
#ifdef ENABLE_TRAINING
#include "core/framework/program_region.h"
#endif

namespace onnxruntime {

namespace fbs {
struct SessionState;
}  // namespace fbs

class ExecutionProviders;
class KernelDef;
class OpKernel;
class NodeIndexInfo;
struct SequentialExecutionPlan;
struct MemoryPatternGroup;
class DeviceStreamCollection;
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
class MemoryInfo;
#endif

/**
 * SessionState should be modified by the inference session class only.
 * It is supposed to be passed by const-ref only to all the executors.
 * This class owns all the initializers.
 * Brief usage:
 *   SessionState s(...);
 *   <process subgraphs to populate subgraph SessionState instances>
 *   <run transformers or any other graph editing steps>
 *   for(...) // copy initializers from GraphProto format in Graph to OrtValue format in SessionState
        s.AddInitializedTensor(...);
 *   s.CleanInitializedTensorsFromGraph(); // remove GraphProto instances from Graph if not needed
 *
 *   s.CreateGraphInfo();
 *   s.CreateKernels(...);
 * Then you can use:
 *   s.GetKernel(...);
 */

// subgraph SessionState. entry for node containing subgraph, with value containing attribute:SessionState pair
// as a node may contain multiple subgraphs (e.g. 'If' has one for both the 'then' and 'else' branches).
using SubgraphSessionStateMap =
    std::unordered_map<onnxruntime::NodeIndex, std::unordered_map<std::string, std::unique_ptr<SessionState>>>;

class SessionState {
 public:
  SessionState(Graph& graph,
               const ExecutionProviders& execution_providers,
               concurrency::ThreadPool* thread_pool,
               concurrency::ThreadPool* inter_op_thread_pool,
               const DataTransferManager& data_transfer_mgr,
               const logging::Logger& logger,
               profiling::Profiler& profiler,
               const SessionOptions& sess_options,
               PrepackedWeightsContainer* prepacked_weights_container = nullptr,
               AllocatorMap* parent_allocators = nullptr);

  ~SessionState() {
    for (auto& kvp : deleter_for_initialized_tensors_) {
      kvp.second.f(kvp.second.param);
    }
  }

  // Graph viewer. CreateGraphInfo must have been called previously.
  const GraphViewer& GetGraphViewer() const noexcept { return *graph_viewer_; };

  // kernels
  // Get kernel for specified node.
  // It should called right before graph execution only.
  const OpKernel* GetKernel(size_t node_id) const {
    return (node_id < session_kernels_.size()) ? session_kernels_[node_id].get() : nullptr;
  }

  OpKernel* GetMutableKernel(size_t node_id) {
    return (node_id < session_kernels_.size()) ? session_kernels_[node_id].get() : nullptr;
  }

  const ExecutionProviders& GetExecutionProviders() const noexcept { return execution_providers_; }

  /**
    Get the allocator for the given OrtMemoryInfo location
    */
  AllocatorPtr GetAllocator(const OrtMemoryInfo& location) const noexcept;

  /** Get the allocator for a given OrtDevice. The first allocator that matches will be returned. */
  AllocatorPtr GetAllocator(const OrtDevice& device) const noexcept;

  /*
   * Get allocators.
   */
  const AllocatorMap& GetAllocators() const { return *allocators_; }

  void UpdateAllocatorsWithEnvAllocators(const std::vector<AllocatorPtr>&);

  const OrtValueNameIdxMap& GetOrtValueNameIdxMap() const noexcept { return ort_value_name_idx_map_; }

  /**
   * Adds an initialized tensor (weight) so that it can be used by the
   * execution frame to setup the appropriate OrtValue vectors.
   * This function will take a shallow copy of d if d is not NULL.
   * If 'constant' is true the tensor value cannot be overridden by an input at runtime.
   * If 'sparse' is true the tensor value represents a densified weight that was initially stored in the model
   * as sparse tensor.
   */
  Status AddInitializedTensor(int ort_value_index, const OrtValue& ort_value, const OrtCallback* d, bool constant, bool sparse);

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

#if !defined(DISABLE_SPARSE_TENSORS)
  bool IsSparseInitializer(int ort_value_index) const;
#endif

#ifdef ENABLE_TRAINING
  // This is referenced in training::TrainingSession. Should be removed when this class is removed.
  /**
    Get some initialized tensors (weights).
    @param interested_weights The names of the weights to retrieve.
    @param allow_missing_weights Whether to allow names in interested_weights
           with no corresponding weight.
    @param[out] retrieved_weights The retrieved weights.
    @return The status of the operation.
    */
  Status GetInitializedTensors(
      const std::unordered_set<std::string>& interested_weights,
      bool allow_missing_weights, NameMLValMap& retrieved_weights) const;

  /**
    Get some initialized tensors (weights).
    Any names in interested_weights with no corresponding weight are ignored.
    */
  NameMLValMap GetInitializedTensors(const std::unordered_set<std::string>& interested_weights) const;
#endif

  // execution plan. nullptr until FinalizeSessionState is called
  const SequentialExecutionPlan* GetExecutionPlan() const;

  const std::vector<AllocPlanPerValue>& GetPerValueAllocPlan() const;

  /**
  Get the logger for this session.
  Falls back to returning Logging::LoggingManager::DefaultLogger if SetLogger has not been called.
  */
  const logging::Logger& Logger() const noexcept { return logger_; }

  /**
  Get the profiler for this session. It needs to be enabled via the InferenceSession to perform
  profiling actions.
  */
  profiling::Profiler& Profiler() const noexcept { return profiler_; }

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  MemoryProfiler* GetMemoryProfiler() const noexcept { return memory_profiler_; }

  void SetMemoryProfiler(MemoryProfiler* memory_profiler) noexcept {
    memory_profiler_ = memory_profiler;
  }
#endif

  /**
  Get cached memory pattern based on input shapes
  Must be called only when all values contain tensors
  In training scenarios, the cache may be updated so
  the callers would receive a copy of inferred shapes
  made under mutex being held. In inference scenarios,
  it is not mutable, we do not obtain a lock and simply get a pointer
  w/o copying a hashtable
  */
  const MemoryPatternGroup* GetMemoryPatternGroup(
      gsl::span<const OrtValue> tensor_inputs,
      gsl::span<const int> feed_mlvalue_idxs,
      const InlinedHashMap<int, TensorShape>*& inferred_shapes) const;

  /**
  Set generated memory pattern with a given input shapes.
  Const as it's an internal cache update only.
  All inputs must represent Tensors
  */
  Status UpdateMemoryPatternGroupCache(gsl::span<const OrtValue> tensor_inputs,
                                       MemoryPatternGroup mem_patterns) const;

  bool GetUseDeterministicCompute() const { return sess_options_.use_deterministic_compute; }

  /**
  Get enable memory pattern flag
  */
  bool GetEnableMemoryPattern() const;

  /**
  Get enable memory re-use flag.
  */

  bool GetEnableMemoryReuse() const;

  /**
  Update enable_mem_pattern_ flag according to the presence of graph inputs' shape
  If any one of the graph input is shapeless, enable_mem_pattern_ will be set to false
  */
  void ResolveMemoryPatternFlag();

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

  using NameNodeInfoMapType = InlinedHashMap<std::string, InlinedVector<NodeInfo>>;

  common::Status AddInputNameToNodeInfoMapping(const std::string& input_name, const NodeInfo& node_info);
  common::Status GetInputNodeInfo(const std::string& input_name, InlinedVector<NodeInfo>& node_info_vec) const;
  const NameNodeInfoMapType& GetInputNodeInfoMap() const;

  void AddOutputNameToNodeInfoMapping(const std::string& output_name, const NodeInfo& node_info);
  common::Status GetOutputNodeInfo(const std::string& output_name, InlinedVector<NodeInfo>& node_info_vec) const;
  const NameNodeInfoMapType& GetOutputNodeInfoMap() const;

  // Get the KernelCreateInfo entry for a node. SessionState must be finalized before calling.
  const KernelCreateInfo& GetNodeKernelCreateInfo(NodeIndex node_index) const;

  /// Return SessionState for the given Node index and attribute name if found.
  const SessionState* GetSubgraphSessionState(NodeIndex index, const std::string& attribute_name) const;

  concurrency::ThreadPool* GetThreadPool() const noexcept { return thread_pool_; }
  concurrency::ThreadPool* GetInterOpThreadPool() const noexcept { return inter_op_thread_pool_; }

  const FuncManager& GetFuncMgr() const noexcept { return fused_funcs_mgr_; }
  FuncManager& GetMutableFuncMgr() noexcept { return fused_funcs_mgr_; }

  const DataTransferManager& GetDataTransferMgr() const noexcept { return data_transfer_mgr_; }

  InlinedVector<BufferUniquePtr>& GetMutableWeightsBuffers() noexcept { return weights_buffers_; }

  const NodeIndexInfo& GetNodeIndexInfo() const;
#ifdef ENABLE_TRAINING
  void UpdateToBeExecutedRange(gsl::span<int const> fetch_mlvalue_idxs);
  const InlinedHashSet<NodeIndex>* GetToBeExecutedRange(gsl::span<int const> fetch_mlvalue_idxs) const;
#endif

  Status FinalizeSessionState(const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
                              const KernelRegistryManager& kernel_registry_manager,
                              bool remove_initializers = true,
                              bool saving_ort_format = false);

  SessionState* Parent() {
    return parent_;
  }

  // Clear all removable attributes if they exists.
  // The function logs the list of removable attributes for every node.
  void PruneRemovableAttributes();

  size_t GetNumberOfPrepacksCounter() const {
    return number_of_prepacks_counter_;
  }

  size_t GetUsedSharedPrePackedWeightCounter() const {
    return used_shared_pre_packed_weights_counter_;
  }

  const KernelCreateInfoMap& GetKernelCreateInfoMap() const {
    return kernel_create_info_map_;
  }

  const SubgraphSessionStateMap& GetSubgraphSessionStateMap() const {
    return subgraph_session_states_;
  }

#ifdef ORT_ENABLE_STREAM
  std::unique_ptr<DeviceStreamCollection> AcquireDeviceStreamCollection() const;

  void RecycleDeviceStreamCollection(std::unique_ptr<DeviceStreamCollection> device_stream_collection) const;

  IStreamCommandHandleRegistry& GetStreamHandleRegistryInstance() const {
    return *stream_handles_registry_;
  }
#endif

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
  void
  IncrementGraphExecutionCounter() {
    ++graph_executions_counter_;
  }

  size_t GetGraphExecutionCounter() const {
    return graph_executions_counter_;
  }
#endif

  const SessionOptions& GetSessionOptions() const { return sess_options_; }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SessionState);

  // Populate OrtValueNameIdxMap and create the graph viewer.
  void CreateGraphInfo();

  // create kernels using info in kernel_create_info_map_
  Status CreateKernels(const KernelRegistryManager& custom_registry_manager);

  // remove TensorProto versions of initializers from Graph instance
  // (replaced byOrtValue instances in initialized_tensors_)
  void CleanInitializedTensorsFromGraph();

  /**
   * Prepack the constant initialized tensors for better performance.
   * The original constant initialized tensors will be removed to save memory.
   */
  Status PrepackConstantInitializedTensors(InlinedHashMap<std::string, size_t>& constant_initializers_use_count,
                                           const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map);

  SessionState* GetMutableSubgraphSessionState(onnxruntime::NodeIndex index, const std::string& attribute_name);

  Status CreateSubgraphSessionState();

  void AddSubgraphSessionState(onnxruntime::NodeIndex index, const std::string& attribute_name,
                               std::unique_ptr<SessionState> session_state);

  Status PopulateKernelCreateInfo(const KernelRegistryManager& kernel_registry_manager,
                                  bool saving_ort_format);

  Status FinalizeSessionStateImpl(const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
                                  const KernelRegistryManager& kernel_registry_manager,
                                  _In_opt_ const Node* parent_node,
                                  const SessionOptions& session_options,
                                  bool remove_initializers,
                                  InlinedHashMap<std::string, size_t>& constant_initializers_use_count,
                                  const InlinedHashMap<OrtValueName, OrtDevice>& outer_scope_node_arg_to_location_map = {},
                                  bool graph_info_already_created = false);

#ifdef ENABLE_TRAINING
  Status GeneratePatternGroupCache(
      gsl::span<const OrtValue> inputs,
      gsl::span<const int> feed_mlvalue_idxs,
      MemoryPatternGroup& output,
      InlinedHashMap<int, TensorShape>& inferred_shapes) const;
#endif

  // KernelCreateInfo for each node so we do kernel lookup once
  KernelCreateInfoMap kernel_create_info_map_;

  // fused_funcs_mgr_ must live longer than the session_kernels_, becaues a kernel could be created from this manager
  FuncManager fused_funcs_mgr_;

  // cache of the constructed kernels to avoid spending construction time per executor
  std::vector<std::unique_ptr<OpKernel>> session_kernels_;
  Graph& graph_;
  std::optional<GraphViewer> graph_viewer_;  // GraphViewer for const access to Graph

  const ExecutionProviders& execution_providers_;

  // currently the allocator type is an implementation detail and we don't make any behavioral choices based on it,
  // so exclude it from the key comparison for allocator_idx_map_.
  // we also don't expect to have two allocators with the same name, one using an arena and one not.
  struct OrtMemoryInfoLessThanIgnoreNameAndAllocType {
    bool operator()(const OrtMemoryInfo& lhs, const OrtMemoryInfo& rhs) const {
      // if (lhs.alloc_type != rhs.alloc_type)
      //   return lhs.alloc_type < rhs.alloc_type;
      if (lhs.mem_type != rhs.mem_type)
        return lhs.mem_type < rhs.mem_type;

      if (lhs.id != rhs.id)
        return lhs.id < rhs.id;

      if (lhs.device != rhs.device) {
        // id should always == device.id so ignore that
        if (lhs.device.Type() != rhs.device.Type())
          return lhs.device.Type() < rhs.device.Type();

        // this is the allocator mem type and not the kernel mem type that OrtMemoryInfo.mem_type represents
        return lhs.device.MemType() < rhs.device.MemType();
      }

      return false;
    }
  };

  // using std::map as OrtDevice would need a custom hash function to be used with std::unordered_map,
  // and as this isn't considered performance critical currently it's not worth the maintenance overhead of adding one.
  // We do get an allocator from ExecutionFrame so this is looked up frequently, however there most likely aren't many
  // entries in the map
  // SessionState will contain other SessionState objects for subgraph. The unique ptr will be initialized only the
  // SessionState object is in the parent graph, the raw pointer will be initialized when session state is in parent
  // graph (from the unique ptr) or in the subgraph (from the raw pointer from parent session state). The raw pointer
  // will be used all the way to access std::map<OrtDevice, AllocatorPtr>, unique pointer is only releasing the resource
  // when the parent session state is releasing.
  std::unique_ptr<AllocatorMap> allocators_unique_ptr_;
  AllocatorMap* allocators_;

  OrtValueNameIdxMap ort_value_name_idx_map_;

  // initialized tensors
  std::unordered_map<int, OrtValue> initialized_tensors_;  // key is ort_value_index
  // subset of initialized_tensors_ that are constant and cannot be overridden at runtime
  std::unordered_map<int, OrtValue> constant_initialized_tensors_;

#if !defined(DISABLE_SPARSE_TENSORS)
  // This is an auxiliary lookup to check if the OrtValue was actually a sparse tensor
  // this is needed because we currently convert all sparse initializer into dense Tensors
  // if and when we actually place SparseTensor instances (we should) into OrtValues, we
  // will not need this structure.
  InlinedHashSet<int> sparse_initialized_tensors_;
#endif

  // This data structure is for uninitializing string tensors and
  // munmap memory region and close file descriptor
  InlinedHashMap<int, OrtCallback> deleter_for_initialized_tensors_;
  InlinedVector<BufferUniquePtr> weights_buffers_;
  std::optional<SequentialExecutionPlan> p_seq_exec_plan_;

  const logging::Logger& logger_;
  profiling::Profiler& profiler_;

#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
  MemoryProfiler* memory_profiler_;
#endif

  // switch for enable memory pattern optimization or not.
  bool enable_mem_pattern_;

  // lock for the mem_patterns_
  mutable OrtMutex mem_patterns_lock_;
  // cache for the generated mem_patterns. key is calculated based on input shapes.
  // must be a node based container as a pointer is cached.
  mutable NodeHashMap<int64_t, MemoryPatternGroup> mem_patterns_;
  // This is mutable under mutex in training scenarios so execution frame would make a copy
  // of the value when created.
#ifdef ENABLE_TRAINING
  mutable NodeHashMap<int64_t, InlinedHashMap<int, TensorShape>> shape_patterns_;
#else
  NodeHashMap<int64_t, InlinedHashMap<int, TensorShape>> shape_patterns_;
#endif

  NameNodeInfoMapType input_names_to_nodeinfo_mapping_;
  NameNodeInfoMapType output_names_to_nodeinfo_mapping_;

  SubgraphSessionStateMap subgraph_session_states_;

  // either threadpool could be nullptr
  concurrency::ThreadPool* const thread_pool_{};
  concurrency::ThreadPool* const inter_op_thread_pool_{};

  const DataTransferManager& data_transfer_mgr_;

  const SessionOptions& sess_options_;

  std::optional<NodeIndexInfo> node_index_info_;

  // Container to store pre-packed weights to share between sessions.
  // The life-cycle of the cache itself is maintained by the user and the user will ensure
  // the cache is valid until any session reliant on it is still in scope.
  // prepacked_weights_container_ can be nullptr if no caching is required for prepacked weights
  PrepackedWeightsContainer* const prepacked_weights_container_{};

#ifdef ENABLE_TRAINING
// Needed for ORTTrainer. Should be removed along with ORTTrainer code
#ifndef DISABLE_ABSEIL
  InlinedHashMap<InlinedVector<int>, InlinedHashSet<NodeIndex>> to_be_executed_nodes_;
#else
  std::map<InlinedVector<int>, InlinedHashSet<NodeIndex>> to_be_executed_nodes_;
#endif
#endif

  SessionState* parent_ = nullptr;
  // Assign each graph in each session an unique id.
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  int graph_id_ = 0;
  int next_graph_id_ = 1;

  void GenerateGraphId() {
    SessionState* p = this;
    while (p->parent_ != nullptr) p = p->parent_;
    graph_id_ = p->next_graph_id_++;
  }
#endif

  // Counter for number of times pre-packing of weights was performed across kernels
  // part the model
  size_t number_of_prepacks_counter_ = 0;

  // Counter for number of times a shared version of the pre-packed weight corresponding to
  // a constant initialized weight was used by the session state
  size_t used_shared_pre_packed_weights_counter_ = 0;

#ifdef DEBUG_NODE_INPUTS_OUTPUTS
  // Counter for number of times the session graph has been executed
  size_t graph_executions_counter_ = 0;
#endif

#ifdef ORT_ENABLE_STREAM
  std::unique_ptr<IStreamCommandHandleRegistry> stream_handles_registry_;

  // lock for the device stream pool
  mutable OrtMutex device_stream_pool_mutex_;
  mutable std::vector<std::unique_ptr<DeviceStreamCollection>> device_stream_pool_;
  // flag to indicate whether current session using any EP that create device stream dynamically.
  bool has_device_stream_enabled_ep_ = false;
#endif
};

}  // namespace onnxruntime
