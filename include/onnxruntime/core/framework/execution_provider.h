// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/tensor.h"
#include "core/framework/data_transfer.h"

namespace onnxruntime {

class GraphViewer;
class Node;
struct ComputeCapability;
class KernelRegistry;
class KernelRegistryManager;
}  // namespace onnxruntime
#endif

#include "core/framework/provider_options.h"
#include "core/framework/func_api.h"
#include "core/framework/allocatormgr.h"

namespace onnxruntime {

/**
   Logical device representation.
*/
using AllocatorMap = std::unordered_map<int, AllocatorPtr>;
using MemoryInfoSet = std::set<OrtMemoryInfo>;

// if we are export the fused function to dll, the function will still in the same binary as onnxruntime
// use std function to give execution provider some chance to capture some state.
using CreateFunctionStateFunc = std::function<int(ComputeContext*, FunctionState*)>;
using ComputeFunc = std::function<Status(FunctionState, const OrtApi*, OrtKernelContext*)>;
using DestroyFunctionStateFunc = std::function<void(FunctionState)>;

struct NodeComputeInfo {
  CreateFunctionStateFunc create_state_func;
  ComputeFunc compute_func;
  DestroyFunctionStateFunc release_state_func;
};

class IExecutionProvider {
 protected:
  IExecutionProvider(const std::string& type, bool use_metadef_id_creator = false)
      : type_{type} {
    if (use_metadef_id_creator) {
      metadef_id_generator_ = std::make_unique<ModelMetadefIdGenerator>();
    }
  }

 public:
  virtual ~IExecutionProvider() = default;

  /**
     Get all IAllocators for <*this> execution provider.
  */
  const std::vector<AllocatorPtr>& GetAllocators() const {
    return allocator_list_;
  }

  /**
   * Get an allocator with specified device id and MemType. Return nullptr if it doesn't exist
   */
  virtual AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const;

  /**
   * Returns a data transfer object that implements methods to copy to and
   * from this device.
   * If no copy is required for the successful operation of this provider,
   * return a nullptr.
   */
  virtual std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const {
    return nullptr;
  }

  /**
     Get execution provider's capability for the specified <graph>.
     Return a bunch of IndexedSubGraphs <*this> execution provider can run if
     the sub-graph contains only one node or can fuse to run if the sub-graph
     contains more than one node. The node indexes contained in sub-graphs may
     have overlap, and it's ONNXRuntime's responsibility to do the partition
     and decide whether a node will be assigned to <*this> execution provider.
  */
  virtual std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& kernel_registries) const;

  /**
     Get kernel registry per execution provider type.
     The KernelRegistry share pointer returned is shared across sessions.

     NOTE: this is a tricky but final solution to achieve following goals,
     1. The execution provider type based kernel registry should be shared
     across sessions.
     Only one copy of this kind of kernel registry exists in ONNXRuntime
     with multiple sessions/models.
     2. Adding an execution provider into ONNXRuntime does not need to touch ONNXRuntime
     frameowrk/session code.
     3. onnxruntime (framework/session) does not depend on any specific
     execution provider lib.
  */
  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const { return nullptr; }

  /**
     Get the device id of current execution provider
  */
  virtual int GetDeviceId() const { return -1; };

  /**
     Get execution provider's configuration options.
   */
  virtual ProviderOptions GetProviderOptions() const { return {}; }

  /**
     Returns an opaque handle whose exact type varies based on the provider
     and is interpreted accordingly by the corresponding kernel implementation.
     For Direct3D operator kernels, this may return an IUnknown supporting
     QueryInterface to ID3D12GraphicsCommandList1.
  */
  virtual const void* GetExecutionHandle() const noexcept {
    return nullptr;
  }

  /**
     @return type of the execution provider; should match that set in the node
     through the SetExecutionProvider API. Example valid return values are:
     kCpuExecutionProvider, kCudaExecutionProvider
  */
  const std::string& Type() const { return type_; }

  /**
     Blocks until the device has completed all preceding requested tasks.
     Currently this is primarily used by the IOBinding object to ensure that all
     inputs have been copied to the device before execution begins.
  */
  virtual common::Status Sync() const { return Status::OK(); }

  /**
     Called when InferenceSession::Run started
     NOTE that due to async execution in provider, the actual work of previous
     Run may not be finished on device This function should be regarded as the
     point after which a new Run would start to submit commands from CPU
  */
  virtual common::Status OnRunStart() { return Status::OK(); }

  /**
     Called when InferenceSession::Run ended
     NOTE that due to async execution in provider, the actual work of this Run
     may not be finished on device This function should be regarded as the point
     that all commands of current Run has been submmited by CPU
  */
  virtual common::Status OnRunEnd() { return Status::OK(); }

  /**
     Called when session creation is complete
     This provides an opportunity for execution providers to optionally synchronize and
     clean up its temporary resources to reduce memory and ensure the first run is fast.
  */
  virtual common::Status OnSessionInitializationEnd() { return Status::OK(); }

  virtual common::Status SetComputeStream(void*) { return Status::OK(); }
  virtual void* GetComputeStream() const { return nullptr; }

  void InsertAllocator(AllocatorPtr allocator);
  void ReplaceAllocator(AllocatorPtr allocator);
  // TODO: temparary sulotion, need to unify the interface in EP and AllocatorManager
  void TryInsertAllocator(AllocatorPtr allocator);

  // creation of a fused node is not supported in a minimal build, so any EP enabled in that scenario must support
  // compilation via GraphViewer instances.
#if !defined(ORT_MINIMAL_BUILD)
  /**
  Given a list of fused_node, return create_state/compute/release_state func for each node.
  */
  virtual common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                 std::vector<NodeComputeInfo>& node_compute_funcs);

  /**
  Given a list of fused_node, return a dll that expose functions for each node.
  For each node, there should be three symbols:
     Create_State_${node_name}
     Compute_${node_name}
     Release_State_${node_name}
  */
  virtual common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                                 std::string& dll_path);

#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  struct FusedNodeAndGraph {
    const std::reference_wrapper<onnxruntime::Node> fused_node;
    // GraphViewer that filters the full graph to the nodes that are covered by 'node'
    const std::reference_wrapper<GraphViewer> filtered_graph;
  };

  /**
  Given a collection of fused Nodes and the respective GraphViewer instance for the nodes that were fused,
  return create_state/compute/release_state func for each node.
  @remarks This is an optional interface that is only needed if the execution provider compiles nodes
           in a scenario involving the minimal build. i.e. on a mobile or embedded device with ORT format model.

           Do NOT cache the GraphViewer in FusedNodeAndGraph.filtered_graph in any of the NodeComputeInfo functions
           as it is only valid for the duration of the call to Compile.
  */
  virtual common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                 std::vector<NodeComputeInfo>& node_compute_funcs);
#endif

  // Fusion approach that is suppported
  enum class FusionStyle {
    // The node fusion will create an onnxruntime::Function based Node that contains a completely new Graph instance
    // in the Node body. The original nodes and initializers are copied to the new Graph instance in Function::Body().
    // A GraphProto can be produced from the Node body.
    Function,

    // The node fusion will create a new Node that defines the inputs and outputs using the IndexedSubGraph
    // that GetCapability returned. The Node will not be onnxruntime::Function based so will have no Body().
    // Instead a GraphViewer that filters the full Graph to the fused Nodes will be created.
    // This is significantly cheaper as it doesn't incur the cost of creating a new Graph instance,
    // and can be supported in a minimal build.
    FilteredGraphViewer
  };

  virtual FusionStyle GetFusionStyle() const {
    // existing EPs use this mode so default to it.
    // newer EPs that can use the cheaper approach, or need to run in a minimal build, should override to return
    // FilteredGraphViewer
    return FusionStyle::Function;
  }

  void SetLogger(const logging::Logger* logger) {
    logger_ = logger;
  }

  const logging::Logger* GetLogger() const {
    return logger_;
  }

  /** Generate a unique id that can be used in a MetaDef name. Values are unique for a model instance. 
   The model hash is also returned if you wish to include that in the MetaDef name to ensure uniqueness across models.
   @param graph_viewer[in] Graph viewer that GetCapability was called with. Can be for the main graph or nested graph.
   @param model_hash[out] Returns the hash for the main (i.e. top level) graph in the model. 
                          This is created using the model path if available, 
                          or the model input names and the output names from all nodes in the main graph.
   @remarks e.g. the TensorRT Execution Provider is used in multiple sessions and the underlying infrastructure caches
            compiled kernels, so the name must be unique and deterministic across models and sessions.
            NOTE: Ideally this would be a protected method, but to work across the EP bridge it has to be public and 
                  virtual, and ModelMetadefIdGenerator but be defined in the header as well.
   */
  virtual int GenerateMetaDefId(const onnxruntime::GraphViewer& graph_viewer, uint64_t& model_hash) const;

  /**
     Register allocators used for EP
     TODO: Used for CUDA & TRT only for now, will have one more PR to apply this for all EPs.
     EPs will have a shared pointer to allocator_manager, allocator_managerall will be the only place for allocators
  */
  virtual void RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager);

 private:
  const std::string type_;
  AllocatorMap allocators_;
  MemoryInfoSet mem_info_set_;  // to ensure only allocators with unique OrtMemoryInfo are registered in the provider.
  //It will be set when this object is registered to a session
  const logging::Logger* logger_ = nullptr;
  // convenience list of the allocators so GetAllocatorList doesn't have to build a new vector each time
  // contains the same instances as allocators_
  std::vector<AllocatorPtr> allocator_list_;

  // helper to generate ids that are unique to model and deterministic, even if the execution provider is shared across
  // multiple sessions.
  class ModelMetadefIdGenerator {
   public:
    int GenerateId(const onnxruntime::GraphViewer& graph_viewer, uint64_t& model_hash);

   private:
    std::unordered_map<uint64_t, int64_t> main_graph_hash_;  // map graph instance hash to model contents hash
    std::unordered_map<int64_t, int> model_metadef_id_;      // current unique id for model
  };

  std::unique_ptr<ModelMetadefIdGenerator> metadef_id_generator_;
};
}  // namespace onnxruntime
