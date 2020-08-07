// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include "gsl/gsl"

#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/tensor.h"
#include "core/framework/func_api.h"
#include "core/framework/data_transfer.h"

namespace onnxruntime {
class GraphViewer;
class Node;
struct ComputeCapability;
class KernelRegistry;
class KernelRegistryManager;

/**
   Logical device representation.
*/
typedef std::map<int, AllocatorPtr> AllocatorMap;

// if we are export the fused function to dll, the function will still in the same binary as onnxruntime
// use std function to give execution provider some chance to capture some state.
using CreateFunctionStateFunc = std::function<int(ComputeContext*, FunctionState*)>;
using ComputeFunc = std::function<Status(FunctionState, const OrtApi*, OrtKernelContext*)>;
using DestroyFunctionStateFunc = std::function<void(FunctionState)>;

//unordered maps
using UnorderedMapStringToString = std::unordered_map<std::string, std::string>;

//data types for execution provider options
using ProviderOptionsVector = std::vector<UnorderedMapStringToString>;  
using ProviderOptionsMap = std::unordered_map<std::string, UnorderedMapStringToString>;  

struct NodeComputeInfo {
  CreateFunctionStateFunc create_state_func;
  ComputeFunc compute_func;
  DestroyFunctionStateFunc release_state_func;
};

class IExecutionProvider {
 protected:
  IExecutionProvider(const std::string& type) : type_{type} {}

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
  virtual std::shared_ptr<KernelRegistry> GetKernelRegistry() const;

  /**
     Get the device id of current execution provider
  */
  virtual int GetDeviceId() const { return -1; };

  /**
     Get execution provider's configurations. 
   */
  const UnorderedMapStringToString& GetProviderOptions() const { return provider_options_; }

  /**
     Store execution provider's configurations. 
   */
  void SetProviderOptions(UnorderedMapStringToString& options) { 
    provider_options_ = options;
  }

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
  virtual common::Status Sync() const;

  /**
     Called when InferenceSession::Run started
     NOTE that due to async execution in provider, the actual work of previous
     Run may not be finished on device This function should be regarded as the
     point after which a new Run would start to submit commands from CPU
  */
  virtual common::Status OnRunStart();

  /**
     Called when InferenceSession::Run ended
     NOTE that due to async execution in provider, the actual work of this Run
     may not be finished on device This function should be regarded as the point
     that all commands of current Run has been submmited by CPU
  */
  virtual common::Status OnRunEnd();

  /**
     Called when session creation is complete
     This provides an opportunity for execution providers to optionally synchronize and
     clean up its temporary resources to reduce memory and ensure the first run is fast.
  */
  virtual common::Status OnSessionInitializationEnd();

  void InsertAllocator(AllocatorPtr allocator);

  /**
  Given a list of fused_node, return create_state/compute/release_state func for each node.
  */
  virtual common::Status Compile(const std::vector<onnxruntime::Node*>& fused_node,
                                 std::vector<NodeComputeInfo>& node_compute_funcs);

  /**
  Given a list of fused_node, return a dll that expose functions for each node.
  For each node, there should be three symbols:
     Create_State_${node_name}
     Compute_${node_name}
     Release_State_${node_name}
  */
  virtual common::Status Compile(const std::vector<onnxruntime::Node*>& fused_node,
                                 std::string& dll_path);

  void SetLogger(const logging::Logger* logger) {
    logger_ = logger;
  }

  const logging::Logger* GetLogger() const {
    return logger_;
  }

 private:
  const std::string type_;
  AllocatorMap allocators_;
  //It will be set when this object is registered to a session
  const logging::Logger* logger_ = nullptr;
  // convenience list of the allocators so GetAllocatorList doesn't have to build a new vector each time
  // contains the same instances as allocators_
  std::vector<AllocatorPtr> allocator_list_;
  // It will be set when constructor is being called
  UnorderedMapStringToString provider_options_;
};
}  // namespace onnxruntime
