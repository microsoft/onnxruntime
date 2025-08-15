// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <vector>
#include <string>

#include "core/common/common.h"
#include "core/common/basic_types.h"
#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/framework/execution_provider.h"
#include "core/framework/data_transfer_manager.h"
#include "core/platform/device_discovery.h"
#include "core/platform/threadpool.h"

#include "core/session/abi_devices.h"
#include "core/session/plugin_ep/ep_library.h"
#include "core/session/onnxruntime_c_api.h"

struct OrtThreadingOptions;
namespace onnxruntime {
class EpFactoryInternal;
class InferenceSession;
struct IExecutionProviderFactory;
struct OrtAllocatorImplWrappingIAllocator;
struct SessionOptions;

namespace plugin_ep {
class DataTransfer;
}  // namespace plugin_ep

/**
   Provides the runtime environment for onnxruntime.
   Create one instance for the duration of execution.
*/
class Environment {
 public:
  /**
     Create and initialize the runtime environment.
    @param logging manager instance that will enable per session logger output using
    session_options.session_logid as the logger id in messages.
    If nullptr, the default LoggingManager MUST have been created previously as it will be used
    for logging. This will use the default logger id in messages.
    See core/common/logging/logging.h for details, and how LoggingManager::DefaultLogger works.
    @param tp_options optional set of parameters controlling the number of intra and inter op threads for the global
    threadpools.
    @param create_global_thread_pools determine if this function will create the global threadpools or not.
  */
  static Status Create(std::unique_ptr<logging::LoggingManager> logging_manager,
                       std::unique_ptr<Environment>& environment,
                       const OrtThreadingOptions* tp_options = nullptr,
                       bool create_global_thread_pools = false);

  /**
   * Set the global threading options for the environment, if no global thread pools have been created yet.
   *
   * This function is not safe to call simultaneously from multiple threads, and will return a FAIL status on all calls
   * after the first.
   * @param tp_options set of parameters controlling the number of intra and inter op threads for the global
    threadpools.
   */
  Status SetGlobalThreadingOptions(const OrtThreadingOptions& tp_options);

  logging::LoggingManager* GetLoggingManager() const {
    return logging_manager_.get();
  }

  void SetLoggingManager(std::unique_ptr<onnxruntime::logging::LoggingManager> logging_manager) {
    logging_manager_ = std::move(logging_manager);
  }

  onnxruntime::concurrency::ThreadPool* GetIntraOpThreadPool() const {
    return intra_op_thread_pool_.get();
  }

  onnxruntime::concurrency::ThreadPool* GetInterOpThreadPool() const {
    return inter_op_thread_pool_.get();
  }

  bool EnvCreatedWithGlobalThreadPools() const {
    return create_global_thread_pools_;
  }

  /**
   * Registers an allocator for sharing between multiple sessions.
   * Return an error if an allocator with the same OrtMemoryInfo is already registered.
   */
  Status RegisterAllocator(OrtAllocator* allocator);

  /**
   * Creates and registers an allocator for sharing between multiple sessions.
   * Return an error if an allocator with the same OrtMemoryInfo is already registered.
   */
  Status CreateAndRegisterAllocator(const OrtMemoryInfo& mem_info, const OrtArenaCfg* arena_cfg = nullptr);

  /**
   * Returns the list of registered allocators in this env.
   */
  const std::vector<AllocatorPtr>& GetRegisteredSharedAllocators() const {
    return shared_allocators_;
  }

  /**
   * Returns an AllocatorPtr for a shared IAllocator based allocator if it matches the memory info.
   * The OrtMemoryInfo name and whether it's an arena or device allocator is ignored in the lookup, as is the
   * alignment.
   * The user calling this function is not expected to know the alignment, and we expect the allocator instance to be
   * created with a valid alignment for the device.
   */
  AllocatorPtr GetRegisteredSharedAllocator(const OrtMemoryInfo& mem_info) const;

  /**
   * Removes registered allocator that was previously registered for sharing between multiple sessions.
   */
  Status UnregisterAllocator(const OrtMemoryInfo& mem_info);

  Environment() = default;

  /**
   * Create and register an allocator, specified by provider_type, for sharing between multiple sessions.
   * Return an error if an allocator with the same OrtMemoryInfo is already registered.
   * For provider_type please refer core/graph/constants.h
   */
  Status CreateAndRegisterAllocatorV2(const std::string& provider_type, const OrtMemoryInfo& mem_info,
                                      const std::unordered_map<std::string, std::string>& options,
                                      const OrtArenaCfg* arena_cfg = nullptr);

#if !defined(ORT_MINIMAL_BUILD)
  Status RegisterExecutionProviderLibrary(const std::string& registration_name, const ORTCHAR_T* lib_path);
  Status UnregisterExecutionProviderLibrary(const std::string& registration_name);

  // convert an OrtEpFactory* to EpFactoryInternal* if possible.
  EpFactoryInternal* GetEpFactoryInternal(OrtEpFactory* factory) const {
    // we're comparing pointers so the reinterpret_cast should be safe
    auto it = internal_ep_factories_.find(reinterpret_cast<EpFactoryInternal*>(factory));
    return it != internal_ep_factories_.end() ? *it : nullptr;
  }

  const std::vector<const OrtEpDevice*>& GetOrtEpDevices() const {
    return execution_devices_;
  }

  Status CreateSharedAllocator(const OrtEpDevice& ep_device,
                               OrtDeviceMemoryType mem_type, OrtAllocatorType allocator_type,
                               const OrtKeyValuePairs* allocator_options, OrtAllocator** allocator);
  Status ReleaseSharedAllocator(const OrtEpDevice& ep_device, OrtDeviceMemoryType mem_type);

  const DataTransferManager& GetDataTransferManager() const {
    return data_transfer_mgr_;
  }
#endif  // !defined(ORT_MINIMAL_BUILD)

  // return a shared allocator from a plugin EP or custom allocator added with RegisterAllocator
  Status GetSharedAllocator(const OrtMemoryInfo& mem_info, OrtAllocator*& allocator);

  ~Environment();

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Environment);

  Status Initialize(std::unique_ptr<logging::LoggingManager> logging_manager,
                    const OrtThreadingOptions* tp_options = nullptr,
                    bool create_global_thread_pools = false);

  Status RegisterAllocatorImpl(AllocatorPtr allocator);
  Status UnregisterAllocatorImpl(const OrtMemoryInfo& mem_info, bool error_if_not_found = true);
  Status CreateSharedAllocatorImpl(const OrtEpDevice& ep_device,
                                   const OrtMemoryInfo& memory_info, OrtAllocatorType allocator_type,
                                   const OrtKeyValuePairs* allocator_options, OrtAllocator** allocator,
                                   bool replace_existing);

  std::unique_ptr<logging::LoggingManager> logging_manager_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> intra_op_thread_pool_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> inter_op_thread_pool_;
  bool create_global_thread_pools_{false};

  mutable std::mutex mutex_;

  // shared allocators from various sources.
  // CreateAndRegisterAllocator[V2]: IAllocator allocators created by ORT
  // RegisterAllocator: IAllocatorImplWrappingOrtAllocator custom allocators registered by the user.
  //                    TODO: How can we detect registration of an allocator from an InferenceSession?
  // OrtEpDevice: We create a default shared IAllocatorImplWrappingOrtAllocator for each OrtEpDevice memory info.
  std::vector<AllocatorPtr> shared_allocators_;

  // RegisterAllocator and CreateSharedAllocator pointers. Used for GetSharedAllocator.
  // Every instance here is also in shared_allocators_.
  std::unordered_set<OrtAllocator*> shared_ort_allocators_;

  // OrtAllocator wrapped CPUAllocator::DefaultInstance that is returned by GetSharedAllocator when no plugin EP is
  // providing a CPU allocator.
  std::unique_ptr<OrtAllocatorImplWrappingIAllocator> default_cpu_ort_allocator_;

  using OrtAllocatorUniquePtr = std::unique_ptr<OrtAllocator, std::function<void(OrtAllocator*)>>;

#if !defined(ORT_MINIMAL_BUILD)
  // register EPs that are built into the ORT binary so they can take part in AutoEP selection
  // added to ep_libraries
  Status CreateAndRegisterInternalEps();

  Status RegisterExecutionProviderLibrary(const std::string& registration_name,
                                          std::unique_ptr<EpLibrary> ep_library,
                                          const std::vector<EpFactoryInternal*>& internal_factories = {});

  struct EpInfo {
    // calls EpLibrary::Load
    // for each factory gets the OrtEpDevice instances and adds to execution_devices
    // internal_factory is set if this is an internal EP
    static Status Create(std::unique_ptr<EpLibrary> library_in, std::unique_ptr<EpInfo>& out,
                         const std::vector<EpFactoryInternal*>& internal_factories = {});

    // removes entries for this library from execution_devices
    // calls EpLibrary::Unload
    ~EpInfo();

    std::unique_ptr<EpLibrary> library;
    std::vector<std::unique_ptr<OrtEpDevice>> execution_devices;
    std::vector<OrtEpFactory*> factories;
    std::vector<EpFactoryInternal*> internal_factories;    // factories that can create IExecutionProvider instances
    std::vector<plugin_ep::DataTransfer*> data_transfers;  // data transfer instances for this EP.

   private:
    EpInfo() = default;
  };

  // registration name to EpInfo for library
  std::unordered_map<std::string, std::unique_ptr<EpInfo>> ep_libraries_;

  // combined set of OrtEpDevices for all registered OrtEpFactory instances
  // std::vector so we can use directly in GetEpDevices.
  // inefficient when EPs are unregistered but that is not expected to be a common operation.
  std::vector<const OrtEpDevice*> execution_devices_;

  // lookup set for internal EPs so we can create an IExecutionProvider directly
  std::unordered_set<EpFactoryInternal*> internal_ep_factories_;

  DataTransferManager data_transfer_mgr_;  // plugin EP IDataTransfer instances

#endif  // !defined(ORT_MINIMAL_BUILD)
};

}  // namespace onnxruntime
