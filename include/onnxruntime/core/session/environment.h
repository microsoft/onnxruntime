// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <filesystem>
#include <memory>

#include "core/common/common.h"
#include "core/common/basic_types.h"
#include "core/common/logging/logging.h"
#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/framework/execution_provider.h"
#include "core/platform/device_discovery.h"
#include "core/platform/threadpool.h"

#include "core/session/abi_devices.h"
#include "core/session/ep_library.h"
#include "core/session/onnxruntime_c_api.h"

struct OrtThreadingOptions;
namespace onnxruntime {
class EpFactoryInternal;
class InferenceSession;
struct IExecutionProviderFactory;
struct SessionOptions;

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
  Status RegisterAllocator(AllocatorPtr allocator);

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
#endif  // !defined(ORT_MINIMAL_BUILD)
  ~Environment();

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Environment);

  Status Initialize(std::unique_ptr<logging::LoggingManager> logging_manager,
                    const OrtThreadingOptions* tp_options = nullptr,
                    bool create_global_thread_pools = false);

  std::unique_ptr<logging::LoggingManager> logging_manager_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> intra_op_thread_pool_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> inter_op_thread_pool_;
  bool create_global_thread_pools_{false};
  std::vector<AllocatorPtr> shared_allocators_;

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
    std::vector<EpFactoryInternal*> internal_factories;  // factories that can create IExecutionProvider instances

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
#endif  // !defined(ORT_MINIMAL_BUILD)
};

}  // namespace onnxruntime
