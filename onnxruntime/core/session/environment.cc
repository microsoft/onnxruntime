// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/environment.h"

#include <array>

#include "core/common/basic_types.h"
#include "core/framework/allocator.h"
#include "core/framework/allocator_utils.h"
#include "core/framework/error_code_helper.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"
#include "core/platform/device_discovery.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/allocator_adapters.h"
#include "core/session/inference_session.h"
#include "core/session/ep_factory_internal.h"
#include "core/session/ep_library_internal.h"
#include "core/session/ep_library_plugin.h"
#include "core/session/ep_library_provider_bridge.h"
#include "core/session/ort_apis.h"
#include "core/session/utils.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/operator_sets_ml.h"
#include "core/graph/contrib_ops/internal_nhwc_onnx_schemas.h"
#include "core/graph/contrib_ops/ms_opset.h"
#include "core/graph/contrib_ops/onnx_deprecated_opset.h"
#if defined(ENABLE_TRAINING_OPS)
#include "onnx/defs/operator_sets_training.h"
#endif
#endif
#ifndef DISABLE_CONTRIB_OPS
#include "core/graph/contrib_ops/contrib_defs.h"
#endif
#ifdef USE_DML
#include "core/graph/dml_ops/dml_defs.h"
#endif

#include "core/platform/env.h"
#include "core/util/thread_utils.h"

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
#include "core/platform/tracing.h"
#endif

#if defined(ENABLE_TRAINING_OPS)
#include "orttraining/core/graph/training_op_defs.h"
#endif
#ifdef ENABLE_TRAINING
#include "orttraining/core/graph/gradient_builder_registry.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "orttraining/core/graph/optimizer_graph_builder_registry.h"
#include "orttraining/core/graph/loss_function_registry.h"
#include "orttraining/core/optimizer/graph_transformer_registry.h"
#endif

#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#endif
namespace onnxruntime {
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

std::once_flag schemaRegistrationOnceFlag;
#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
ProviderInfo_CUDA& GetProviderInfo_CUDA();
#endif  // defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)

namespace {
// Ignore whether there is an arena wrapping the allocator by excluding OrtMemoryInfo.alloc_type from the comparison
static bool AreOrtMemoryInfosEquivalent(
    const OrtMemoryInfo& left, const OrtMemoryInfo& right,
    bool match_name = true) {
  return left.mem_type == right.mem_type &&
         left.device == right.device &&
         (!match_name || strcmp(left.name, right.name) == 0);
}

std::vector<AllocatorPtr>::const_iterator FindExistingAllocator(const std::vector<AllocatorPtr>& allocators,
                                                                const OrtMemoryInfo& mem_info,
                                                                bool match_name = true) {
  auto ite = std::find_if(std::begin(allocators),
                          std::end(allocators),
                          [&mem_info, match_name](const AllocatorPtr& alloc_ptr) {
                            // We want to do the equality checking of 2 OrtMemoryInfos sans the OrtAllocatorType field.
                            // This is because we want to avoid registering two allocators for the same device that just
                            // differ on OrtAllocatorType.
                            // To be more specific, we want to avoid the scenario where the user calls CreateAndRegiserAllocator()
                            // and registers the ORT-internal arena allocator and then tries to register their own custom
                            // allocator using RegisterAllocator() for the same device that has an OrtAllocatorType as
                            // OrtDeviceAllocator (which is the only accepted value while registering a custom allocator).
                            // If we allowed this, it could potentially cause a lot of confusion as to which shared allocator
                            // to use for that device and we want to avoid having any ugly logic around this.
                            return AreOrtMemoryInfosEquivalent(alloc_ptr->Info(), mem_info, match_name);
                          });

  return ite;
}

std::unordered_set<OrtAllocator*>::const_iterator FindExistingAllocator(const std::unordered_set<OrtAllocator*>& allocators,
                                                                        const OrtMemoryInfo& mem_info,
                                                                        bool match_name = true) {
  return std::find_if(std::begin(allocators),
                      std::end(allocators),
                      [&mem_info, match_name](const OrtAllocator* alloc_ptr) {
                        const auto* alloc_mem_info = alloc_ptr->Info(alloc_ptr);
                        return AreOrtMemoryInfosEquivalent(*alloc_mem_info, mem_info, match_name);
                      });
}
}  // namespace

Status Environment::Create(std::unique_ptr<logging::LoggingManager> logging_manager,
                           std::unique_ptr<Environment>& environment,
                           const OrtThreadingOptions* tp_options,
                           bool create_global_thread_pools) {
  environment = std::make_unique<Environment>();
  auto status = environment->Initialize(std::move(logging_manager), tp_options, create_global_thread_pools);
  return status;
}

Status Environment::RegisterAllocator(OrtAllocator* allocator) {
  std::lock_guard<std::mutex> lock{mutex_};

  auto allocator_ptr = std::make_shared<IAllocatorImplWrappingOrtAllocator>(allocator);

  // for the public API we always want to replace any existing allocator for the device.
  auto status = RegisterAllocatorImpl(allocator_ptr);

  // update shared_ort_allocators_
  if (status.IsOK()) {
    if (auto it = FindExistingAllocator(shared_ort_allocators_, *allocator->Info(allocator), /*match_name*/ true);
        it != shared_ort_allocators_.end()) {
      shared_ort_allocators_.erase(it);
    }

    shared_ort_allocators_.insert(allocator);
  }

  return status;
}

Status Environment::RegisterAllocatorImpl(AllocatorPtr allocator) {
  const auto& mem_info = allocator->Info();

  const bool match_name = false;
  if (FindExistingAllocator(shared_allocators_, mem_info, match_name) != shared_allocators_.end()) {
    ORT_RETURN_IF_ERROR(UnregisterAllocatorImpl(mem_info, match_name));
  }

  shared_allocators_.push_back(std::move(allocator));

  return Status::OK();
}
Status Environment::UnregisterAllocator(const OrtMemoryInfo& mem_info) {
  std::lock_guard<std::mutex> lock{mutex_};

  return UnregisterAllocatorImpl(mem_info);
}

Status Environment::UnregisterAllocatorImpl(const OrtMemoryInfo& mem_info, bool error_if_not_found) {
  auto it = FindExistingAllocator(shared_allocators_, mem_info);

  if (error_if_not_found && it == shared_allocators_.end()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "No allocator for this device has been registered for sharing.");
  }

  // we need to remove from shared_ort_allocators_ first in case the entry in shared_allocators_ owns the pointer in
  // shared_ort_allocators_
  // e.g. a plug-in EP allocator is an IAllocatorImplWrappingOrtAllocator that owns the OrtAllocator* created by the EP
  // so when we remove that from shared_allocators_ we release the OrtAllocator instance.

  // shared_ort_allocators_ are internal only so never an error if there's no match
  auto it2 = FindExistingAllocator(shared_ort_allocators_, mem_info);
  if (it2 != shared_ort_allocators_.end()) {
    shared_ort_allocators_.erase(it2);
  }

  shared_allocators_.erase(it);

  return Status::OK();
}

Status Environment::CreateAndRegisterAllocator(const OrtMemoryInfo& mem_info, const OrtArenaCfg* arena_cfg) {
  // TODO should we allow sharing of non-CPU allocators?
  if (mem_info.device.Type() != OrtDevice::CPU) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Only CPU devices are supported. Please call CreateAndRegisterAllocatorV2() for other device.");
  }

  // determine if arena should be used
  const bool create_arena = DoesCpuAllocatorSupportArenaUsage()
                                ? (mem_info.alloc_type == OrtArenaAllocator)
                                : false;

  AllocatorPtr allocator_ptr;
  // create appropriate DeviceAllocatorRegistrationInfo and allocator based on create_arena
  if (create_arena) {
    // defaults in case arena_cfg is nullptr (not supplied by the user)
    size_t max_mem = 0;
    int arena_extend_strategy = -1;
    int initial_chunk_size_bytes = -1;
    int max_dead_bytes_per_chunk = -1;
    int initial_growth_chunk_size_bytes = -1;
    int64_t max_power_of_two_extend_bytes = -1L;

    // override with values from the user supplied arena_cfg object
    if (arena_cfg) {
      max_mem = arena_cfg->max_mem;
      arena_extend_strategy = arena_cfg->arena_extend_strategy;
      initial_chunk_size_bytes = arena_cfg->initial_chunk_size_bytes;
      max_dead_bytes_per_chunk = arena_cfg->max_dead_bytes_per_chunk;
      initial_growth_chunk_size_bytes = arena_cfg->initial_growth_chunk_size_bytes;
      max_power_of_two_extend_bytes = arena_cfg->max_power_of_two_extend_bytes;
    }

    OrtArenaCfg l_arena_cfg{max_mem, arena_extend_strategy, initial_chunk_size_bytes, max_dead_bytes_per_chunk,
                            initial_growth_chunk_size_bytes, max_power_of_two_extend_bytes};
    AllocatorCreationInfo alloc_creation_info{
        [mem_info](int) { return std::make_unique<CPUAllocator>(mem_info); },
        0,
        create_arena,
        l_arena_cfg};
    allocator_ptr = CreateAllocator(alloc_creation_info);
  } else {
    AllocatorCreationInfo alloc_creation_info{[mem_info](int) { return std::make_unique<CPUAllocator>(mem_info); },
                                              0, create_arena};
    allocator_ptr = CreateAllocator(alloc_creation_info);
  }

  std::lock_guard<std::mutex> lock{mutex_};
  return RegisterAllocatorImpl(allocator_ptr);
}

Status Environment::Initialize(std::unique_ptr<logging::LoggingManager> logging_manager,
                               const OrtThreadingOptions* tp_options,
                               bool create_global_thread_pools) {
  auto status = Status::OK();

  logging_manager_ = std::move(logging_manager);

  // create thread pools
  if (create_global_thread_pools) {
    ORT_RETURN_IF_ERROR(SetGlobalThreadingOptions(*tp_options));
  }

  ORT_TRY {
#if !defined(ORT_MINIMAL_BUILD)
    // Register Microsoft domain with min/max op_set version as 1/1.
    std::call_once(schemaRegistrationOnceFlag, []() {
      auto& domainToVersionRangeInstance = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
      if (domainToVersionRangeInstance.Map().find(onnxruntime::kMSDomain) == domainToVersionRangeInstance.Map().end()) {
        // External shared providers may have already added kMSDomain
        domainToVersionRangeInstance.AddDomainToVersion(onnxruntime::kMSDomain, 1, 1);
      }
      domainToVersionRangeInstance.AddDomainToVersion(onnxruntime::kMSExperimentalDomain, 1, 1);
      domainToVersionRangeInstance.AddDomainToVersion(onnxruntime::kMSNchwcDomain, 1, 1);

      // we have static registrations for NHWC versions of ONNX operators so this domain needs to extend to the
      // latest ONNX version
      auto onnx_version = domainToVersionRangeInstance.LastReleaseVersionMap()
                              .find(ONNX_NAMESPACE::ONNX_DOMAIN)
                              ->second;
      domainToVersionRangeInstance.AddDomainToVersion(onnxruntime::kMSInternalNHWCDomain, 1, onnx_version);

      domainToVersionRangeInstance.AddDomainToVersion(onnxruntime::kPytorchAtenDomain, 1, 1);
#ifdef USE_DML
      domainToVersionRangeInstance.AddDomainToVersion(onnxruntime::kMSDmlDomain, 1, 1);
#endif
// Register contributed schemas.
// The corresponding kernels are registered inside the appropriate execution provider.
#ifndef DISABLE_CONTRIB_OPS
      RegisterOpSetSchema<contrib::OpSet_Microsoft_ver1>();
      RegisterOpSetSchema<contrib::OpSet_ONNX_Deprecated>();
      // internal opset that has NHWC versions of ONNX operators
      RegisterOpSetSchema<internal_nhwc_onnx::OpSet_Internal_NHWC_ONNX>();
      contrib::RegisterContribSchemas();
#endif

#ifdef USE_DML
      dml::RegisterDmlSchemas();
#endif
      RegisterOnnxOperatorSetSchema();

#ifndef DISABLE_ML_OPS
      RegisterOnnxMLOperatorSetSchema();
#endif

#if defined(ENABLE_TRAINING_OPS)
      RegisterOnnxTrainingOperatorSetSchema();
#endif

#if defined(ENABLE_TRAINING_OPS)
      // preserve this order until <training schemas>: this depends on operatorsetschema registration.
      training::RegisterTrainingOpSchemas();
#endif
#ifdef ENABLE_TRAINING
      training::GradientBuilderRegistry::GetInstance().RegisterGradientBuilders();
      training::LossFunctionRegistry::GetInstance().RegisterNonOperatorLossFunctions();
      training::OptimizerBuilderRegistry::GetInstance().RegisterBuilders();
      training::OptimizerGraphBuilderRegistry::GetInstance().RegisterGraphBuilders();
      // <training schemas>
      // This was added for a partner team and is most probably no longer in use.
      // Can be removed once we have the confirmation.
      training::GraphTransformerRegistry::GetInstance().RegisterExternalGraphTransformers();
#endif
    });

    // Register MemCpy schema;

    // These ops are internal-only, so register outside of onnx

    std::vector<std::string> all_fixed_size_types;

    std::vector<std::string> all_tensor_types = OpSchema::all_tensor_types_ir9();
    std::vector<std::string> all_sequence_types = OpSchema::all_tensor_sequence_types();
    all_fixed_size_types.insert(all_fixed_size_types.end(), all_tensor_types.begin(), all_tensor_types.end());
    all_fixed_size_types.insert(all_fixed_size_types.end(), all_sequence_types.begin(), all_sequence_types.end());
    all_fixed_size_types.emplace_back("seq(tensor(bfloat16))");
    all_fixed_size_types.erase(std::remove_if(all_fixed_size_types.begin(), all_fixed_size_types.end(),
                                              [](const std::string& s) { return s.find("string") != std::string::npos; }),
                               all_fixed_size_types.end());

    ORT_ATTRIBUTE_UNUSED ONNX_OPERATOR_SCHEMA(MemcpyFromHost)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output", "T")
        .TypeConstraint(
            "T",
            all_fixed_size_types,
            "Constrain to all fixed size tensor and sequence types. If the dtype attribute is not provided this must be a valid output type.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .SetDoc(R"DOC(
Internal copy node
)DOC");

    ORT_ATTRIBUTE_UNUSED ONNX_OPERATOR_SCHEMA(MemcpyToHost)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output", "T")
        .TypeConstraint(
            "T",
            all_fixed_size_types,
            "Constrain to all fixed size tensor and sequence types. If the dtype attribute is not provided this must be a valid output type.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .SetDoc(R"DOC(
Internal copy node
)DOC");

#endif  // !defined(ORT_MINIMAL_BUILD)
    // fire off startup telemetry (this call is idempotent)
    const Env& env = Env::Default();
    env.GetTelemetryProvider().LogProcessInfo();

#if !defined(ORT_MINIMAL_BUILD)
    // register internal EPs for autoep selection
    ORT_RETURN_IF_ERROR(CreateAndRegisterInternalEps());
#endif
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ONNXRUNTIME, common::RUNTIME_EXCEPTION, std::string{"Exception caught: "} + ex.what());
    });
  }
  ORT_CATCH(...) {
    status = Status{ONNXRUNTIME, common::RUNTIME_EXCEPTION};
  }
  return status;
}

Status Environment::SetGlobalThreadingOptions(const OrtThreadingOptions& tp_options) {
  if (create_global_thread_pools_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Global thread pools have already been created, cannot replace them");
  }
  create_global_thread_pools_ = true;
  OrtThreadPoolParams to = tp_options.intra_op_thread_pool_params;
  if (to.name == nullptr) {
    to.name = ORT_TSTR("intra-op");
  }
  intra_op_thread_pool_ = concurrency::CreateThreadPool(&Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);
  to = tp_options.inter_op_thread_pool_params;
  if (to.name == nullptr) {
    to.name = ORT_TSTR("inter-op");
  }
  inter_op_thread_pool_ = concurrency::CreateThreadPool(&Env::Default(), to, concurrency::ThreadPoolType::INTER_OP);
  return Status::OK();
}

Status Environment::CreateAndRegisterAllocatorV2(const std::string& provider_type, const OrtMemoryInfo& mem_info,
                                                 const std::unordered_map<std::string, std::string>& options,
                                                 const OrtArenaCfg* arena_cfg) {
  if (provider_type == onnxruntime::kCpuExecutionProvider) {
    ORT_UNUSED_PARAMETER(options);
    return CreateAndRegisterAllocator(mem_info, arena_cfg);
  }

#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
  if (provider_type == onnxruntime::kCudaExecutionProvider) {
    if (mem_info.device.MemType() == OrtDevice::MemType::HOST_ACCESSIBLE) {
      AllocatorPtr allocator_ptr = GetProviderInfo_CUDA().CreateCUDAPinnedAllocator(
          static_cast<int16_t>(mem_info.device.Id()),
          onnxruntime::CUDA_PINNED);
      return RegisterAllocatorImpl(allocator_ptr);
    } else {
      CUDAExecutionProviderInfo cuda_ep_info;
      GetProviderInfo_CUDA().CUDAExecutionProviderInfo__FromProviderOptions(options, cuda_ep_info);
      CUDAExecutionProviderExternalAllocatorInfo external_info = cuda_ep_info.external_allocator_info;
      AllocatorPtr allocator_ptr = GetProviderInfo_CUDA().CreateCudaAllocator(
          static_cast<int16_t>(mem_info.device.Id()),
          arena_cfg->max_mem,
          static_cast<ArenaExtendStrategy>(arena_cfg->arena_extend_strategy),
          external_info, arena_cfg);
      return RegisterAllocatorImpl(allocator_ptr);
    }
  }
#endif

  return Status{ONNXRUNTIME, common::INVALID_ARGUMENT,
                provider_type + " is not implemented in CreateAndRegisterAllocatorV2()"};
}

Environment::~Environment() = default;

AllocatorPtr Environment::GetRegisteredSharedAllocator(const OrtMemoryInfo& mem_info) const {
  std::lock_guard<std::mutex> lock{mutex_};

  auto it = FindExistingAllocator(shared_allocators_, mem_info, /*match_name*/ false);
  return it != shared_allocators_.end() ? *it : nullptr;
}

Status Environment::GetSharedAllocator(const OrtMemoryInfo& mem_info, OrtAllocator*& allocator) {
  std::lock_guard<std::mutex> lock{mutex_};

  // doesn't matter whether we match a custom allocator or an EP allocator so match_name is false
  auto it = FindExistingAllocator(shared_ort_allocators_, mem_info, /*match_name*/ false);
  allocator = it != shared_ort_allocators_.end() ? *it : nullptr;

  // use the default CPU allocator if there's no custom or EP provided CPU allocator
  if (!allocator && (mem_info.device.Type() == OrtDevice::CPU &&
                     mem_info.device.MemType() == OrtDevice::MemType::DEFAULT)) {
    if (!default_cpu_ort_allocator_) {
      auto cpu_ort_allocator = std::make_unique<OrtAllocatorImplWrappingIAllocator>(CPUAllocator::DefaultInstance());
      default_cpu_ort_allocator_ = std::move(cpu_ort_allocator);
    }

    allocator = default_cpu_ort_allocator_.get();
  }

  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD)
Status Environment::RegisterExecutionProviderLibrary(const std::string& registration_name,
                                                     std::unique_ptr<EpLibrary> ep_library,
                                                     const std::vector<EpFactoryInternal*>& internal_factories) {
  if (ep_libraries_.count(registration_name) > 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "library is already registered under ", registration_name);
  }

  auto status = Status::OK();

  ORT_TRY {
    // create the EpInfo which loads the library if required
    std::unique_ptr<EpInfo> ep_info = nullptr;
    ORT_RETURN_IF_ERROR(EpInfo::Create(std::move(ep_library), ep_info));

    // add the pointers to the OrtEpDevice instances to our global list
    execution_devices_.reserve(execution_devices_.size() + ep_info->execution_devices.size());
    for (const auto& ed : ep_info->execution_devices) {
      execution_devices_.push_back(ed.get());

      // add shared allocators so they're available without an inference session being required.
      // we don't replace an existing allocator as we just need one to exist for the OrtMemoryInfo and we don't want
      // to blow away any custom allocators previously added by the user.
      if (ed->device_memory_info != nullptr) {
        ORT_RETURN_IF_ERROR(CreateSharedAllocatorImpl(*ed, *ed->device_memory_info, OrtDeviceAllocator, nullptr,
                                                      nullptr, /*replace_existing*/ false));
      }

      if (ed->host_accessible_memory_info != nullptr) {
        ORT_RETURN_IF_ERROR(CreateSharedAllocatorImpl(*ed, *ed->host_accessible_memory_info, OrtDeviceAllocator,
                                                      nullptr, nullptr, /*replace_existing*/ false));
      }
    }

    for (const auto& internal_factory : internal_factories) {
      internal_ep_factories_.insert(internal_factory);
    }

    ep_libraries_[registration_name] = std::move(ep_info);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to register EP library under '", registration_name, "' with error: ", ex.what());
    });
  }

  return status;
}

Status Environment::CreateAndRegisterInternalEps() {
  auto internal_ep_libraries = EpLibraryInternal::CreateInternalEps();
  for (auto& ep_library : internal_ep_libraries) {
    // we do a std::move in the function call so need a valid pointer for the args after the move
    auto* internal_library_ptr = ep_library.get();
    ORT_RETURN_IF_ERROR(RegisterExecutionProviderLibrary(internal_library_ptr->RegistrationName(),
                                                         std::move(ep_library),
                                                         {&internal_library_ptr->GetInternalFactory()}));
  }

  return Status::OK();
}

Status Environment::RegisterExecutionProviderLibrary(const std::string& registration_name, const ORTCHAR_T* lib_path) {
  std::lock_guard<std::mutex> lock{mutex_};

  std::vector<EpFactoryInternal*> internal_factories = {};
  std::unique_ptr<EpLibrary> ep_library;

  // This will create an EpLibraryPlugin or an EpLibraryProviderBridge depending on what the library supports.
  ORT_RETURN_IF_ERROR(LoadPluginOrProviderBridge(registration_name, lib_path, ep_library,
                                                 internal_factories));

  return RegisterExecutionProviderLibrary(registration_name, std::move(ep_library), internal_factories);
}

Status Environment::UnregisterExecutionProviderLibrary(const std::string& ep_name) {
  std::lock_guard<std::mutex> lock{mutex_};

  if (ep_libraries_.count(ep_name) == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Execution provider library: ", ep_name, " was not registered.");
  }

  auto status = Status::OK();

  ORT_TRY {
    auto ep_info = std::move(ep_libraries_[ep_name]);
    // remove from map and global list of OrtEpDevice* before unloading so we don't get a leftover entry if
    // something goes wrong in any of the following steps..
    ep_libraries_.erase(ep_name);

    for (auto* internal_factory : ep_info->internal_factories) {
      internal_ep_factories_.erase(internal_factory);
    }

    for (const auto& ed : ep_info->execution_devices) {
      // remove from global list of OrtEpDevices
      if (auto it = std::find(execution_devices_.begin(), execution_devices_.end(), ed.get());
          it != execution_devices_.end()) {
        execution_devices_.erase(it);
      }

      // unregister any shared allocators.
      // match only the OrtEpDevice allocator in case the user registered a custom allocator with matching info.
      const bool error_if_not_found = false;
      if (ed->device_memory_info != nullptr) {
        ORT_RETURN_IF_ERROR(UnregisterAllocatorImpl(*ed->device_memory_info, error_if_not_found));
      }

      if (ed->host_accessible_memory_info != nullptr) {
        ORT_RETURN_IF_ERROR(UnregisterAllocatorImpl(*ed->host_accessible_memory_info, error_if_not_found));
      }
    }

    ep_info.reset();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to unregister EP library: ", ep_name, " with error: ",
                               ex.what());
    });
  }

  return status;
}

Status Environment::CreateSharedAllocator(const OrtEpDevice& ep_device,
                                          OrtDeviceMemoryType mem_type, OrtAllocatorType allocator_type,
                                          const OrtKeyValuePairs* allocator_options,
                                          OrtAllocator** allocator_out) {
  auto* memory_info = mem_type == OrtDeviceMemoryType_DEFAULT ? ep_device.device_memory_info
                                                              : ep_device.host_accessible_memory_info;
  if (memory_info == nullptr) {
    return Status(ONNXRUNTIME, ORT_INVALID_ARGUMENT, "Invalid memory type for OrtEpDevice.");
  }

  std::lock_guard<std::mutex> lock{mutex_};
  return CreateSharedAllocatorImpl(ep_device, *memory_info, allocator_type, allocator_options, allocator_out,
                                   /*replace_existing*/ true);
}

Status Environment::CreateSharedAllocatorImpl(const OrtEpDevice& ep_device,
                                              const OrtMemoryInfo& memory_info,
                                              OrtAllocatorType allocator_type,
                                              const OrtKeyValuePairs* allocator_options,
                                              OrtAllocator** allocator_out,
                                              bool replace_existing) {
  // if we're replacing an existing allocator we don't care who added it
  if (auto it = FindExistingAllocator(shared_allocators_, memory_info, /*match_name*/ false);
      it != shared_allocators_.end()) {
    if (!replace_existing) {
      return Status::OK();
    }

    shared_allocators_.erase(it);
  }

  // clear out any exact match in the internal shared allocators
  if (auto it = FindExistingAllocator(shared_ort_allocators_, memory_info, /*match_name*/ true);
      it != shared_ort_allocators_.end()) {
    shared_ort_allocators_.erase(it);
  }

  OrtAllocator* allocator = nullptr;
  auto* ort_status = ep_device.ep_factory->CreateAllocator(ep_device.ep_factory, &memory_info, allocator_options,
                                                           &allocator);
  if (ort_status != nullptr) {
    return ToStatusAndRelease(ort_status);
  }

  if (allocator_out != nullptr) {
    *allocator_out = allocator;
  }

  auto ort_allocator = OrtAllocatorUniquePtr(allocator,
                                             [&ep_device](OrtAllocator* allocator) {
                                               ep_device.ep_factory->ReleaseAllocator(ep_device.ep_factory, allocator);
                                             });

  AllocatorPtr shared_allocator;

  if (allocator_type == OrtArenaAllocator) {
    // wrap with arena
    OrtArenaCfg arena_cfg;
    if (allocator_options != nullptr) {
      auto status = OrtArenaCfg::FromKeyValuePairs(*allocator_options, arena_cfg);
    }

    // pending Stream support being added to plugin EP API in separate PR
    // ep_device.ep_factory->IsStreamAware(ep_device.ep_factory);
    bool stream_aware_arena = false;

    AllocatorCreationInfo alloc_creation_info{
        [&ort_allocator](int) -> std::unique_ptr<IAllocator> {
          return std::make_unique<IAllocatorImplWrappingOrtAllocator>(std::move(ort_allocator));
        },
        /*unused*/ -1,  // arg to the lambda above that is ignored as the device id comes from the allocator
        /*create_arena*/ true,
        arena_cfg,
        stream_aware_arena,
    };

    shared_allocator = CreateAllocator(alloc_creation_info);
  } else {
    shared_allocator = std::make_shared<IAllocatorImplWrappingOrtAllocator>(std::move(ort_allocator));
  }

  shared_ort_allocators_.insert(allocator);
  shared_allocators_.push_back(std::move(shared_allocator));

  return Status::OK();
}

Status Environment::ReleaseSharedAllocator(const OrtEpDevice& ep_device, OrtDeviceMemoryType mem_type) {
  auto* memory_info = mem_type == OrtDeviceMemoryType_DEFAULT ? ep_device.device_memory_info
                                                              : ep_device.host_accessible_memory_info;
  if (memory_info == nullptr) {
    return Status(ONNXRUNTIME, ORT_INVALID_ARGUMENT, "Invalid memory type for OrtEpDevice.");
  }

  auto status = UnregisterAllocator(*memory_info);

  return status;
}

namespace {
std::vector<const OrtHardwareDevice*> SortDevicesByType() {
  auto& devices = DeviceDiscovery::GetDevices();
  std::vector<const OrtHardwareDevice*> sorted_devices;
  sorted_devices.reserve(devices.size());

  const auto select_by_type = [&](OrtHardwareDeviceType type) {
    for (const auto& device : devices) {
      if (device.type == type) {
        sorted_devices.push_back(&device);
      }
    }
  };

  select_by_type(OrtHardwareDeviceType_NPU);
  select_by_type(OrtHardwareDeviceType_GPU);
  select_by_type(OrtHardwareDeviceType_CPU);

  return sorted_devices;
}
}  // namespace

Status Environment::EpInfo::Create(std::unique_ptr<EpLibrary> library_in, std::unique_ptr<EpInfo>& out,
                                   const std::vector<EpFactoryInternal*>& internal_factories) {
  if (!library_in) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "EpLibrary was null");
  }

  out.reset(new EpInfo());  // can't use make_unique with private ctor
  EpInfo& instance = *out;
  instance.library = std::move(library_in);
  instance.internal_factories = internal_factories;

  ORT_RETURN_IF_ERROR(instance.library->Load());
  const auto& factories = instance.library->GetFactories();

  // OrtHardwareDevice instances to pass to GetSupportedDevices. sorted by type to be slightly more structured.
  // the set of hardware devices is static so this can also be static.
  const static std::vector<const OrtHardwareDevice*> sorted_devices = SortDevicesByType();

  for (auto* factory_ptr : factories) {
    ORT_ENFORCE(factory_ptr != nullptr, "Factory pointer was null. EpLibrary should prevent this. Library:",
                instance.library->RegistrationName());

    auto& factory = *factory_ptr;

    std::array<OrtEpDevice*, 8> ep_devices{nullptr};
    size_t num_ep_devices = 0;
    ORT_RETURN_IF_ERROR(ToStatusAndRelease(
        factory.GetSupportedDevices(&factory, sorted_devices.data(), sorted_devices.size(),
                                    ep_devices.data(), ep_devices.size(), &num_ep_devices)));

    for (size_t i = 0; i < num_ep_devices; ++i) {
      if (ep_devices[i] != nullptr) {                            // should never happen but just in case...
        instance.execution_devices.emplace_back(ep_devices[i]);  // take ownership
      }
    }
  }

  return Status::OK();
}

Environment::EpInfo::~EpInfo() {
  execution_devices.clear();
  auto status = library->Unload();
  if (!status.IsOK()) {
    LOGS_DEFAULT(WARNING) << "Failed to unload EP library registered under '" << library->RegistrationName()
                          << "' with error: " << status.ErrorMessage();
  }
}

#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace onnxruntime
