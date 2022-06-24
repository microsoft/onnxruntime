// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/environment.h"
#include "core/session/allocator_adapters.h"
#include "core/framework/allocatormgr.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/operator_sets_ml.h"
#include "core/graph/contrib_ops/internal_nhwc_onnx_opset.h"
#include "core/graph/contrib_ops/ms_opset.h"
#include "core/graph/contrib_ops/onnx_deprecated_opset.h"
#if defined(ENABLE_TRAINING) || defined(ENABLE_TRAINING_OPS)
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

#if defined(ENABLE_TRAINING) || defined(ENABLE_TRAINING_OPS)
#include "orttraining/core/graph/training_op_defs.h"
#endif
#ifdef ENABLE_TRAINING
#include "orttraining/core/graph/gradient_builder_registry.h"
#include "orttraining/core/graph/loss_function_registry.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "orttraining/core/graph/optimizer_graph_builder_registry.h"
#include "orttraining/core/optimizer/graph_transformer_registry.h"

#endif

namespace onnxruntime {
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

std::once_flag schemaRegistrationOnceFlag;

Status Environment::Create(std::unique_ptr<logging::LoggingManager> logging_manager,
                           std::unique_ptr<Environment>& environment,
                           const OrtThreadingOptions* tp_options,
                           bool create_global_thread_pools) {
  environment = std::make_unique<Environment>();
  auto status = environment->Initialize(std::move(logging_manager), tp_options, create_global_thread_pools);
  return status;
}

// Ugly but necessary for instances where we want to check equality of two OrtMemoryInfos
// without accounting for OrtAllocatorType in the equality checking process.
// TODO: Should we remove the OrtAllocatorType field from the OrtMemoryInfo struct to
// avoid such problems and also remove the unintuitive phenomenon of binding an allocator
// type to OrtMemoryInfo (which loosely is just device info) ?
static bool AreOrtMemoryInfosEquivalent(
    const OrtMemoryInfo& left, const OrtMemoryInfo& right,
    bool include_allocator_type_for_equivalence_checking = true) {
  if (include_allocator_type_for_equivalence_checking) {
    return left == right;
  } else {
    return left.mem_type == right.mem_type &&
           left.id == right.id &&
           left.device == right.device &&
           strcmp(left.name, right.name) == 0;
  }
}

Status Environment::RegisterAllocator(AllocatorPtr allocator) {
  const auto& mem_info = allocator->Info();

  if (mem_info.device.Type() != OrtDevice::CPU) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Only CPU allocators can be shared between "
                           "multiple sessions for now.");
  }

  // We don't expect millions of allocators getting registered. Hence linear search should be fine.
  auto ite = std::find_if(std::begin(shared_allocators_),
                          std::end(shared_allocators_),
                          [&mem_info](const AllocatorPtr& alloc_ptr) {
                            // We want to do the equality checking of 2 OrtMemoryInfos sans the OrtAllocatorType field.
                            // This is because we want to avoid registering two allocators for the same device that just
                            // differ on OrtAllocatorType.
                            // To be more specific, we want to avoid the scenario where the user calls CreateAndRegiserAllocator()
                            // and registers the ORT-internal arena allocator and then tries to register their own custom
                            // allocator using RegisterAllocator() for the same device that has an OrtAllocatorType as
                            // OrtDeviceAllocator (which is the only accepted value while registering a custom allocator).
                            // If we allowed this, it could potentially cause a lot of confusion as to which shared allocator
                            // to use for that device and we want to avoid having any ugly logic around this.
                            return AreOrtMemoryInfosEquivalent(alloc_ptr->Info(), mem_info, false);
                          });

  if (ite != shared_allocators_.end()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "An allocator for this device has already been registered for sharing.");
  }

  shared_allocators_.insert(ite, allocator);

  return Status::OK();
}

Status Environment::CreateAndRegisterAllocator(const OrtMemoryInfo& mem_info, const OrtArenaCfg* arena_cfg) {
  // TODO should we allow sharing of non-CPU allocators?
  if (mem_info.device.Type() != OrtDevice::CPU) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Only CPU devices are supported for now.");
  }

  // determine if arena should be used
  bool create_arena = mem_info.alloc_type == OrtArenaAllocator;

#if defined(USE_JEMALLOC) || defined(USE_MIMALLOC)
  // We use these allocators instead of the arena
  create_arena = false;
#elif !(defined(__amd64__) || defined(_M_AMD64))
  // Disable Arena allocator for x86_32 build because it may run into infinite loop when integer overflow happens
  create_arena = false;
#endif

  AllocatorPtr allocator_ptr;
  // create appropriate DeviceAllocatorRegistrationInfo and allocator based on create_arena
  if (create_arena) {
    // defaults in case arena_cfg is nullptr (not supplied by the user)
    size_t max_mem = 0;
    int arena_extend_strategy = -1;
    int initial_chunk_size_bytes = -1;
    int max_dead_bytes_per_chunk = -1;
    int initial_growth_chunk_size_bytes = -1;

    // override with values from the user supplied arena_cfg object
    if (arena_cfg) {
      max_mem = arena_cfg->max_mem;

      arena_extend_strategy = arena_cfg->arena_extend_strategy;
      // validate the value here
      if (!(arena_extend_strategy == -1 || arena_extend_strategy == 0 || arena_extend_strategy == 1)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Received invalid value for arena extend strategy."
                               " Valid values can be either 0, 1 or -1.");
      }

      initial_chunk_size_bytes = arena_cfg->initial_chunk_size_bytes;
      max_dead_bytes_per_chunk = arena_cfg->max_dead_bytes_per_chunk;
      initial_growth_chunk_size_bytes = arena_cfg->initial_growth_chunk_size_bytes;
    }

    OrtArenaCfg l_arena_cfg{max_mem, arena_extend_strategy, initial_chunk_size_bytes, max_dead_bytes_per_chunk,
                            initial_growth_chunk_size_bytes};
    AllocatorCreationInfo alloc_creation_info{
        [mem_info](int) { return std::make_unique<CPUAllocator>(mem_info); },
        0,
        create_arena,
        l_arena_cfg};
    allocator_ptr = CreateAllocator(alloc_creation_info);
  } else {
    AllocatorCreationInfo alloc_creation_info{[](int) { return std::make_unique<CPUAllocator>(); },
                                              0, create_arena};
    allocator_ptr = CreateAllocator(alloc_creation_info);
  }

  return RegisterAllocator(allocator_ptr);
}

Status Environment::UnregisterAllocator(const OrtMemoryInfo& mem_info) {
  auto ite = std::find_if(std::begin(shared_allocators_),
                          std::end(shared_allocators_),
                          [&mem_info](const AllocatorPtr& alloc_ptr) {
                            // See comment in RegisterAllocator() as to why we
                            // use this method of OrtMemoryInfo equality checking
                            return AreOrtMemoryInfosEquivalent(alloc_ptr->Info(), mem_info, false);
                          });

  if (ite == shared_allocators_.end()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                  "No allocator for this device has been registered for sharing.");
  }

  shared_allocators_.erase(ite);

  return Status::OK();
}

Status Environment::Initialize(std::unique_ptr<logging::LoggingManager> logging_manager,
                               const OrtThreadingOptions* tp_options,
                               bool create_global_thread_pools) {
  auto status = Status::OK();

  logging_manager_ = std::move(logging_manager);

  // create thread pools
  if (create_global_thread_pools) {
    create_global_thread_pools_ = true;
    OrtThreadPoolParams to = tp_options->intra_op_thread_pool_params;
    if (to.name == nullptr) {
      to.name = ORT_TSTR("intra-op");
    }
    intra_op_thread_pool_ = concurrency::CreateThreadPool(&Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);
    to = tp_options->inter_op_thread_pool_params;
    if (to.name == nullptr) {
      to.name = ORT_TSTR("inter-op");
    }
    inter_op_thread_pool_ = concurrency::CreateThreadPool(&Env::Default(), to, concurrency::ThreadPoolType::INTER_OP);
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
#ifndef ORT_MINIMAL_BUILD
      RegisterOpSetSchema<contrib::OpSet_Microsoft_ver1>();
      RegisterOpSetSchema<contrib::OpSet_ONNX_Deprecated>();
      // internal opset that has NHWC versions of ONNX operators
      RegisterOpSetSchema<internal_nhwc_onnx::OpSet_Internal_NHWC_ONNX>();
#endif
      contrib::RegisterContribSchemas();
#endif

#ifdef USE_DML
      dml::RegisterDmlSchemas();
#endif
      RegisterOnnxOperatorSetSchema();

#ifndef DISABLE_ML_OPS
      RegisterOnnxMLOperatorSetSchema();
#endif

#if defined(ENABLE_TRAINING) || defined(ENABLE_TRAINING_OPS)
      RegisterOnnxTrainingOperatorSetSchema();
#endif

#if defined(ENABLE_TRAINING) || defined(ENABLE_TRAINING_OPS)
      // preserve this order until <training schemas>: this depends on operatorsetschema registration.
      training::RegisterTrainingOpSchemas();
#endif
#ifdef ENABLE_TRAINING
      training::GradientBuilderRegistry::GetInstance().RegisterGradientBuilders();
      training::LossFunctionRegistry::GetInstance().RegisterNonOperatorLossFunctions();
      training::OptimizerBuilderRegistry::GetInstance().RegisterBuilders();
      training::OptimizerGraphBuilderRegistry::GetInstance().RegisterGraphBuilders();
      // <training schemas>
      training::GraphTransformerRegistry::GetInstance().RegisterExternalGraphTransformers();
#endif
    });

    // Register MemCpy schema;

    // These ops are internal-only, so register outside of onnx
    static std::vector<std::string> all_fixed_size_types = []() {
      std::vector<std::string> all_types;
      std::vector<std::string> all_tensor_types = OpSchema::all_tensor_types_with_bfloat();
      std::vector<std::string> all_sequence_types = OpSchema::all_tensor_sequence_types();
      all_types.insert(all_types.end(), all_tensor_types.begin(), all_tensor_types.end());
      all_types.insert(all_types.end(), all_sequence_types.begin(), all_sequence_types.end());
      all_types.emplace_back("seq(tensor(bfloat16))");
      all_types.erase(std::remove_if(all_types.begin(), all_types.end(),
                      [](const std::string& s) { return s.find("string") != std::string::npos; }), all_types.end());
      return all_types; }();

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

}  // namespace onnxruntime
