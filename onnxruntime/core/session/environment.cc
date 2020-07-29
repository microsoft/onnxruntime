// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/environment.h"
#include "core/framework/allocatormgr.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/operator_sets_ml.h"
#include "onnx/defs/operator_sets_training.h"
#ifndef DISABLE_CONTRIB_OPS
#include "core/graph/contrib_ops/contrib_defs.h"
#endif
#ifdef ML_FEATURIZERS
#include "core/graph/featurizers_ops/featurizers_defs.h"
#endif
#ifdef USE_DML
#include "core/graph/dml_ops/dml_defs.h"
#endif

#include "core/platform/env.h"
#include "core/util/thread_utils.h"

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
#include "core/platform/tracing.h"
#endif

#ifdef ENABLE_TRAINING
#include "orttraining/core/graph/training_op_defs.h"
#include "orttraining/core/graph/gradient_builder_registry.h"
#include "orttraining/core/graph/loss_function_registry.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "orttraining/core/graph/optimizer_graph_builder_registry.h"
#endif

namespace onnxruntime {
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

std::once_flag schemaRegistrationOnceFlag;

Status Environment::Create(std::unique_ptr<logging::LoggingManager> logging_manager,
                           std::unique_ptr<Environment>& environment,
                           const OrtThreadingOptions* tp_options,
                           bool create_global_thread_pools) {
  environment = std::unique_ptr<Environment>(new Environment());
  auto status = environment->Initialize(std::move(logging_manager), tp_options, create_global_thread_pools);
  return status;
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

  try {
    // Register Microsoft domain with min/max op_set version as 1/1.
    std::call_once(schemaRegistrationOnceFlag, []() {
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSDomain, 1, 1);
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSNchwcDomain, 1, 1);
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSFeaturizersDomain, 1, 1);
#ifdef USE_DML
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSDmlDomain, 1, 1);
#endif
// Register contributed schemas.
// The corresponding kernels are registered inside the appropriate execution provider.
#ifndef DISABLE_CONTRIB_OPS
      contrib::RegisterContribSchemas();
#endif
#ifdef ML_FEATURIZERS
      featurizers::RegisterMSFeaturizersSchemas();
#endif
#ifdef USE_DML
      dml::RegisterDmlSchemas();
#endif
      RegisterOnnxOperatorSetSchema();

#ifndef DISABLE_ML_OPS
      RegisterOnnxMLOperatorSetSchema();
#endif

#ifdef ENABLE_TRAINING
      RegisterOnnxTrainingOperatorSetSchema();
#endif

#ifdef ENABLE_TRAINING
      // preserve this order: this depends on operatorsetschema registration.
      training::RegisterTrainingOpSchemas();
      training::GradientBuilderRegistry::GetInstance().RegisterGradientBuilders();
      training::LossFunctionRegistry::GetInstance().RegisterNonOperatorLossFunctions();
      training::OptimizerBuilderRegistry::GetInstance().RegisterBuilders();
      training::OptimizerGraphBuilderRegistry::GetInstance().RegisterGraphBuilders();
#endif
    });

    // Register MemCpy schema;

    // These ops are internal-only, so register outside of onnx
    ORT_ATTRIBUTE_UNUSED ONNX_OPERATOR_SCHEMA(MemcpyFromHost)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .SetDoc(R"DOC(
Internal copy node
)DOC");

    ORT_ATTRIBUTE_UNUSED ONNX_OPERATOR_SCHEMA(MemcpyToHost)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output", "T")
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .SetDoc(R"DOC(
Internal copy node
)DOC");

    // fire off startup telemetry (this call is idempotent)
    const Env& env = Env::Default();
    env.GetTelemetryProvider().LogProcessInfo();
  } catch (std::exception& ex) {
    status = Status{ONNXRUNTIME, common::RUNTIME_EXCEPTION, std::string{"Exception caught: "} + ex.what()};
  } catch (...) {
    status = Status{ONNXRUNTIME, common::RUNTIME_EXCEPTION};
  }

  return status;
}

}  // namespace onnxruntime
