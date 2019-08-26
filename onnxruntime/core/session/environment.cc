// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/environment.h"
#include "core/framework/allocatormgr.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/operator_sets-ml.h"
#ifndef DISABLE_CONTRIB_OPS
#include "core/graph/contrib_ops/contrib_defs.h"
#endif

namespace onnxruntime {
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

OrtMutex Environment::s_env_mutex_;
bool Environment::s_env_initialized_ = false;
logging::Logger* Environment::s_default_logger_ = nullptr;
  
Status Environment::Initialize(const std::string& default_logger_id) {
  std::lock_guard<OrtMutex> guard(s_env_mutex_);
  if (s_env_initialized_) {
    return Status{ONNXRUNTIME, common::FAIL, "Runtime environment has already been created."};
  }

  auto status = Status::OK();
  try {
    // Register Microsoft domain with min/max op_set version as 1/1.
    ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSDomain, 1, 1);
    ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSNchwcDomain, 1, 1);

    // Register contributed op schemas.
    // The corresponding kernels are registered inside the appropriate execution provider.
#ifndef DISABLE_CONTRIB_OPS
    contrib::RegisterContribSchemas();
#endif

    // Register ONNX op schemas.
    RegisterOnnxOperatorSetSchema();
    RegisterOnnxMLOperatorSetSchema();

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
    } catch (std::exception& ex) {
      status = Status{ONNXRUNTIME, common::RUNTIME_EXCEPTION, std::string{"Exception caught: "} + ex.what()};
    } catch (...) {
      status = Status{ONNXRUNTIME, common::RUNTIME_EXCEPTION};
  }

  // Create default logger.
  s_default_logger_ = loggingManager_->CreateLogger(default_logger_id).release();
  s_env_initialized_ = true;
  
  return status;
}

std::exception Environment::LogFatalAndCreateException(const char* category,
                                                       const CodeLocation& location,
                                                       const char* format_str, ...) {
  std::string exception_msg;

  // create Capture in separate scope so it gets destructed (leading to log output) before we throw.
  {
    ::onnxruntime::logging::Capture c{DefaultLogger(),
                                      ::onnxruntime::logging::Severity::kFATAL, category,
                                      ::onnxruntime::logging::DataType::SYSTEM, location};
    va_list args;
    va_start(args, format_str);

    c.ProcessPrintf(format_str, args);
    va_end(args);

    exception_msg = c.Message();
  }

  return OnnxRuntimeException(location, exception_msg);
}
Environment::~Environment() {
  std::lock_guard<OrtMutex> guard(s_env_mutex_);  
  delete s_default_logger_;
  s_default_logger_ = nullptr;
  ::google::protobuf::ShutdownProtobufLibrary();  
}

}  // namespace onnxruntime
