// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/environment.h"
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

std::once_flag schemaRegistrationOnceFlag;

std::atomic<bool> Environment::is_initialized_{false};

Status Environment::Create(std::unique_ptr<Environment>& environment) {
  environment = std::unique_ptr<Environment>(new Environment());
  auto status = environment->Initialize();
  return status;
}

Status Environment::Initialize() {
  auto status = Status::OK();

  try {
    // Register Microsoft domain with min/max op_set version as 1/1.
    std::call_once(schemaRegistrationOnceFlag, []() {
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSDomain, 1, 1);
      // Register contributed schemas.
      // The corresponding kernels are registered inside the appropriate execution provider.
#ifndef DISABLE_CONTRIB_OPS
      contrib::RegisterContribSchemas();
#endif
      RegisterOnnxOperatorSetSchema();
      RegisterOnnxMLOperatorSetSchema();
    });
    //TODO:put all of the following things into call_once
    // Register MVN operator for backward compatibility.
    // Experimental operator does not have history kept in ONNX. Unfortunately, RS5 takes bunch of experimental operators
    // in onnx as production ops. MVN is one of them. Now (9/26/2018) MVN is a production function in ONNX. The experimental
    // MVN op was removed. The history has to be kept locally as below.
    ORT_ATTRIBUTE_UNUSED ONNX_OPERATOR_SCHEMA(MeanVarianceNormalization)
        .SetDoc(R"DOC(Perform mean variance normalization.)DOC")
        .Attr("across_channels", "If 1, mean and variance are computed across channels. Default is 0.", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("normalize_variance", "If 0, normalize the mean only.  Default is 1.", AttributeProto::INT, static_cast<int64_t>(1))
        .Input(0, "input", "Input tensor of shape [N,C,H,W]", "T")
        .Output(0, "output", "Result, has same shape and type as input", "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput);

    ORT_ATTRIBUTE_UNUSED ONNX_OPERATOR_SCHEMA(ScaledTanh)
        .Attr("alpha", "Scaling value", AttributeProto::FLOAT, OPTIONAL)
        .Attr("beta", "Scaling value", AttributeProto::FLOAT, OPTIONAL)
        .Input(0, "input", "Input tensor", "T")
        .Output(
            0,
            "output",
            "The scaled hyperbolic tangent values of the input tensor "
            "computed element-wise",
            "T")
        .TypeConstraint(
            "T",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput);

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

    is_initialized_ = true;
  } catch (std::exception& ex) {
    status = Status{ONNXRUNTIME, common::RUNTIME_EXCEPTION, std::string{"Exception caught: "} + ex.what()};
  } catch (...) {
    status = Status{ONNXRUNTIME, common::RUNTIME_EXCEPTION};
  }

  return status;
}

Environment::~Environment() {
  ::google::protobuf::ShutdownProtobufLibrary();
}

}  // namespace onnxruntime
