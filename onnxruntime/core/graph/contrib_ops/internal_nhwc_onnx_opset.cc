// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "internal_nhwc_onnx_opset.h"

#include "onnx/defs/operator_sets.h"

#include "core/graph/constants.h"
#include "core/graph/contrib_ops/nhwc_inference_context.h"

namespace onnxruntime {
namespace internal_nhwc_onnx {

using contrib::NhwcInferenceContext;
using RegistrationFunc = std::function<void(ONNX_NAMESPACE::OpSchema&&)>;

namespace {

template <typename F>
void RegisterNHWCSchema(const RegistrationFunc& f, ::ONNX_NAMESPACE::OpSchema&& schema) {
  // Need to copy the inferencing function from the temporary OpSchema object
  auto onnx_inferencing_func = schema.GetTypeAndShapeInferenceFunction();
  f(std::move(::ONNX_NAMESPACE::OpSchema(schema)
                  .TypeAndShapeInferenceFunction([onnx_inferencing_func](ONNX_NAMESPACE::InferenceContext& ctx) {
                    // use the NHWC inferencing context to convert input 0 and output 0 to NCHW
                    // so the ONNX shape inferencing can be used. Once that completes, the call to PropagateOutputShape
                    // will convert the inferred shape from NCHW to NHWC
                    NhwcInferenceContext nhwc_ctx(ctx);
                    onnx_inferencing_func(nhwc_ctx);
                    nhwc_ctx.PropagateOutputShape();
                  })
                  .SetDomain(onnxruntime::kMSInternalNHWCDomain)));
}

void RegisterNHWCSchemaWithActivation(const RegistrationFunc& f, ::ONNX_NAMESPACE::OpSchema&& schema) {
  auto onnx_inferencing_func = schema.GetTypeAndShapeInferenceFunction();
  f(std::move(::ONNX_NAMESPACE::OpSchema(schema)
                  .Attr("activation", "", ONNX_NAMESPACE::AttributeProto::STRING, ONNX_NAMESPACE::OPTIONAL_VALUE)
                  .Attr("activation_params", "", ONNX_NAMESPACE::AttributeProto::FLOATS, ONNX_NAMESPACE::OPTIONAL_VALUE)
                  .TypeAndShapeInferenceFunction([onnx_inferencing_func](ONNX_NAMESPACE::InferenceContext& ctx) {
                    NhwcInferenceContext nhwc_ctx(ctx);
                    onnx_inferencing_func(nhwc_ctx);
                    nhwc_ctx.PropagateOutputShape();
                  })
                  .SetDomain(onnxruntime::kMSInternalNHWCDomain)));
}
}  // namespace

#define REGISTER_NHWC_SCHEMA(RegistrationFn, Op, SinceVersion) \
  RegisterNHWCSchema(                                          \
      RegistrationFn,                                          \
      ::ONNX_NAMESPACE::GetOpSchema<                           \
          ::ONNX_NAMESPACE::ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, SinceVersion, Op)>())

#define REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(RegistrationFn, Op, SinceVersion) \
  RegisterNHWCSchemaWithActivation(                                            \
      RegistrationFn,                                                          \
      ::ONNX_NAMESPACE::GetOpSchema<                                           \
          ::ONNX_NAMESPACE::ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, SinceVersion, Op)>())

void OpSet_Internal_NHWC_ONNX::ForEachSchema(const std::function<void(ONNX_NAMESPACE::OpSchema&&)>& fn) {
  // if the operator may be fused with an activation, use the WITH_ACTIVATION variant to add optional attributes
  // for the activation parameters.
  // For now we only register operators from opset 11 on. Models can easily have their opset updated using ONNX tools
  // so supporting older opsets is unnecessary.
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, Conv, 11);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, MaxPool, 11);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, MaxPool, 12);

  // TODO: Add other layout sensitive ops when needed. Those are:
  //   QLinearConv,
  //   BatchNormalization,
  //   AveragePool, GlobalAveragePool, GlobalMaxPool,
  //   LRN,
  //   GridSample
  //   DepthToSpace, SpaceToDepth
}

}  // namespace internal_nhwc_onnx
}  // namespace onnxruntime
