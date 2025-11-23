// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/contrib_ops/internal_nhwc_onnx_schemas.h"

#include "onnx/defs/operator_sets.h"

#include "core/graph/constants.h"
#include "core/graph/contrib_ops/nhwc_inference_context.h"
#include "core/graph/contrib_ops/ms_schema.h"  // contrib::GetOpSchema

#ifndef ORT_MINIMAL_BUILD

namespace onnxruntime {
namespace contrib {
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearAveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, 1, QLinearConvTranspose);
}  // namespace contrib
namespace internal_nhwc_onnx {

using contrib::NhwcInferenceContext;
using RegistrationFunc = std::function<void(ONNX_NAMESPACE::OpSchema&&)>;

namespace {

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

// Registration function that uses the default InferenceContext to leverage the default ONNX type/shape inferencing
// with the kMSInternalNHWCDomain domain. Used by NHWC Resize operator.
void RegisterNCHWSchemaWithNHWCDomain(const RegistrationFunc& f, ::ONNX_NAMESPACE::OpSchema&& schema) {
  auto onnx_inferencing_func = schema.GetTypeAndShapeInferenceFunction();
  f(std::move(::ONNX_NAMESPACE::OpSchema(schema)
                  .TypeAndShapeInferenceFunction([onnx_inferencing_func](ONNX_NAMESPACE::InferenceContext& ctx) {
                    onnx_inferencing_func(ctx);
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

#define REGISTER_NHWC_SCHEMA_FROM_MSDOMAIN(RegistrationFn, Op, SinceVersion) \
  RegisterNHWCSchema(                                                        \
      RegistrationFn,                                                        \
      contrib::GetOpSchema<                                                  \
          contrib::ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Microsoft, SinceVersion, Op)>())

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

#define REGISTER_NCHW_SCHEMA_WITH_NHWC_DOMAIN(RegistrationFn, Op, SinceVersion) \
  RegisterNCHWSchemaWithNHWCDomain(                                             \
      RegistrationFn,                                                           \
      ::ONNX_NAMESPACE::GetOpSchema<                                            \
          ::ONNX_NAMESPACE::ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, SinceVersion, Op)>())

void OpSet_Internal_NHWC_ONNX::ForEachSchema(const std::function<void(ONNX_NAMESPACE::OpSchema&&)>& fn) {
  // if the operator may be fused with an activation, use the WITH_ACTIVATION variant to add optional attributes
  // for the activation parameters.
  // We mainly register operators from opset 11 on . Models can easily have their opset updated using ONNX tools
  // so supporting older opsets is unnecessary.
  // Older opsets are included on a per-operator basis as needed.

  // NOTE: This should be in sync with GetLayoutSensitiveOps in
  // /onnxruntime/core/optimizer/transpose_optimization/transpose_optimizer.cc
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, AveragePool, 7);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, AveragePool, 10);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, AveragePool, 11);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, AveragePool, 19);

  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, BatchNormalization, 7);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, BatchNormalization, 9);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, BatchNormalization, 14);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, BatchNormalization, 15);

  REGISTER_NHWC_SCHEMA(fn, DepthToSpace, 1);
  REGISTER_NHWC_SCHEMA(fn, DepthToSpace, 11);
  REGISTER_NHWC_SCHEMA(fn, DepthToSpace, 13);

  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, InstanceNormalization, 6);

  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, Conv, 1);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, Conv, 11);

  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, ConvTranspose, 1);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, ConvTranspose, 11);

  REGISTER_NHWC_SCHEMA(fn, GlobalAveragePool, 1);
  REGISTER_NHWC_SCHEMA(fn, GlobalLpPool, 2);
  REGISTER_NHWC_SCHEMA(fn, GlobalMaxPool, 1);

  REGISTER_NHWC_SCHEMA(fn, GridSample, 16);
  REGISTER_NHWC_SCHEMA(fn, GridSample, 20);

  REGISTER_NHWC_SCHEMA(fn, LRN, 1);
  REGISTER_NHWC_SCHEMA(fn, LRN, 13);

  REGISTER_NHWC_SCHEMA(fn, LpPool, 11);
  REGISTER_NHWC_SCHEMA(fn, LpPool, 18);

  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, MaxPool, 1);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, MaxPool, 8);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, MaxPool, 10);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, MaxPool, 11);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, MaxPool, 12);

  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, MaxUnpool, 9);
  REGISTER_NHWC_SCHEMA_WITH_ACTIVATION(fn, MaxUnpool, 11);

  REGISTER_NHWC_SCHEMA(fn, QLinearConv, 10);

  REGISTER_NHWC_SCHEMA(fn, SpaceToDepth, 1);
  REGISTER_NHWC_SCHEMA(fn, SpaceToDepth, 13);

  // The REGISTER_NCHW_SCHEMA_WITH_NHWC_DOMAIN macro uses the default ONNX type/shape inferencing, which assumes
  // data layouts in NCHW. We use the default ONNX type/shape inferencing for NHWC Resize to avoid having to modify
  // the 'scales' or 'sizes' inputs within a custom InferenceContext class.
  //
  // An alternative could have been to leave NHWC Resize in the ONNX domain (instead of the internal NHWC domain).
  // However, the internal NHWC domain is necessary to allow EPs to detect and reject Resize ops with unsupported
  // shapes after layout transformation.
  //
  // NHWC Resize is currently used by the QNN EP.
  REGISTER_NCHW_SCHEMA_WITH_NHWC_DOMAIN(fn, Resize, 11);
  REGISTER_NCHW_SCHEMA_WITH_NHWC_DOMAIN(fn, Resize, 13);
  REGISTER_NCHW_SCHEMA_WITH_NHWC_DOMAIN(fn, Resize, 18);
  REGISTER_NCHW_SCHEMA_WITH_NHWC_DOMAIN(fn, Resize, 19);

  // internal QLinear ops
  REGISTER_NHWC_SCHEMA_FROM_MSDOMAIN(fn, QLinearAveragePool, 1);
  REGISTER_NHWC_SCHEMA_FROM_MSDOMAIN(fn, QLinearConvTranspose, 1);

  // FormatTransform operator for OneDNN blocked format support
  fn(std::move(::ONNX_NAMESPACE::OpSchema()
                   .SetName("FormatTransform")
                   .SetDomain(onnxruntime::kMSInternalNHWCDomain)
                   .SinceVersion(1)
                   .SetDoc("Transform tensor between plain (NCHW) and OneDNN blocked formats (nChw4c, ABcd16a4b).")
                   .Attr("src_format", "Source format: Plain, nChw4c, or ABcd16a4b",
                         ONNX_NAMESPACE::AttributeProto::STRING)
                   .Attr("dst_format", "Destination format: Plain, nChw4c, or ABcd16a4b",
                         ONNX_NAMESPACE::AttributeProto::STRING)
                   .Input(0, "X", "Input tensor", "T")
                   .Output(0, "Y", "Output tensor with transformed layout", "T")
                   .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"},
                                   "Constrain input and output types to floating-point tensors.")
                   .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                     ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
                     if (!ONNX_NAMESPACE::hasInputShape(ctx, 0)) {
                       return;
                     }

                     const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
                     if (input_shape.dim_size() != 4) {
                       fail_shape_inference("FormatTransform requires 4D input tensor (NCHW)");
                     }

                     auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
                     output_shape->clear_dim();  // Clear any existing dimensions before writing

                     // Get destination format attribute
                     std::string dst_format;
                     if (ctx.getAttribute("dst_format") != nullptr) {
                       dst_format = ctx.getAttribute("dst_format")->s();
                     }

                     // Calculate output shape with padding if needed
                     if (dst_format == "nChw4c") {
                       // Pad channels (dimension 1) to multiple of 4
                       *output_shape->add_dim() = input_shape.dim(0);

                       if (input_shape.dim(1).has_dim_value()) {
                         int64_t C = input_shape.dim(1).dim_value();
                         int64_t padded_C = ((C + 3) / 4) * 4;
                         output_shape->add_dim()->set_dim_value(padded_C);
                       } else {
                         // Dynamic channel dimension - can't compute padding statically
                         output_shape->add_dim()->set_dim_param(input_shape.dim(1).dim_param());
                       }

                       *output_shape->add_dim() = input_shape.dim(2);
                       *output_shape->add_dim() = input_shape.dim(3);
                     } else if (dst_format == "ABcd16a4b") {
                       // Pad N (dimension 0) to multiple of 16 and C (dimension 1) to multiple of 4
                       if (input_shape.dim(0).has_dim_value()) {
                         int64_t N = input_shape.dim(0).dim_value();
                         int64_t padded_N = ((N + 15) / 16) * 16;
                         output_shape->add_dim()->set_dim_value(padded_N);
                       } else {
                         output_shape->add_dim()->set_dim_param(input_shape.dim(0).dim_param());
                       }

                       if (input_shape.dim(1).has_dim_value()) {
                         int64_t C = input_shape.dim(1).dim_value();
                         int64_t padded_C = ((C + 3) / 4) * 4;
                         output_shape->add_dim()->set_dim_value(padded_C);
                       } else {
                         output_shape->add_dim()->set_dim_param(input_shape.dim(1).dim_param());
                       }

                       *output_shape->add_dim() = input_shape.dim(2);
                       *output_shape->add_dim() = input_shape.dim(3);
                     } else {
                       // Plain or other formats: no padding needed
                       for (int i = 0; i < input_shape.dim_size(); ++i) {
                         *output_shape->add_dim() = input_shape.dim(i);
                       }
                     }
                   })));

  // not all schema are registered here. For part of layout insensitive ops
  // we will use onnx schema directly, for others, like fused-node/qdq-group
  // we may leverage internal schema or create on the fly.
}

}  // namespace internal_nhwc_onnx
}  // namespace onnxruntime

#endif  // !ORT_MINIMAL_BUILD
