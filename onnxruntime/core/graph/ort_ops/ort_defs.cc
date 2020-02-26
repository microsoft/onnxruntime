// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/ort_ops/ort_defs.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/graph/constants.h"
#include "core/optimizer/matmul_prepacking.h"

#include <core/mlas/inc/mlas.h>

#include "onnx/defs/schema.h"



#define ONNX_ORT_OPERATOR_SCHEMA(name) \
  ONNX_ORT_OPERATOR_SCHEMA_UNIQ_HELPER(__COUNTER__, name)
#define ONNX_ORT_OPERATOR_SCHEMA_UNIQ_HELPER(Counter, name) \
  ONNX_ORT_OPERATOR_SCHEMA_UNIQ(Counter, name)
#define ONNX_ORT_OPERATOR_SCHEMA_UNIQ(Counter, name)         \
  static ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce( \
      op_schema_register_once##name##Counter) ONNX_UNUSED =      \
      ONNX_NAMESPACE::OpSchema(#name, __FILE__, __LINE__)

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::OpSchema;


namespace onnxruntime {

static MLAS_GEMM_PARAMETERS GetGemmParameters(const ONNX_NAMESPACE::InferenceContext& ctx) {
  OpNodeProtoHelper<ONNX_NAMESPACE::InferenceContext> helper(&ctx);
  MLAS_GEMM_PARAMETERS mlas_params;
  auto status = GemmParamsFromNodeAttributes(helper, mlas_params);
  if (!status.IsOK()) {
    fail_shape_inference("Could not get MLAS_GEMM_PARAMS from node: " + status.ErrorMessage());
  }
  return mlas_params;
}

static void InferPackForGemm(ONNX_NAMESPACE::InferenceContext& ctx) {
  ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput(ctx);
  auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  if (output_shape->dim_size() < 2) {
    fail_shape_inference("Input rank must be at least 2");
  }

  output_shape->mutable_dim()->RemoveLast();
  const auto mlas_params = GetGemmParameters(ctx);
  output_shape->mutable_dim(output_shape->dim_size() - 1)->set_dim_value(
    mlas_params.PackedSize
  );
}

static void InferMatmulPrepacked(ONNX_NAMESPACE::InferenceContext& ctx) {
  if (!ONNX_NAMESPACE::hasInputShape(ctx, 0)) {
    return;
  }

  ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput(ctx);
  auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
  if (output_shape->dim_size() == 0) {
    fail_shape_inference("Input rank must be at least 1");
  }

  const auto mlas_params = GetGemmParameters(ctx);
  output_shape->mutable_dim(output_shape->dim_size() - 1)->set_dim_value(mlas_params.N);
}

#define ADD_GEMM_PARAMS \
    .Attr("K", "K dimension", AttributeProto::INT) \
    .Attr("N", "N dimension", AttributeProto::INT) \
    .Attr("PackedSize", "Total size of the last two packed dimensions of matrix B", AttributeProto::INT) \
    .Attr("PackedStrideN", "N stride for packing", AttributeProto::INT) \
    .Attr("PackedStrideK", "K stride for packing", AttributeProto::INT) \


void RegisterOrtSchemas() {
  ONNX_ORT_OPERATOR_SCHEMA(MatMulPrepacked)
    .SetDomain(kOnnxRuntimeDomain)
    .SinceVersion(1)
    .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
    .SetDoc("ONNX runtime internal operator: MatMul with an argument that has been \"pre-packed\" for GEMM")
    ADD_GEMM_PARAMS
    .Input(0, "A", "N-dimensional matrix A", "T")
    .Input(1, "PackedB", "N-1 dimensional tensor representing an N-dimensional matrix B", "T")
    .Input(2, "OriginalB", "Original N-dimensional matrix B", "T", OpSchema::Optional)
    .Output(0, "Y", "Matrix multiply results from A*B", "T")
    .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors.")
    .TypeAndShapeInferenceFunction(InferMatmulPrepacked);

  ONNX_ORT_OPERATOR_SCHEMA(PackForGemm)
    .SetDomain(kOnnxRuntimeDomain)
    .SinceVersion(1)
    .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
    .SetDoc("ONNX runtime internal operator: \"pre-pack\" a constant matrix for GEMM operator")
    ADD_GEMM_PARAMS
    .Input(0, "B", "N-dimensional matrix B", "T")
    .Output(0, "PackedB", "N-1 dimensional tensor representing the matrix B; with last two dimensions packed", "T")
    .TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors.")
    .TypeAndShapeInferenceFunction(InferPackForGemm);
}

}
