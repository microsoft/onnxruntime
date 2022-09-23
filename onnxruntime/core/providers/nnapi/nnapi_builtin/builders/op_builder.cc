// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_builder.h"

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"
#include "helper.h"
#include "model_builder.h"
#include "op_support_checker.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

using namespace op_builder_helpers;


/* #pragma region CreateGetOpBuilders

// The reason we use macros to create OpBuilders is for easy exclusion in build if certain op(s) are not used
// such that we can reduce binary size.
// This is for multiple ops share the same OpBuilder, we only need create one for all of them
#define NNAPI_EP_ADD_SHARED_OP_BUILDER(OP_TYPE, BUILDER_NAME) \
  BUILDER_NAME::CreateSharedOpBuilder(OP_TYPE, op_registrations);

// This is for ops with dedicated OpBuilder
#define NNAPI_EP_ADD_SINGLE_OP_BUILDER(OP_TYPE, BUILDER_NAME)                                 \
  do {                                                                                        \
    op_registrations.builders.push_back(std::make_unique<BUILDER_NAME>());                    \
    op_registrations.op_builder_map.emplace(OP_TYPE, op_registrations.builders.back().get()); \
  } while (0)

static OpBuilderRegistrations CreateOpBuilderRegistrations() {
  OpBuilderRegistrations op_registrations;

  // Builders handle a single op
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("BatchNormalization", BatchNormalizationOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Cast", CastOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Clip", ClipOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Concat", ConcatOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("DepthToSpace", DepthToSpaceOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("DequantizeLinear", DequantizeLinearOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Elu", EluOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Flatten", FlattenOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Gather", GatherOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Identity", IdentityOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("LRN", LRNOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Pad", PadOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("QuantizeLinear", QuantizeLinearOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Relu", ReluOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Reshape", ReshapeOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Resize", ResizeOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Slice", SliceOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Softmax", SoftMaxOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Squeeze", SqueezeOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Transpose", TransposeOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Unsqueeze", UnsqueezeOpBuilder);

  // Builders shared among similar ops
  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Add", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Div", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Mul", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Pow", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("PRelu", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearAdd", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearMul", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Sub", BinaryOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("AveragePool", PoolOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("GlobalAveragePool", PoolOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("GlobalMaxPool", PoolOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("MaxPool", PoolOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearAveragePool", PoolOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Conv", ConvOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearConv", ConvOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Gemm", GemmOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("MatMul", GemmOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearMatMul", GemmOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Abs", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Exp", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Floor", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Log", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Neg", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearSigmoid", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Sigmoid", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Sin", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Sqrt", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Tanh", UnaryOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Max", MinMaxOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Min", MinMaxOpBuilder);
  }

  return op_registrations;
}

const std::unordered_map<std::string, const IOpBuilder*>& GetOpBuilders() {
  static const OpBuilderRegistrations op_registrations = CreateOpBuilderRegistrations();
  return op_registrations.op_builder_map;
}

#pragma endregion */

}  // namespace nnapi
}  // namespace onnxruntime
