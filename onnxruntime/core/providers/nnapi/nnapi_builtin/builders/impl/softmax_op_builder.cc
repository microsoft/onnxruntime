// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_builder.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

using namespace op_builder_helpers;

class SoftMaxOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related

 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }
  bool HasSupportedInputOutputsImpl(
      const GraphViewer& graph_viewer, const NodeUnit& node_unit,
      const OpSupportCheckParams& params) const override;

  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }

  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

// Add operator related

void SoftMaxOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (IsQuantizedOp(node_unit)) {
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);   // x_scale, x_zp
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
  }
}

Status SoftMaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto android_feature_level = model_builder.GetEffectiveFeatureLevel();

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  NodeAttrHelper helper(node_unit);
  int32_t axis = helper.Get("axis", 1);

  // Check if the quantization scale and ZP are correct
  float x_scale = 0.0f;
  int32_t x_zero_point = 0;
  float y_scale = 0.0f;
  int32_t y_zero_point = 0;
  if (IsQuantizedOp(node_unit)) {
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        model_builder.GetGraphViewer(), node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));

    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));

    y_scale = 1.f / 256;
  }

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  const float beta = 1.f;
  ADD_SCALAR_OPERAND(model_builder, input_indices, beta);

  auto input_shape = shaper[input];
  if (axis < 0) {
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, input_shape.size()));
  }

  // if opset < 13 we may need to manually coerce into 2D. we can skip this IFF it's already 2D and axis == 1.
  // otherwise we need the coercion to create an input shape that works with axis == 1, which will work with any
  // NNAPI version.
  // e.g. if 2D and axis is 0 we coerce to shape {1 , dim0 + dim1}
  //      if 4D and axis is 2 we coerce to shape {dim0 + dim1, dim2 + dim3}
  if (node_unit.SinceVersion() < 13 && !(input_shape.size() == 2 && axis == 1)) {
    // Reshape to 2D based on axis
    const SafeInt<uint32_t> safe1(1);
    uint32_t dim0 = std::accumulate(input_shape.cbegin(), input_shape.cbegin() + axis, safe1, std::multiplies());
    uint32_t dim1 = std::accumulate(input_shape.cbegin() + axis, input_shape.cend(), safe1, std::multiplies());
    Shape input2d_shape{dim0, dim1};

    std::string shape2d_name = model_builder.GetUniqueName(node_unit.Name() + input + "_2D_shape");
    std::string reshape2d_output_name = model_builder.GetUniqueName(node_unit.Name() + input + "_2D");

    ORT_RETURN_IF_ERROR(op_builder_helpers::AddNnapiReshape(model_builder, input, shape2d_name,
                                                            {narrow<int32_t>(dim0), narrow<int32_t>(dim1)},
                                                            reshape2d_output_name));

    input_indices[0] = operand_indices.at(reshape2d_output_name);  // replace input 1 with 2d output

    std::string softmax2d_output_name = model_builder.GetUniqueName("softmax_" + reshape2d_output_name);
    const OperandType output_operand_type(operand_types.at(input).type, input2d_shape, y_scale, y_zero_point);
    shaper.AddShape(softmax2d_output_name, input2d_shape);

    ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_SOFTMAX, input_indices,
                                                   {softmax2d_output_name}, {output_operand_type}));

    // add Reshape back to original shape
    const Shape& output_shape = shaper[output];

    // convert from uint32_t to int32_t
    std::vector<int32_t> reshape_output_shape;
    reshape_output_shape.reserve(output_shape.size());
    std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(reshape_output_shape),
                   [](uint32_t dim) { return narrow<int32_t>(dim); });

    std::string reshape_output_name = model_builder.GetUniqueName(node_unit.Name() + output + "_shape");
    ORT_RETURN_IF_ERROR(op_builder_helpers::AddNnapiReshape(model_builder, softmax2d_output_name,
                                                            reshape_output_name, reshape_output_shape,
                                                            output));

  } else {
    if (android_feature_level > ANEURALNETWORKS_FEATURE_LEVEL_2) {
      // you can only specify axis for android api level 29+
      ADD_SCALAR_OPERAND(model_builder, input_indices, axis);
    }

    const OperandType output_operand_type(operand_types.at(input).type, shaper[output], y_scale, y_zero_point);
    ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_SOFTMAX, input_indices,
                                                   {output}, {output_operand_type}));
  }
  return Status::OK();
}

// Operator support related

bool SoftMaxOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQSoftmax;
}

bool SoftMaxOpBuilder::IsOpSupportedImpl(const GraphViewer& /* graph_viewer */, const NodeUnit& node_unit,
                                         const OpSupportCheckParams& params) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  if (node_unit.SinceVersion() < 13) {
    // if opset < 13 the ONNX spec coerces to 2D based on axis, so we will manually do that when adding to the model
    // and use axis of 1.
  } else {
    const auto input_size = narrow<int32_t>(input_shape.size());

    if (input_size > 4) {
      LOGS_DEFAULT(VERBOSE) << "Softmax only supports maximum 4d input. input is " << input_size << "d";
      return false;
    }

    NodeAttrHelper helper(node_unit);
    int32_t axis = helper.Get("axis", 1);
    if (axis < 0) {
      axis = static_cast<int32_t>(HandleNegativeAxis(axis, input_shape.size()));
    }

    if (params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
      // 2D or 4D input is supported with axis of the last dim
      if (input_size != 2 && input_size != 4) {
        LOGS_DEFAULT(VERBOSE) << "Softmax only support 2d or 4d shape, input has " << input_size << "d shape";
        return false;
      }

      if (axis != input_size - 1) {
        LOGS_DEFAULT(VERBOSE) << "Softmax only supports axis of the last dim on Android API level "
                              << params.android_feature_level << ". input axis: " << axis;
        return false;
      }
    }
  }

  return true;
}

bool SoftMaxOpBuilder::HasSupportedInputOutputsImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                                    const OpSupportCheckParams& params) const {
  if (!IsQuantizedOp(node_unit)) {
    return BaseOpBuilder::HasSupportedInputOutputsImpl(graph_viewer, node_unit, params);
  }

  if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0}, params, ArgType::kInput)) {
    return false;
  }

  if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0}, params, ArgType::kOutput)) {
    return false;
  }

  // NNAPI requires the scale be 1.f/256 and zero point to be 0
  if (!HasRequiredScaleAndZeroPoint(
          graph_viewer,
          MakeString("Op [", node_unit.OpType(), "] name [", node_unit.Name(), "]'s output 0 "),
          node_unit.Outputs()[0], node_unit.ModelPath(),
          1.f / 256 /* required_scale */, 0 /* required_zp */)) {
    return false;
  }

  return true;
}

void CreateSoftMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SoftMaxOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
