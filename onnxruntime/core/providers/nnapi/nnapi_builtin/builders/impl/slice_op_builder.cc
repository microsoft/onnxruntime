// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_builder.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

class SliceOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  // We only support slice from opset 10
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 10; }

  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

// Add operator related

void SliceOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  // Skip everything except input0 for Slice
  const auto& inputs = node_unit.Inputs();
  model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());  // starts
  model_builder.AddInitializerToSkip(inputs[2].node_arg.Name());  // ends
  if (inputs.size() > 3) {
    model_builder.AddInitializerToSkip(inputs[3].node_arg.Name());  // axes
    if (inputs.size() > 4) {
      model_builder.AddInitializerToSkip(inputs[4].node_arg.Name());  // steps
    }
  }
}

Status SliceOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& inputs = node_unit.Inputs();
  const auto input_shape = shaper[inputs[0].node_arg.Name()];
  TensorShapeVector input_shape_64(input_shape.cbegin(), input_shape.cend());
  SliceOp::PrepareForComputeMetadata compute_metadata(input_shape_64);

  {
    // We need to copy the data from the starts/ends/axes/steps initializers to int64 vectors
    // to be used in shared PrepareForCompute function to calculate the output shape
    // and normalize inputs, for example, input can be starts/ends/steps for certain axes,
    // PrepareForCompute can generate standard starts/ends/steps/axes for each axes
    TensorShapeVector input_starts;
    TensorShapeVector input_ends;
    TensorShapeVector input_axes;
    TensorShapeVector input_steps;

    const auto CopyInputData = [&inputs, &model_builder](size_t input_idx, TensorShapeVector& data) {
      data.clear();

      // This is an optional input, return empty vector
      if (inputs.size() <= input_idx)
        return Status::OK();

      const auto& input_name = inputs[input_idx].node_arg.Name();
      const auto& initializers(model_builder.GetInitializerTensors());

      const auto& tensor = *initializers.at(input_name);
      Initializer unpacked_tensor(tensor, model_builder.GetGraphViewer().ModelPath());
      const auto data_type = tensor.data_type();
      if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
        auto tensor_data = unpacked_tensor.DataAsSpan<int64_t>();
        data.insert(data.end(), tensor_data.begin(), tensor_data.end());
      } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
        auto tensor_data = unpacked_tensor.DataAsSpan<int32_t>();
        data.insert(data.end(), tensor_data.begin(), tensor_data.end());
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Data type for starts and ends inputs' is not supported in this build. Got ",
                               data_type);
      }

      return Status::OK();
    };

    ORT_RETURN_IF_ERROR(CopyInputData(1, input_starts));
    ORT_RETURN_IF_ERROR(CopyInputData(2, input_ends));
    ORT_RETURN_IF_ERROR(CopyInputData(3, input_axes));
    ORT_RETURN_IF_ERROR(CopyInputData(4, input_steps));
    ORT_RETURN_IF_ERROR(
        SliceOp::PrepareForComputeHelper(input_starts, input_ends, input_axes, input_steps, compute_metadata));
  }

  // output shape is of type uint32_t, convert from int64 compute_metadata.output_dims_
  Shape nnapi_output_shape;
  nnapi_output_shape.reserve(compute_metadata.output_dims_.size());
  std::transform(compute_metadata.output_dims_.cbegin(), compute_metadata.output_dims_.cend(),
                 std::back_inserter(nnapi_output_shape),
                 [](int64_t i) { return SafeInt<uint32_t>(i); });

  const auto& input = inputs[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  // No shape inference for Slice, everything is calculated here, we only need to add the output shape
  // to the shaper
  shaper.AddShape(output, nnapi_output_shape);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

  // begin/end/strides of ANEURALNETWORKS_STRIDED_SLICE have the same shape
  Shape param_dimen = {static_cast<uint32_t>(input_shape.size())};

  // helper function to add begin/end/strides of ANEURALNETWORKS_STRIDED_SLICE
  const auto AddOperand = [&model_builder, &node_unit, &input_indices, &operand_indices](
                              const char* name, const Shape& shape, const gsl::span<const int64_t>& param_raw_data) {
    std::vector<int32_t> param_data;
    param_data.reserve(param_raw_data.size());
    std::transform(param_raw_data.begin(), param_raw_data.end(),
                   std::back_inserter(param_data),
                   [](int64_t i) { return SafeInt<int32_t>(i); });
    std::string param_name = model_builder.GetUniqueName(node_unit.Name() + name);
    OperandType param_operand_type(Type::TENSOR_INT32, shape);
    ORT_RETURN_IF_ERROR(
        model_builder.AddOperandFromPersistMemoryBuffer(param_name, param_data.data(), param_operand_type));
    input_indices.push_back(operand_indices.at(param_name));
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(AddOperand("starts", param_dimen, compute_metadata.starts_));  // nnapi_begin

  // NNAPI has 2 slice operations
  // - ANEURALNETWORKS_SLICE
  //    Simpler and faster version of slice without steps, available from ANEURALNETWORKS_FEATURE_LEVEL_3
  //    Use this one if no step other than 1 is used in ONNX slice
  // - ANEURALNETWORKS_STRIDED_SLICE
  //    More comprehensive version, available from ANEURALNETWORKS_FEATURE_LEVEL_2
  int op_code = ANEURALNETWORKS_STRIDED_SLICE;
  if (std::all_of(compute_metadata.steps_.cbegin(),
                  compute_metadata.steps_.cend(),
                  [](int64_t i) { return i == 1; }) &&
      model_builder.GetEffectiveFeatureLevel() > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    op_code = ANEURALNETWORKS_SLICE;
    // the nnapi size of the slice in this case is the output shape
    ORT_RETURN_IF_ERROR(AddOperand("sizes", param_dimen, compute_metadata.output_dims_));  // nnapi_sizes
  } else {
    // ** The special treatment of ends **
    // The nnapi_end need some special handling, based on the current undocumented design of
    // ANEURALNETWORKS_STRIDED_SLICE
    // For ORT, for a single axis, after SliceOp::PrepareForCompute, and the step is negative,
    // and the last element for slice is at the beginning of the axis (we are slicing backwards)
    // The end for this axis will be -1
    // For NNAPI, it is not documented that end can be negative,
    // see https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a89695302f8b1e7ae7ce8f4d8c0b8a752
    // However, the actual NNAPI StridedSlice has some odd implementations,
    // See https://android.googlesource.com/platform/frameworks/ml/+/5b525d4d9100819d87447bd2c2a0bcfdd62899ee/nn/common/operations/StridedSlice.cpp#177
    // and, https://android.googlesource.com/platform/frameworks/ml/+/5b525d4d9100819d87447bd2c2a0bcfdd62899ee/nn/common/include/OperationsUtils.h#262
    // If a negative end is no less than -dim (dimension of the axis), it will be treated as an index counting from
    // the end, for example, dim = 5, and end = -1, the end will be normalized to 4, which will cause
    // incorrect result, so here we have to make the end = -dim - 1 such that it will not be treated as
    // an index counting from the end.
    auto ends = compute_metadata.ends_;
    for (size_t i = 0, limit = ends.size(); i < limit; ++i) {
      if (ends[i] == -1) {
        ends[i] = -static_cast<int32_t>(input_shape[i] + 1);
      }
    }
    ORT_RETURN_IF_ERROR(AddOperand("ends", param_dimen, ends));                      // nnapi_end
    ORT_RETURN_IF_ERROR(AddOperand("steps", param_dimen, compute_metadata.steps_));  // nnapi_strides
    // We do not use the following inputs in ANEURALNETWORKS_STRIDED_SLICE, set them all to 0
    ADD_SCALAR_OPERAND(model_builder, input_indices, 0);  // begin_mask
    ADD_SCALAR_OPERAND(model_builder, input_indices, 0);  // end_mask
    ADD_SCALAR_OPERAND(model_builder, input_indices, 0);  // shrink_axis_mask
  }
  return model_builder.AddOperation(op_code, input_indices, {output}, {output_operand_type});
}

// Operator support related

bool SliceOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                       const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  if (input_shape.size() > 4) {
    LOGS_DEFAULT(VERBOSE) << "Slice only supports 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  // TODO, replace with std::find when we switch to c++17
  if (std::any_of(input_shape.cbegin(), input_shape.cend(), [](int32_t i) { return i == 0; })) {
    LOGS_DEFAULT(VERBOSE) << "Slice doesn't support dynamic input shape";
    return false;
  }

  if (!CheckIsConstantInitializer(graph_viewer, node_unit, node_unit.Inputs()[1].node_arg.Name(), "starts")) {
    return false;
  }
  if (!CheckIsConstantInitializer(graph_viewer, node_unit, node_unit.Inputs()[2].node_arg.Name(), "ends")) {
    return false;
  }
  const auto& inputs = node_unit.Inputs();
  if (inputs.size() > 3) {
    if (!CheckIsConstantInitializer(graph_viewer, node_unit, node_unit.Inputs()[3].node_arg.Name(), "axes")) {
      return false;
    }
    if (inputs.size() > 4) {
      if (!CheckIsConstantInitializer(graph_viewer, node_unit, node_unit.Inputs()[4].node_arg.Name(), "steps")) {
        return false;
      }
    }
  }

  return true;
}

void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SliceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
