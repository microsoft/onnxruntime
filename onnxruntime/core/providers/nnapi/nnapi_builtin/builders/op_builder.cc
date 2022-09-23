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















#pragma region op_clip

class ClipOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateClipOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ClipOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void ClipOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  if (inputs.size() > 1)
    model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());  // min

  if (inputs.size() > 2)
    model_builder.AddInitializerToSkip(inputs[2].node_arg.Name());  // max
}

Status ClipOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  if (Contains(model_builder.GetFusedActivations(), input)) {
    LOGS_DEFAULT(VERBOSE) << "Clip Node [" << node_unit.Name() << "] fused";
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type);
    return Status::OK();
  }

  float min, max;
  GetClipMinMax(model_builder.GetInitializerTensors(), node_unit.GetNode(), min, max,
                logging::LoggingManager::DefaultLogger());

  int32_t op_code;
  if (min == 0.0f && max == 6.0f)
    op_code = ANEURALNETWORKS_RELU6;
  else if (min == -1.0f && max == 1.0f)
    op_code = ANEURALNETWORKS_RELU1;
  else
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ClipOpBuilder, unsupported input [", min, ", ", max, "].",
                           "We should not reach here, ClipOpBuilder::IsOpSupportedImpl should have caught this.");

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

#pragma endregion

#pragma region op_Resize

class ResizeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void CreateResizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ResizeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

bool ResizeOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQResize;
}

void ResizeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  if (IsQuantizedOp(node_unit)) {
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[0].quant_param);               // x_scale, x_zp
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
  }

  // We don't really use ROI here, so add them to skipped list
  model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());  // ROI

  // We will still add scales to the skipped list even sizes are present
  // since there is no use of it, we will not process it later
  model_builder.AddInitializerToSkip(inputs[2].node_arg.Name());  // scales

  if (inputs.size() > 3)
    model_builder.AddInitializerToSkip(inputs[3].node_arg.Name());  // sizes
}

Status ResizeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node_unit);
  const auto& inputs = node_unit.Inputs();
  const auto android_feature_level = model_builder.GetNNAPIFeatureLevel();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  auto input = inputs[0].node_arg.Name();
  bool use_nchw = model_builder.UseNCHW();
  ORT_RETURN_IF_ERROR(IsOpInRequiredLayout(use_nchw, node_unit));

  // Check if the quantization scale and ZP is correct
  if (IsQuantizedOp(node_unit)) {
    float x_scale = 0.0f;
    int32_t x_zero_point = 0;
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
  }

  bool is_linear_resize = helper.Get("mode", "nearest") == "linear";

  int32_t operationCode = is_linear_resize ? ANEURALNETWORKS_RESIZE_BILINEAR
                                           : ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR;

  const auto coord_trans_mode = helper.Get("coordinate_transformation_mode", "half_pixel");
  bool using_half_pixel = coord_trans_mode == "half_pixel";
  bool using_align_corners = coord_trans_mode == "align_corners";

  // if the node domain is NHWC it means all the node inputs are converted to NHWC format by the layout transformer.
  // pick the index for height and width based on the format.
  int h_idx = use_nchw ? 2 : 1;
  int w_idx = use_nchw ? 3 : 2;

  if (inputs.size() == 3) {  // we are using scales
    const auto& scales_name = inputs[2].node_arg.Name();
    const auto& scales_tensor = *initializers.at(scales_name);
    std::vector<uint8_t> unpacked_tensor;
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(scales_tensor, unpacked_tensor));
    const float* scales_data = reinterpret_cast<const float*>(unpacked_tensor.data());
    ORT_RETURN_IF_ERROR(
        shaper.ResizeUsingScales(input, scales_data[h_idx], scales_data[w_idx], use_nchw, output));
  } else {  // we are using sizes
    const auto& sizes_name = inputs[3].node_arg.Name();
    const auto& sizes_tensor = *initializers.at(sizes_name);
    std::vector<uint8_t> unpacked_tensor;
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(sizes_tensor, unpacked_tensor));
    const int64_t* sizes_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
    ORT_RETURN_IF_ERROR(
        shaper.ResizeUsingOutputSizes(input, SafeInt<uint32_t>(sizes_data[h_idx]), SafeInt<uint32_t>(sizes_data[w_idx]), use_nchw, output));
  }

  const auto& output_shape = shaper[output];
  int32_t output_h = output_shape[h_idx];
  int32_t output_w = output_shape[w_idx];

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, output_w);
  ADD_SCALAR_OPERAND(model_builder, input_indices, output_h);

  if (android_feature_level > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    // using nchw is only available on API level 29
    ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);
  }

  // Currently we only support align_corners and half_pixel on bilinear resize
  // TODO, investigate nearest neighbor resize difference between NNAPI(based on TF) and ONNX
  if (is_linear_resize) {
    if (android_feature_level > ANEURALNETWORKS_FEATURE_LEVEL_3 && (using_align_corners || using_half_pixel)) {
      ADD_SCALAR_OPERAND(model_builder, input_indices, using_align_corners);
      if (using_half_pixel)
        ADD_SCALAR_OPERAND(model_builder, input_indices, using_half_pixel);
    }
  }

  OperandType output_operand_type = operand_types.at(input);
  output_operand_type.SetDimensions(output_shape);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(operationCode, input_indices,
                                                 {output}, {output_operand_type}));

  return Status::OK();
}

#pragma endregion

#pragma region op_gather

class GatherOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateGatherOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GatherOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void GatherOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  const auto& indices_name = inputs[1].node_arg.Name();
  int32_t indices_data_type;
  GetType(node_unit.Inputs()[1].node_arg, indices_data_type);
  if (Contains(model_builder.GetInitializerTensors(), indices_name) &&
      indices_data_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    // Skip the second input `indices` for Gather if it is an initializer
    model_builder.AddInitializerToSkip(indices_name);
  }
}

Status GatherOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  const auto& input1 = node_unit.Inputs()[0].node_arg.Name();
  const auto& input2 = node_unit.Inputs()[1].node_arg.Name();  // "indices"
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  NodeAttrHelper helper(node_unit);
  int32_t rank = static_cast<int32_t>(shaper[input1].size());
  int32_t axis = static_cast<int32_t>(HandleNegativeAxis(helper.Get("axis", 0), rank));

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));
  ADD_SCALAR_OPERAND(model_builder, input_indices, axis);

  int32_t indices_data_type;
  GetType(node_unit.Inputs()[1].node_arg, indices_data_type);
  if (Contains(model_builder.GetInitializerTensors(), input2) &&
      indices_data_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    // Add indices operand into nnapi
    const auto& indices_tensor = *initializers.at(input2);
    std::vector<uint8_t> unpacked_tensor;
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(indices_tensor, unpacked_tensor));

    const auto data_type = indices_tensor.data_type();
    const auto indices_shape = indices_tensor.dims();
    uint32_t size = 1;
    Shape indices_dimen;
    indices_dimen.reserve(indices_tensor.dims_size());
    for (auto i = 0; i < indices_tensor.dims_size(); i++) {
      size *= SafeInt<uint32_t>(indices_shape[i]);
      indices_dimen.push_back(static_cast<uint32_t>(indices_shape[i]));
    }

    std::vector<int32_t> indices(size);
    // see https://gist.github.com/shafik/848ae25ee209f698763cffee272a58f8#type-punning-arrays for the usage of memcpy here
    if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      for (uint32_t i = 0; i < size; i++) {
        int64_t index_i64;
        memcpy(&index_i64, unpacked_tensor.data() + i * sizeof(int64_t), sizeof(int64_t));
        indices[i] = SafeInt<int32_t>(index_i64);
      }
    } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      for (uint32_t i = 0; i < size; i++) {
        int32_t index;
        memcpy(&index, unpacked_tensor.data() + i * sizeof(int32_t), sizeof(int32_t));
        indices[i] = SafeInt<int32_t>(index);
      }
    }

    OperandType indices_operand_type(Type::TENSOR_INT32, indices_dimen);
    ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(input2, indices.data(), indices_operand_type));
  }
  input_indices.push_back(operand_indices.at(input2));
  ORT_RETURN_IF_ERROR(shaper.Gather(input1, input2, axis, output));
  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output]);

  return model_builder.AddOperation(ANEURALNETWORKS_GATHER, input_indices,
                                    {output}, {output_operand_type});
}

#pragma endregion

#pragma region op_minmax

class MinMaxOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  static Status AddMinMaxOperator(ModelBuilder& model_builder, const NodeUnit& node_unit,
                                  const std::string& input1, const std::string& input2);
};

void CreateMinMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<MinMaxOpBuilder>(
      op_type, op_registrations,
      {
          "Min",
          "Max",
      });
}

/* static */ Status MinMaxOpBuilder::AddMinMaxOperator(ModelBuilder& model_builder, const NodeUnit& node_unit,
                                                       const std::string& input1, const std::string& input2) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  const auto& op_type(node_unit.OpType());
  int32_t op_code;
  if (op_type == "Min")
    op_code = ANEURALNETWORKS_MINIMUM;
  else if (op_type == "Max")
    op_code = ANEURALNETWORKS_MAXIMUM;
  else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MinMaxOpBuilder, unknown op: ", op_type);
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // input 1
  input_indices.push_back(operand_indices.at(input2));  // input 2
  ORT_RETURN_IF_ERROR(shaper.Eltwise(input1, input2, output));
  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}));

  return Status::OK();
}

Status MinMaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  std::string input1 = inputs[0].node_arg.Name();
  std::string input2 = inputs[1].node_arg.Name();

  return AddMinMaxOperator(model_builder, node_unit, input1, input2);
}

#pragma endregion

#pragma region op_elu

class EluOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateEluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<EluOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

Status EluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  NodeAttrHelper helper(node_unit);
  const auto alpha = helper.Get("alpha", 1.0f);
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, alpha);
  return model_builder.AddOperation(ANEURALNETWORKS_ELU, input_indices,
                                    {output}, {output_operand_type});
}

#pragma endregion

#pragma region op_slice

class SliceOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SliceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

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
  const auto& input_shape = shaper[inputs[0].node_arg.Name()];
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
      std::vector<uint8_t> unpacked_tensor;
      ORT_RETURN_IF_ERROR(
          onnxruntime::utils::UnpackInitializerData(tensor, model_builder.GetGraphViewer().ModelPath(),
                                                    unpacked_tensor));
      size_t tensor_byte_size = unpacked_tensor.size();
      const auto data_type = tensor.data_type();
      if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
        const int64_t* tensor_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
        size_t size = tensor_byte_size / sizeof(int64_t);
        data.insert(data.end(), tensor_data, tensor_data + size);
      } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
        const int32_t* tensor_data = reinterpret_cast<const int32_t*>(unpacked_tensor.data());
        size_t size = tensor_byte_size / sizeof(int32_t);
        data.insert(data.end(), tensor_data, tensor_data + size);
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

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

  // begin/end/strides of ANEURALNETWORKS_STRIDED_SLICE have the same shape
  Shape param_dimen = {static_cast<uint32_t>(input_shape.size())};

  // helper function to add begin/end/strides of ANEURALNETWORKS_STRIDED_SLICE
  const auto AddOperand = [&model_builder, &node_unit, &input_indices, &operand_indices](
                              const char* name, const Shape& shape, const gsl::span<const int64_t>& param_raw_data) {
    std::vector<int32_t> param_data;
    param_data.reserve(param_raw_data.size());
    std::transform(param_raw_data.cbegin(), param_raw_data.cend(),
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
      model_builder.GetNNAPIFeatureLevel() > ANEURALNETWORKS_FEATURE_LEVEL_2) {
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

#pragma endregion

#pragma region op_pad

class PadOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreatePadOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<PadOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void PadOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());  // pads
  if (inputs.size() > 2) {
    model_builder.AddInitializerToSkip(inputs[2].node_arg.Name());  // constant_value
  }
}

Status PadOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper = model_builder.GetShaper();
  const auto& operand_indices = model_builder.GetOperandIndices();
  const auto& operand_types = model_builder.GetOperandTypes();
  const auto& inputs = node_unit.Inputs();
  const auto& outputs = node_unit.Outputs();

  std::vector<uint32_t> input_indices{};

  // `data` input
  const auto& data = inputs[0].node_arg.Name();
  input_indices.push_back(operand_indices.at(data));

  // `pads` input
  // convert from [begin_1, begin_2, ..., end_1, end_2, ...] to [begin_1, end_1, begin_2, end_2, ...]
  // convert from int64_t to int32_t
  const auto& data_shape = shaper[data];
  const uint32_t data_rank = SafeInt<uint32_t>(data_shape.size());

  const auto& pads = inputs[1].node_arg.Name();
  const auto* pads_initializer = model_builder.GetConstantInitializer(pads);
  ORT_RETURN_IF_NOT(pads_initializer, "pads must be a constant");

  std::vector<uint8_t> pads_initializer_raw_data{};
  ORT_RETURN_IF_ERROR(utils::UnpackInitializerData(*pads_initializer, pads_initializer_raw_data));
  // assume pads_initializer has int64 data, per ONNX spec
  ORT_RETURN_IF_NOT(pads_initializer_raw_data.size() == 2 * data_rank * sizeof(int64_t),
                    "Expected pads initializer size in bytes: ", 2 * data_rank * sizeof(int64_t),
                    ", actual: ", pads_initializer_raw_data.size());

  std::vector<int32_t> converted_pads_data{};
  converted_pads_data.reserve(2 * data_rank);

  auto copy_and_convert = [](const void* raw_i64_src,
                             std::back_insert_iterator<decltype(converted_pads_data)> i32_dst) {
    int64_t i64;
    memcpy(&i64, raw_i64_src, sizeof(i64));
    *i32_dst = SafeInt<int32_t>(i64);
  };

  for (size_t i = 0; i < data_rank; ++i) {
    copy_and_convert(&pads_initializer_raw_data[i * sizeof(int64_t)],
                     std::back_inserter(converted_pads_data));

    copy_and_convert(&pads_initializer_raw_data[(i + data_rank) * sizeof(int64_t)],
                     std::back_inserter(converted_pads_data));
  }

  const Shape converted_pads_shape{data_rank, 2};
  const OperandType converted_pads_operand_type{Type::TENSOR_INT32, converted_pads_shape};
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(pads, converted_pads_data.data(),
                                                                      converted_pads_operand_type));
  input_indices.push_back(operand_indices.at(pads));

  // `constant_value` input
  float pad_value = 0.0f;
  if (inputs.size() > 2 && inputs[2].node_arg.Exists()) {
    const auto& constant_value = inputs[2].node_arg.Name();
    const auto* constant_value_initializer = model_builder.GetConstantInitializer(constant_value);
    ORT_RETURN_IF_NOT(constant_value_initializer, "constant_value must be a constant");

    std::vector<uint8_t> pad_value_raw_data{};
    ORT_RETURN_IF_ERROR(utils::UnpackInitializerData(*constant_value_initializer, pad_value_raw_data));
    // assume constant_value_initializer has float data
    // ONNX spec says it matches `data` input type, and op support checker limits that to float
    ORT_RETURN_IF_NOT(pad_value_raw_data.size() == sizeof(float),
                      "Expected constant_value initializer size in bytes: ", sizeof(float),
                      ", actual size: ", pad_value_raw_data.size());
    memcpy(&pad_value, pad_value_raw_data.data(), sizeof(float));
  }

  ADD_SCALAR_OPERAND(model_builder, input_indices, pad_value);

  const auto& output = outputs[0].node_arg.Name();

  ORT_RETURN_IF_ERROR(shaper.Pad(data, converted_pads_data, output));

  const OperandType output_operand_type{operand_types.at(data).type, shaper[output]};
  const auto op_code = ANEURALNETWORKS_PAD_V2;

  return model_builder.AddOperation(op_code, input_indices, {output}, {output_operand_type});
}

#pragma endregion op_pad

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
