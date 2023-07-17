// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/cpu/tensor/slice_helper.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

class SliceOpBuilder : public BaseOpBuilder {
 public:
  SliceOpBuilder() : BaseOpBuilder("SliceOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SliceOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       bool is_quantized_model,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool is_quantized_model,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const;
  void GetDataFromAttribute(const NodeUnit& node_unit,
                            TensorShapeVector& raw_starts,
                            TensorShapeVector& raw_ends,
                            TensorShapeVector& raw_axes) const;
  typedef struct {
    int32_t begin, end, stride;
  } Range;
  mutable std::vector<Range> ranges_;
};

Status SliceOpBuilder::ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  size_t input_count = node_unit.Inputs().size();
  // Op set 9 only has 1 input with starts, ends, axes attribute
  // Op set > 9, starts, ends, axes are from node input
  if (input_count > 1) {
    // Skip the first input. All other input need to be initializer
    for (size_t i = 1; i < input_count; i++) {
      const auto& next_input = node_unit.Inputs()[i].node_arg.Name();
      if (!qnn_model_wrapper.IsInitializerInput(next_input)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN desn't support dynamic slice.");
      }
    }
  }

  return Status::OK();
}

void SliceOpBuilder::GetDataFromAttribute(const NodeUnit& node_unit,
                                          TensorShapeVector& raw_starts,
                                          TensorShapeVector& raw_ends,
                                          TensorShapeVector& raw_axes) const {
  NodeAttrHelper node_helper(node_unit);
  auto starts = node_helper.Get("starts", std::vector<int64_t>{0});
  raw_starts.assign(starts.begin(), starts.end());
  auto ends = node_helper.Get("ends", std::vector<int64_t>{0});
  raw_ends.assign(ends.begin(), ends.end());
  if (node_helper.HasAttr("axes")) {
    auto axes = node_helper.Get("axes", std::vector<int64_t>{0});
    raw_axes.assign(axes.begin(), axes.end());
  }
}

// Note: For ONNX Slice operation the expected number of inputs is between 3 and 5
Status SliceOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     const logging::Logger& logger,
                                     bool is_quantized_model,
                                     std::vector<std::string>& input_names,
                                     bool do_op_validation) const {
  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplictOpCheck(qnn_model_wrapper, node_unit));
  }
  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  utils::InitializeQuantizeParam(quantize_param, is_quantized_model);
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  TensorShapeVector raw_starts;
  TensorShapeVector raw_ends;
  TensorShapeVector raw_axes;
  TensorShapeVector raw_steps;
  std::vector<uint32_t> input0_shape;

  auto inputs = node_unit.Inputs();
  auto input_count = inputs.size();
  // Opset 9, only 1 input, starts, ends, axes are in attribute
  if (1 == input_count) {
    GetDataFromAttribute(node_unit, raw_starts, raw_ends, raw_axes);
  }

  for (size_t input_i = 0; input_i < input_count; ++input_i) {
    auto& input_name = inputs[input_i].node_arg.Name();
    if (input_name.empty()) {
      // Ignore unspecified/unused optional input
      continue;
    }
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added or the input is not named, skip it: " << input_name;
      input_names.push_back(input_name);
      ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[input_i].node_arg, input0_shape), "Cannot get shape");
      continue;
    }

    const auto* type_proto = inputs[input_i].node_arg.TypeAsProto();
    ORT_RETURN_IF_ERROR(utils::GetQnnDataType(is_quantized_model, type_proto, qnn_data_type));

    std::vector<uint32_t> input_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[input_i].node_arg, input_shape), "Cannot get shape");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessQuantizationParameter(inputs[input_i].quant_param,
                                                                     quantize_param.scaleOffsetEncoding.scale,
                                                                     quantize_param.scaleOffsetEncoding.offset),
                      "Cannot get quantization parameter");

    std::vector<uint8_t> unpacked_tensor;
    bool is_initializer_input = qnn_model_wrapper.IsInitializerInput(input_name);
    if (is_initializer_input) {
      const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(input_name);
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));
      size_t tensor_byte_size = unpacked_tensor.size();
      const auto data_type = input_tensor->data_type();
      TensorShapeVector data;
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
      if (input_i == 0) {
        // Do nothing!
      } else if (input_i == 1) {
        // Starts
        raw_starts = data;
        continue;
      } else if (input_i == 2) {
        // Ends
        raw_ends = data;
        continue;
      } else if (input_i == 3) {
        // Axes
        raw_axes = data;
        continue;
      } else if (input_i == 4) {
        // Steps
        raw_steps = data;
        continue;
      }
    }
    input0_shape = input_shape;

    input_names.push_back(input_name);
    Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, input_name);
    Qnn_QuantizeParams_t quantize_params = QNN_QUANTIZE_PARAMS_INIT;
    QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, quantize_params,
                                         std::move(input_shape), std::move(unpacked_tensor));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }
  TensorShapeVector input_dimensions(input0_shape.cbegin(), input0_shape.cend());
  onnxruntime::SliceOp::PrepareForComputeMetadata compute_metadata(input_dimensions);
  ORT_RETURN_IF_ERROR(
      SliceOp::PrepareForComputeHelper(raw_starts, raw_ends, raw_axes, raw_steps, compute_metadata));
  ranges_.clear();
  for (size_t i = 0; i < input_dimensions.size(); i++) {
    auto start = static_cast<int32_t>(compute_metadata.starts_[i]);
    auto end = static_cast<int32_t>(compute_metadata.ends_[i]);
    auto step = static_cast<int32_t>(compute_metadata.steps_[i]);
    ranges_.push_back(Range({start, end, step}));
  }
  return Status::OK();
}

Status SliceOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                   const NodeUnit& node_unit,
                                                   std::vector<std::string>&& input_names,
                                                   const logging::Logger& logger,
                                                   bool is_quantized_model,
                                                   bool do_op_validation) const {
  std::vector<uint32_t> ranges_dims{static_cast<uint32_t>(ranges_.size()), 3};
  std::vector<uint32_t> ranges_data;
  for (auto range : ranges_) {
    ranges_data.push_back(static_cast<uint32_t>(range.begin));
    ranges_data.push_back(static_cast<uint32_t>(range.end));
    ranges_data.push_back(static_cast<uint32_t>(range.stride));
  }
  QnnParamWrapper ranges_paramwrapper(node_unit.Index(),
                                      node_unit.Name(),
                                      qnn_def::ranges,
                                      std::move(ranges_dims),
                                      std::move(ranges_data),
                                      true);
  std::string param_tensor_name(ranges_paramwrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(ranges_paramwrapper));
  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper,
                                     node_unit,
                                     std::move(input_names),
                                     {param_tensor_name},
                                     logger,
                                     is_quantized_model,
                                     do_op_validation, GetQnnOpType(node_unit.OpType())));
  return Status::OK();
}

void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SliceOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
