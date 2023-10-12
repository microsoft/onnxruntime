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
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const;
  void GetDataFromAttribute(const NodeUnit& node_unit,
                            TensorShapeVector& raw_starts,
                            TensorShapeVector& raw_ends,
                            TensorShapeVector& raw_axes) const;
};

Status SliceOpBuilder::ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit) const {
  size_t input_count = node_unit.Inputs().size();

  // Opset < 10: Only has 1 data input. The starts, ends, and axes values are attributes.
  // Opset >= 10: Everything is an input. The data, starts, and ends inputs are required.
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

// Gets the data from initializer inputs (e.g., starts, ends, axes, or steps) as a TensorShapeVector.
static Status GetInitializerInputData(const NodeUnitIODef& input, const QnnModelWrapper& qnn_model_wrapper,
                                      TensorShapeVector& output) {
  OnnxInputInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(input, input_info));
  ORT_RETURN_IF_NOT(input_info.is_initializer,
                    "QNN requires the starts, ends, axes, and steps inputs to be initializers");

  std::vector<uint8_t> initializer_bytes;

  // Note: UnpackInitializerData() uses ORT's protobuf utilities, which ensure that the initializer bytes are
  // contiguous, aligned, and in the appropriate endianness. This is necessary to be able to reinterpret bytes
  // as an array of larger elements.
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_info.initializer_tensor, initializer_bytes));

  const auto data_type = input_info.initializer_tensor->data_type();
  Status status;

  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      gsl::span<const int64_t> elements = qnn::utils::ReinterpretBytesAsSpan<int64_t>(initializer_bytes.data(),
                                                                                      initializer_bytes.size());
      output.insert(output.end(), elements.begin(), elements.end());
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      gsl::span<const int32_t> elements = qnn::utils::ReinterpretBytesAsSpan<int32_t>(initializer_bytes.data(),
                                                                                      initializer_bytes.size());
      output.insert(output.end(), elements.begin(), elements.end());
      break;
    }
    default:
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Data type ", data_type,
                               " is not supported for Slice initializer input ", input.node_arg.Name().c_str());
      break;
  }

  return status;
}

// Note: For ONNX Slice operation the expected number of inputs is between 3 and 5
Status SliceOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     const logging::Logger& logger,
                                     std::vector<std::string>& input_names,
                                     bool do_op_validation) const {
  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplictOpCheck(qnn_model_wrapper, node_unit));
  }

  // Only need to add input 0. The other inputs (if any) contain static data that is passed to QNN APIs
  // as static parameters.
  return ProcessInput(qnn_model_wrapper, node_unit.Inputs()[0], logger, input_names);
}

Status SliceOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                   const NodeUnit& node_unit,
                                                   std::vector<std::string>&& input_names,
                                                   const logging::Logger& logger,
                                                   bool do_op_validation) const {
  // Extract starts, ends, axes, and steps data from attributes (opset < 10) or initializer inputs (opset >= 10).
  TensorShapeVector raw_starts;
  TensorShapeVector raw_ends;
  TensorShapeVector raw_axes;
  TensorShapeVector raw_steps;

  const auto& inputs = node_unit.Inputs();
  const size_t input_count = inputs.size();

  // Opset 9 only has 1 input. The starts, ends, axes values are attributes.
  if (node_unit.SinceVersion() < 10) {
    GetDataFromAttribute(node_unit, raw_starts, raw_ends, raw_axes);
  } else {
    constexpr size_t starts_index = 1;
    constexpr size_t ends_index = 2;
    constexpr size_t axes_index = 3;
    constexpr size_t steps_index = 4;

    // Starts input (required).
    ORT_RETURN_IF_ERROR(GetInitializerInputData(inputs[starts_index], qnn_model_wrapper, raw_starts));

    // Ends input (required).
    ORT_RETURN_IF_ERROR(GetInitializerInputData(inputs[ends_index], qnn_model_wrapper, raw_ends));

    // Axes input (optional).
    if (input_count > axes_index && !inputs[axes_index].node_arg.Name().empty()) {
      ORT_RETURN_IF_ERROR(GetInitializerInputData(inputs[axes_index], qnn_model_wrapper, raw_axes));
    }

    // Steps input (optional).
    if (input_count > steps_index && !inputs[steps_index].node_arg.Name().empty()) {
      ORT_RETURN_IF_ERROR(GetInitializerInputData(inputs[steps_index], qnn_model_wrapper, raw_steps));
    }
  }

  std::vector<uint32_t> input0_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input0_shape),
                    "Cannot get shape for Slice input 0.");

  TensorShapeVector input_dimensions(input0_shape.cbegin(), input0_shape.cend());
  onnxruntime::SliceOp::PrepareForComputeMetadata compute_metadata(input_dimensions);
  ORT_RETURN_IF_ERROR(SliceOp::PrepareForComputeHelper(raw_starts, raw_ends, raw_axes, raw_steps, compute_metadata));

  const size_t input_rank = input_dimensions.size();
  std::vector<uint32_t> ranges_dims{static_cast<uint32_t>(input_rank), 3};
  std::vector<uint32_t> ranges_data;
  ranges_data.reserve(input_rank);

  for (size_t i = 0; i < input_rank; i++) {
    ranges_data.push_back(static_cast<uint32_t>(compute_metadata.starts_[i]));
    ranges_data.push_back(static_cast<uint32_t>(compute_metadata.ends_[i]));
    ranges_data.push_back(static_cast<uint32_t>(compute_metadata.steps_[i]));
  }

  QnnParamWrapper ranges_paramwrapper(node_unit.Index(),
                                      node_unit.Name(),
                                      QNN_OP_STRIDED_SLICE_PARAM_RANGES,
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
                                     do_op_validation, GetQnnOpType(node_unit.OpType())));
  return Status::OK();
}

void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SliceOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
