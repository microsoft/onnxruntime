// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

namespace {

// QNN-EP COPY START
// Below implementations are directly copied from core/providers/cpu/tensor/slice_helper.h.
struct PrepareForComputeMetadata {
  explicit PrepareForComputeMetadata(gsl::span<const int64_t> input_dimensions)
      : input_dimensions_(input_dimensions),
        ends_(input_dimensions.begin(), input_dimensions.end()),
        output_dims_(input_dimensions.begin(), input_dimensions.end()) {
    size_t dimension_count = input_dimensions.size();
    starts_.resize(dimension_count, 0);
    steps_.resize(dimension_count, 1);
  }

  gsl::span<const int64_t> input_dimensions_;
  std::vector<int64_t> starts_;
  std::vector<int64_t> ends_;
  std::vector<int64_t> steps_;
  std::vector<int64_t> output_dims_;
  std::vector<int64_t> flattened_input_dims_;
  std::vector<int64_t>* p_flattened_input_dims_ = &flattened_input_dims_;
  std::vector<int64_t> flattened_output_dims_;
  std::vector<int64_t>* p_flattened_output_dims_ = &flattened_output_dims_;
};

inline Ort::Status PrepareForComputeHelper(const gsl::span<const int64_t>& raw_starts,
                                           const gsl::span<const int64_t>& raw_ends,
                                           const gsl::span<const int64_t>& raw_axes,
                                           const gsl::span<const int64_t>& raw_steps,
                                           PrepareForComputeMetadata& compute_metadata) {
  // Initialize axes to the provided axes attribute or to the default sequence
  std::vector<int64_t> axes;
  if (raw_axes.empty()) {
    // axes are omitted, they are set to[0, ..., ndim - 1]
    axes.reserve(raw_starts.size());
    for (int64_t i = 0, limit = raw_starts.size(); i < limit; ++i) {
      axes.push_back(i);
    }
  } else {
    axes.assign(raw_axes.begin(), raw_axes.end());
  }

  // Iterate through the provided axes and override the start/end/steps ranges
  using AxesSet = InlinedHashSet<int64_t>;
  const auto axes_count = axes.size();
  AxesSet unique_axes;
  unique_axes.reserve(axes_count);

  const auto dimension_count = compute_metadata.input_dimensions_.size();
  for (size_t axis_index = 0; axis_index < axes_count; ++axis_index) {
    const auto axis = axes[axis_index] < 0 ? axes[axis_index] + static_cast<int64_t>(dimension_count) : axes[axis_index];
    if (axis >= static_cast<int64_t>(dimension_count) || axis < 0)
      return MAKE_EP_FAIL("'axes' has an axis outside of the tensor dimension count");
    auto p = unique_axes.insert(axis);
    if (!p.second)
      return MAKE_EP_FAIL("'axes' has duplicates");
    const auto dim_value = compute_metadata.input_dimensions_[gsl::narrow<size_t>(axis)];

    // process step
    auto step = axis_index < raw_steps.size() ? raw_steps[axis_index] : 1;
    if (step == 0)
      return MAKE_EP_FAIL("'step' value cannot be 0");

    if (dim_value == 0) {
      // shape with empty dim. only output_dims_ matters but set everything for completeness
      compute_metadata.steps_[gsl::narrow<size_t>(axis)] = step;
      compute_metadata.starts_[gsl::narrow<size_t>(axis)] = 0;
      compute_metadata.ends_[gsl::narrow<size_t>(axis)] = 0;
      compute_metadata.output_dims_[gsl::narrow<size_t>(axis)] = 0;
      continue;
    }

    // clamp step to avoid overflow if there's a stupidly large value (which will be multiplied in SliceImpl)
    // as long as the clamped value is >= the size of the dimension a single step will push us past the end
    step = std::clamp(step, -dim_value, dim_value);

    compute_metadata.steps_[gsl::narrow<size_t>(axis)] = step;

    // process start
    auto start = raw_starts[axis_index];
    if (start < 0)
      start += dim_value;
    if (step < 0)
      compute_metadata.starts_[gsl::narrow<size_t>(axis)] = std::clamp(start, int64_t{0}, dim_value - 1);
    else
      compute_metadata.starts_[gsl::narrow<size_t>(axis)] = std::clamp(start, int64_t{0}, dim_value);

    // process end
    auto end = raw_ends[axis_index];
    // INT_MAX has a special meaning for end according to spec
    // equivalent to 'None' in numpy
    // it represent slicing to the end of the dimension
    if (end == std::numeric_limits<int32_t>::max() ||
        end == std::numeric_limits<int64_t>::max()) {
      end = step < 0 ? -1 : dim_value;
    } else {
      if (end < 0)
        end += dim_value;
      if (step < 0)
        end = std::clamp(end, int64_t{-1}, dim_value);
      else
        end = std::clamp(end, int64_t{0}, dim_value);
    }

    compute_metadata.ends_[gsl::narrow<size_t>(axis)] = end;

    // find output dim value for this axis
    const auto temp = static_cast<int64_t>(ceil(1.0 * (compute_metadata.ends_[gsl::narrow<size_t>(axis)] - compute_metadata.starts_[gsl::narrow<size_t>(axis)]) / step));
    if (temp < 0)
      compute_metadata.output_dims_[gsl::narrow<size_t>(axis)] = 0;
    else
      compute_metadata.output_dims_[gsl::narrow<size_t>(axis)] = temp;
  }

  return Ort::Status();
}
// QNN-EP COPY END

}  // namespace

class SliceOpBuilder : public BaseOpBuilder {
 public:
  SliceOpBuilder() : BaseOpBuilder("SliceOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SliceOpBuilder);

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit) const;
  void GetDataFromAttribute(const OrtNodeUnit& node_unit,
                            std::vector<int64_t>& raw_starts,
                            std::vector<int64_t>& raw_ends,
                            std::vector<int64_t>& raw_axes) const;
};

Ort::Status SliceOpBuilder::ExplictOpCheck(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& node_unit) const {
  size_t input_count = node_unit.Inputs().size();

  // Opset < 10: Only has 1 data input. The starts, ends, and axes values are attributes.
  // Opset >= 10: Everything is an input. The data, starts, and ends inputs are required.
  if (input_count > 1) {
    // Skip the first input. All other input need to be initializer
    for (size_t i = 1; i < input_count; i++) {
      const auto& next_input = node_unit.Inputs()[i].name;
      if (!qnn_model_wrapper.IsConstantInput(next_input)) {
        return MAKE_EP_FAIL("QNN doesn't support dynamic slice.");
      }
    }
  }

  return Ort::Status();
}

void SliceOpBuilder::GetDataFromAttribute(const OrtNodeUnit& node_unit,
                                          std::vector<int64_t>& raw_starts,
                                          std::vector<int64_t>& raw_ends,
                                          std::vector<int64_t>& raw_axes) const {
  OrtNodeAttrHelper node_helper(node_unit);
  auto starts = node_helper.Get("starts", std::vector<int64_t>{0});
  raw_starts.assign(starts.begin(), starts.end());
  auto ends = node_helper.Get("ends", std::vector<int64_t>{0});
  raw_ends.assign(ends.begin(), ends.end());
  if (node_helper.HasAttr("axes")) {
    auto axes = node_helper.Get("axes", std::vector<int64_t>{0});
    raw_axes.assign(axes.begin(), axes.end());
  }
}

// Gets the data from initializer inputs (e.g., starts, ends, axes, or steps) as a std::vector<int64_t>.
static Ort::Status GetInitializerInputData(const OrtNodeUnitIODef& input, const QnnModelWrapper& qnn_model_wrapper,
                                           std::vector<int64_t>& output) {
  const auto& input_name = input.name;
  const bool is_constant = qnn_model_wrapper.IsConstantInput(input_name);
  RETURN_IF_NOT(is_constant, ("Expected input " + input_name + " to be an initializer.").c_str());
  const OrtValueInfo* initializer_valueinfo = nullptr;
  RETURN_IF_ERROR(qnn_model_wrapper.FindInitializer(input_name, &initializer_valueinfo));

  // Deserialize initializer into byte buffer
  std::vector<uint8_t> initializer_bytes;
  RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(initializer_valueinfo, initializer_bytes));

  // Copy Tensor of int32_t or int64_t elems into output (int64_ts).
  ONNXTensorElementDataType onnx_type = input.type;
  if (onnx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    gsl::span<const int64_t> tensor_elems = ReinterpretAsSpan<int64_t, uint8_t>(initializer_bytes);
    output.insert(output.end(), tensor_elems.begin(), tensor_elems.end());
  } else if (onnx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    gsl::span<const int32_t> tensor_elems = ReinterpretAsSpan<int32_t, uint8_t>(initializer_bytes);
    output.insert(output.end(), tensor_elems.begin(), tensor_elems.end());
  } else {
    return MAKE_EP_FAIL(("Data type " + std::to_string(onnx_type) +
                         " is not supported for Slice initializer input " + input.name)
                            .c_str());
  }

  return Ort::Status();
}

// Note: For ONNX Slice operation the expected number of inputs is between 3 and 5
Ort::Status SliceOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          const Ort::Logger& logger,
                                          std::vector<std::string>& input_names,
                                          bool do_op_validation) const {
  if (do_op_validation) {
    RETURN_IF_ERROR(ExplictOpCheck(qnn_model_wrapper, node_unit));
  }

  // Only need to add input 0. The other inputs (if any) contain static data that is passed to QNN APIs
  // as static parameters.
  return ProcessInput(qnn_model_wrapper, node_unit.Inputs()[0], logger, input_names);
}

Ort::Status SliceOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                        const OrtNodeUnit& node_unit,
                                                        std::vector<std::string>&& input_names,
                                                        const Ort::Logger& logger,
                                                        bool do_op_validation) const {
  // Extract starts, ends, axes, and steps data from attributes (opset < 10) or initializer inputs (opset >= 10).
  std::vector<int64_t> raw_starts;
  std::vector<int64_t> raw_ends;
  std::vector<int64_t> raw_axes;
  std::vector<int64_t> raw_steps;

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
    RETURN_IF_ERROR(GetInitializerInputData(inputs[starts_index], qnn_model_wrapper, raw_starts));

    // Ends input (required).
    RETURN_IF_ERROR(GetInitializerInputData(inputs[ends_index], qnn_model_wrapper, raw_ends));

    // Axes input (optional).
    if (input_count > axes_index && inputs[axes_index].Exists()) {
      RETURN_IF_ERROR(GetInitializerInputData(inputs[axes_index], qnn_model_wrapper, raw_axes));
    }

    // Steps input (optional).
    if (input_count > steps_index && inputs[steps_index].Exists()) {
      RETURN_IF_ERROR(GetInitializerInputData(inputs[steps_index], qnn_model_wrapper, raw_steps));
    }
  }

  std::vector<uint32_t> input0_shape;
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].shape, input0_shape),
                "Cannot get shape for Slice input 0.");

  std::vector<int64_t> input_dimensions(input0_shape.cbegin(), input0_shape.cend());
  PrepareForComputeMetadata compute_metadata(input_dimensions);
  RETURN_IF_ERROR(PrepareForComputeHelper(raw_starts, raw_ends, raw_axes, raw_steps, compute_metadata));

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
  RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper,
                                 node_unit,
                                 std::move(input_names),
                                 {param_tensor_name},
                                 logger,
                                 do_op_validation, GetQnnOpType(node_unit.OpType())));
  return Ort::Status();
}

void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<SliceOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
