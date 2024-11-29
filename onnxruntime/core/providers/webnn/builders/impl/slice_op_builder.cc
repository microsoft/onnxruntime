// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class SliceOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  // TODO: Support Slice opset < 10, which uses attributes for starts and ends.
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 10; }
};

// Add operator related.

void SliceOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Skip all initializer except the first inputs(data).
  for (size_t i = 1; i < node.InputDefs().size(); i++) {
    model_builder.AddInitializerToSkip(node.InputDefs()[i]->Name());
  }
}

Status SliceOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  auto rank = input_shape.size();
  NodeAttrHelper helper(node);

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());

  // Copy the data from the starts/ends/axes/steps initializers.
  std::vector<int64_t> input_starts;
  std::vector<int64_t> input_ends;
  std::vector<int64_t> input_axes;
  std::vector<int64_t> input_steps;
  SliceOp::PrepareForComputeMetadata compute_metadata(input_shape);
  const auto CopyInputData = [&input_defs, &model_builder, &logger](size_t input_idx, std::vector<int64_t>& data,
                                                                    bool is_required = false) {
    data.clear();
    std::string input_name;
    // This is an optional input, return empty vector.
    if (!is_required) {
      if (input_defs.size() <= input_idx)
        return Status::OK();
      input_name = input_defs[input_idx]->Name();
      if (input_name.empty())
        return Status::OK();
    }
    input_name = input_defs[input_idx]->Name();
    const auto& initializers(model_builder.GetInitializerTensors());
    const auto& tensor = *initializers.at(input_name);
    if (!ReadIntArrayFrom1DTensor(tensor, data, logger)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Data type for starts and ends inputs is not supported in this build.");
    }

    return Status::OK();
  };
  ORT_RETURN_IF_ERROR(CopyInputData(1, input_starts, true));
  ORT_RETURN_IF_ERROR(CopyInputData(2, input_ends, true));
  ORT_RETURN_IF_ERROR(CopyInputData(3, input_axes));
  ORT_RETURN_IF_ERROR(CopyInputData(4, input_steps));
  ORT_RETURN_IF_ERROR(
      SliceOp::PrepareForComputeHelper(input_starts, input_ends, input_axes, input_steps, compute_metadata));

  // Check if reverse op is needed.
  std::vector<uint32_t> reverse_axes;
  emscripten::val reverse_output = input;
  for (size_t i = 0; i < rank; ++i) {
    if (compute_metadata.steps_[i] < 0) {
      reverse_axes.push_back(SafeInt<uint32_t>(i));
      compute_metadata.steps_[i] = -compute_metadata.steps_[i];
      compute_metadata.starts_[i] = input_shape[i] - 1 - compute_metadata.starts_[i];
      compute_metadata.ends_[i] = input_shape[i] - 1 - compute_metadata.ends_[i];
    }
  }
  if (!reverse_axes.empty()) {
    emscripten::val reverse_options = emscripten::val::object();
    reverse_options.set("axes", emscripten::val::array(reverse_axes));
    reverse_options.set("label", node.Name() + "_reverse");
    reverse_output = model_builder.GetBuilder().call<emscripten::val>("reverse", input, reverse_options);
  }

  // Check if slice op is needed.
  bool is_slice_required = false;
  for (size_t i = 0; i < rank; ++i) {
    if (compute_metadata.steps_[i] != 1 || compute_metadata.starts_[i] != 0 ||
        compute_metadata.ends_[i] != input_shape[i]) {
      is_slice_required = true;
      break;
    }
  }

  emscripten::val output = reverse_output;
  if (is_slice_required) {
    std::vector<uint32_t> starts = GetVecUint32FromVecInt64(compute_metadata.starts_);
    std::vector<uint32_t> steps = GetVecUint32FromVecInt64(compute_metadata.steps_);
    std::vector<uint32_t> sizes(rank);
    std::transform(compute_metadata.ends_.cbegin(), compute_metadata.ends_.cend(), compute_metadata.starts_.cbegin(),
                   sizes.begin(), [](int64_t i, int64_t j) { return SafeInt<uint32_t>(i - j); });

    emscripten::val options = emscripten::val::object();
    options.set("strides", emscripten::val::array(steps));
    options.set("label", node.Name());
    output = model_builder.GetBuilder().call<emscripten::val>("slice", reverse_output, emscripten::val::array(starts),
                                                              emscripten::val::array(sizes), options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

bool SliceOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                       const WebnnDeviceType /* device_type */, const logging::Logger& logger) const {
  const auto& name = node.Name();
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return false;
  }

  if (input_defs.size() < 3) {
    LOGS(logger, VERBOSE) << op_type << " [" << name << "] requires at least 3 inputs (data, starts, ends) but got "
                          << input_defs.size();
    return false;
  }

  // Inputs: starts, ends, axes, and steps must be constant initializers if present.
  for (size_t i = 1; i < input_defs.size(); i++) {
    // Optional tensors (axes, steps) can be indicated by an empty name, just ignore it.
    const std::string input_name = GetTensorName(input_defs, i);
    if (!input_name.empty() && !Contains(initializers, input_name)) {
      LOGS(logger, VERBOSE) << "Input [" << input_name << "] of " << op_type << " [" << name
                            << "] must be known as initializer";
      return false;
    }
  }

  return true;
}

void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SliceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
