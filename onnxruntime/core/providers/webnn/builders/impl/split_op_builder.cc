// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class SplitOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

// Add operator related.

void SplitOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Skip split initializer if present.
  if (node.InputDefs().size() > 1) {
    model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
  }
}

Status SplitOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                             const Node& node,
                                             const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val output_array;
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  const size_t rank = input_shape.size();
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  NodeAttrHelper helper(node);
  int32_t axis = helper.Get("axis", 0);
  axis = SafeInt<int32_t>(HandleNegativeAxis(axis, rank));
  options.set("axis", axis);

  uint32_t split_count = 0;
  std::vector<uint32_t> splits = helper.Get("split", std::vector<uint32_t>{});

  // Read either the split count or explicit split lengths from the various attributes over opset versions.
  if (helper.HasAttr("num_outputs")) {
    split_count = helper.Get("num_outputs", 0);
  } else if (GetTensorName(input_defs, 1).size()) {
    const auto& initializers(model_builder.GetInitializerTensors());
    const auto& split_tensor = *initializers.at(input_defs[1]->Name());
    ORT_RETURN_IF_NOT(ReadIntArrayFrom1DTensor(split_tensor, splits, logger), "Cannot get input for split.");
  } else if (!helper.HasAttr("split")) {
    split_count = node.OutputDefs().size();
  }

  // Check that the splits evenly divide.
  if (split_count > 0 && splits.empty() && input_shape[axis] % split_count != 0) {
    // Divide inputs into variable size outputs:
    splits.insert(splits.end(), split_count - 1, narrow<uint32_t>(input_shape[axis]) / split_count);
    splits.insert(splits.end(), narrow<uint32_t>(input_shape[axis]) % split_count);
  }

  if (splits.empty()) {
    output_array = model_builder.GetBuilder().call<emscripten::val>(
        "split", input, split_count, options);
  } else {
    output_array = model_builder.GetBuilder().call<emscripten::val>(
        "split", input, emscripten::val::array(splits), options);
  }

  for (size_t i = 0, count = output_array["length"].as<size_t>(); i < count; i++) {
    model_builder.AddOperand(node.OutputDefs()[i]->Name(), std::move(output_array[i]));
  }
  return Status::OK();
}

// Operator support related.

bool SplitOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                       const Node& node,
                                       const WebnnDeviceType /* device_type */,
                                       const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get input's shape.";
    return false;
  }
  const size_t rank = input_shape.size();

  NodeAttrHelper helper(node);
  int32_t axis = helper.Get("axis", 0);
  axis = SafeInt<int32_t>(HandleNegativeAxis(axis, rank));
  std::vector<uint32_t> split = helper.Get("split", std::vector<uint32_t>{});

  const std::string split_name = GetTensorName(input_defs, 1);
  // Inputs contain optional 'split' input.
  if (!split_name.empty()) {
    if (!Contains(initializers, split_name)) {
      LOGS(logger, VERBOSE) << "The split must be a constant initializer.";
      return false;
    }
    // Values should be >= 0. Sum of the values must be equal to the dim value at 'axis' specified.
    const auto& split_tensor = *initializers.at(input_defs[1]->Name());
    if (split_tensor.data_type() != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      LOGS(logger, VERBOSE) << "The type of tensor's element data must be INT64.";
      return false;
    }
    if (!ReadIntArrayFrom1DTensor(split_tensor, split, logger)) {
      LOGS(logger, VERBOSE) << "Cannot get split.";
      return false;
    }
  } else {
    if (helper.HasAttr("num_outputs")) {
      // Split has 'num_outputs' attribute when opset is 18.
      const int32_t num_outputs = helper.Get("num_outputs", 1);
      if (num_outputs < 1) {
        LOGS(logger, VERBOSE) << "The 'num_outputs' must be a positive integer.";
        return false;
      }
    } else {
      const auto opset = node.SinceVersion();
      if (opset >= 18) {
        LOGS(logger, VERBOSE) << "The 'num_outputs' should be specified when 'split' isn't specified.";
        return false;
      }
    }
  }

  if (!split.empty()) {
    int64_t sum = 0;
    // TODO: Allow 0 size dimensions.
    // https://github.com/webmachinelearning/webnn/issues/391
    for (uint32_t split_value : split) {
      if (split_value <= 0) {
        LOGS(logger, VERBOSE) << "Value of split should be greater than 0.";
        return false;
      }
      sum += split_value;
    }
    if (sum != input_shape[axis]) {
      LOGS(logger, VERBOSE) << "Sum of the split's values must be equal to the dim value at 'axis' specified.";
      return false;
    }
  }
  return true;
}

bool SplitOpBuilder::HasSupportedOutputsImpl(const Node& node,
                                             const emscripten::val& wnn_limits,
                                             const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();
  const auto& op_type = node.OpType();
  int32_t output_type = 0;

  if (GetType(*output_defs[0], output_type, logger)) {
    // Chromium has changed the output name of split from 'output' to 'outputs',
    // to avoid breaking the existing API, we need to check both names.
    std::string wnn_output_name = wnn_limits["split"]["output"].isUndefined() ? "outputs" : "output";
    return IsDataTypeSupportedByOp(op_type, output_type, wnn_limits, wnn_output_name, "outputs", logger);
  }

  return false;
}

void CreateSplitOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SplitOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
