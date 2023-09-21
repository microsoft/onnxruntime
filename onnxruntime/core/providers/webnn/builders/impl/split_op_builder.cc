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
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;

  int GetMinSupportedOpSet(const Node& node) const override;
};

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

  NodeAttrHelper helper(node);
  int32_t axis = helper.Get("axis", 0);
  axis = SafeInt<int32_t>(HandleNegativeAxis(axis, rank));
  options.set("axis", axis);

  if (input_defs.size() == 2) {
    // Inputs contains optional 'split' input
    std::vector<int32_t> splits;
    const auto& initializers(model_builder.GetInitializerTensors());
    const auto& split_tensor = *initializers.at(input_defs[1]->Name());
    ORT_RETURN_IF_NOT(ReadIntArrayFrom1DTensor(split_tensor, splits, logger), "Cannot get split.");
    output_array = model_builder.GetBuilder().call<emscripten::val>("split",
                                                                    input,
                                                                    emscripten::val::array(splits),
                                                                    options);
    ORT_RETURN_IF_NOT(output_array["length"].as<int32_t>() == static_cast<int32_t>(splits.size()),
                      "The size of outputs must be equal to the size of 'split' input.");
  } else {
    if (helper.HasAttr("num_outputs")) {
      const int32_t num_outputs = helper.Get("num_outputs", 1);
      ORT_RETURN_IF_NOT(num_outputs > 0, "The 'num_outputs' must be a positive integer.");
      if (input_shape[axis] % num_outputs == 0) {
        // The 'num_outputs' evenly divide the dim value at 'axis' specified.
        output_array = model_builder.GetBuilder().call<emscripten::val>("split",
                                                                        input,
                                                                        num_outputs,
                                                                        options);
      } else {
        std::vector<int64_t> mapping_split;
        mapping_split.insert(mapping_split.begin(), num_outputs - 1, input_shape[axis] / num_outputs);
        mapping_split.insert(mapping_split.end(), input_shape[axis] % num_outputs);
        std::vector<int32_t> converted_splits;
        std::transform(mapping_split.cbegin(), mapping_split.cend(),
                       std::back_inserter(converted_splits),
                       [](int64_t dim) -> int32_t { return SafeInt<int32_t>(dim); });
        output_array = model_builder.GetBuilder().call<emscripten::val>("split",
                                                                        input,
                                                                        emscripten::val::array(converted_splits),
                                                                        options);
      }
      ORT_RETURN_IF_NOT(output_array["length"].as<int32_t>() == num_outputs,
                        "The size of outputs must be equal to 'num_outputs'.");
    } else {
      // w/o 'split' input for opset 13
      // Refer to https://github.com/microsoft/onnxruntime/blob/a7ad859e3ab60bddfcf2fefa96bfcb550f0fc04c/onnxruntime/core/providers/dml/OperatorAuthorHelper/OperatorHelper.cpp#L984-L989
      // split input stream equally across output streams.
      const auto& output_defs = node.OutputDefs();
      const size_t output_count = output_defs.size();
      output_array = model_builder.GetBuilder().call<emscripten::val>("split",
                                                                      input, static_cast<int32_t>(output_count),
                                                                      options);
      ORT_RETURN_IF_NOT(output_array["length"].as<size_t>() == output_count,
                        "The size of outputs must be equal to the count of output nodes.");
    }
  }
  for (size_t i = 0, count = output_array["length"].as<size_t>(); i < count; i++) {
    model_builder.AddOperand(node.OutputDefs()[i]->Name(), std::move(output_array[i]));
  }
  return Status::OK();
}

// Operator support related.

int SplitOpBuilder::GetMinSupportedOpSet(const Node& /* node */) const {
  // Since opset 13, Split has optional 'split' input.
  return 13;
}

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

  if (input_defs.size() == 2) {
    // Inputs contains optional 'split' input
    const auto& split_name = input_defs[1]->Name();
    if (!Contains(initializers, split_name)) {
      LOGS(logger, VERBOSE) << "The split must be a constant initializer.";
      return false;
    }
    // Values should be >= 0. Sum of the values must be equal to the dim value at 'axis' specified.
    std::vector<int64_t> split;
    const auto& split_tensor = *initializers.at(input_defs[1]->Name());
    if (split_tensor.data_type() != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      LOGS(logger, VERBOSE) << "The type of tensor's element data must be INT64.";
      return false;
    }
    if (!ReadIntArrayFrom1DTensor(split_tensor, split, logger)) {
      LOGS(logger, VERBOSE) << "Cannot get split.";
      return false;
    }
    int64_t sum = 0;
    for (size_t i = 0; i < split.size(); i++) {
      if (split[i] < 0) {
        LOGS(logger, VERBOSE) << "Value of split should be greater than or equal to 0.";
        return false;
      }
      sum += split[i];
    }
    if (sum != input_shape[axis]) {
      LOGS(logger, VERBOSE) << "Sum of the split's values must be equal to the dim value at 'axis' specified.";
      return false;
    }
  } else if (input_defs.size() == 1) {
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
  return true;
}

void CreateSplitOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SplitOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
