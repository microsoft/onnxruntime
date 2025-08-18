// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class NormalizationOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer&, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

Status NormalizationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                     const Node& node,
                                                     const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();
  ORT_RETURN_IF_NOT(input_defs.size() >= 2, op_type, " requires at least two inputs.");

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  const auto rank = input_shape.size();

  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  const size_t scale_input_index = op_type == "SkipSimplifiedLayerNormalization" ? 2 : 1;
  emscripten::val scale = model_builder.GetOperand(input_defs[scale_input_index]->Name());
  options.set("scale", scale);

  const size_t bias_input_index = op_type == "SkipSimplifiedLayerNormalization" ? 3 : 2;
  emscripten::val bias = emscripten::val::undefined();
  if (TensorExists(input_defs, bias_input_index)) {
    // Bias input exists.
    bias = model_builder.GetOperand(input_defs[bias_input_index]->Name());
    options.set("bias", bias);
  }

  NodeAttrHelper helper(node);
  const auto epsilon = helper.Get("epsilon", 1e-05f);
  options.set("epsilon", epsilon);

  emscripten::val output = emscripten::val::undefined();
  if (op_type == "BatchNormalization") {
    ORT_RETURN_IF_NOT(input_defs.size() == 5, "BatchNormalization requires five inputs.");
    emscripten::val mean = model_builder.GetOperand(input_defs[3]->Name());
    emscripten::val variance = model_builder.GetOperand(input_defs[4]->Name());
    output = model_builder.GetBuilder().call<emscripten::val>("batchNormalization", input, mean, variance, options);
  } else if (op_type == "LayerNormalization" ||
             op_type == "SimplifiedLayerNormalization" ||
             op_type == "SkipSimplifiedLayerNormalization") {
    int64_t axis = helper.Get("axis", -1);
    axis = HandleNegativeAxis(axis, rank);
    std::vector<uint32_t> axes(rank - SafeInt<uint32_t>(axis));
    std::iota(axes.begin(), axes.end(), axis);

    if (op_type == "LayerNormalization") {
      options.set("axes", emscripten::val::array(axes));
      output = model_builder.GetBuilder().call<emscripten::val>("layerNormalization", input, options);
    } else {  // SimplifiedLayerNormalization or SkipSimplifiedLayerNormalization
      /**
      WebNN doesn't support SimplifiedLayerNormalization or SkipSimplifiedLayerNormalization.
      So decompose it into a series of ops:
          X --> Pow --> ReduceMean --> Add --> Sqrt --> Div -> Mul -> Add (optional)
                ^          ^           ^                ^      ^       ^
                |          |           |                |      |       |
               Y:2        axis     B:epsilon           A:X  A:scale  B:bias

      If it is SkipSimplifiedLayerNormalization, X should be input_skip_bias_sum:
      input_skip_bias_sum = X + skip + bias (if it exists)
      */

      int32_t input_type;
      ORT_RETURN_IF_NOT(GetType(*input_defs[0], input_type, logger), "Cannot get input type");
      emscripten::val common_options = emscripten::val::object();

      if (input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
        // Decomposed *SimplifiedLayerNormalization may lose precision if its data type is float16.
        // So cast all inputs to float32 to ensure precision.
        common_options.set("label", node.Name() + "_cast_input_to_fp32");
        input = model_builder.GetBuilder().call<emscripten::val>("cast", input,
                                                                 emscripten::val("float32"), common_options);

        common_options.set("label", node.Name() + "_cast_scale_to_fp32");
        scale = model_builder.GetBuilder().call<emscripten::val>("cast", scale,
                                                                 emscripten::val("float32"), common_options);

        if (!bias.isUndefined()) {
          common_options.set("label", node.Name() + "_cast_bias_to_fp32");
          bias = model_builder.GetBuilder().call<emscripten::val>("cast", bias,
                                                                  emscripten::val("float32"), common_options);
        }
      }

      // If it is SkipSimplifiedLayerNormalization, add the skip and bias (if it exists) to the input.
      if (op_type == "SkipSimplifiedLayerNormalization") {
        emscripten::val skip = model_builder.GetOperand(input_defs[1]->Name());
        if (input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
          // Cast skip to float32
          common_options.set("label", node.Name() + "_cast_skip_to_fp32");
          skip = model_builder.GetBuilder().call<emscripten::val>("cast", skip,
                                                                  emscripten::val("float32"), common_options);
        }
        common_options.set("label", node.Name() + "_add_skip");
        input = model_builder.GetBuilder().call<emscripten::val>("add", input, skip, common_options);
        if (!bias.isUndefined()) {
          common_options.set("label", node.Name() + "_add_skip_bias");
          input = model_builder.GetBuilder().call<emscripten::val>("add", input, bias, common_options);
        }

        // Add SkipSimplifiedLayerNormalization's output input_skip_bias_sum if it exists.
        // Now input equals to input_skip_bias_sum.
        if (TensorExists(output_defs, 3)) {
          emscripten::val input_skip_bias_sum = input;
          if (input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
            // Cast input_skip_bias_sum back to float16.
            common_options.set("label", node.Name() + "_cast_input_skip_bias_sum_to_fp16");
            input_skip_bias_sum = model_builder.GetBuilder().call<emscripten::val>("cast", input_skip_bias_sum,
                                                                                   emscripten::val("float16"),
                                                                                   common_options);
          }
          model_builder.AddOperand(output_defs[3]->Name(), input_skip_bias_sum);
        }
      }

      // Pow
      emscripten::val pow_constant =
          model_builder.CreateOrGetConstant<float>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 2);
      common_options.set("label", node.Name() + "_pow");
      emscripten::val pow =
          model_builder.GetBuilder().call<emscripten::val>("pow", input, pow_constant, common_options);

      // ReduceMean
      emscripten::val reduce_options = emscripten::val::object();
      reduce_options.set("axes", emscripten::val::array(axes));
      reduce_options.set("keepDimensions", true);
      reduce_options.set("label", node.Name() + "_reduceMean");
      emscripten::val reduce_mean = model_builder.GetBuilder().call<emscripten::val>("reduceMean", pow, reduce_options);

      // Add
      emscripten::val add_constant =
          model_builder.CreateOrGetConstant<float>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, epsilon);
      common_options.set("label", node.Name() + "_add");
      emscripten::val add =
          model_builder.GetBuilder().call<emscripten::val>("add", reduce_mean, add_constant, common_options);

      // Sqrt
      common_options.set("label", node.Name() + "_sqrt");
      emscripten::val sqrt = model_builder.GetBuilder().call<emscripten::val>("sqrt", add, common_options);

      // Div
      common_options.set("label", node.Name() + "_div");
      emscripten::val div = model_builder.GetBuilder().call<emscripten::val>("div", input, sqrt, common_options);

      // Mul
      common_options.set("label", node.Name() + "_mul");
      output = model_builder.GetBuilder().call<emscripten::val>("mul", scale, div, common_options);

      // Add (if bias exists)
      if (!bias.isUndefined()) {
        common_options.set("label", node.Name() + "_add_bias");
        output = model_builder.GetBuilder().call<emscripten::val>("add", output, bias, common_options);
      }

      if (input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
        // Cast output back to float16.
        common_options.set("label", node.Name() + "_cast_output_to_fp16");
        output = model_builder.GetBuilder().call<emscripten::val>("cast", output,
                                                                  emscripten::val("float16"), common_options);
      }
    }
  } else if (op_type == "InstanceNormalization") {
    // WebNN spec only supports 4D input for instanceNormalization.
    // Supports 3D input by prepending 1 size dimension.
    // For models with dimensions greater than 4, they will be reshaped into 4D.
    constexpr size_t webnn_shape_rank = 4;
    if (input_shape.size() != webnn_shape_rank) {
      std::vector<uint32_t> new_shape;
      new_shape.reserve(std::max(input_shape.size(), webnn_shape_rank));
      std::transform(input_shape.begin(), input_shape.end(),
                     std::back_inserter(new_shape),
                     [](int64_t dim) -> uint32_t { return SafeInt<uint32_t>(dim); });

      ptrdiff_t excess_rank = new_shape.size() - webnn_shape_rank;
      auto insertion_point = new_shape.begin() + 3;
      if (input_shape.size() < webnn_shape_rank) {
        // Pad the shape with extra 1's to satisfy WebNN v1's rank requirements.
        new_shape.insert(insertion_point, -excess_rank, 1);
      } else {
        // Fold the extra range to fit within WebNN v1's rank requirements.
        uint32_t sum = std::accumulate(
            insertion_point, insertion_point + excess_rank + 1, 1, std::multiplies<uint32_t>());
        new_shape.erase(insertion_point, insertion_point + excess_rank);
        *insertion_point = sum;
      }
      emscripten::val reshape_input_options = emscripten::val::object();
      reshape_input_options.set("label", node.Name() + "_reshape_input");
      input = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                               input,
                                                               emscripten::val::array(new_shape),
                                                               reshape_input_options);
    }

    output = model_builder.GetBuilder().call<emscripten::val>("instanceNormalization", input, options);
    // Reshape back to the original output shape for 3D input.
    if (input_shape.size() != 4) {
      std::vector<uint32_t> output_shape = GetNarrowedIntFromInt64<uint32_t>(input_shape);
      emscripten::val reshape_output_options = emscripten::val::object();
      reshape_output_options.set("label", node.Name() + "reshape_output");
      output = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                                output,
                                                                emscripten::val::array(output_shape),
                                                                reshape_output_options);
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported normalization op: ", op_type);
  }
  model_builder.AddOperand(output_defs[0]->Name(), std::move(output));

  return Status::OK();
}

// Operator support related.

bool NormalizationOpBuilder::IsOpSupportedImpl(const GraphViewer&,
                                               const Node& node,
                                               const WebnnDeviceType /* device_type */,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  NodeAttrHelper helper(node);

  if (input_defs.size() < 2) {
    LOGS(logger, VERBOSE) << op_type << " requires at least two inputs.";
    return false;
  }

  const auto& output_defs = node.OutputDefs();
  if (op_type == "SkipSimplifiedLayerNormalization") {
    if (output_defs.size() > 4) {
      LOGS(logger, VERBOSE) << "SkipSimplifiedLayerNormalization output count must not exceed 4.";
      return false;
    }
    if (TensorExists(output_defs, 1) || TensorExists(output_defs, 2)) {
      // Output mean and inv_std_var are used for training mode, which is not supported.
      LOGS(logger, VERBOSE) << "SkipSimplifiedLayerNormalization's output mean and inv_std_var are not supported.";
      return false;
    }
  } else {
    if (output_defs.size() != 1) {
      LOGS(logger, VERBOSE) << op_type << " output count must be one.";
      return false;
    }
  }

  if (op_type == "BatchNormalization" && helper.Get("training_mode", 0)) {
    LOGS(logger, VERBOSE) << "BatchNormalization with training_mode set to true is not supported.";
    return false;
  }

  return true;
}

bool NormalizationOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                                    const emscripten::val& wnn_limits,
                                                    const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string_view op_type = node.OpType();

  std::vector<int32_t> input_types;
  bool all_types_valid = true;

  // Iterate through all inputs and check their existence and types
  for (size_t i = 0; i <= input_defs.size(); ++i) {
    if (TensorExists(input_defs, i)) {
      int32_t input_type;
      if (!GetType(*input_defs[i], input_type, logger)) {
        all_types_valid = false;
        break;
      }
      input_types.push_back(input_type);
    }
  }

  // Return false if any input type is invalid
  if (!all_types_valid) {
    return false;
  }

  // Check if all input data types are the same
  if (!AreDataTypesSame(op_type, input_types, logger)) {
    return false;
  }

  if (op_type == "SimplifiedLayerNormalization" || op_type == "SkipSimplifiedLayerNormalization") {
    // SkipSimplifiedLayerNormalization and SimplifiedLayerNormalization are supported by decomposed WebNN ops.
    // Check if the input data type is supported by each decomposed WebNN op.
    // Decomposed ops include: "Add", "Div", "Mul", "Pow", "ReduceMean" and "Sqrt".
    for (const std::string_view decomposed_op_type : decomposed_op_map.at(op_type)) {
      const std::string_view webnn_op_type = GetWebNNOpType(decomposed_op_type);
      const std::string_view webnn_input_name = GetWebNNOpFirstInputName(decomposed_op_type);
      if (!IsDataTypeSupportedByWebNNOp(
              op_type, webnn_op_type, input_types[0], wnn_limits, webnn_input_name, "input", logger)) {
        return false;
      }
    }

    std::vector<int64_t> input_shape;
    if (!GetShape(*input_defs[0], input_shape, logger)) {
      return false;
    }
    // It's complicated to check all the decomposed ops' input rank support.
    // Ensure at least the first input rank is supported by the decomposed ops (pow and div accept the first input).
    return IsInputRankSupported(wnn_limits, "pow", "a", input_shape.size(), node.Name(), logger) &&
           IsInputRankSupported(wnn_limits, "div", "a", input_shape.size(), node.Name(), logger);
  } else {
    bool is_data_type_supported = IsDataTypeSupportedByOp(op_type, input_types[0], wnn_limits, "input", "X", logger);
    if (op_type == "InstanceNormalization") {
      // Skip input rank check for InstanceNormalization, as we will reshape the input to 4D if necessary.
      return is_data_type_supported;
    }

    // For other ops, check both data type and input rank compatibility.
    bool is_input_rank_supported = IsInputRankSupportedByOp(node, wnn_limits, logger);
    return is_input_rank_supported && is_data_type_supported;
  }
}

bool NormalizationOpBuilder::HasSupportedOutputsImpl(const Node& node,
                                                     const emscripten::val& wnn_limits,
                                                     const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();
  const std::string_view op_type = node.OpType();
  int32_t output_type = 0;
  if (!GetType(*output_defs[0], output_type, logger)) {
    return false;
  }

  if (op_type == "SimplifiedLayerNormalization" || op_type == "SkipSimplifiedLayerNormalization") {
    // Check if the output data type is supported by every decomposed WebNN op.
    for (const std::string_view decomposed_op_type : decomposed_op_map.at(op_type)) {
      const std::string_view webnn_op_type = GetWebNNOpType(decomposed_op_type);
      if (!IsDataTypeSupportedByWebNNOp(op_type, webnn_op_type, output_type, wnn_limits, "output", "output", logger)) {
        return false;
      }
    }
    return true;
  } else {
    return IsDataTypeSupportedByOp(op_type, output_type, wnn_limits, "output", "Output", logger);
  }
}

void CreateNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  constexpr static std::string_view op_types[] =
      {
          "BatchNormalization",
          "InstanceNormalization",
          "LayerNormalization",
          "SimplifiedLayerNormalization",
          "SkipSimplifiedLayerNormalization",
      };

  op_registrations.builders.push_back(std::make_unique<NormalizationOpBuilder>());
  for (const auto& op_type : op_types) {
    op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
