// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class GruOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /*device_type*/, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

void GruOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  if (node.InputDefs().size() > 4 && node.InputDefs()[4]->Exists()) {
    model_builder.AddInitializerToSkip(node.InputDefs()[4]->Name());  // sequence_lens
    model_builder.AddInputToSkip(node.InputDefs()[4]->Name());
  }
}

Status GruOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                           const logging::Logger& logger) const {
  NodeAttrHelper helper(node);
  uint32_t hidden_size = helper.Get("hidden_size", 1);

  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input's shape");
  uint32_t steps = static_cast<uint32_t>(input_shape[0]);
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val weight = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val recurrent_weight = model_builder.GetOperand(input_defs[2]->Name());

  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  options.set("layout", emscripten::val("zrn"));

  if (input_defs.size() > 3 && input_defs[3]->Exists()) {
    emscripten::val bias = model_builder.GetOperand(input_defs[3]->Name());
    emscripten::val split_options = emscripten::val::object();
    split_options.set("label", node.Name() + "_split");
    split_options.set("axis", 1);
    // Split it to bias and recurrentBias.
    emscripten::val splitted_biases =
        model_builder.GetBuilder().call<emscripten::val>("split", bias, /*splits*/ 2, split_options);
    options.set("bias", splitted_biases[0]);
    options.set("recurrentBias", splitted_biases[1]);
  }

  if (input_defs.size() > 5 && input_defs[5]->Exists()) {
    options.set("initialHiddenState", model_builder.GetOperand(input_defs[5]->Name()));
  }

  bool linear_before_reset = !!helper.Get("linear_before_reset ", 0);
  options.set("resetAfter", linear_before_reset);

  const auto& output_defs = node.OutputDefs();
  bool has_Y = output_defs.size() > 0 && output_defs[0]->Exists();
  bool has_Y_h = output_defs.size() > 1 && output_defs[1]->Exists();
  options.set("returnSequence", has_Y);

  std::string direction = helper.Get("direction", "forward");
  if (direction == "forward") {
    options.set("direction", emscripten::val("forward"));
  } else if (direction == "reverse") {
    options.set("direction", emscripten::val("backward"));
  } else if (direction == "bidirectional") {
    options.set("direction", emscripten::val("both"));
  }

  if (helper.HasAttr("activations")) {
    const auto activations = helper.Get("activations", std::vector<std::string>{"Sigmoid", "Tanh"});
    emscripten::val recurrent_network_activations = emscripten::val::array();
    for (size_t i = 0; i < 2; ++i) {
      const std::string& activation = activations[i];
      if (activation == "Relu") {
        recurrent_network_activations.call<void>("push", emscripten::val("relu"));
      } else if (activation == "Sigmoid") {
        recurrent_network_activations.call<void>("push", emscripten::val("sigmoid"));
      } else if (activation == "Tanh") {
        recurrent_network_activations.call<void>("push", emscripten::val("tanh"));
      }
    }

    options.set("activations", recurrent_network_activations);
  }

  emscripten::val outputs = model_builder.GetBuilder().call<emscripten::val>("gru", input, weight, recurrent_weight,
                                                                             steps, hidden_size, options);

  if (has_Y) {
    model_builder.AddOperand(output_defs[0]->Name(), outputs[1]);
  }
  if (has_Y_h) {
    model_builder.AddOperand(output_defs[1]->Name(), outputs[0]);
  }

  return Status::OK();
}

bool GruOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                     const WebnnDeviceType /*device_type*/, const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 3) {
    LOGS(logger, ERROR) << "GRU: input size must greater than or equal to 3";
    return false;
  }

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger) || input_shape.empty()) {
    LOGS(logger, ERROR) << "Cannot get input's shape";
    return false;
  }
  int32_t steps = static_cast<int32_t>(input_shape[0]);

  if (input_defs.size() > 4 && input_defs[4]->Exists()) {
    if (!Contains(initializers, input_defs[4]->Name())) {
      LOGS(logger, ERROR) << "GRU: sequence_lens must be constant";
      return false;
    }

    const auto& sequence_lens_tensor = *initializers.at(input_defs[4]->Name());
    std::vector<int32_t> sequence_lens;
    if (!ReadIntArrayFrom1DTensor(sequence_lens_tensor, sequence_lens, logger)) {
      LOGS(logger, ERROR) << "Cannot read sequence lens tensor";
      return false;
    }
    if (!std::all_of(sequence_lens.begin(), sequence_lens.end(),
                     [steps](int32_t lens) -> bool { return steps == lens; })) {
      LOGS(logger, ERROR) << "GRU: every sequence length must be equal to input shape[0]";
      return false;
    }
  }

  NodeAttrHelper helper(node);
  if (helper.HasAttr("activations")) {
    const auto activations = helper.Get("activations", std::vector<std::string>{"Sigmoid", "Tanh"});

    if (activations.size() >= 4) {
      if (activations[0] != activations[2] || activations[1] != activations[3]) {
        LOGS(logger, ERROR) << "GRU: forward and reverse directions must have the same activations";
        return false;
      }
    }

    const InlinedHashSet<std::string> supported_activations = {"Relu", "Tanh", "Sigmoid"};
    if (!std::all_of(activations.begin(), activations.end(),
                     [&supported_activations](const std::string& activation) -> bool {
                       return supported_activations.contains(activation);
                     })) {
      LOGS(logger, ERROR) << "GRU: activations must be one of Relu, Tanh, Sigmoid";
      return false;
    }
  }

  if (helper.Get("clip", std::numeric_limits<float>::max()) != std::numeric_limits<float>::max()) {
    LOGS(logger, ERROR) << "GRU: clip is not supported";
    return false;
  }

  if (helper.Get("layout", 0) != 0) {
    LOGS(logger, ERROR) << "GRU: batchwise (layout == 1) is not supported";
    return false;
  }

  return true;
}

bool GruOpBuilder::HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                          const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  int32_t input_X_type = 0;          // input data type
  int32_t input_W_type = 0;          // weight data type
  int32_t input_R_type = 0;          // recurrent weight data type
  int32_t input_B_type = 0;          // bias data type
  int32_t input_initial_h_type = 0;  // initial hidden state data type
  bool has_input_B = input_defs.size() > 3 && input_defs[3]->Exists();
  bool has_input_initial_h = input_defs.size() > 5 && input_defs[5]->Exists();

  if (!GetType(*input_defs[0], input_X_type, logger) ||
      !GetType(*input_defs[1], input_W_type, logger) ||
      !GetType(*input_defs[2], input_R_type, logger) ||
      (has_input_B && !GetType(*input_defs[3], input_B_type, logger)) ||
      // input_defs[4] refers to sequence_lens and is a fixed data type of int32.
      (has_input_initial_h && !GetType(*input_defs[5], input_initial_h_type, logger))) {
    return false;
  }

  InlinedVector<int32_t, 5> input_types = {input_X_type, input_W_type, input_R_type};
  if (has_input_B) {
    input_types.push_back(input_B_type);
  }
  if (has_input_initial_h) {
    input_types.push_back(input_initial_h_type);
  }
  if (!AreInputDataTypesSame(op_type, input_types, logger)) {
    return false;
  }

  return IsDataTypeSupportedByOp(op_type, input_X_type, wnn_limits, "input", "X", logger);
}

bool GruOpBuilder::HasSupportedOutputsImpl(const Node& node,
                                           const emscripten::val& wnn_limits,
                                           const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();
  const auto& op_type = node.OpType();
  int32_t Y_type = 0;
  int32_t Y_h_type = 0;
  bool has_Y = output_defs.size() > 0 && output_defs[0]->Exists();
  bool has_Y_h = output_defs.size() > 1 && output_defs[1]->Exists();

  bool Y_supported = has_Y && GetType(*output_defs[0], Y_type, logger);
  bool Y_h_supported = has_Y_h && GetType(*output_defs[1], Y_h_type, logger);

  if (Y_supported && !Y_h_supported) {
    return IsDataTypeSupportedByOp(op_type, Y_type, wnn_limits, "outputs", "Y", logger);
  } else if (!Y_supported && Y_h_supported) {
    return IsDataTypeSupportedByOp(op_type, Y_h_type, wnn_limits, "outputs", "Y_h", logger);
  } else if (Y_supported && Y_h_supported) {
    if (Y_type != Y_h_type) {
      LOGS(logger, VERBOSE) << "[GRU] Output data types must be the same.";
      return false;
    }
    return IsDataTypeSupportedByOp(op_type, Y_type, wnn_limits, "outputs", "Y", logger);
  } else {
    LOGS(logger, VERBOSE) << "[GRU] No output found.";
    return false;
  }
}

void CreateGruOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GruOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
