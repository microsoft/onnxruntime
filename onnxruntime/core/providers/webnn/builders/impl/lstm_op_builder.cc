// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime::webnn {

class LstmOpBuilder : public BaseOpBuilder {
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
};

void LstmOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  if (node.InputDefs().size() > 4 && node.InputDefs()[4]->Exists()) {
    model_builder.AddInitializerToSkip(node.InputDefs()[4]->Name());  // sequence_lens
    model_builder.AddInputToSkip(node.InputDefs()[4]->Name());
  }
}

Status LstmOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
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
  options.set("layout", emscripten::val("iofg"));

  if (input_defs.size() > 3 && input_defs[3]->Exists()) {
    emscripten::val bias = model_builder.GetOperand(input_defs[3]->Name());
    emscripten::val split_options = emscripten::val::object();
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
  if (input_defs.size() > 6 && input_defs[6]->Exists()) {
    options.set("initialCellState", model_builder.GetOperand(input_defs[6]->Name()));
  }
  if (input_defs.size() > 7 && input_defs[7]->Exists()) {
    options.set("peepholeWeight", model_builder.GetOperand(input_defs[7]->Name()));
  }

  std::string direction = helper.Get("direction", "forward");
  if (direction == "forward") {
    options.set("direction", emscripten::val("forward"));
  } else if (direction == "reverse") {
    options.set("direction", emscripten::val("backward"));
  } else if (direction == "bidirectional") {
    options.set("direction", emscripten::val("both"));
  }

  const auto& output_defs = node.OutputDefs();
  bool has_Y = output_defs.size() > 0 && output_defs[0]->Exists();
  bool has_Y_h = output_defs.size() > 1 && output_defs[1]->Exists();
  bool has_Y_c = output_defs.size() > 2 && output_defs[2]->Exists();
  if (has_Y) {
    options.set("returnSequence", true);
  }

  if (helper.HasAttr("activations")) {
    const auto activations = helper.Get("activations", std::vector<std::string>{"Sigmoid", "Tanh", "Tanh"});
    const auto activation_alpha = helper.Get("activation_alpha", std::vector<float>{});
    const auto activation_beta = helper.Get("activation_beta", std::vector<float>{});

    auto get_value_or_default = [](std::vector<float>::const_iterator& entry,
                                   const std::vector<float>::const_iterator& end,
                                   float def_val) -> float { return entry == end ? def_val : *entry++; };

    auto alpha_iter = activation_alpha.begin();
    auto beta_iter = activation_beta.begin();
    const auto alpha_iter_end = activation_alpha.end();
    const auto beta_iter_end = activation_beta.end();

    emscripten::val opt_activations = emscripten::val::array();
    for (size_t i = 0; i < 3; ++i) {
      const std::string& activation = activations[i];
      if (activation == "Affine") {
        emscripten::val affine_options = emscripten::val::object();
        affine_options.set("alpha", get_value_or_default(alpha_iter, alpha_iter_end, 1.0));
        affine_options.set("beta", get_value_or_default(beta_iter, beta_iter_end, 0));
        opt_activations.call<void>("push", model_builder.GetBuilder().call<emscripten::val>("linear", affine_options));
      } else if (activation == "Elu") {
        emscripten::val elu_options = emscripten::val::object();
        elu_options.set("alpha", get_value_or_default(alpha_iter, alpha_iter_end, 1.0));
        opt_activations.call<void>("push", model_builder.GetBuilder().call<emscripten::val>("elu", elu_options));
      } else if (activation == "HardSigmoid") {
        emscripten::val hard_sigmoid_options = emscripten::val::object();
        hard_sigmoid_options.set("alpha", get_value_or_default(alpha_iter, alpha_iter_end, 0.2));
        hard_sigmoid_options.set("beta", get_value_or_default(beta_iter, beta_iter_end, 0.5));
        opt_activations.call<void>(
            "push", model_builder.GetBuilder().call<emscripten::val>("hardSigmoid", hard_sigmoid_options));
      } else if (activation == "LeakyRelu") {
        emscripten::val leaky_relu_options = emscripten::val::object();
        leaky_relu_options.set("alpha", get_value_or_default(alpha_iter, alpha_iter_end, 0.01));
        opt_activations.call<void>("push",
                                   model_builder.GetBuilder().call<emscripten::val>("leakyRelu", leaky_relu_options));
      } else if (activation == "Relu") {
        opt_activations.call<void>("push", model_builder.GetBuilder().call<emscripten::val>("relu"));
      } else if (activation == "Sigmoid") {
        opt_activations.call<void>("push", model_builder.GetBuilder().call<emscripten::val>("sigmoid"));
      } else if (activation == "Softplus") {
        opt_activations.call<void>("push", model_builder.GetBuilder().call<emscripten::val>("softplus"));
      } else if (activation == "Softsign") {
        opt_activations.call<void>("push", model_builder.GetBuilder().call<emscripten::val>("softsign"));
      } else if (activation == "Tanh") {
        opt_activations.call<void>("push", model_builder.GetBuilder().call<emscripten::val>("tanh"));
      }
    }

    options.set("activations", opt_activations);
  }

  emscripten::val outputs = model_builder.GetBuilder().call<emscripten::val>("lstm", input, weight, recurrent_weight,
                                                                             steps, hidden_size, options);

  if (has_Y) {
    model_builder.AddOperand(output_defs[0]->Name(), outputs[2]);
  }
  if (has_Y_h) {
    model_builder.AddOperand(output_defs[1]->Name(), outputs[0]);
  }
  if (has_Y_c) {
    model_builder.AddOperand(output_defs[2]->Name(), outputs[1]);
  }

  return Status::OK();
}

bool LstmOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                      const WebnnDeviceType /*device_type*/, const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    LOGS(logger, ERROR) << "Cannot get input's shape";
    return false;
  }
  int32_t steps = static_cast<int32_t>(input_shape[0]);

  if (input_defs.size() > 4 && input_defs[4]->Exists()) {
    if (!Contains(initializers, input_defs[4]->Name())) {
      LOGS(logger, ERROR) << "LSTM: sequence_lens must be constant";
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
      LOGS(logger, ERROR) << "LSTM: every sequence length must be equal to input shape[0]";
      return false;
    }
  }

  NodeAttrHelper helper(node);
  if (helper.HasAttr("activations")) {
    const auto activations = helper.Get("activations", std::vector<std::string>{"Sigmoid", "Tanh", "Tanh"});

    if (activations.size() >= 6) {
      if (activations[0] != activations[3] || activations[1] != activations[4] || activations[2] != activations[5]) {
        LOGS(logger, ERROR) << "LSTM: forward and reverse directions must have the same activations";
        return false;
      }
      // TODO(shiyi9801): support activation_alpha and activation_beta when provided 6 activations.
      if (helper.HasAttr("activation_alpha") || helper.HasAttr("activation_beta")) {
        LOGS(logger, ERROR)
            << "LSTM: activation_alpha and activation_beta are not supported when provided 6 activations";
        return false;
      }
    }

    const InlinedHashSet<std::string> supported_activations = {"Affine", "Relu", "LeakyRelu", "Tanh", "Sigmoid",
                                                               "HardSigmoid", "Elu", "Softsign", "Softplus"};
    if (!std::all_of(activations.begin(), activations.end(),
                     [&supported_activations](const std::string& activation) -> bool {
                       return supported_activations.contains(activation);
                     })) {
      LOGS(logger, ERROR) << "LSTM: activations must be one of Affine, Relu, LeakyRelu, Tanh, Sigmoid, HardSigmoid, "
                             "Elu, Softsign, Softplus";
      return false;
    }
  }

  if (helper.Get("clip", std::numeric_limits<float>::max()) != std::numeric_limits<float>::max()) {
    LOGS(logger, ERROR) << "LSTM: clip is not supported";
    return false;
  }

  if (helper.Get("input_forget", 0) != 0) {
    LOGS(logger, ERROR) << "LSTM: input_forget == 1 is not supported";
    return false;
  }

  if (helper.Get("layout", 0) != 0) {
    LOGS(logger, ERROR) << "LSTM: batchwise (layout == 1) is not supported";
    return false;
  }

  return true;
}

void CreateLstmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<LstmOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace onnxruntime::webnn
