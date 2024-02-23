// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class ConcatOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
};

// Add operator related.

Status ConcatOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const Node& node,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ConcatOpBuilder::AddToModelBuilderImpl, cannot get input shape");
  }
  auto rank = input_shape.size();
  NodeAttrHelper helper(node);
  uint32_t axis = static_cast<uint32_t>(HandleNegativeAxis(helper.Get("axis", 1), rank));

  const size_t num_inputs = input_defs.size();
  std::vector<emscripten::val> inputs;
  for (const auto* input : input_defs) {
    LOGS(logger, VERBOSE) << "input name " << input->Name();
    inputs.push_back(model_builder.GetOperand(input->Name()));
  }

  emscripten::val output = emscripten::val::undefined();
  if (num_inputs <= 4 || model_builder.GetPreferredLayout() == DataLayout::NCHW) {
    output = model_builder.GetBuilder().call<emscripten::val>("concat", emscripten::val::array(inputs), axis);
  } else {
    // WebNN XNNPack backend only supports the concat with inputs number <= 4,
    // decomposing the Concat with inputs number > 4 into multiple WebNN concat ops.
    size_t remaining_inputs = num_inputs;
    size_t max_inputs = 4;
    while (remaining_inputs > 0) {
      std::vector<emscripten::val> chunk_inputs;

      // Push the last concated output to the next chunk_inputs.
      if (output != emscripten::val::undefined()) {
        chunk_inputs.push_back(output);
        max_inputs = 3;
      }

      size_t chunk_size = std::min(remaining_inputs, max_inputs);

      for (size_t i = 0; i < chunk_size; i++) {
        chunk_inputs.push_back(inputs[num_inputs - remaining_inputs + i]);
      }

      output = model_builder.GetBuilder().call<emscripten::val>("concat", emscripten::val::array(chunk_inputs), axis);
      remaining_inputs -= chunk_size;
    }
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

void CreateConcatOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ConcatOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
