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

class TriangularOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};


Status TriangularOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
    const auto& input_defs = node.InputDefs();
    emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
    emscripten::val output = emscripten::val::object();
    std::vector<int64_t> input_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
    const auto input_size = input_shape.size();
    NodeAttrHelper helper(node);


    if (input_defs.size() > 1) {
      // Optional input axes is provided, use axes initializer data.
      const auto& initializers(model_builder.GetInitializerTensors());
      const auto& diagonal = *initializers.at(input_defs[1]->Name());
    }

    int32_t upper = helper.Get("upper", 1);

    emscripten::val options = emscripten::val::object();
    options.set("upper", upper);
    options.set("diagonal", diagonal == 0);


    if (input_size != 2) {
      /* Implementation of tril/triu in numpy
      1. generate a triangle 0-1 mask (tri)
      2. mask the tensor (where)
      */

      // CHECK: how to define a all-0/1 matrix
      emscripten::val desc = emscripten::val::object();
      ORT_RETURN_IF_NOT(SetWebnnDataType(desc, ONNX_NAMESPACE::TensorProto_DataType_INT64), "Unsupported data type");
      emscripten::val dims = emscripten::val::array(std::vector<uint32_t>{input_shape[-2], input_shape[-1]});
      desc.set("dimensions", dims);
      std::vector<int64_t> ones(input_shape[-2]*input_shape[-1],1);
      std::vector<int64_t> zeros(input_shape[-2]*input_shape[-1],0);
      emscripten::val ones_buffer = emscripten::val::global("BigInt64Array").new_(emscripten::val::array(ones));
      emscripten::val zeros_buffer = emscripten::val::global("BigInt64Array").new_(emscripten::val::array(zeros));

      emscripten::val mask = model_builder.GetBuilder().call<emscripten::val>("constant", desc, ones_buffer);
      emscripten::val others = model_builder.GetBuilder().call<emscripten::val>("constant", desc, zeros_buffer);
      mask = model_builder.GetBuilder().call<emscripten::val>("triangular", mask, options);
      output = model_builder.GetBuilder().call<emscripten::val>("where", mask, input, others);
    }
    else {
      output = model_builder.GetBuilder().call<emscripten::val>("triangular", input, options);
    }

    model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
    return Status::OK();
}

// Operator support related.
bool TriangularOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */,
                                         const Node& node,
                                         const WebnnDeviceType /* device_type */,
                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;
  const auto input_size = input_shape.size();
  if (input_size < 2) {
    LOGS(logger, VERBOSE) << "Triangular only support input size >= 2d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

void CreateTriangularOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<TriangularOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
