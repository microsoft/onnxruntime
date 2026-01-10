// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "core/optimizer/initializer.h"

#include "base_op_builder.h"
#include "builder_utils.h"

namespace onnxruntime {
namespace webnn {

class ReductionOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  // Allow axes potentially being empty inputs that are ignored during processing.
  ReductionOpBuilder() : BaseOpBuilder(/*allow empty inputs*/ true) {}
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

// Add operator related.
void ReductionOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() > 1) {
    model_builder.AddInitializerToSkip(input_defs[1]->Name());  // axes
    model_builder.AddInputToSkip(input_defs[1]->Name());        // axes
  }
}

// Add operator related.

Status ReductionOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                 const Node& node,
                                                 const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  const auto input_rank = input_shape.size();

  NodeAttrHelper helper(node);
  const auto keep_dims = helper.Get("keepdims", 1);

  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  options.set("keepDimensions", keep_dims == 1);

  std::vector<int64_t> axes_data;
  const auto opset = node.SinceVersion();
  const auto& op_type = node.OpType();
  if (opset >= 18 || (op_type == "ReduceSum" && opset >= 13)) {
    // 'axes' is an optional input.
    std::vector<int64_t> axes_shape;
    if (TensorExists(input_defs, 1)) {
      ORT_RETURN_IF_NOT(GetShape(*input_defs[1], axes_shape, logger), "Cannot get shape of input axes");
      if (axes_shape[0] != 0) {
        // Optional input axes is provided and we already ensure it is an initializer.
        // Use that initializer data.
        const auto& initializers(model_builder.GetInitializerTensors());
        const auto& axes_tensor = *initializers.at(input_defs[1]->Name());
        Initializer axes_initializer(axes_tensor);
        const auto axes_data_span = axes_initializer.DataAsSpan<int64_t>();
        axes_data = HandleNegativeAxes(axes_data_span, input_rank);
      }
    }
  } else {
    if (helper.HasAttr("axes")) {
      axes_data = GetResolvedAxes(helper, input_rank);
    }
  }

  // When axes is not provided or is empty, check the 'noop_with_empty_axes' attribute:
  // - If it is false, perform reduction over all dimensions.
  //   (In WebNN, this means the 'axes' option is not set.)
  // - If it is true, no reduction is applied, but other operations are still performed.
  //   (In WebNN, this requires setting 'axes' to an empty array.)
  if (!axes_data.empty() || helper.Get("noop_with_empty_axes", 0) == 1) {
    options.set("axes", emscripten::val::array(GetNarrowedIntFromInt64<uint32_t>(axes_data)));
  }

  const std::string_view webnn_op_type = GetWebNNOpType(op_type);
  ORT_RETURN_IF(webnn_op_type.empty(), "Cannot get WebNN op type");

  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>(
      std::string(webnn_op_type).c_str(), input, options);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.
bool ReductionOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer,
                                           const Node& node,
                                           const WebnnDeviceType /* device_type */,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  if (TensorExists(input_defs, 1)) {
    std::vector<int64_t> axes_shape;
    if (!GetShape(*input_defs[1], axes_shape, logger)) {
      LOGS(logger, VERBOSE) << "Cannot get shape of input axes";
      return false;
    }

    if (axes_shape.size() != 1) {
      LOGS(logger, VERBOSE) << "Input axes of " << node.OpType() << " must be 1D";
      return false;
    }

    const std::string axes_name = GetTensorName(input_defs, 1);
    // If the optional input 'axes' is provided and not empty, it must be an initializer.
    if (axes_shape[0] != 0 && !graph_viewer.GetConstantInitializer(axes_name)) {
      LOGS(logger, VERBOSE) << "Input axes of " << node.OpType() << " must be a constant";
      return false;
    }
  }

  return true;
}

void CreateReductionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "ReduceL1",
          "ReduceL2",
          "ReduceLogSum",
          "ReduceLogSumExp",
          "ReduceMax",
          "ReduceMean",
          "ReduceMin",
          "ReduceProd",
          "ReduceSum",
          "ReduceSumSquare",
      };

  op_registrations.builders.push_back(std::make_unique<ReductionOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
