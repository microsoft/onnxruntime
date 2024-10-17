// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class ArgMaxOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

 public:
  bool SupportsMLProgram() const override { return true; }
};

Status ArgMaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const Node& node,
                                              const logging::Logger& /* logger */) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);
  const auto& graph_viewer = model_builder.GetGraphViewer();

  NodeAttrHelper helper(node);
  const int64_t axis = helper.Get("axis", 0);
  const int64_t keepdims = helper.Get("keepdims", 1);
  const bool removedim = keepdims != 1;

#if defined(COREML_ENABLE_MLPROGRAM)
  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;
    // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#module-coremltools.converters.mil.mil.ops.defs.iOS15.reduction

    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "reduce_argmax");
    AddOperationInput(*op, "x", node.InputDefs()[0]->Name());
    AddOperationInput(*op, "axis", model_builder.AddScalarConstant(op->type(), "axis", axis));
    AddOperationInput(*op, "keep_dims", model_builder.AddScalarConstant(op->type(), "keep_dims", bool(keepdims)));
    if (node.GetOutputEdgesCount() == 1) {
      auto it = node.OutputEdgesBegin();
      const auto* next_node_in_partition = &(it->GetNode());
      // If Argmax's successive node is a Cast from int64 to int32 output, we fuse it
      if (next_node_in_partition != nullptr && next_node_in_partition->OpType() == "Cast") {
        // Skip the cast's input/argmax's output
        AddOperationOutput(*op, *next_node_in_partition->OutputDefs()[0]);
        model_builder.AddOperation(std::move(op));
        return Status::OK();
      }
    }
    // shall we add cast here?
    // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_unary.cast
    AddOperationOutput(*op, *node.OutputDefs()[0]);
    model_builder.AddOperation(std::move(op));
  } else
#endif  // (COREML_ENABLE_MLPROGRAM)
  {
    auto* coreml_argmax = layer->mutable_argmax();
    coreml_argmax->set_axis(axis);
    coreml_argmax->set_removedim(removedim);

    // There are two cases here:
    // 1. Special Case (ArgMax-Cast(from int64 to int32)), we fuse the Argmax's output/Cast's input
    // (We still have this special case here because CoreML model does not have Cast)
    // 2. Otherwise, we add Argmax layer normally
    if (node.GetOutputEdgesCount() == 1) {
      auto it = node.OutputEdgesBegin();
      const auto* next_node_in_partition = graph_viewer.GetNode(it->GetNode().Index());
      // If Argmax's successive node is a Cast from int64 to int32 output
      // The 'cast to' type is checked when determining operator support (see CastOpBuilder::IsOpSupportedImpl())
      //   so we omit the check here
      if (next_node_in_partition != nullptr && next_node_in_partition->OpType() == "Cast") {
        // Skip the cast's input/argmax's output
        *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
        *layer->mutable_output()->Add() = next_node_in_partition->OutputDefs()[0]->Name();
        model_builder.AddLayer(std::move(layer));
        return Status::OK();
      }
    }

    *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

    model_builder.AddLayer(std::move(layer));
  }
  return Status::OK();
}

bool ArgMaxOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                        const logging::Logger& logger) const {
  // Attribute `select_last_index` of ArgMax op is not supported
  NodeAttrHelper helper(node);
  const auto select_last_index = helper.Get("select_last_index", 0);
  if (select_last_index != 0) {
    LOGS(logger, VERBOSE) << "select_last_index for ArgMax is not supported";
    return false;
  }

  // If there are multiple downstream nodes and cast (toint32) is one of them
  // not supported, exit here
  // Otherwise, for general multiple downstream nodes, supported
  if (node.GetOutputEdgesCount() > 1) {
    for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
      const auto& op_type = it->GetNode().OpType();
      if (op_type == "Cast") {
        // Check if the output type of cast node is int32
        NodeAttrHelper output_helper(it->GetNode());
        const auto cast_to_type = output_helper.Get("to", ONNX_NAMESPACE::TensorProto::UNDEFINED);
        if (cast_to_type == ONNX_NAMESPACE::TensorProto::INT32) {
          LOGS(logger, VERBOSE) << "Argmax has both cast and other downstream nodes.";
          return false;
        }
      }
    }
  }

#if defined(COREML_ENABLE_MLPROGRAM)
  if (input_params.create_mlprogram) {
    if (node.GetOutputEdgesCount() == 1) {
      auto it = node.OutputEdgesBegin();
      const auto& op_type = it->GetNode().OpType();
      if (op_type == "Cast") {
        // Check if the output type of cast node is int32
        NodeAttrHelper output_helper(it->GetNode());
        const auto cast_to_type = output_helper.Get("to", ONNX_NAMESPACE::TensorProto::UNDEFINED);
        if (cast_to_type == ONNX_NAMESPACE::TensorProto::INT32) {
          return true;
        } else {
          return false;
        }
      }
    } else {
      return false;
    }
  }
#endif

  return true;
}

void CreateArgMaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ArgMaxOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
