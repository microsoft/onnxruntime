// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE
#include <onnx/defs/attr_proto_util.h>

#include "core/common/safeint.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/insert_gather_for_sceloss.h"
#include "core/optimizer/compute_optimizer/upstream_gather_actors.h"
#include "core/optimizer/compute_optimizer/upstream_reshape_actors.h"

namespace onnxruntime::optimizer::compute_optimizer {

Status InsertGatherBeforeSceLoss::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                                            const logging::Logger& logger) const {
  LOG_DEBUG_INFO(logger, "Enter InsertGatherBeforeSceLoss");

  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();
  for (auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr)
      // node was removed.
      continue;

    auto& node = *node_ptr;

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "SoftmaxCrossEntropyLossInternal", {1}, kMSDomain) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    auto logits_shape = node.MutableInputDefs()[0]->Shape();
    auto labels_shape = node.MutableInputDefs()[1]->Shape();
    if (logits_shape == nullptr || labels_shape == nullptr) {
      LOG_DEBUG_INFO(logger, "Skip SoftmaxCrossEntropyLossInternal node " + node.Name() +
                                 " due to undefined input shapes.");
      continue;
    }

    if (logits_shape->dim_size() != 2 || labels_shape->dim_size() != 1) {
      LOG_DEBUG_INFO(logger, "Skip SoftmaxCrossEntropyLossInternal node " + node.Name() +
                                 " due to unsupported input shape ranks.");
      continue;
    }

    std::string reduction = node.GetAttributes().at("reduction").s();
    bool need_reset_shape = false;
    if (reduction.compare("mean") == 0) {
      // loss output is a scalar, don't need reset shape.
    } else if (reduction.compare("none") == 0) {
      need_reset_shape = true;
    } else {
      LOG_DEBUG_INFO(logger, "Skip SoftmaxCrossEntropyLossInternal node " + node.Name() +
                                 " due to unsupported reduction type.");
      continue;
    }

    if (need_reset_shape) {
      std::vector<const Node*> sce_out0_consumers = graph.GetConsumerNodes(node.OutputDefs()[0]->Name());
      if (sce_out0_consumers.size() != 0 || graph.IsOutput(node.OutputDefs()[0])) {
        LOG_DEBUG_INFO(logger, "Skip SoftmaxCrossEntropyLossInternal node " + node.Name() +
                                   " due to loss [reduction=none] output is graph output or consumed by other nodes.");
        continue;
      }
    }

    std::vector<const Node*> sce_out1_consumers = graph.GetConsumerNodes(node.OutputDefs()[1]->Name());
    if (sce_out1_consumers.size() != 0 || graph.IsOutput(node.OutputDefs()[1])) {
      LOG_DEBUG_INFO(logger, "Skip SoftmaxCrossEntropyLossInternal node " + node.Name() +
                                 " due to logit output is graph output or consumed by other nodes.");
      continue;
    }

    const Node* labels_producer = graph.GetProducerNode(node.MutableInputDefs()[1]->Name());
    // Skip if already inserted a Gather node.
    if (labels_producer && graph_utils::IsSupportedOptypeVersionAndDomain(
                               *labels_producer, "Gather", {1, 11, 13}, kOnnxDomain)) {
      LOG_DEBUG_INFO(logger, "Skip SoftmaxCrossEntropyLossInternal node " + node.Name() +
                                 " due to labels input is already consumed by a Gather node.");
      continue;
    }

    if (node.InputDefs().size() >= 4 && !graph_utils::IsConstantInitializer(
                                            graph, node.InputDefs()[3]->Name(), /* check_outer_scope */ false)) {
      LOG_DEBUG_INFO(logger, "Skip SoftmaxCrossEntropyLossInternal node " + node.Name() +
                                 " due to target padding idx is non-constant initializer.");
      continue;
    }

    // SoftmaxCrossEntropyLossInternal op definition guarantees that the the first dimension of both inputs must match,
    // we don't do the check explicitly here.

    LOG_DEBUG_INFO(logger, "Inserting Sub+NonZero nodes for filtering valid tokens");
    InlinedVector<NodeArg*> input_args;
    input_args.reserve(node.InputDefs().size());
    input_args.push_back(node.MutableInputDefs()[1]);
    input_args.push_back(node.MutableInputDefs()[3]);
    InlinedVector<NodeArg*> output_args;
    output_args.push_back(
        &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("label_sub_result"), node.MutableInputDefs()[1]->TypeAsProto()));

    Node& sub_node = graph.AddNode(graph.GenerateNodeName("labels_sub"), "Sub", "label sub padding idx", input_args,
                                   output_args, nullptr, kOnnxDomain);
    ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(sub_node), "Failed to set op schema for " + sub_node.Name());
    sub_node.SetExecutionProviderType(node.GetExecutionProviderType());

    auto non_zero_out_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("labels_filter_pad_result"), node.MutableInputDefs()[1]->TypeAsProto());
    Node& non_zero_node = graph.AddNode(graph.GenerateNodeName("labels_filter_pad"), "NonZero", "labels filtering padding idx",
                                        {sub_node.MutableOutputDefs()[0]},
                                        {non_zero_out_arg}, nullptr, kOnnxDomain);
    ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(non_zero_node), "Failed to set op schema for " + non_zero_node.Name());

    // 1D input NonZero generate output of shape (1,valid_token_count).
    ONNX_NAMESPACE::TensorShapeProto non_zero_output_shape;
    non_zero_output_shape.add_dim()->set_dim_value(1);
    non_zero_output_shape.add_dim()->set_dim_param("valid_token_count");
    non_zero_out_arg->SetShape(non_zero_output_shape);
    non_zero_node.SetExecutionProviderType(node.GetExecutionProviderType());

    bool opset_lower_than_13 = GetONNXOpSetVersion(graph) < 13;
    onnxruntime::NodeAttributes attributes;
    if (opset_lower_than_13) {
      attributes["axes"] = ONNX_NAMESPACE::MakeAttribute("axes", std::vector<int64_t>{0});
    }
    InlinedVector<NodeArg*> squeeze_input_args;
    squeeze_input_args.push_back(non_zero_out_arg);
    if (opset_lower_than_13) {
      squeeze_input_args.push_back(CreateUnsqueezeAxesInitializer(graph, {0}));
    }

    auto squeeze_out_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("squeeze_adaptor"),
                                                     non_zero_out_arg->TypeAsProto());
    Node& squeeze_node = graph.AddNode(graph.GenerateNodeName("squeeze_adaptor"), "Squeeze", "nonzero_squeezer",
                                       squeeze_input_args, {squeeze_out_arg}, &attributes, kOnnxDomain);
    ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(squeeze_node),
                "Failed to set op schema for " + squeeze_node.Name());

    // 1D input NonZero generate output of shape (1,valid_token_count).
    ONNX_NAMESPACE::TensorShapeProto squeeze_output_shape;
    squeeze_output_shape.add_dim()->set_dim_param("valid_token_count");
    squeeze_out_arg->SetShape(squeeze_output_shape);
    squeeze_node.SetExecutionProviderType(node.GetExecutionProviderType());

    // Insert the Gather node on the two inputs.
    for (int i = 0; i < 2; ++i) {
      LOG_DEBUG_INFO(logger, "Inserting Gather node on SoftmaxCrossEntropyLossInternal node's " +
                                 std::to_string(i) + "-th input");
      InlinedVector<NodeArg*> input_args;
      input_args.reserve(2);
      input_args.push_back(node.MutableInputDefs()[i]);
      input_args.push_back(squeeze_out_arg);

      InlinedVector<NodeArg*> output_args;
      output_args.push_back(
          &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("label_filter_result"),
                                    node.MutableInputDefs()[i]->TypeAsProto()));

      /* new node input index to connect to node's input node*/
      int new_gather_input_index_to_connect = 0;
      /* new node output index to connect to node*/
      int new_gather_output_index_to_connect = 0;
      Node* new_gather_node = InsertIntermediateNodeOnDestInput(
          graph, node,
          i,
          new_gather_input_index_to_connect,
          new_gather_output_index_to_connect,
          graph.GenerateNodeName("LabelsFilter"),
          "Gather",
          "Gather node to filter invalid tokens.",
          input_args,
          output_args,
          {},
          kOnnxDomain,
          logger);

      new_gather_node->SetExecutionProviderType(node.GetExecutionProviderType());
      auto gather_out_arg = new_gather_node->MutableOutputDefs()[0];
      UpdateSliceOutputShape(*gather_out_arg,
                             0 - gather_out_arg->Shape()->dim_size() /*reversed axis*/,
                             squeeze_output_shape.dim(0));

      std::cout << "@@@@@@@@@@@@@@@@@@@@ non_zero_node" << std::to_string(non_zero_node.MutableOutputDefs()[0]->Shape() == nullptr) << std::endl;
    }

    if (need_reset_shape) {
      LOG_DEBUG_INFO(logger, "Updating loss output shape for SoftmaxCrossEntropyLossInternal");
      auto sce_out0_arg = node.MutableOutputDefs()[0];
      UpdateSliceOutputShape(*sce_out0_arg, 0 - sce_out0_arg->Shape()->dim_size() /*reversed axis*/,
                             squeeze_output_shape.dim(0));
    }

    modified = true;
  }

  LOG_DEBUG_INFO(logger, "Exit InsertGatherBeforeSceLoss");
  return Status::OK();
}

}  // namespace onnxruntime::optimizer::compute_optimizer

#endif
