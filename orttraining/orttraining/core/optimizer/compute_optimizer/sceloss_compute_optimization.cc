// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING
#include <onnx/defs/attr_proto_util.h>

#include "orttraining/core/optimizer/compute_optimizer/sceloss_compute_optimization.h"

#include "core/graph/graph_utils.h"
#include "core/framework/random_seed.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/shared_utils.h"
#include "core/optimizer/compute_optimizer/upstream_gather_actors.h"

namespace onnxruntime {

// Put utilities in anonymous namespace.
namespace {
NodeArg* InsertNodesForValidLabelIndices(Graph& graph, Node& node, NodeArg* label_input, NodeArg* reduce_index_input) {
  InlinedVector<NodeArg*> input_args{label_input, reduce_index_input};

  InlinedVector<NodeArg*> output_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("label_sub_result"),
                                                                node.MutableInputDefs()[1]->TypeAsProto())};

  Node& sub_node = graph.AddNode(graph.GenerateNodeName("labels_sub"), "Sub", "label sub padding idx", input_args,
                                 output_args, nullptr, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(sub_node), "Failed to set op schema for " + sub_node.Name());
  sub_node.SetExecutionProviderType(node.GetExecutionProviderType());

  auto non_zero_out_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("labels_filter_pad_result"),
                                                    node.MutableInputDefs()[1]->TypeAsProto());

  Node& non_zero_node = graph.AddNode(graph.GenerateNodeName("labels_filter_pad"), "NonZero",
                                      "labels filtering padding idx",
                                      {sub_node.MutableOutputDefs()[0]},
                                      {non_zero_out_arg}, nullptr, kOnnxDomain);

  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(non_zero_node),
              "Failed to set op schema for " + non_zero_node.Name());

  const std::string dim_name = MakeString("valid_label_count_", utils::GetRandomSeed());

  // 1D input NonZero generates output of shape (1,valid_token_count).
  ONNX_NAMESPACE::TensorShapeProto non_zero_output_shape;
  non_zero_output_shape.add_dim()->set_dim_value(1);
  non_zero_output_shape.add_dim()->set_dim_param(dim_name);
  non_zero_out_arg->SetShape(non_zero_output_shape);
  non_zero_node.SetExecutionProviderType(node.GetExecutionProviderType());

  InlinedVector<NodeArg*> squeeze_input_args;
  squeeze_input_args.push_back(non_zero_out_arg);

  bool opset_lower_than_13 = onnxruntime::optimizer::compute_optimizer::GetONNXOpSetVersion(graph) < 13;
  onnxruntime::NodeAttributes attributes;
  if (opset_lower_than_13) {
    attributes["axes"] = ONNX_NAMESPACE::MakeAttribute("axes", std::vector<int64_t>{0});
  } else {
    squeeze_input_args.push_back(onnxruntime::optimizer::compute_optimizer::CreateInitializerFromVector(
        graph, {1}, {0}, graph.GenerateNodeArgName("axes")));
  }

  auto squeeze_out_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("squeeze_adaptor"),
                                                   non_zero_out_arg->TypeAsProto());
  Node& squeeze_node = graph.AddNode(graph.GenerateNodeName("squeeze_adaptor"), "Squeeze", "nonzero_squeezer",
                                     squeeze_input_args, {squeeze_out_arg}, &attributes, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(squeeze_node),
              "Failed to set op schema for " + squeeze_node.Name());

  // After Squeeze, the shape becomes (valid_token_count).
  ONNX_NAMESPACE::TensorShapeProto squeeze_output_shape;
  squeeze_output_shape.add_dim()->set_dim_param(dim_name);
  squeeze_out_arg->SetShape(squeeze_output_shape);
  squeeze_node.SetExecutionProviderType(node.GetExecutionProviderType());

  return squeeze_out_arg;
}
}  // namespace

Status InsertGatherBeforeSceLoss::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                                            const logging::Logger& logger) const {
  LOG_DEBUG_INFO(logger, "Enter InsertGatherBeforeSceLoss");

  GraphViewer graph_viewer(graph);
  [[maybe_unused]] size_t handled_sce_node_count = 0;  // For summary
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr)
      // node was removed.
      continue;

    auto& node = *node_ptr;

    bool is_onnx_sce = graph_utils::IsSupportedOptypeVersionAndDomain(node, "SoftmaxCrossEntropyLoss", {12, 13});
    bool is_internal_sce = graph_utils::IsSupportedOptypeVersionAndDomain(node, "SoftmaxCrossEntropyLossInternal", {1},
                                                                          kMSDomain);

    if ((!is_onnx_sce && !is_internal_sce) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    // Check whether this SCE node is handled or not.
    const Node* labels_producer = graph.GetProducerNode(node.MutableInputDefs()[1]->Name());
    // Skip if already inserted a ShrunkenGather node.
    if (labels_producer && graph_utils::IsSupportedOptypeVersionAndDomain(
                               *labels_producer, "ShrunkenGather", {1}, kMSDomain)) {
      LOG_DEBUG_INFO(logger, "Skip node " + node.Name() + "(" + node.OpType() +
                                 ") due to labels input is already consumed by a ShrunkenGather node.");
      continue;
    }

    // Check shape requirements.
    auto logits_shape = node.MutableInputDefs()[0]->Shape();
    auto labels_shape = node.MutableInputDefs()[1]->Shape();
    if (logits_shape == nullptr || labels_shape == nullptr) {
      LOG_DEBUG_INFO(logger, "Skip node " + node.Name() + "(" + node.OpType() +
                                 ") due to undefined input shapes.");
      continue;
    }

    if (logits_shape->dim_size() != 2 || labels_shape->dim_size() != 1) {
      LOG_DEBUG_INFO(logger, "Skip node " + node.Name() + "(" + node.OpType() +
                                 ") due to unsupported input shape ranks.");
      continue;
    }

    // Check attribute and input requirements.
    std::string reduction = node.GetAttributes().at("reduction").s();
    if (reduction.compare("mean") == 0 || reduction.compare("sum") == 0) {
      // loss output is a scalar, don't need reset shape.
    } else {
      LOG_DEBUG_INFO(logger, "Skip node " + node.Name() + "(" + node.OpType() +
                                 ") due to loss [reduction=" + reduction + "].");
      continue;
    }

    NodeArg* ignore_index_node_arg = nullptr;
    if (is_internal_sce) {
      if (node.InputDefs().size() < 4 || !graph_utils::IsConstantInitializer(
                                             graph, node.InputDefs()[3]->Name(), /* check_outer_scope */ false)) {
        LOG_DEBUG_INFO(logger, "Skip node " + node.Name() + "(" + node.OpType() +
                                   ") due to target padding idx is non-constant initializer. Input count: " + std::to_string(node.InputDefs().size()));
        continue;
      }
      ignore_index_node_arg = node.MutableInputDefs()[3];
    } else {
      const auto ignore_index_attr = node.GetAttributes().find("ignore_index");
      if (ignore_index_attr != node.GetAttributes().end()) {
        int64_t ignore_index_value = (*ignore_index_attr).second.i();
        ignore_index_node_arg = onnxruntime::optimizer::compute_optimizer::CreateInitializerFromVector(
            graph, {}, {ignore_index_value}, graph.GenerateNodeArgName("ignore_index"));
      } else {
        LOG_DEBUG_INFO(logger, "Skip node " + node.Name() + "(" + node.OpType() +
                                   ") due to missing ignore_index attribute.");
        continue;
      }
    }

    std::vector<const Node*> sce_out1_consumers = graph.GetConsumerNodes(node.OutputDefs()[1]->Name());
    if (sce_out1_consumers.size() != 0 || graph.IsOutput(node.OutputDefs()[1])) {
      LOG_DEBUG_INFO(logger, "Skip node " + node.Name() + "(" + node.OpType() +
                                 ") due to log_prob output is graph output or consumed by other nodes.");
      continue;
    }

    // SoftmaxCrossEntropyLossInternal op definition guarantees that the the first dimension of both inputs must match,
    // we don't do the check explicitly here.

    LOG_DEBUG_INFO(logger, "Inserting Sub+NonZero nodes for filtering valid tokens");

    // It is possible a label input is used by multiple SoftmaxCrossEntropyLossInternal nodes, here we will create a
    // subgraph retrieving valid tokens for each SoftmaxCrossEntropyLossInternal node.
    // The duplication will be removed by CSE graph transformers.
    NodeArg* valid_labels_input_arg =
        InsertNodesForValidLabelIndices(graph, node, node.MutableInputDefs()[1], ignore_index_node_arg);

    // Insert the ShrunkenGather node on the two inputs.
    for (int i = 0; i < 2; ++i) {
      InlinedVector<NodeArg*> input_args;
      input_args.reserve(2);
      input_args.push_back(node.MutableInputDefs()[i]);
      input_args.push_back(valid_labels_input_arg);

      InlinedVector<NodeArg*> output_args;
      output_args.push_back(
          &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("label_filter_result"),
                                    node.MutableInputDefs()[i]->TypeAsProto()));

      /* new node input index to connect to node's input node*/
      int new_gather_input_index_to_connect = 0;
      /* new node output index to connect to node*/
      int new_gather_output_index_to_connect = 0;
      Node* new_gather_node = onnxruntime::optimizer::compute_optimizer::InsertIntermediateNodeOnDestInput(
          graph, node,
          i,
          new_gather_input_index_to_connect,
          new_gather_output_index_to_connect,
          graph.GenerateNodeName("LabelsFilter"),
          "ShrunkenGather",
          "ShrunkenGather node to filter invalid tokens.",
          input_args,
          output_args,
          {},
          kMSDomain,
          logger);

      new_gather_node->SetExecutionProviderType(node.GetExecutionProviderType());
      auto gather_out_arg = new_gather_node->MutableOutputDefs()[0];

      onnxruntime::optimizer::compute_optimizer::UpdateSliceOutputShape(
          *gather_out_arg, 0, valid_labels_input_arg->Shape()->dim(0));
    }

    modified = true;
    handled_sce_node_count += 1;
  }

  LOG_DEBUG_INFO(logger, "Exit InsertGatherBeforeSceLoss, handled " + std::to_string(handled_sce_node_count) +
                             " SCE nodes");

  return Status::OK();
}

}  // namespace onnxruntime

#endif
