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

Status InsertGatherBeforeSceLoss::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                                            const logging::Logger& logger) const {
  LOG_DEBUG_INFO(logger, "Enter InsertGatherBeforeSceLoss");

  if (sparse_label_input_names_.size() == 0) {
    LOG_DEBUG_INFO(logger, "Exit InsertGatherBeforeSceLoss, no sparse label input names.");
    return Status::OK();
  }

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

    const NodeArg* label_input_arg = node.InputDefs()[1];

    // Check whether this SCE node is handled or not.
    const Node* labels_producer = graph.GetProducerNode(label_input_arg->Name());
    // Skip if already inserted a ShrunkenGather node.
    if (labels_producer && graph_utils::IsSupportedOptypeVersionAndDomain(
                               *labels_producer, "ShrunkenGather", {1}, kMSDomain)) {
      LOG_DEBUG_INFO(logger, "Skip node " + node.Name() + "(" + node.OpType() +
                                 ") due to labels input is already consumed by a ShrunkenGather node.");
      continue;
    }

    // Label input can be a graph input or from a Reshape node taking a graph input as its data input.
    if (labels_producer && graph_utils::IsSupportedOptypeVersionAndDomain(
                               *labels_producer, "Reshape", {1, 5, 13, 14}, kOnnxDomain)) {
      label_input_arg = labels_producer->InputDefs()[0];
    }
    // Then check if the label input is graph input and in the sparse label input list.
    if (!graph.IsInputsIncludingInitializers(label_input_arg) ||
        std::find(sparse_label_input_names_.begin(), sparse_label_input_names_.end(),
                  label_input_arg->Name()) == sparse_label_input_names_.end()) {
      LOG_DEBUG_INFO(logger, "Skip node " + node.Name() + "(" + node.OpType() +
                                 ") due to labels input is not a graph input or not in the sparse label input list.");
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
                                   ") due to target padding idx is non-constant initializer. Input count: " +
                                   std::to_string(node.InputDefs().size()));
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

    // SoftmaxCrossEntropyLossInternal op definition guarantees that the first dimension of both inputs must match,
    // we don't do the check explicitly here.

    LOG_DEBUG_INFO(logger, "Inserting Sub+NonZero nodes for filtering valid tokens");

    // It is possible a label input is used by multiple SoftmaxCrossEntropyLossInternal nodes, here we will create a
    // subgraph retrieving valid tokens for each SoftmaxCrossEntropyLossInternal node.
    // The duplication will be removed by CSE graph transformers.
    NodeArg* valid_labels_input_arg =
        onnxruntime::optimizer::compute_optimizer::InsertNodesForValidIndices(graph, node.MutableInputDefs()[1], ignore_index_node_arg, node.GetExecutionProviderType());

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

  LOGS(logger, INFO) << "Total handled SCE node count:  " << handled_sce_node_count;

  return Status::OK();
}

}  // namespace onnxruntime

#endif
