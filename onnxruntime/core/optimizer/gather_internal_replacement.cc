// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// gather_internal_replacement.cc

#include "core/optimizer/gather_internal_replacement.h"

#include "core/common/logging/logging.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

namespace {

const char* GatherGradOpType("GatherGrad");
const char* GatherName("Gather");

}

bool GatherInternalReplacement::SatisfyCondition(const Graph&, const Node& gather_internal_node, const logging::Logger&) const
{
    // If the gather_internal_node is not connected to a GatherGrad node, then the extra
    // precomputed GatherInternal outputs are not needed. And the node can be replaced
    // by a simple Gather node.
    for (auto output_node = gather_internal_node.OutputNodesBegin();
         output_node != gather_internal_node.OutputNodesEnd();
         ++output_node)
    {
        if (output_node->OpType() == GatherGradOpType)
        {
            // GatherGrad node detected, return false.
            return false;
        }
    }

    // gather_internal_node is not connected to the GatherGrad node.
    return true;
}

Status GatherInternalReplacement::Apply(Graph& graph, Node& gather_internal_node, RewriteRuleEffect& rule_effect, const logging::Logger&) const
{
    // Prepare the inputs and outputs for the new Gather node to be added.
    const auto& gather_internal_inputs = gather_internal_node.MutableInputDefs();
    const auto& gather_internal_outputs = gather_internal_node.MutableOutputDefs();
    const std::vector<NodeArg*> gather_outputs(gather_internal_outputs.cbegin(),
                                               gather_internal_outputs.cbegin()+1);

    // Add the node based on all gather_internal_node inputs and only a single output.
    Node& gather_node = graph.AddNode(graph.GenerateNodeName(GatherName),
                                      GatherName,
                                      "Gather that was re-added after replacing the GatherInternal node.",
                                      gather_internal_inputs,
                                      gather_outputs,
                                      &gather_internal_node.GetAttributes(),
                                      kOnnxDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    gather_node.SetExecutionProviderType(gather_internal_node.GetExecutionProviderType());

    // Remove the gather_internal_node from the graph.
    const auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(gather_internal_node);
    graph_utils::GraphEdge::RemoveGraphEdges(graph, output_edges);
    graph.RemoveNode(gather_internal_node.Index());
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;

    return Status::OK();
}

}  // namespace onnxruntime
