// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// gather_replacement.cc

#include "orttraining/core/optimizer/gather_replacement.h"

#include "core/common/logging/logging.h"
#include "core/optimizer/rewrite_rule.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

namespace {

const char* NumSegmentsNodeArgName = "num_segments";
const char* SegmentOffsetssNodeArgName = "segment_offsets";
const char* LastSegmentPartialSegmentCountNodeArgName = "last_segment_partial_segment_count";
const char* LastSegmentPartialSegmentOffsetNodeArgName = "last_segment_partial_segment_offset";
const char* PerSegmentPartialSegmentCountsNodeArgName = "per_segment_partial_segment_counts";
const char* PerSegmentPartialSegmentOffsetsNodeArgName = "per_segment_partial_segment_offsets";
const char* DXIndeicesSortedNorArgName = "dX_indices_sorted";
const char* DYIndeicesSortedNorArgName = "dY_indices_sorted";
const char* GatherInternalName("GatherInternal");


std::vector<NodeArg*> create_gather_internal_output_node_args(const std::vector<NodeArg*>& gather_inputs,
                                                              const std::vector<NodeArg*>& gather_outputs,
                                                              Graph& graph)
{
    // Prepare all the gather internal outputs
    ONNX_NAMESPACE::TypeProto num_segments_type_proto,
                              segment_offsets_type_proto,
                              last_segment_partial_segment_offset_type_proto,
                              last_segment_partial_segment_count_type_proto,
                              per_segment_partial_segment_counts_type_proto,
                              per_segment_partial_segment_offsets_type_proto,
                              dX_indices_sorted_type_proto,
                              dY_indices_sorted_type_proto;

    num_segments_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
    num_segments_type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    segment_offsets_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);

    last_segment_partial_segment_count_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
    last_segment_partial_segment_count_type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    last_segment_partial_segment_offset_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
    last_segment_partial_segment_offset_type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    per_segment_partial_segment_counts_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);

    per_segment_partial_segment_offsets_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);

    dX_indices_sorted_type_proto.mutable_tensor_type()->set_elem_type(gather_inputs[1]->TypeAsProto()->tensor_type().elem_type());

    dY_indices_sorted_type_proto.mutable_tensor_type()->set_elem_type(gather_inputs[1]->TypeAsProto()->tensor_type().elem_type());


    std::vector<NodeArg*> gather_internal_outputs(gather_outputs);
    gather_internal_outputs.insert(gather_internal_outputs.end(), {
        &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(NumSegmentsNodeArgName), &num_segments_type_proto),
        &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(SegmentOffsetssNodeArgName), &segment_offsets_type_proto),
        &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(LastSegmentPartialSegmentCountNodeArgName), &last_segment_partial_segment_count_type_proto),
        &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(LastSegmentPartialSegmentOffsetNodeArgName), &last_segment_partial_segment_offset_type_proto),
        &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(PerSegmentPartialSegmentCountsNodeArgName), &per_segment_partial_segment_counts_type_proto),
        &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(PerSegmentPartialSegmentOffsetsNodeArgName), &per_segment_partial_segment_offsets_type_proto),
        &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(DXIndeicesSortedNorArgName), &dX_indices_sorted_type_proto),
        &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(DYIndeicesSortedNorArgName), &dY_indices_sorted_type_proto)});

    return gather_internal_outputs;
}


} // unnamed namespace

bool GatherReplacement::SatisfyCondition(const Graph&, const Node&, const logging::Logger&) const
{
    return true;
}

Status GatherReplacement::Apply(Graph& graph, Node& gather_node, RewriteRuleEffect& rule_effect, const logging::Logger&) const
{
    // Prepare the inputs and outputs for the new GatherInternal node to be added.
    const auto& gather_inputs = gather_node.MutableInputDefs();
    const auto& gather_outputs = gather_node.MutableOutputDefs();

    const auto gather_internal_outputs =
        create_gather_internal_output_node_args(gather_inputs, gather_outputs, graph);

    // Add the node based on all gather_node inputs and outputs collected.
    Node& gather_internal_node = graph.AddNode(graph.GenerateNodeName(GatherInternalName),
                                               GatherInternalName,
                                               "GatherInternal that was added replacing the Gather node for extra outputs",
                                               gather_inputs,
                                               gather_internal_outputs,
                                               &gather_node.GetAttributes(),
                                               kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    gather_internal_node.SetExecutionProviderType(gather_node.GetExecutionProviderType());
 
    // Remove the gather_node from the graph.
    const auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(gather_node);
    graph_utils::GraphEdge::RemoveGraphEdges(graph, output_edges);
    graph.RemoveNode(gather_node.Index());
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;

    return Status::OK();
}

}  // namespace onnxruntime
