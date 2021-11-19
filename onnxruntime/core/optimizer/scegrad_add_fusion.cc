// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/scegrad_add_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnxruntime;

namespace onnxruntime {

Status SCEGradWithAdd::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    // 1. find matched subgraph, and all nodes' output is not graph output, and shapes of sum's 2inputs should be same.
    // the subgraph is : SoftmaxCrossEntropyLossInternalGrad > Reshape > Sum(only has two inputs)
    auto scegrad_ptr = graph.GetNode(node_index);
    if ((scegrad_ptr == nullptr) or
        (scegrad_ptr->OpType() != "SoftmaxCrossEntropyLossInternalGrad") or (scegrad_ptr->GetInputEdgesCount() == 6) or (scegrad_ptr->GetOutputEdgesCount() != 1))
      continue;
    auto reshape_ptr = graph.GetNode(scegrad_ptr->OutputNodesBegin()->Index());
    if ((reshape_ptr->OpType() != "Reshape") or (reshape_ptr->GetOutputEdgesCount() != 1))
      continue;
    auto sum_ptr = graph.GetNode(reshape_ptr->OutputNodesBegin()->Index());
    if ((sum_ptr->OpType() != "Sum") or (sum_ptr->GetInputEdgesCount() != 2))
      continue;
    if ((graph.NodeProducesGraphOutput(*scegrad_ptr) or graph.NodeProducesGraphOutput(*reshape_ptr)))
      continue;

    auto sum_input_defs = sum_ptr->InputDefs();
    if (sum_input_defs[0]->Shape() != sum_input_defs[0]->Shape()){
        continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(*scegrad_ptr, modified, graph_level, logger));

    // found added_data from sum's input. sum should have only two input
    NodeArg* added_data_def = nullptr;
    if(reshape_ptr->OutputDefs()[0] == sum_input_defs[0])
      added_data_def = const_cast<NodeArg*> (sum_input_defs[1]);
    else
      added_data_def = const_cast<NodeArg*> (sum_input_defs[0]);

    graph_utils::RemoveNodeOutputEdges(graph, *scegrad_ptr);
    std::vector<NodeArg*> new_scegrad_node_inputs = {scegrad_ptr->MutableInputDefs()};
    new_scegrad_node_inputs.push_back(added_data_def);
    std::vector<NodeArg*> new_scegrad_node_outputs = {&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("scegrad_fused_output"), scegrad_ptr->MutableOutputDefs()[0]->TypeAsProto())};
    Node& new_scegrad_node = graph.AddNode(graph.GenerateNodeName(scegrad_ptr->Name() + "fused_with_add"),
                                               std::string("SoftmaxCrossEntropyLossInternalGrad"),
                                               std::string("fused SCEGrad with Add"),
                                               new_scegrad_node_inputs,
                                               new_scegrad_node_outputs,
                                               &scegrad_ptr->GetAttributes(),
                                               kMSDomain);

    new_scegrad_node.SetExecutionProviderType(scegrad_ptr->GetExecutionProviderType());
    reshape_ptr->MutableInputDefs()[0] = new_scegrad_node.MutableOutputDefs()[0];
    graph_utils::FinalizeNodeFusion(graph, *reshape_ptr, *sum_ptr);

    graph.RemoveNode(scegrad_ptr->Index());
    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
