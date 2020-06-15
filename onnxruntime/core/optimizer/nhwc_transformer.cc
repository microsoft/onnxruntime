// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_ACL) || defined(USE_ARMNN)

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nhwc_transformer.h"

#include "core/providers/acl/acl_common.h"

// ACL
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

// NEON
#include "arm_compute/runtime/NEON/functions/NEPermute.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

enum DataLayout {
  NchwLayout,
  NhwcLayout,
  InheritLayout
};

class NhwcTransformerImpl {
 public:
  NhwcTransformerImpl(Graph& graph, std::string provider, const logging::Logger& _logger) noexcept :
    graph_(graph), provider_(provider), logger(_logger) {};
  void Transform(Node& node);
  void Finalize(bool& modified);

 private:
  NodeIndex InsertPermuteParentNode(Node& node, const Node::EdgeEnd* edge, bool bNHWC);
  NodeIndex InsertPermuteChildNode(Node& node, bool bNHWC);
  NodeIndex ReplaceNode(Node& node);
  void PermuteWeights(NodeArg* src, NodeArg** dst, const std::string&);

  DataLayout RequiredLayout(const Node& node);
  bool SuportsReplacementNHWC(const Node& node);
  bool RequiresWeightsPermutation(const Node& node);

  std::map<NodeIndex, bool> nodes_layout;
  Graph& graph_;
  std::string provider_;
  const logging::Logger& logger;
};

NodeIndex NhwcTransformerImpl::InsertPermuteParentNode(Node& node, const Node::EdgeEnd* edge, bool bNHWC) {

  const Node* parent_node = NULL;
  int srcIdx = 0, dstIdx = 0;
  if (edge) {
    parent_node = &edge->GetNode();
    srcIdx = edge->GetSrcArgIndex();
    dstIdx = edge->GetDstArgIndex();
  }

  LOGS(logger, VERBOSE) << "NHWC Insert: "
      << (parent_node ? parent_node->OpType() : "*")
      << ":" << srcIdx
      << (bNHWC ? " [ReorderInput:0] " : " [ReorderOutput:0] ")
      << node.OpType()
      << ":" << dstIdx;

  auto& input_defs = node.MutableInputDefs();
  std::string new_input_def_name = graph_.GenerateNodeArgName("input");
  auto* new_input_arg = &graph_.GetOrCreateNodeArg(new_input_def_name, input_defs[0]->TypeAsProto());

  Node& permute_node = graph_.AddNode(graph_.GenerateNodeName(bNHWC ? "PermuteNHWC" : "PermuteNCHW"),
                                     bNHWC ? "ReorderInput" : "ReorderOutput",
                                     bNHWC ? "ReorderIntput" : "ReorderOutput",
                                     {input_defs[dstIdx]},
                                     {new_input_arg},
                                     nullptr,
                                     kMSNhwcDomain);
  permute_node.SetExecutionProviderType(provider_);

  if (parent_node) {
    graph_.RemoveEdge(parent_node->Index(), node.Index(), srcIdx, dstIdx);
  }
  input_defs[dstIdx] = new_input_arg;

  graph_.AddEdge(permute_node.Index(), node.Index(), 0, dstIdx);
  if (parent_node) {
    graph_.AddEdge(parent_node->Index(), permute_node.Index(), srcIdx, 0);
  }

  return permute_node.Index();
}

NodeIndex NhwcTransformerImpl::InsertPermuteChildNode(Node& node, bool bNHWC) {

  LOGS(logger, VERBOSE) << "NHWC Insert: " << node.OpType()
    << (bNHWC ? " [ReorderInput] " : " [ReorderOutput] ") << "*:0";

  auto& output_defs = node.MutableOutputDefs();
  std::string new_output_def_name = graph_.GenerateNodeArgName("output");
  auto* new_output_arg = &graph_.GetOrCreateNodeArg(new_output_def_name, output_defs[0]->TypeAsProto());

  Node& permute_node = graph_.AddNode(graph_.GenerateNodeName(bNHWC ? "PermuteNCHW" : "PermuteNHWC"),
                                     bNHWC ? "ReorderIntput" : "ReorderOutput",
                                     bNHWC ? "ReorderIntput" : "ReorderOutput",
                                     {new_output_arg},
                                     {output_defs[0]},
                                     nullptr,
                                     kMSNhwcDomain);
  permute_node.SetExecutionProviderType(provider_);
  output_defs[0] = new_output_arg;
  graph_.AddEdge(node.Index(), permute_node.Index(), 0, 0);

  return permute_node.Index();
}

bool NhwcTransformerImpl::RequiresWeightsPermutation(const Node& node) {
   return ((node.GetExecutionProviderType() == kAclExecutionProvider ||
            node.GetExecutionProviderType() == kArmNNExecutionProvider) &&
           (node.OpType() == "Conv" ||
            node.OpType() == "FusedConv"));
}

NodeIndex NhwcTransformerImpl::ReplaceNode(Node& node) {

  auto& input_defs = node.MutableInputDefs();
  auto& output_defs = node.MutableOutputDefs();

  LOGS(logger, VERBOSE) << "NHWC >> Replace " << node.OpType();

  Node& newNode = graph_.AddNode(graph_.GenerateNodeName(node.Name()),
                                     node.OpType(),
                                     node.OpType(),
                                     input_defs,
                                     output_defs,
                                     &node.GetAttributes(),
                                     kMSNhwcDomain);
  newNode.SetExecutionProviderType(node.GetExecutionProviderType());

  // Permute weights
  if (RequiresWeightsPermutation(node))
    PermuteWeights(input_defs[1], &newNode.MutableInputDefs()[1], node.GetExecutionProviderType());

  NodeIndex oldIndex = node.Index();
  const std::vector<std::reference_wrapper<Node>> replacedNode({node});
  graph_utils::FinalizeNodeFusion(graph_, replacedNode, newNode);
  nodes_layout.erase(oldIndex);

  return newNode.Index();
}

void NhwcTransformerImpl::PermuteWeights(NodeArg *input_def, NodeArg** nhwc_conv_W_arg, __attribute__ ((unused)) const std::string& execution_provider) {
  // Require that the weights tensor be static.
  const ONNX_NAMESPACE::TensorProto* conv_W_tensor_proto = nullptr;
  if (!graph_.GetInitializedTensor(input_def->Name(), conv_W_tensor_proto) ||
      (conv_W_tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) ||
      (conv_W_tensor_proto->dims_size() != 4) ||
      (execution_provider == kArmNNExecutionProvider && conv_W_tensor_proto->dims(1) == 1)) {

    return;
  }

  Initializer conv_W{*conv_W_tensor_proto, graph_.ModelPath()};

  arm_compute::Tensor weights;
  arm_compute::Tensor new_weights;

  arm_compute::TensorShape initial_shape(conv_W.dims()[3], conv_W.dims()[2], conv_W.dims()[1], conv_W.dims()[0]);

  weights.allocator()->init(arm_compute::TensorInfo(initial_shape, arm_compute::Format::F32));

  arm_compute::NEPermute permutationLayer;
  permutationLayer.configure(&weights, &new_weights,
    (conv_W_tensor_proto->dims(1) == 1) ? arm_compute::PermutationVector(3,2,0,1) : arm_compute::PermutationVector(2,0,1));

  weights.allocator()->import_memory(conv_W.data<float>());

  new_weights.allocator()->allocate();

  permutationLayer.run();

  weights.allocator()->free();

  float* reordered_filter = reinterpret_cast<float*>(new_weights.buffer());

  ONNX_NAMESPACE::TensorProto nhwc_conv_W_tensor_proto;

  nhwc_conv_W_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  nhwc_conv_W_tensor_proto.set_name(graph_.GenerateNodeArgName("nhwc_conv_W_arg"));
  nhwc_conv_W_tensor_proto.set_raw_data(reordered_filter, new_weights.info()->tensor_shape().total_size() * sizeof(float));

  for (size_t i = 0; i < 4; i++) {
    nhwc_conv_W_tensor_proto.add_dims(conv_W.dims()[i]);
  }

  new_weights.allocator()->free();

  graph_.AddInitializedTensor(nhwc_conv_W_tensor_proto);

  *nhwc_conv_W_arg = &graph_.GetOrCreateNodeArg(nhwc_conv_W_tensor_proto.name(), nullptr);
}


bool NhwcTransformerImpl::SuportsReplacementNHWC(const Node& node) {
   return ((node.GetExecutionProviderType() == kAclExecutionProvider ||
            node.GetExecutionProviderType() == kArmNNExecutionProvider) &&
           (node.OpType() == "Conv" ||
            node.OpType() == "FusedConv" ||
            node.OpType() == "MaxPool" ||
            node.OpType() == "AveragePool" ||
            node.OpType() == "GlobalMaxPool" ||
            node.OpType() == "GlobalAveragePool"));
}

DataLayout NhwcTransformerImpl::RequiredLayout(const Node& node) {
  // Default to NCHW to cover all cases
  DataLayout layout = NchwLayout;

  if (SuportsReplacementNHWC(node)) {
     layout = NhwcLayout;
  } else if (node.OpType() == "Add" ||
             node.OpType() == "Sum") {
     for (auto it = node.InputNodesBegin(), end = node.InputNodesEnd(); it != end && layout == NchwLayout; ++it) {
       auto itLayout = nodes_layout.find(it->Index());
       if (itLayout != nodes_layout.end())
         layout = itLayout->second ? NhwcLayout : NchwLayout;
     }
  } else {
     if (node.OpType() == "Clip" ||
         node.OpType() == "Elu" ||
         node.OpType() == "HardSigmoid" ||
         node.OpType() == "LeakyRelu" ||
         node.OpType() == "Relu" ||
         node.OpType() == "Selu" ||
         node.OpType() == "Sigmoid" ||
         node.OpType() == "Softplus" ||
         node.OpType() == "Softsign" ||
         node.OpType() == "Tanh" ||
         node.OpType() == "PRelu" ||
         node.OpType() == "RandomNormal" ||
         node.OpType() == "RandomUniform" ||
         node.OpType() == "RandomNormalLike" ||
         node.OpType() == "RandomUniformLike" ||
         node.OpType() == "Multinomial")
       layout = InheritLayout;
  }

  return layout;
}

void NhwcTransformerImpl::Transform(Node& node) {
  long unsigned int node_idx = node.Index();

  LOGS(logger, VERBOSE) << "NHWC " << node.OpType() << " "  << node.GetExecutionProviderType();

  DataLayout required_layout = RequiredLayout(node);
  bool bRequiresNHWC = (required_layout == NhwcLayout);
  bool bNodeNHWC = false;

  // create temporary container to allow inseration without alteration
  std::vector<const Node::EdgeEnd*> inputEdges;
  for (auto it = node.InputEdgesBegin(), end = node.InputEdgesEnd(); it != end; ++it)
    inputEdges.push_back(&(*it));

  for (std::vector<const Node::EdgeEnd*>::iterator it = inputEdges.begin(), end = inputEdges.end(); it != end; ++it) {
    const Node::EdgeEnd* edge = *it;

    bool bParentNHWC = true;

    auto itParentLayout = nodes_layout.find(edge->GetNode().Index());
    if (itParentLayout != nodes_layout.end())
      bParentNHWC = itParentLayout->second;

    switch (required_layout) {
    case InheritLayout:
        bNodeNHWC = bParentNHWC;
        break;
    case NhwcLayout:
    case NchwLayout: {
        if (bRequiresNHWC != bParentNHWC) {
          NodeIndex permute_index = InsertPermuteParentNode(node, edge, bRequiresNHWC);
          nodes_layout.insert(std::make_pair(permute_index, bRequiresNHWC));
        }
        bNodeNHWC = bRequiresNHWC;
        break;
      }
    }
  }

  // first node
  if (node.InputNodesBegin() == node.InputNodesEnd()) {
    switch (required_layout) {
    case InheritLayout:
      bNodeNHWC = false;
      break;
    case NhwcLayout:
    case NchwLayout:
        if (bRequiresNHWC) {
          NodeIndex permute_index = InsertPermuteParentNode(node, NULL, true);
          nodes_layout.insert(std::make_pair(permute_index, true));
        }
        bNodeNHWC = bRequiresNHWC;
        break;
    }
  }

  // Replace specific ops
  if (SuportsReplacementNHWC(node)) {
    node_idx = ReplaceNode(node);
    nodes_layout.insert(std::make_pair(node_idx, true));
    bNodeNHWC = true;
  }

  nodes_layout.insert(std::make_pair(node_idx, bNodeNHWC));

  Node& latest_node = *graph_.GetNode(node_idx);

  if (latest_node.OutputNodesBegin() == latest_node.OutputNodesEnd() &&
      bNodeNHWC) {
    NodeIndex perute_index = InsertPermuteChildNode(latest_node, false);
    nodes_layout.insert(std::make_pair(perute_index, false));
  }
}

void NhwcTransformerImpl::Finalize(__attribute__ ((unused)) bool& modified) {
  modified = false;
}

Status NhwcTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {

  std::string provider;
  modified = false;

  for(auto it = registered_execution_providers_.begin(); it != registered_execution_providers_.end(); ++it) {
    if (*it == kAclExecutionProvider || *it == kArmNNExecutionProvider) {
      provider = *it;
      break;
    }
  }

  if (!provider.empty()) {
    NhwcTransformerImpl impl(graph, provider, logger);
    GraphViewer graph_viewer(graph);

    for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
      Node* node = graph.GetNode(index);

      // check that node hasn't already been removed
      if (!node)
        continue;

      ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));
      impl.Transform(*node);
    }

    impl.Finalize(modified);
  }

  return Status::OK();
}

}  // namespace onnxruntime
#endif
