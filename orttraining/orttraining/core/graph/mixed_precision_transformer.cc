// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/mixed_precision_transformer.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "orttraining/core/graph/gradient_builder_base.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/optimizer/initializer.h"
#include "onnx/defs/attr_proto_util.h"

namespace onnxruntime {
namespace training {

// Goals of the mixed-precision-transformer: Replace full-precision (FP32) arithmetic by
// low-precision (FP16) arithmetic as appropriate/required. Currently, the plan is to use
// FP16, by default, and use FP32 only in the following exceptional situations:
// (a) Due to the unavailability of FP16 ops or kernels for some ops such as Trainable/Dropout.
// (b) Due to the usefulness of full-precision in some ops such as SparseSoftmaxCrossEntropy.
// Note that in the long term, it may be useful to extend ops such as ReduceSum with
// an attribute to indicate that it should use a higher-precision accumulator internally
// (e.g., to indicate that a FP16 ReduceSum should internally use 32bit precision).

// Ideally (a) should be computed from schema registries of all available ops & kernels.
// Currently, this information is hard-coded via the stage1_fp32_node_args parameter below.
// The choice for (b) is supplied via the stage2_fp32_node_args parameter below.

// Functions introduce further choices in terms of the precision we use for function parameters.
// We handle functions just like ops: if we want a function to use FP32 parameters, it should
// be indicated using stage2_fp32_node_args.

// The following is a list of ops, as well as functions, that will
// continue to use 32-bit precision. Others will used reduced precision.
// Loss Ops and loss grad Ops are now handled by LossSubgraph, so currently this set is empty.
// If in the future there is new FP32 Op, we can add it here without changing code on other place.
static const std::unordered_set<std::string> FP32_Nodes = {"Sigmoid", "Softmax", "SoftmaxCrossEntropyLoss"};

bool IsFP32Node(const Node* node) {
  return FP32_Nodes.find(node->OpType()) != FP32_Nodes.cend();
}

// At present, we use these table to identify which input needs to be keep in FP32
static const std::unordered_map<std::string, std::vector<int>> stage1_fp32_node_args = {
    {"TrainableDropout", {1}},
    {"TrainableDropoutGrad", {2}},
    {"Dropout", {1}},
    {"DropoutGrad", {2}},
    {"SoftmaxCrossEntropyLoss", {0}},
};

// Currently the list here is same as stage1 above due to empty FP32_Nodes.
// It's possibile we will have more FP32 nodes added, this map will also be extended.
static const std::unordered_map<std::string, std::vector<int>> stage2_fp32_node_args = {
    {"TrainableDropout", {1}},
    {"TrainableDropoutGrad", {2}},
    {"Dropout", {1}},
    {"DropoutGrad", {2}},
    {"Sigmoid", {0}},
    {"Softmax", {0}},
    {"SoftmaxCrossEntropyLoss", {0}},
};

bool IsFP32(const std::unordered_map<std::string, std::vector<int>>& map, std::string opname, int argnum) {
  auto it = map.find(opname);
  if (it == map.cend()) {
    return false;
  } else {
    const auto index_it = std::find(it->second.cbegin(), it->second.cend(), argnum);
    return (index_it != it->second.cend());
  }
}

static const std::string loss_scale_input = "loss_scale";

static const std::unordered_set<std::string> loss_subgraph_entry_nodes = {
    };

static const std::unordered_set<std::string> loss_subgraph_exit_nodes = {
    };

static bool IsLossSubgraphEntryNode(const Node* node) {
  return loss_subgraph_entry_nodes.find(node->OpType()) != loss_subgraph_entry_nodes.cend();
}

static bool IsLossSubgraphExitNode(const Node* node) {
  return loss_subgraph_exit_nodes.find(node->OpType()) != loss_subgraph_exit_nodes.cend();
}

// Separate the consumer nodes of `arg` into two groups: FP32 vs FP16
// The argument `fp32_node_args_by_op_type` specifies the cases where the `arg` should be 32-bit float using op type.
// The argument `fp32_node_args_by_node` specifies the cases where the `arg` should be 32-bit float using node pointer.
static void GetConsumerNodeInputs(onnxruntime::Graph& graph,
                                  const std::unordered_map<std::string, std::vector<int>>& fp32_node_args_by_op_type,
                                  const std::unordered_map<Node*, std::vector<int>>& fp32_node_args_by_node,
                                  const NodeArg* arg,
                                  std::vector<std::pair<Node*, int>>& fp16_inputs,
                                  std::vector<std::pair<Node*, int>>& fp32_inputs) {
  std::vector<Node*> consumer_nodes = graph.GetMutableConsumerNodes(arg->Name());
  for (Node* node : consumer_nodes) {
    int node_arg_slot = -1;
    for (int i = 0; i < static_cast<int>(node->InputDefs().size()); i++) {
      if (node->InputDefs()[i] == arg) {
        node_arg_slot = i;
        break;
      }
    }

    if (node_arg_slot == -1) {
      continue;
    }

    auto it = fp32_node_args_by_op_type.find(node->OpType());
    if (it != fp32_node_args_by_op_type.cend() &&
        std::find(it->second.cbegin(), it->second.cend(), node_arg_slot) != it->second.cend()) {
      fp32_inputs.push_back({node, node_arg_slot});
    } else {
      auto it2 = fp32_node_args_by_node.find(node);
      if (it2 != fp32_node_args_by_node.cend() &&
          std::find(it2->second.cbegin(), it2->second.cend(), node_arg_slot) != it2->second.cend()) {
        fp32_inputs.push_back({node, node_arg_slot});
      } else {
        fp16_inputs.push_back({node, node_arg_slot});
      }
    }
  }
}

static void RewireCastedNodeArg(onnxruntime::Graph& graph,
                                Node* cast_node,
                                const Node* producer_node,
                                int producer_node_arg_index,
                                Node* consumer_node,
                                int consumer_node_arg_index) {
  auto& consumer_inputs = consumer_node->MutableInputDefs();
  if (producer_node != nullptr) {
    graph.RemoveEdge(producer_node->Index(), consumer_node->Index(), producer_node_arg_index, consumer_node_arg_index);
  }
  consumer_inputs[consumer_node_arg_index] = cast_node->MutableOutputDefs()[0];
  graph.AddEdge(cast_node->Index(), consumer_node->Index(), 0, consumer_node_arg_index);
}

// This function tries casting `arg` to `element_type`.
// The argument `fp32_node_args_by_op_type` specifies the cases where the `arg` should be 32-bit float using op type.
// The argument `fp32_node_args_by_node` specifies the cases where the `arg` should be 32-bit float using node pointer.
static Status CastNodeArg(onnxruntime::Graph& graph,
                          const std::unordered_map<std::string, std::vector<int>>& fp32_node_args_by_op_type,
                          const std::unordered_map<Node*, std::vector<int>>& fp32_node_args_by_node,
                          NodeArg* arg,
                          ONNX_NAMESPACE::TensorProto_DataType elem_type) {
  if (arg == nullptr) {
    return Status::OK();
  }
  ORT_ENFORCE(elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
                  elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
                  elem_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16,
              "elem_type should be float, bfloat16 or float16");

  // Get consumer nodes of the input `arg`
  std::vector<std::pair<Node*, int>> fp16_inputs;
  std::vector<std::pair<Node*, int>> fp32_inputs;
  GetConsumerNodeInputs(graph, fp32_node_args_by_op_type, fp32_node_args_by_node, arg, fp16_inputs, fp32_inputs);
  if (((elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 || elem_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) && fp16_inputs.empty()) ||
      (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT && fp32_inputs.empty())) {
    return Status::OK();
  }

  // Create output arg of Cast
  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(elem_type);
  std::string output_name = graph.GenerateNodeArgName(arg->Name());
  const std::string cast_node_name = graph.GenerateNodeName("cast_" + output_name);

  output_name += (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ? "_f32" : "_f16");
  NodeArg& output = graph.GetOrCreateNodeArg(output_name, &type_proto);

  // Create Cast node
  NodeAttributes attrs;
  attrs["to"] = ONNX_NAMESPACE::MakeAttribute("to", int64_t(elem_type));
  Node& cast_node = graph.AddNode(cast_node_name, "Cast", "", {arg}, {&output}, &attrs);

  // Find node arg index in producer
  Node* producer_node = graph.GetMutableProducerNode(arg->Name());
  int producer_node_arg_index = 0;
  if (producer_node != nullptr) {
    while (producer_node_arg_index < static_cast<int>(producer_node->OutputDefs().size()) &&
           producer_node->OutputDefs()[producer_node_arg_index] != arg) {
      producer_node_arg_index++;
    }
    ORT_RETURN_IF_NOT(producer_node_arg_index != static_cast<int>(producer_node->OutputDefs().size()));
  }

  // Update consumer
  if (!fp16_inputs.empty() || !fp32_inputs.empty()) {
    if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 || elem_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
      std::vector<Node*> fp16_nodes;
      fp16_nodes.reserve(fp16_inputs.size());
      for (const auto& kv : fp16_inputs) {
        RewireCastedNodeArg(graph, &cast_node, producer_node, producer_node_arg_index, kv.first, kv.second);
        fp16_nodes.emplace_back(kv.first);
      }

      std::vector<Node*> fp32_nodes;
      fp32_nodes.reserve(fp32_inputs.size() + 1);
      fp32_nodes.emplace_back(&cast_node);
      for (const auto& kv : fp32_inputs) {
        fp32_nodes.emplace_back(kv.first);
      }
      graph.UpdateConsumerNodes(arg->Name(), fp32_nodes);
      graph.UpdateConsumerNodes(output_name, fp16_nodes);
    } else {
      std::vector<Node*> fp32_nodes;
      fp32_nodes.reserve(fp32_inputs.size());
      for (const auto& kv : fp32_inputs) {
        RewireCastedNodeArg(graph, &cast_node, producer_node, producer_node_arg_index, kv.first, kv.second);
        fp32_nodes.emplace_back(kv.first);
      }

      std::vector<Node*> fp16_nodes;
      fp16_nodes.reserve(fp16_inputs.size() + 1);
      fp16_nodes.push_back(&cast_node);
      for (const auto& kv : fp16_inputs) {
        fp16_nodes.emplace_back(kv.first);
      }

      graph.UpdateConsumerNodes(arg->Name(), fp16_nodes);
      graph.UpdateConsumerNodes(output_name, fp32_nodes);
    }
  } else {
    // Make sure it is not one of graph outputs, otherwise, graph outputs need to be updated.
    ORT_RETURN_IF_NOT(std::find(graph.GetOutputs().cbegin(), graph.GetOutputs().cend(), arg) == graph.GetOutputs().cend());
  }

  // Update producer
  if (producer_node != nullptr) {
    graph.AddEdge(producer_node->Index(), cast_node.Index(), producer_node_arg_index, 0);
  }
  graph.UpdateProducerNode(output_name, cast_node.Index());

  return Status::OK();
}

struct LossSubgraph {
  // All nodes belong to this subgraph.
  std::unordered_set<Node*> nodes_;

  // NodeArgs that are inputs of this subgraph from outside, which need to be converted to FP32.
  std::unordered_set<NodeArg*> to_fp32_inputs_;

  // NodeArgs that are outputs of this subgraph to outside, which need to be converted to FP16.
  std::unordered_set<NodeArg*> to_fp16_outputs_;

  // Nodes that take float input from outside of subgraph, the input indices are also saved.
  // It's useful when calling CastNodeArg, so FP32 inputs will no need to be converted.
  std::unordered_map<Node*, std::vector<int>> fp32_node_args_;

  LossSubgraph(Graph& graph) {
    GraphViewer graph_viewer(graph);
    const auto& order = graph_viewer.GetNodesInTopologicalOrder();

    // Get the nodes related to loss scale. It's a Mul node and it's grad nodes.
    // We initialize loss subgraph only when there is loss scale as input.
    std::vector<Node*> loss_scale_consumers = graph.GetMutableConsumerNodes(loss_scale_input);
    if (loss_scale_consumers.size() == 0) {
      return;
    }

    nodes_.insert(loss_scale_consumers.begin(), loss_scale_consumers.end());
    for (Node* node : loss_scale_consumers) {
      for (const NodeArg* output : node->OutputDefs()) {
        std::vector<Node*> level2_consumers = graph.GetMutableConsumerNodes(output->Name());
        nodes_.insert(level2_consumers.begin(), level2_consumers.end());
      }
    }

    // The node number here depends on how to implement the gradient of Mul.
    // Add this check here for safety at certain level.
    ORT_ENFORCE(nodes_.size() == 3,
                "The node number of the loss scale and it's grad subgraph is expected to be 3.");

    // Check if graph contains any loss Op from the white-list.
    // If not, then above loss scale related nodes are all we need.
    bool has_loss_subgraph_entry_node = false;
    for (auto index : order) {
      if (IsLossSubgraphEntryNode(graph.GetNode(index))) {
        has_loss_subgraph_entry_node = true;
        break;
      }
    }

    // If it contains one or more loss Ops from white-list, travel the graph again to get the whole loss subgraph.
    if (has_loss_subgraph_entry_node) {
      for (auto index : order) {
        Node* node = graph.GetNode(index);
        if (IsLossSubgraphEntryNode(node) || IsLossSubgraphExitNode(node)) {
          nodes_.insert(node);
        } else {
          // For other nodes, if it consumes any output of any node from loss subgraph, it also belongs to loss subgraph.
          bool part_of_loss_subgraph = false;
          for (NodeArg* input : node->MutableInputDefs()) {
            Node* producer_node = graph.GetMutableProducerNode(input->Name());
            if (producer_node != nullptr &&
                !IsLossSubgraphExitNode(producer_node) &&
                nodes_.find(producer_node) != nodes_.cend()) {
              part_of_loss_subgraph = true;
              break;
            }
          }

          if (part_of_loss_subgraph) {
            nodes_.insert(node);
          }
        }
      }
    }

    // We now have all the nodes of the loss subgraph. Now get all float inputs from outside.
    for (Node* node : nodes_) {
      int index = 0;
      for (NodeArg* input : node->MutableInputDefs()) {
        if (input->Name() != loss_scale_input &&  // loss_scale input will keep FP32, no need to handle here.
            input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
          // If its producer is from outside, it's one of the inputs of this subgraph.
          Node* producer_node = graph.GetMutableProducerNode(input->Name());
          if (producer_node == nullptr || nodes_.find(producer_node) == nodes_.cend()) {
            to_fp32_inputs_.insert(input);
            if (fp32_node_args_.find(node) == fp32_node_args_.cend()) {
              fp32_node_args_[node] = {index};
            } else {
              fp32_node_args_[node].push_back(index);
            }
          }
        }

        index++;
      }

      // Get all float outputs to outside of the subgraph. They will be converted to FP16.
      for (NodeArg* output : node->MutableOutputDefs()) {
        if (output->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
            !ContainsAllConsumers(graph, output->Name())) {
          to_fp16_outputs_.insert(output);
        }
      }
    }
  }

  bool Contains(Node* node) {
    return nodes_.find(node) != nodes_.cend();
  }

  // Check if this loss subgraph contains all the consumers of given Arg.
  bool ContainsAllConsumers(Graph& graph, const std::string arg_name) {
    std::vector<Node*> consumer_nodes = graph.GetMutableConsumerNodes(arg_name);
    for (Node* node : consumer_nodes) {
      if (nodes_.find(node) == nodes_.cend()) {
        return false;
      }
    }

    return true;
  }

  // For those inputs and constants that are already handled, remove them from the to_fp32 list.
  void RemoveFromToFP32Inputs(const std::string& arg_name) {
    auto it = to_fp32_inputs_.begin();
    while (it != to_fp32_inputs_.end()) {
      if ((*it)->Name() == arg_name) {
        it = to_fp32_inputs_.erase(it);
      } else {
        ++it;
      }
    }
  }

  std::unordered_map<Node*, std::vector<int>>& GetFP32NodeArgs() {
    return fp32_node_args_;
  }

  // Once all inputs, constants, and function calls are handled, it's time to convert all
  // inputs to FP32, and convert all outputs to FP16.
  Status ProcessInputsAndOutputs(Graph& graph) {
    for (auto* node_arg : to_fp32_inputs_) {
      ORT_RETURN_IF_ERROR(CastNodeArg(graph,
                                      stage1_fp32_node_args,
                                      fp32_node_args_,
                                      node_arg,
                                      ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    }

    for (auto* node_arg : to_fp16_outputs_) {
      ORT_RETURN_IF_ERROR(CastNodeArg(graph,
                                      stage1_fp32_node_args,
                                      fp32_node_args_,
                                      node_arg,
                                      ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
    }

    return Status::OK();
  }
};

Status TransformConstants(Graph& graph, ONNX_NAMESPACE::TensorProto_DataType fp16_type, LossSubgraph* p_loss_subgraph = nullptr) {
  // This pass does not require topological sort order: okay to visit nodes in any order.
  // We identify nodeargs to be converted to FP16 first, and then convert them separately
  // to avoid modifying the graph while iterating through it.
  std::unordered_set<NodeArg*> toFP16;
  for (auto& node : graph.Nodes()) {
    // Ignore any node in loss subgraph.
    if (p_loss_subgraph != nullptr && p_loss_subgraph->Contains(&node)) {
      continue;
    }

    const std::string& optype = node.OpType();
    // TODO: Why do we need to handle "Cast" here?
    if ((optype == "Constant") || (optype == "Cast") || (optype == "ConstantOfShape")) {
      for (NodeArg* output : node.MutableOutputDefs()) {
        if (output->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
          // If all consumers are from loss subgraph, don't convert it.
          if (p_loss_subgraph == nullptr || !p_loss_subgraph->ContainsAllConsumers(graph, output->Name())) {
            toFP16.insert(output);
          }

          if (p_loss_subgraph != nullptr) {
            // If it's one of loss subgraph's input, remove it from the to-convert set since it's already handled.
            p_loss_subgraph->RemoveFromToFP32Inputs(output->Name());
          }
        }
      }
    }
  }

  for (auto* tensor : toFP16) {
    ORT_RETURN_IF_ERROR(
        CastNodeArg(graph,
                    stage1_fp32_node_args,
                    p_loss_subgraph != nullptr ?
                        p_loss_subgraph->GetFP32NodeArgs() :
                        std::unordered_map<Node*, std::vector<int>>(),
                    tensor,
                    fp16_type));
  }

  return Status::OK();
}

// Stage 2 transformation: Introduce conversions from FP16 back to FP32 for ops such
// as SparseSoftmaxCrossEntropy where FP32 precision is required.
// Converts fp16 tensor --> Op --> fp16 tensor to
// fp16 tensor --> Cast --> fp32 tensor --> Op --> fp32 tensor --> Cast --> fp16 tensor
Status TransformStage2(Graph& graph, 
                       ONNX_NAMESPACE::TensorProto_DataType fp16_type,
                       const std::unordered_map<Node*, std::vector<int>>& loss_subgraph_fp32_node_args = {}) {
  // This pass does not require topological sort order: okay to visit nodes in any order.
  std::unordered_set<NodeArg*> toFP16, toFP32;
  for (auto& node : graph.Nodes()) {
    if (IsFP32Node(&node)) {
      for (NodeArg* input : node.MutableInputDefs()) {
        // TODO: Shouldn't we check stage2_fp32_node_args to conditionally transform this?
        if (input->TypeAsProto()->tensor_type().elem_type() == fp16_type)
          toFP32.insert(input);
      }

      for (NodeArg* output : node.MutableOutputDefs()) {
        // TODO: This currently assumes that all outputs of FP32 ops are FP32.
        if (output->TypeAsProto()->tensor_type().elem_type() == fp16_type)
          toFP16.insert(output);
      }
    }
  }
  for (auto* tensor : toFP32)
    ORT_RETURN_IF_ERROR(CastNodeArg(graph,
                                    stage2_fp32_node_args,
                                    loss_subgraph_fp32_node_args,
                                    tensor,
                                    ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  for (auto* tensor : toFP16)
    ORT_RETURN_IF_ERROR(CastNodeArg(graph,
                                    stage2_fp32_node_args,
                                    loss_subgraph_fp32_node_args,
                                    tensor,
                                    fp16_type));
  return Status::OK();
}

static Status HandleFunctionCalls(Graph& graph, ONNX_NAMESPACE::TensorProto_DataType fp16_type, LossSubgraph* p_loss_subgraph = nullptr);

// TODO: Ideally, we should not need to transform a function-body here.
// Ideally, for any full-precision function F, there should be a corresponding 16-bit precision
// version of F too: that is, the type-signature of F should include both the full-precision and
// low-precision. Thus, transforming the types of actual-parameters of a call to F should be
// sufficient. We explicitly transform a function body here due to a couple of limitations.
// (a) First, the existing function-mechanism does not allow us to express the body of the
// full-precision and low-precision function because of the treatment of constants.
// (b) The existing ORT pipeline specializes function-bodies to the types of actual-parameters
// eagerly during Graph resolution. Ideally, the function-body specialization should be delayed
// until after mixed-precision-transformation or any transformation that changes types.
// Once (a) and (b) are addressed elsewhere, we can simplify the treatment here.
// In cases where we do need to transform the function-body, we should ideally inline the transformed
// body if its transformed semantics does not match the original semantics (or rename the function):
// otherwise, we may end up using a kernel with the original semantics erroneously.

static Status HandleFunctionBody(const Function& node_func, ONNX_NAMESPACE::TensorProto_DataType fp16_type) {
  const Graph& fn_body = node_func.Body();
  // TODO: eliminate use of const_casts
  Graph& graph = const_cast<Graph&>(fn_body);
  // TODO: The resolve below is likely unnecessary.
  ORT_RETURN_IF_ERROR(graph.Resolve());

  // Stage 1 for functions:
  // Update the types of inputs of function body graph:
  const std::string& fn_name = node_func.OpSchema().Name();
  int argnum = 0;
  for (const NodeArg* input : graph.GetInputs()) {
    // Reduce input type to lower precision (unless specified as FP32 by stage2_fp32_node_args).
    onnx::TypeProto type = *(input->TypeAsProto());
    if (type.has_tensor_type() && type.tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      if (!IsFP32(stage2_fp32_node_args, fn_name, argnum)) {
        type.mutable_tensor_type()->set_elem_type(fp16_type);
        graph.SetNodeArgType(const_cast<NodeArg&>(*input), type);
        // Introduce cast to full-precision if required:
        // TODO: fix const_cast; Graph doesn't provide us a method "GetMutableInputs".
        NodeArg* mutable_input = const_cast<NodeArg*>(input);
        CastNodeArg(graph,
                    stage1_fp32_node_args,
                    std::unordered_map<Node*, std::vector<int>>(),
                    mutable_input,
                    ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      }
    }

    ++argnum;
  }

  ORT_RETURN_IF_ERROR(TransformConstants(graph, fp16_type));

  // End of stage 1. Update types of intermediate-values and return-values:
  Graph::ResolveOptions options;
  options.override_types = true;
  ORT_RETURN_IF_ERROR(graph.Resolve(options));

  // Stage 2:
  ORT_RETURN_IF_ERROR(TransformStage2(graph, fp16_type));

  // Recursively transform nested function call bodies.
  ORT_RETURN_IF_ERROR(HandleFunctionCalls(graph, fp16_type));

  // Update types of intermediate-values and return-values:
  auto status = graph.Resolve(options);
  return status;
}

static Status HandleFunctionCalls(Graph& graph, ONNX_NAMESPACE::TensorProto_DataType fp16_type, LossSubgraph* p_loss_subgraph) {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();
  for (auto index : order) {
    Node* node = graph.GetNode(index);
    // Bodies of FP32 Functions are not transformed.
    if (IsFP32Node(node) ||
        (p_loss_subgraph != nullptr && p_loss_subgraph->Contains(node))) {
      continue;
    }

    const Function* node_func = node->GetFunctionBody();
    if (nullptr != node_func) {
      ORT_RETURN_IF_ERROR(HandleFunctionBody(*node_func, fp16_type));
    }
  }

  return Status::OK();
}

// Create FP16 NodeArg and update the consumers of arg with new FP16 NodeArg.
static NodeArg* CreateFP16NodeArgAndUpdateConsumers(Graph& graph,
                                                    const std::unordered_map<std::string, std::vector<int>>& fp32_node_args_by_op_type,
                                                    const std::unordered_map<Node*, std::vector<int>>& fp32_node_args_by_node,
                                                    const NodeArg* arg,
                                                    ONNX_NAMESPACE::TensorProto_DataType fp16_type) {
  ORT_ENFORCE(arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              "data type is not float");
  // Create FP16 Node Arg
  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(fp16_type);
  type_proto.mutable_tensor_type()->mutable_shape()->CopyFrom(*arg->Shape());
  std::string arg_name = arg->Name() + "_fp16";
  NodeArg& new_arg = graph.GetOrCreateNodeArg(arg_name, &type_proto);

  // Check consumer nodes
  std::vector<std::pair<Node*, int>> fp16_inputs;
  std::vector<std::pair<Node*, int>> fp32_inputs;
  GetConsumerNodeInputs(graph, fp32_node_args_by_op_type, fp32_node_args_by_node, arg, fp16_inputs, fp32_inputs);
  if (fp16_inputs.empty()) {
    return nullptr;
  }

  for (auto kv : fp16_inputs) {
    kv.first->MutableInputDefs()[kv.second] = &new_arg;
  }

  return &new_arg;
}

Status TransformGraphForMixedPrecision(Graph& graph,
                                       const std::unordered_set<std::string>& weights_to_train,
                                       bool use_fp16_initializer,
                                       std::unordered_map<std::string, NodeArg*>& fp32_weight_name_to_fp16_node_arg,
                                       ONNX_NAMESPACE::TensorProto_DataType fp16_type) {
  //only fp16 and bfloat16 supported
  ORT_ENFORCE(fp16_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 || fp16_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
  // Stag 0: Initialize loss subgraph.
  LossSubgraph loss_subgraph(graph);

  // Stage 1: Convert whole graph including forward and backward to FP16
  // Insert Cast node to convert inputs from FP32 to FP16
  // If all consumers are from loss graph, don't convert it, and remove it from To-32 loss graph inputs.
  for (const NodeArg* input : graph.GetInputs()) {
    // Input loss_scale will always keep as FP32.
    if (input->Name() != loss_scale_input &&
        input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      // If all consumers are from loss subgraph, no need to convert.
      if (!loss_subgraph.ContainsAllConsumers(graph, input->Name())) {
        ORT_RETURN_IF_ERROR(
            CastNodeArg(graph,
                        stage1_fp32_node_args,
                        loss_subgraph.GetFP32NodeArgs(),
                        graph.GetNodeArg(input->Name()),
                        fp16_type));
      }

      // Remove it from the to-convert set since it's already handled.
      loss_subgraph.RemoveFromToFP32Inputs(input->Name());
    }
  }

  // Convert initializers including trainable weights from FP32 to FP16
  const auto& initialized_tensors = graph.GetAllInitializedTensors();
  std::unordered_map<std::string, NodeArg*> fp32_weight_name_to_fp16_node_arg_result{};
  std::vector<std::pair<std::string, const ONNX_NAMESPACE::TensorProto*>> fp16_initializers;
  for (const auto& kv : initialized_tensors) {
    NodeArg* input = graph.GetNodeArg(kv.first);
    if (input->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      // If all consumers are from loss graph, don't convert it.
      if (!loss_subgraph.ContainsAllConsumers(graph, input->Name())) {
        if (use_fp16_initializer) {
          NodeArg* fp16_weight_arg = CreateFP16NodeArgAndUpdateConsumers(graph,
                                                                         stage1_fp32_node_args,
                                                                         loss_subgraph.GetFP32NodeArgs(),
                                                                         input,
                                                                         fp16_type);
          if (fp16_weight_arg != nullptr) {
            fp16_initializers.emplace_back(fp16_weight_arg->Name(), kv.second);
            const auto it = weights_to_train.find(kv.first);
            if (it != weights_to_train.cend()) {
              fp32_weight_name_to_fp16_node_arg_result[kv.first] = fp16_weight_arg;
            }
          }
        } else {
          ORT_RETURN_IF_ERROR(CastNodeArg(graph,
                                          stage1_fp32_node_args,
                                          loss_subgraph.GetFP32NodeArgs(),
                                          input,
                                          fp16_type));
        }
      }

      // Remove it from the to-convert set since it's already handled.
      loss_subgraph.RemoveFromToFP32Inputs(input->Name());
    }
  }

  // Add new FP16 initializers to the graph
  for (const auto& kv : fp16_initializers) {
    const ONNX_NAMESPACE::TensorProto* tensor_proto = kv.second;
    Initializer initializer(*tensor_proto, graph.ModelPath());
    ONNX_NAMESPACE::TensorProto weight_tensor_proto = fp16_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ? initializer.ToFP16(kv.first) : initializer.ToBFloat16(kv.first);
    graph.AddInitializedTensor(weight_tensor_proto);
  }

  // Handle pipeline case
  for (auto& node : graph.Nodes()) {
    // For send and recv node, if the tensor being sent or received is FP32, update its
    // attribute and change it to FP16.
    if ((!node.OpType().compare("Send") || !node.OpType().compare("Recv")) && !loss_subgraph.Contains(&node)) {
      auto& attributes = node.GetMutableAttributes();
      auto* element_type = &(attributes.find("element_types")->second);
      int ints_size = element_type->ints_size();
      for (int i = 0; i < ints_size; ++i) {
        if (element_type->ints(i) == static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT)) {
          element_type->set_ints(i, static_cast<int64_t>(fp16_type));
          // Need to resolve and populate the new type through the graph.
          graph.SetGraphResolveNeeded();
        }
      }
    }
  }

  // Handle implicit data type casting nodes such as Cast, ConstantOfShape
  ORT_RETURN_IF_ERROR(TransformConstants(graph, fp16_type, &loss_subgraph));

  // Handle function body
  ORT_RETURN_IF_ERROR(HandleFunctionCalls(graph, fp16_type, &loss_subgraph));

  // Handle loss graph inputs and outputs.
  ORT_RETURN_IF_ERROR(loss_subgraph.ProcessInputsAndOutputs(graph));

  // At this point, the model has been transformed to a valid FP16 model.

  Graph::ResolveOptions options;
  options.initializer_names_to_preserve = &weights_to_train;
  options.override_types = true;

  ORT_RETURN_IF_ERROR(graph.Resolve(options));

  TransformStage2(graph, fp16_type, loss_subgraph.GetFP32NodeArgs());

  ORT_RETURN_IF_ERROR(graph.Resolve(options));

  fp32_weight_name_to_fp16_node_arg = std::move(fp32_weight_name_to_fp16_node_arg_result);

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
