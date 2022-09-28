// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "core/common/inlined_containers.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace {

/**
 * @brief Create new initializer using partial or all values of the given shape vector.
 *
 * @param graph Graph to iterate and operate on.
 * @param node Node whose output can be replaced by a constant initializer.
 * @param dim_values Vector of int values, representing a shape.
 * @param start Start offset where we copy int value from.
 * @param length Number of int values to be copies into newly created initializer.
 * @param create_scalar_for_single_value Bool value indicates whether initializer should be created as a scalar.
 *     Only applicable when length param is 1. Default to be false.
 */
void CreateInitializerFromShapeVector(Graph& graph,
                                      NodeArg* arg_to_be_replaced,
                                      const TensorShapeVector& dim_values,
                                      int64_t start,
                                      int64_t length,
                                      bool create_scalar_for_single_value = false) {
  bool is_scalar = length == 1 && create_scalar_for_single_value;

  // Create new TensorProto.
  ONNX_NAMESPACE::TensorProto constant_tensor_proto;
  constant_tensor_proto.set_name(arg_to_be_replaced->Name());
  constant_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  if (!is_scalar) {
    constant_tensor_proto.add_dims(length);
  }
  constant_tensor_proto.set_raw_data(dim_values.data() + start, length * sizeof(int64_t));

  // Add initializer into Graph.
  graph.AddInitializedTensor(constant_tensor_proto);

  // Update the output arg shape.
  ONNX_NAMESPACE::TensorShapeProto new_shape;
  if (!is_scalar) {
    new_shape.add_dim()->set_dim_value(length);
  }
  arg_to_be_replaced->SetShape(new_shape);
}

/**
 * @brief Deal with negatives and clamp on indices.
 *
 * @param start Int value representing starting index of Slice or Shape (Opset 15).
 *    Be noted: Opset-15 Shape op supports slicing shape values
 *    Return updated starting index after dealing with negatives.
 * @param end Int value representing ending index (exclusive).
 *    Return updated ending index after dealing with negatives.
 * @param rank Int value representing the length of the dimension (to slice from).
 * @return size_t Length to slice.
 */
size_t HandleSliceOrShape15Indices(int64_t& start, int64_t& end, int64_t rank) {
  // We ascertain the "true" starts/ends (if they were provided)
  //  Deal with negatives and clamp
  start = start < 0 ? start + rank : start;
  start = start < 0 ? 0 : ((start > rank) ? rank : start);

  end = end < 0 ? end + rank : end;
  end = end < 0 ? 0 : ((end > rank) ? rank : end);

  int64_t slice_length = end - start;
  size_t clamped_slice_length = slice_length < 0 ? 0 : static_cast<size_t>(slice_length);
  return clamped_slice_length;
}

bool IsScalarShape(const ONNX_NAMESPACE::TensorShapeProto* input_shape) {
  return input_shape && (input_shape->dim_size() == 0);
}

bool IsSingleValue1DShape(const ONNX_NAMESPACE::TensorShapeProto* input_shape) {
  if (input_shape == nullptr) {
    return false;
  }

  size_t dim_size = static_cast<size_t>(input_shape->dim_size());
  if (dim_size == 1 && utils::HasDimValue(input_shape->dim(0)) && input_shape->dim(0).dim_value() == 1) {
    return true;
  }

  return false;
}

/**
 * @brief Check Shape node can be constant folded or not.
 *
 * @param shape_node Shape node to check.
 * @param dim_values Int vector (tensor shape inferred) containing -1 as symbolic dim.
 * @param start For Shape-15 node, used to return slicing start index; Otherwise, should be 0.
 * @param end For Shape-15 node, used to return slicing end index (exclusive); Otherwise, shuld be rank.
 * @param clamped_slice_length Number of sliced elements.
 * @return true The dimension vaules in dim_values ranging from start to end (exclusive) are all concrete values.
 * @return false Otherwise.
 */
bool IsShapeNodeCanBeConstantFolded(Node& shape_node,
                                    const TensorShapeVector& dim_values,
                                    int64_t& start,
                                    int64_t& end,
                                    size_t& clamped_slice_length) {
  bool can_be_folded = true;

  if (graph_utils::IsSupportedOptypeVersionAndDomain(shape_node, "Shape", {15})) {
    // Opset-15 Shape supports slicing using a 'start' and 'end' attribute
    const auto& shape_attributes = shape_node.GetAttributes();
    for (const auto& attr : shape_attributes) {
      if (attr.first == "start") {
        start = attr.second.i();
      } else if (attr.first == "end") {
        end = attr.second.i();
      }
    }
  }

  clamped_slice_length = HandleSliceOrShape15Indices(start, end, static_cast<int64_t>(dim_values.size()));
  for (size_t i = static_cast<size_t>(start); i < static_cast<size_t>(end); ++i) {
    if (dim_values[i] == -1) {
      can_be_folded = false;
      break;
    }
  }

  return can_be_folded;
}

/**
 * @brief Check Slice node (consuming Shape) can be constant folded or not.
 * Only support 1D single value slice.
 *
 * @param graph Graph to operate.
 * @param slice_node Slice node to check.
 * @param dim_values Int vector (tensor shape inferred) containing -1 as symbolic dim.
 * @param start Used to return slicing start index.
 * @param end Used to return slicing end index (exclusive)
 * @param clamped_slice_length Number of sliced elements.
 * @return true The dimension vaules in dim_values ranging from start to end (exclusive) are all concrete values.
 * @return false Otherwise.
 */
bool IsSliceNodeCanBeConstantFolded(Graph& graph,
                                    Node& slice_node,
                                    const TensorShapeVector& dim_values,
                                    int64_t& start,
                                    int64_t& end,
                                    size_t& clamped_slice_length) {
  NodeArg* starts_input = slice_node.MutableInputDefs()[1];
  NodeArg* ends_input = slice_node.MutableInputDefs()[2];
  NodeArg* axes_input = slice_node.MutableInputDefs().size() > 3 ? slice_node.MutableInputDefs()[3] : nullptr;
  NodeArg* steps_input = slice_node.MutableInputDefs().size() > 4 ? slice_node.MutableInputDefs()[4] : nullptr;

  // We only support 1D slices currently, can be extended further to support other cases.
  if (!IsSingleValue1DShape(starts_input->Shape()) ||
      !IsSingleValue1DShape(ends_input->Shape()) ||
      (axes_input && !IsSingleValue1DShape(axes_input->Shape())) ||
      (steps_input && !IsSingleValue1DShape(steps_input->Shape()))) {
    return false;
  }

  // Try to parse the value and double check.
  InlinedVector<int64_t> starts_values, ends_values, axes_values, steps_values;
  if (!(optimizer_utils::AppendTensorFromInitializer(graph, *starts_input, starts_values, true) &&
        starts_values.size() == 1)) {
    return false;
  }
  if (!(optimizer_utils::AppendTensorFromInitializer(graph, *ends_input, ends_values, true) &&
        ends_values.size() == 1)) {
    return false;
  }
  if (axes_input && !(optimizer_utils::AppendTensorFromInitializer(graph, *axes_input, axes_values, true) &&
                      axes_values.size() == 1 && axes_values[0] == 0)) {
    return false;
  }
  if (steps_input && !(optimizer_utils::AppendTensorFromInitializer(graph, *steps_input, steps_values, true) &&
                       steps_values.size() == 1 && steps_values[0] == 1)) {
    return false;
  }

  start = starts_values[0];
  end = ends_values[0];
  clamped_slice_length = HandleSliceOrShape15Indices(start, end,
                                                     static_cast<int64_t>(dim_values.size()));

  for (size_t i = static_cast<size_t>(start); i < static_cast<size_t>(end); ++i) {
    if (dim_values[i] == -1) {
      return false;
    }
  }

  return true;
}

/**
 * @brief Check Gather node (consuming Shape) can be constant folded or not.
 * Only support 1D single value or scalar value slice.
 *
 * @param graph Graph to operate.
 * @param slice_node Slice node to check.
 * @param dim_values Int vector (tensor shape inferred) containing -1 as symbolic dim.
 * @param gather_index Used to return the index to gather.
 * @param gather_indices_length Always to be 1, since we only operate on 1D single value or scalar value.
 * @param gather_output_rank Used to return the output rank of Gather output.
 * @return true The dimension value at gather_index in dim_values is concreate value.
 * @return false Otherwise.
 */
bool IsGatherNodeCanBeConstantFolded(Graph& graph,
                                     Node& gather_node,
                                     const TensorShapeVector& dim_values,
                                     int64_t& gather_index,
                                     size_t& gather_indices_length,
                                     int& gather_output_rank) {
  NodeArg* indices_input = gather_node.MutableInputDefs()[1];
  auto indices_shape = indices_input->Shape();
  if (!IsScalarShape(indices_shape) && !IsSingleValue1DShape(indices_shape)) {
    // If the indices did not contain one single element, then skip it.
    return false;
  }

  // Try to parse int64 type constant initializers.
  // We only support 1D slices currently, can be extended further to support other cases.
  InlinedVector<int64_t> indices_values;
  if (!(optimizer_utils::AppendTensorFromInitializer(graph, *indices_input, indices_values, true) &&
        indices_values.size() == 1)) {
    return false;
  }

  const ONNX_NAMESPACE::AttributeProto* axis_attr = graph_utils::GetNodeAttribute(gather_node, "axis");
  if (axis_attr && static_cast<int>(axis_attr->i()) != 0) {
    return false;
  }

  gather_index = indices_values[0];
  const int64_t rank = static_cast<int64_t>(dim_values.size());
  gather_index = gather_index < 0 ? gather_index + rank : gather_index;
  gather_indices_length = 1;

  if (dim_values[static_cast<size_t>(gather_index)] < 0) {
    return false;
  }

  gather_output_rank = 1 /* gather input data is a 1-D tensor representing a shape */ + indices_shape->dim_size() - 1;
  return true;
}

// We need to handle a Shape node separately as the input doesn't need to be a constant initializer for
// Shape to be able to be constant folded.
bool ConstantFoldShapeNode(Graph& graph, Node& node, InlinedVector<Node*>& nodes_to_remove,
                           bool enable_enhanced_shape_constant_fold) {
  auto shape = node.MutableInputDefs()[0]->Shape();

  if (shape != nullptr) {
    TensorShapeVector dim_values;
    dim_values.reserve(shape->dim_size());
    bool has_concrete_dim = false;
    for (int dim_index = 0; dim_index < shape->dim_size(); dim_index++) {
      auto dim = shape->dim(dim_index);
      if (utils::HasDimValue(dim)) {
        dim_values.push_back(dim.dim_value());
        has_concrete_dim = true;
      } else {
        // Fill with -1 for symbolic dimension.
        dim_values.push_back(-1);
      }
    }

    int64_t shape_slice_start = 0;
    int64_t shape_slice_end = std::numeric_limits<int64_t>::max();
    size_t shape_slice_length = 0;
    if (IsShapeNodeCanBeConstantFolded(node, dim_values, shape_slice_start, shape_slice_end, shape_slice_length)) {
      CreateInitializerFromShapeVector(graph, node.MutableOutputDefs()[0], dim_values, shape_slice_start,
                                       shape_slice_length);
      nodes_to_remove.push_back(&node);
      return true;
    }

    if (!has_concrete_dim || !enable_enhanced_shape_constant_fold) {
      return false;
    }

    // Check consumer Slice/Gather nodes to see any opportunities for constant folding.
    auto p_ip_node = node.OutputNodesBegin();
    const auto p_ip_node_end = node.OutputNodesEnd();
    InlinedHashSet<const Node*> visited_nodes;
    while (p_ip_node != p_ip_node_end) {
      if (visited_nodes.find(&(*p_ip_node)) != visited_nodes.end()) {
        // Already handled, skip the node.
        ++p_ip_node;
        continue;
      }

      auto& output_node = const_cast<Node&>(*p_ip_node);
      visited_nodes.insert(&output_node);
      ++p_ip_node;

      NodeArg* data_input = output_node.MutableInputDefs()[0];
      // Skip when shape is not used as sliced data.
      if (data_input != node.MutableOutputDefs()[0]) {
        continue;
      }

      int64_t slice_start = 0;
      int64_t slice_end = std::numeric_limits<int64_t>::max();
      size_t slice_length = 0;
      if (graph_utils::IsSupportedOptypeVersionAndDomain(output_node, "Slice", {10, 11, 13}) &&
          IsSliceNodeCanBeConstantFolded(graph, output_node, dim_values, slice_start, slice_end, slice_length)) {
        CreateInitializerFromShapeVector(graph, output_node.MutableOutputDefs()[0], dim_values, slice_start,
                                         slice_length);
        nodes_to_remove.push_back(&output_node);
        continue;
      }

      int64_t gather_index = 0;
      size_t gather_indices_length = 0;
      int gather_output_rank = 0;
      if (graph_utils::IsSupportedOptypeVersionAndDomain(output_node, "Gather", {1, 11, 13}) &&
          IsGatherNodeCanBeConstantFolded(graph, output_node, dim_values, gather_index, gather_indices_length,
                                          gather_output_rank)) {
        CreateInitializerFromShapeVector(graph, output_node.MutableOutputDefs()[0], dim_values, gather_index,
                                         gather_indices_length,
                                         gather_output_rank == 0);
        nodes_to_remove.push_back(&output_node);
        continue;
      }
    }
  }

  return nodes_to_remove.size() > 0;
}
}  // namespace

Status ConstantFolding::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  bool have_updated_nodes = false;
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();

#if !defined(DISABLE_SPARSE_TENSORS)
  std::function<bool(const std::string&)> is_sparse_initializer_check = [&graph](const std::string& name) -> bool {
    return graph.IsSparseInitializer(name);
  };
#endif

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);
    if (!node) {
      continue;
    }

    // avoid to constant fold DequantizeLinear for QDQ format
    if (skip_dequantize_linear_ && node->OpType().compare("DequantizeLinear") == 0) {
      continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    // Updating a node may allow shape inferencing to infer output shapes of following nodes,
    // so re-run the shape inferencing. use have_updated_nodes as that only applies to this Graph
    // (vs. 'modified' which is passed into subgraphs and applies to the main graph and all subgraphs)
    // Ignore any control flow node containing subgraphs as UpdateShapeInference is not intended to be used on it.
    if (have_updated_nodes && !node->ContainsSubgraph()) {
      ORT_RETURN_IF_ERROR(graph.UpdateShapeInference(*node));
    }

    bool converted_to_constant = false;
    InlinedVector<Node*> nodes_to_remove;
    if (node->OpType().compare("Shape") == 0) {
      converted_to_constant = ConstantFoldShapeNode(graph, *node, nodes_to_remove,
                                                    enable_enhanced_shape_constant_fold_);
    } else {
      InitializedTensorSet constant_inputs;

      // we currently constant fold using the CPU EP only.
      // if the node is assigned to a different EP we can run it if it's an ONNX op as we have CPU based
      // implementations for all ONNX ops. If the node/op is from a different op domain or if the CPU implementation
      // does not support the specific input type(s) required by the node (currently we only support a subset of
      // types in some CPU kernels) then we can't proceed with constant folding for the node.
      auto ep_type = node->GetExecutionProviderType();
      bool cpu_ep = ep_type == kCpuExecutionProvider;
      if (!cpu_ep && node->Domain() != kOnnxDomain) {
        continue;
      }

      // Check if constant folding can be applied on this node.
      if (!graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders()) ||
          !optimizer_utils::IsOperationDeterministic(node->Domain(), node->OpType()) ||
          // constant folding does not support executing a node that includes subgraphs (control flow operators,
          // such as If/Loop/Scan, fall into this category). individual nodes in the subgraph will be processed
          // by the Recurse call above
          node->ContainsSubgraph() ||
          !graph_utils::AllNodeInputsAreConstant(graph, *node, constant_inputs, excluded_initializers_)) {
        continue;
      }

#if !defined(DISABLE_SPARSE_TENSORS)
      // Create execution frame for executing constant nodes.
      OptimizerExecutionFrame::Info info({node}, constant_inputs, graph.ModelPath(), execution_provider_,
                                         is_sparse_initializer_check);
#else
      // Create execution frame for executing constant nodes.
      OptimizerExecutionFrame::Info info({node}, constant_inputs, graph.ModelPath(), execution_provider_,
                                         [](std::string const&) { return false; });
#endif

      std::vector<int> fetch_mlvalue_idxs;
      for (const auto* node_out : node->OutputDefs()) {
        fetch_mlvalue_idxs.push_back(info.GetMLValueIndex(node_out->Name()));
      }

      // override the EP assigned to the node so that it will use the CPU kernel for Compute.
      if (!cpu_ep) {
        node->SetExecutionProviderType(kCpuExecutionProvider);
      }

      auto kernel = info.CreateKernel(node);

      // undo the EP change to the value that was assigned at graph partitioning time
      if (!cpu_ep) {
        node->SetExecutionProviderType(ep_type);
      }

      if (kernel == nullptr) {
        LOGS(logger, WARNING) << "Could not find a CPU kernel and hence "
                              << "can't constant fold " << node->OpType() << " node '" << node->Name() << "'";

        // Move on to the next candidate node
        continue;
      }

      OptimizerExecutionFrame frame(info, fetch_mlvalue_idxs);

      OpKernelContext op_kernel_context(&frame, kernel.get(), nullptr, logger);
      ORT_RETURN_IF_ERROR(kernel->Compute(&op_kernel_context));

      std::vector<OrtValue> fetches;
      ORT_RETURN_IF_ERROR(frame.GetOutputs(fetches));

      // Go over all output node args and substitute them with the newly computed tensors, which will be
      // added to the graph as initializers.
      ORT_ENFORCE(fetches.size() == node->OutputDefs().size());
      converted_to_constant = true;
      for (size_t fetch_idx = 0; fetch_idx < fetches.size(); ++fetch_idx) {
        const auto& constant_arg_out = *node->OutputDefs()[fetch_idx];
        // XXX: Add support for SparseTensors outputs when we have sparse outputs
        if (!utils::HasTensorType(*constant_arg_out.TypeAsProto())) {
          LOGS(logger, INFO) << "Unsupported output type of " << constant_arg_out.Type()
                             << ". Can't constant fold " << node->OpType() << " node '" << node->Name() << "'";
          converted_to_constant = false;
          break;
        }
      }

      if (converted_to_constant) {
        for (size_t fetch_idx = 0; fetch_idx < fetches.size(); ++fetch_idx) {
          OrtValue& ort_value = fetches[fetch_idx];
          // Build the TensorProto that corresponds to the computed OrtValue and add it as initializer to the graph.
          auto* constant_arg_out = node->MutableOutputDefs()[fetch_idx];
          const Tensor& out_tensor = ort_value.Get<Tensor>();
          ONNX_NAMESPACE::TensorProto out_tensorproto =
              utils::TensorToTensorProto(out_tensor, constant_arg_out->Name());

          ONNX_NAMESPACE::TensorShapeProto result_shape;
          for (auto& dim : out_tensor.Shape().GetDims()) {
            result_shape.add_dim()->set_dim_value(dim);
          }

          constant_arg_out->SetShape(result_shape);
          graph.AddInitializedTensor(out_tensorproto);
        }

        nodes_to_remove.push_back(node);
      }
    }

    if (converted_to_constant) {
      for (Node* node_to_remove : nodes_to_remove) {
        // Remove single-output node chain for inputs of the node
        auto p_ip_node = node_to_remove->InputNodesBegin();
        const auto p_ip_node_end = node_to_remove->InputNodesEnd();
        while (p_ip_node != p_ip_node_end) {
          const auto& input_node = *p_ip_node;
          // Update the node iterator before removing the corresponding node because removing
          // the node will invalidate the node iterator
          ++p_ip_node;
          graph_utils::RemoveNodesWithOneOutputBottomUp(graph, input_node);
        }

        // Remove the output edges of the constant node and then remove the node itself.
        graph_utils::RemoveNodeOutputEdges(graph, *node_to_remove);
        graph.RemoveNode(node_to_remove->Index());
        modified = true;
        have_updated_nodes = true;
      }
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
