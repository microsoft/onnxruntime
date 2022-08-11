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

ConstantFolding::ConstantFolding(const IExecutionProvider& execution_provider,
                                 bool skip_dequantize_linear,
                                 const InlinedHashSet<std::string_view>& compatible_execution_providers,
                                 const InlinedHashSet<std::string>& excluded_initializers) noexcept
    : GraphTransformer("ConstantFolding", compatible_execution_providers),
      skip_dequantize_linear_(skip_dequantize_linear),
      excluded_initializers_(excluded_initializers),
      execution_provider_(execution_provider) {
}

namespace {

void CreateInitializerFromShapeVector(Graph& graph,
                                      Node& node,
                                      const TensorShapeVector& dim_values,
                                      int64_t start,
                                      int64_t length,
                                      bool create_scalar_for_single_value = false) {
  bool is_scalar = length == 1 && create_scalar_for_single_value;
  ONNX_NAMESPACE::TensorProto shape_constant;
  auto* constant_arg_out = node.MutableOutputDefs()[0];
  shape_constant.set_name(constant_arg_out->Name());
  shape_constant.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  if (!is_scalar) {
    shape_constant.add_dims(length);
  }
  shape_constant.set_raw_data(dim_values.data() + start,
                              length * sizeof(int64_t));
  ONNX_NAMESPACE::TensorShapeProto result_shape;
  if (!is_scalar) {
    result_shape.add_dim()->set_dim_value(length);
  }
  constant_arg_out->SetShape(result_shape);
  graph.AddInitializedTensor(shape_constant);
}

// We ascertain the "true" starts/ends (if they were provided)
// Opset-15 Shape op supports slicing shape values
// start: starting index
// end: ending index (exclusive)
size_t HandleSliceOrShape15Indices(int64_t& start, int64_t& end, size_t data_rank) {
  // Deal with negatives and clamp
  int64_t rank = static_cast<int64_t>(data_rank);
  start = start < 0 ? start + rank : start;
  start = start < 0 ? 0 : ((start > rank) ? rank : start);

  end = end < 0 ? end + rank : end;
  end = end < 0 ? 0 : ((end > rank) ? rank : end);

  int64_t slice_length = end - start;
  size_t clamped_slice_length = slice_length < 0 ? 0 : static_cast<size_t>(slice_length);
  return clamped_slice_length;
}

// We need to handle a Shape node separately as the input doesn't need to be a constant initializer for
// Shape to be able to be constant folded.
bool ConstantFoldShapeNode(Graph& graph, Node& node, InlinedVector<Node*>& nodes_to_remove) {
  auto shape = node.MutableInputDefs()[0]->Shape();
  bool is_concrete_shape = true;
  TensorShapeVector dim_values;

  if (shape != nullptr) {
    dim_values.reserve(shape->dim_size());
    for (int dim_index = 0; dim_index < shape->dim_size(); dim_index++) {
      auto dim = shape->dim(dim_index);
      if (utils::HasDimValue(dim)) {
        dim_values.push_back(dim.dim_value());
      } else {
        is_concrete_shape = false;
        // Fill with -1 for symbolic dimension.
        dim_values.push_back(-1);
      }
    }
  } else {
    is_concrete_shape = false;
  }

  if (is_concrete_shape) {
    int64_t start = 0;
    int64_t end = std::numeric_limits<int64_t>::max();

    if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Shape", {15})) {
      // Opset-15 Shape supports slicing using a 'start' and 'end' attribute
      const auto& shape_attributes = node.GetAttributes();
      for (const auto& attr : shape_attributes) {
        if (attr.first == "start") {
          start = attr.second.i();
        } else if (attr.first == "end") {
          end = attr.second.i();
        }
      }
    }
    size_t clamped_slice_length = HandleSliceOrShape15Indices(start, end, dim_values.size());
    CreateInitializerFromShapeVector(graph, node, dim_values, start, clamped_slice_length);
    nodes_to_remove.push_back(&node);
  } else if (shape != nullptr) {
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

      if (graph_utils::IsSupportedOptypeVersionAndDomain(output_node, "Slice", {10, 11, 13})) {
        NodeArg* starts_input = output_node.MutableInputDefs()[1];
        NodeArg* ends_input = output_node.MutableInputDefs()[2];
        NodeArg* axes_input = output_node.MutableInputDefs()[3];
        NodeArg* steps_input = output_node.MutableInputDefs().size() > 4 ? output_node.MutableInputDefs()[4] : nullptr;
        InlinedVector<int64_t> steps_values;
        if (steps_input && !(optimizer_utils::AppendTensorFromInitializer(graph, *steps_input, steps_values, true) &&
                             steps_values.size() == 0 && steps_values[0] == 1)) {
          continue;
        }

        InlinedVector<int64_t> starts_values, ends_values, axes_values;
        // Try to parse int32/int64 type constant initializers.
        if (!(optimizer_utils::AppendTensorFromInitializer(graph, *starts_input, starts_values, true) &&
              optimizer_utils::AppendTensorFromInitializer(graph, *ends_input, ends_values, true) &&
              optimizer_utils::AppendTensorFromInitializer(graph, *axes_input, axes_values, true))) {
          continue;
        }

        // We only support 1D slices currently, can be extended further to support other cases.
        if (!(starts_values.size() == 1 && ends_values.size() == 1 && axes_values.size() == 1 && axes_values[0] == 0)) {
          continue;
        }

        int64_t start = starts_values[0];
        int64_t end = ends_values[0];
        size_t clamped_slice_length = HandleSliceOrShape15Indices(start, end, dim_values.size());

        // If requested shape is missing, then we can't do anything.
        if (dim_values[start] < 0) {
          continue;
        }

        CreateInitializerFromShapeVector(graph, output_node, dim_values, start, clamped_slice_length);
        nodes_to_remove.push_back(&output_node);

      } else if (graph_utils::IsSupportedOptypeVersionAndDomain(output_node, "Gather", {11, 13})) {
        NodeArg* indices_input = output_node.MutableInputDefs()[1];
        auto indices_shape = indices_input->Shape();
        if (!indices_shape || indices_shape->dim_size() > 1) {
          // If the indices did not contain one single element, then skip it.
          continue;
        }

        // Try to parse int64 type constant initializers.
        InlinedVector<int64_t> indices_values;
        if (!optimizer_utils::AppendTensorFromInitializer(graph, *indices_input, indices_values, true)) {
          continue;
        }

        const ONNX_NAMESPACE::AttributeProto* axis_attr;
        if ((axis_attr = graph_utils::GetNodeAttribute(output_node, "axis")) &&
            static_cast<int>(axis_attr->i()) != 0) {
          continue;
        }

        // We only support 1D slices currently, can be extended further to support other cases.
        if (!(indices_values.size() == 1 && dim_values[indices_values[0]] > 0)) {
          continue;
        }

        int64_t start = indices_values[0];
        int64_t rank = static_cast<int64_t>(dim_values.size());
        start = start < 0 ? start + rank : start;
        size_t clamped_slice_length = 1;

        int gather_output_rank = 1 /* gather input data is a 1-D tensor representing a shape */ +
                                 indices_shape->dim_size() - 1;
        CreateInitializerFromShapeVector(graph, output_node, dim_values, start, clamped_slice_length,
                                         gather_output_rank == 0);
        nodes_to_remove.push_back(&output_node);
      }
    }
  }
  return is_concrete_shape               // all dimensions are concrete values.
         || nodes_to_remove.size() > 0;  // OR concrete dim values usage can be constant folded.
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
      converted_to_constant = ConstantFoldShapeNode(graph, *node, nodes_to_remove);
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
        std::cout << "Removing Node named " << node_to_remove->Name() << node_to_remove->OpType() << std::endl;
        graph.RemoveNode(node_to_remove->Index());
        modified = true;
        have_updated_nodes = true;
      }
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
