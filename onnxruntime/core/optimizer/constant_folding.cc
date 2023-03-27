// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "core/optimizer/constant_folding.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/optimizer/utils.h"
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

// We need to handle a Shape node separately as the input doesn't need to be a constant initializer for
// Shape to be able to be constant folded.
static bool ConstantFoldShapeNode(Graph& graph, Node& node) {
  // Opset-15 Shape supports slicing using a 'start' and 'end' attribute
  const auto& shape_attributes = node.GetAttributes();

  int64_t start = 0;
  int64_t end = std::numeric_limits<int64_t>::max();

  for (const auto& attr : shape_attributes) {
    if (attr.first == "start") {
      start = attr.second.i();
    } else if (attr.first == "end") {
      end = attr.second.i();
    }
  }

  auto shape = node.MutableInputDefs()[0]->Shape();
  bool is_concrete_shape = true;
  std::vector<int64_t> dim_values;
  if (shape != nullptr) {
    for (int dim_index = 0; dim_index < shape->dim_size(); dim_index++) {
      auto dim = shape->dim(dim_index);
      if (!utils::HasDimValue(dim)) {
        is_concrete_shape = false;
        break;
      }
      dim_values.push_back(dim.dim_value());
    }
  } else {
    is_concrete_shape = false;
  }

  if (is_concrete_shape) {
    int64_t rank = static_cast<int64_t>(dim_values.size());

    // We ascertain the "true" starts/ends (if they were provided)
    // Opset-15 Shape op supports slicing shape values

    // Deal with negatives and clamp
    start = start < 0 ? start + rank : start;
    start = start < 0 ? 0 : ((start > rank) ? rank : start);

    end = end < 0 ? end + rank : end;
    end = end < 0 ? 0 : ((end > rank) ? rank : end);

    int64_t slice_length = end - start;
    size_t clamped_slice_length = slice_length < 0 ? 0 : static_cast<size_t>(slice_length);

    ONNX_NAMESPACE::TensorProto shape_constant;
    auto* constant_arg_out = node.MutableOutputDefs()[0];
    shape_constant.set_name(constant_arg_out->Name());
    shape_constant.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    shape_constant.add_dims(clamped_slice_length);
    shape_constant.set_raw_data(dim_values.data() + start,
                                clamped_slice_length * sizeof(int64_t));
    ONNX_NAMESPACE::TensorShapeProto result_shape;
    result_shape.add_dim()->set_dim_value(clamped_slice_length);
    constant_arg_out->SetShape(result_shape);
    graph.AddInitializedTensor(shape_constant);
  }

  return is_concrete_shape;  // convert to constant if this is true
}

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

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    // Updating a node may allow shape inferencing to infer output shapes of following nodes,
    // so re-run the shape inferencing. use have_updated_nodes as that only applies to this Graph
    // (vs. 'modified' which is passed into subgraphs and applies to the main graph and all subgraphs)
    // Ignore any control flow node containing subgraphs as UpdateShapeInference is not intended to be used on it.
    if (have_updated_nodes && !node->ContainsSubgraph()) {
      ORT_RETURN_IF_ERROR(graph.UpdateShapeInference(*node));
    }

    bool converted_to_constant = false;
    if (node->OpType().compare("Shape") == 0) {
      converted_to_constant = ConstantFoldShapeNode(graph, *node);
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
      const auto can_constant_fold_node = [&](const Node& n, bool skip_inputs_constant_check = false) {
        return graph_utils::IsSupportedProvider(n, GetCompatibleExecutionProviders()) &&
               optimizer_utils::IsOperationDeterministic(n.Domain(), n.OpType()) &&
               // constant folding does not support executing a node that includes subgraphs (control flow operators,
               // such as If/Loop/Scan, fall into this category). individual nodes in the subgraph will be processed
               // by the Recurse call above
               !n.ContainsSubgraph() &&
               (skip_inputs_constant_check ||
                graph_utils::AllNodeInputsAreConstant(graph, n, constant_inputs, excluded_initializers_));
      };

      if (!can_constant_fold_node(*node)) {
        continue;
      }

      // if skip_dequantize_linear is true we want to maintain QDQ node units so avoid constant folding
      // DequantizeLinear unless we can fold the whole QDQ node unit
      if (skip_dequantize_linear_ && node->OpType() == "DequantizeLinear") {
        bool can_constant_fold_qdq_node_unit = false;

        // Simplest scenario where the whole QDQ node unit of (DQ -> X -> Q) can be constant folded is if:
        //   - the DQ node does not produce a graph output, and its output is only consumed by X
        //   - X is a deterministic node with a single input and single output
        //   - the output from X is not a graph output and is only consumed by a Q node
        if (optimizer_utils::CheckOutputEdges(graph, *node, 1)) {  // DQ does not produce graph output, single consumer
          const Node& node_x = *node->OutputNodesBegin();
          if (node_x.InputDefs().size() == 1 &&
              node_x.OutputDefs().size() == 1 &&
              optimizer_utils::CheckOutputEdges(graph, node_x, 1)) {
            const Node& probably_q = *node_x.OutputNodesBegin();

            if (probably_q.OpType() == "QuantizeLinear") {
              // the inputs to these nodes are not const yet, but will be if we constant fold,
              // so set skip_const_check to simulate that having happened
              constexpr bool skip_const_check = true;
              can_constant_fold_qdq_node_unit = can_constant_fold_node(node_x, skip_const_check) &&
                                                can_constant_fold_node(probably_q, skip_const_check);
            }
          }
        }

        if (!can_constant_fold_qdq_node_unit) {
          continue;
        }
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
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 6387)
#endif
      OpKernelContext op_kernel_context(&frame, kernel.get(), /*stream*/ nullptr, nullptr, logger);
      ORT_RETURN_IF_ERROR(kernel->Compute(&op_kernel_context));
#ifdef _WIN32
#pragma warning(pop)
#endif

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
          ONNX_NAMESPACE::TensorProto out_tensorproto = utils::TensorToTensorProto(out_tensor, constant_arg_out->Name());

          ONNX_NAMESPACE::TensorShapeProto result_shape;
          for (auto& dim : out_tensor.Shape().GetDims()) {
            result_shape.add_dim()->set_dim_value(dim);
          }

          constant_arg_out->SetShape(result_shape);
          graph.AddInitializedTensor(out_tensorproto);
        }
      }
    }

    if (converted_to_constant) {
      // Remove single-output node chain for inputs of the node
      auto p_ip_node = node->InputNodesBegin();
      const auto p_ip_node_end = node->InputNodesEnd();
      while (p_ip_node != p_ip_node_end) {
        const auto& input_node = *p_ip_node;
        // Update the node iterator before removing the corresponding node because removing
        // the node will invalidate the node iterator
        ++p_ip_node;
        graph_utils::RemoveNodesWithOneOutputBottomUp(graph, input_node);
      }

      // Remove the output edges of the constant node and then remove the node itself.
      graph_utils::RemoveNodeOutputEdges(graph, *node);
      graph.RemoveNode(node->Index());
      modified = true;
      have_updated_nodes = true;
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
