// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "core/optimizer/constant_folding.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/optimizer/utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/common/safeint.h"
#include "core/common/parse_string.h"

using namespace onnxruntime::common;

namespace onnxruntime {

ConstantFolding::ConstantFolding(const IExecutionProvider& execution_provider,
                                 bool skip_dequantize_linear,
                                 const ConfigOptions& config_options,
                                 const InlinedHashSet<std::string_view>& compatible_execution_providers,
                                 const InlinedHashSet<std::string>& excluded_initializers) noexcept
    : ConstantFolding("ConstantFolding", execution_provider, skip_dequantize_linear, config_options, compatible_execution_providers, excluded_initializers) {
}

ConstantFolding::ConstantFolding(const std::string& name,
                                 const IExecutionProvider& execution_provider,
                                 bool skip_dequantize_linear,
                                 const ConfigOptions& config_options,
                                 const InlinedHashSet<std::string_view>& compatible_execution_providers,
                                 const InlinedHashSet<std::string>& excluded_initializers) noexcept
    : GraphTransformer(name, compatible_execution_providers),
      skip_dequantize_linear_(skip_dequantize_linear),
      config_options_(config_options),
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
    utils::SetRawDataInTensorProto(shape_constant, dim_values.data() + start, clamped_slice_length * sizeof(int64_t));
    ONNX_NAMESPACE::TensorShapeProto result_shape;
    result_shape.add_dim()->set_dim_value(clamped_slice_length);
    constant_arg_out->SetShape(result_shape);
    graph_utils::AddInitializerWithOrtValue(graph, shape_constant);
  }

  return is_concrete_shape;  // convert to constant if this is true
}

// This function inlines the appropriate subgraph. It does not literally fold it.
static Status ConstantFoldIfNode(Graph& graph, Node& if_node, const logging::Logger& logger, bool& folded) {
  folded = false;
  // First, find out which subgraph to inline
  // We need to fetch the constant argument.
  assert(if_node.InputDefs().size() == 1);
  const auto* condition_def = if_node.InputDefs()[0];

  // We need to check if the condition is a constant.
  constexpr bool check_outer_scope_true = true;
  const ONNX_NAMESPACE::TensorProto* initializer =
      graph.GetConstantInitializer(condition_def->Name(), check_outer_scope_true);
  if (initializer == nullptr) {
    return Status::OK();
  }

  // This is a boolean initializer with a single element.
  Initializer condition{graph, *initializer};
  ORT_RETURN_IF_NOT(condition.size() == 1, "If node condition initializer: `", condition_def->Name(),
                    "' is expected to have a single boolean element");

  const bool condition_value = *condition.data<bool>();

  auto status = graph.InlineIfSubgraph(condition_value, if_node, logger);

  if (!status.IsOK()) {
    LOGS(logger, WARNING) << "Unable to constant fold. InlineIfSubgraph failed "
                          << " node '" << if_node.Name() << "': "
                          << status.ErrorMessage();
    return status;
  }

  graph_utils::RemoveNodeOutputEdges(graph, if_node);
  graph.RemoveNode(if_node.Index());

  folded = true;
  return status;
}

// Default maximum output size per constant-folded node: 1 GB.
// This prevents malicious models from causing excessive memory allocation during optimization.
static constexpr int64_t kDefaultConstantFoldingMaxOutputSizeInBytes = 1024 * 1024 * 1024;

// Estimate the total output size in bytes for a node using shape inference results.
// Returns -1 if the output size cannot be estimated (e.g., unknown shapes or types).
static int64_t EstimateNodeOutputSizeInBytes(const Node& node) {
  SafeInt<int64_t> total_size = 0;

  for (const auto* output_def : node.OutputDefs()) {
    if (!output_def->Exists()) {
      continue;
    }

    const auto* type_proto = output_def->TypeAsProto();
    if (type_proto == nullptr || !utils::HasTensorType(*type_proto)) {
      return -1;  // Cannot estimate non-tensor or unknown types
    }

    const auto* shape = output_def->Shape();
    if (shape == nullptr) {
      return -1;  // Unknown shape
    }

    auto elem_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
        type_proto->tensor_type().elem_type());
    size_t element_size = utils::GetElementSizeOfTensor(elem_type);
    if (element_size == 0) {
      return -1;  // Unknown element type
    }

    SafeInt<int64_t> num_elements = 1;
    for (int i = 0; i < shape->dim_size(); ++i) {
      const auto& dim = shape->dim(i);
      if (!utils::HasDimValue(dim)) {
        return -1;  // Symbolic dimension
      }
      int64_t dim_value = dim.dim_value();
      if (dim_value < 0) {
        return -1;  // Invalid dimension
      }
      num_elements *= dim_value;
    }

    total_size += num_elements * static_cast<int64_t>(element_size);
  }

  return total_size;
}

// Get the configured max output size from session options, or use the default.
static int64_t GetConstantFoldingMaxOutputSize(const ConfigOptions& config_options) {
  std::string max_size_str = config_options.GetConfigOrDefault(
      kOrtSessionOptionsConstantFoldingMaxOutputSizeInBytes,
      std::to_string(kDefaultConstantFoldingMaxOutputSizeInBytes));

  int64_t max_size = 0;
  if (!TryParseStringWithClassicLocale(max_size_str, max_size) || max_size < 0) {
    max_size = kDefaultConstantFoldingMaxOutputSizeInBytes;
  }

  return max_size;
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

  const int64_t max_output_size = GetConstantFoldingMaxOutputSize(config_options_);

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);
    if (!node || !AllowConstantFolding(*node)) {
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
    if (node->OpType().compare("If") == 0) {
      // This process constant folds the If node only,
      // but inlines the nodes of the corresponding branch graph.
      // It does not convert the node to a constant in a common sense.
      // We call it constant folding because the `If` node constant condition
      // may enable us to inline the corresponding branch graph.
      bool folded = false;
      ORT_RETURN_IF_ERROR(ConstantFoldIfNode(graph, *node, logger, folded));
      if (folded) {
        // Node removal is done within ConstantFoldIfNode()
        modified = true;
        have_updated_nodes = true;
      }
    } else if (node->OpType().compare("Shape") == 0) {
      converted_to_constant = ConstantFoldShapeNode(graph, *node);
    } else {
      InitializedTensorSet constant_inputs;

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

      // Check if the estimated output size exceeds the configured limit.
      // This prevents malicious models from causing excessive memory allocation during constant folding.
      if (max_output_size > 0) {
        int64_t estimated_size = -1;
        try {
          estimated_size = EstimateNodeOutputSizeInBytes(*node);
        } catch (const std::exception&) {
          // SafeInt overflow means the size is astronomically large - definitely skip
          LOGS(logger, WARNING) << "Integer overflow while estimating output size of "
                                << node->OpType() << " node '" << node->Name()
                                << "'. Skipping constant folding for this node.";
          continue;
        }

        if (estimated_size > max_output_size) {
          LOGS(logger, WARNING) << "Skipping constant folding for " << node->OpType()
                                << " node '" << node->Name()
                                << "' because estimated output size (" << estimated_size
                                << " bytes) exceeds the limit (" << max_output_size << " bytes).";
          continue;
        }
        // If estimated_size is -1, we couldn't estimate; we'll check actual size after execution.
      }

#if !defined(DISABLE_SPARSE_TENSORS)
      // Create execution frame for executing constant nodes.
      OptimizerExecutionFrame::Info info({node}, constant_inputs, graph.ModelPath(), execution_provider_,
                                         is_sparse_initializer_check, logger);
#else
      // Create execution frame for executing constant nodes.
      OptimizerExecutionFrame::Info info(
          {node}, constant_inputs, graph.ModelPath(), execution_provider_, [](const std::string&) { return false; },
          logger);
#endif

      std::vector<int> fetch_mlvalue_idxs;
      std::vector<size_t> fetch_to_output_idx;
      fetch_mlvalue_idxs.reserve(node->OutputDefs().size());
      fetch_to_output_idx.reserve(node->OutputDefs().size());

      for (size_t output_idx = 0; output_idx < node->OutputDefs().size(); ++output_idx) {
        const auto* node_out = node->OutputDefs()[output_idx];
        if (!node_out->Exists()) {
          continue;
        }

        const int ort_value_idx = info.GetMLValueIndex(node_out->Name());
        if (ort_value_idx < 0) {
          LOGS(logger, INFO) << "Skipping constant folding for " << node->OpType()
                             << " node '" << node->Name()
                             << "' because some outputs are not present in the graph.";
          fetch_mlvalue_idxs.clear();
          fetch_to_output_idx.clear();
          break;
        }

        fetch_mlvalue_idxs.push_back(ort_value_idx);
        fetch_to_output_idx.push_back(output_idx);
      }

      if (fetch_mlvalue_idxs.empty()) {
        continue;
      }

      const bool node_on_cpu_ep = node->GetExecutionProviderType() == kCpuExecutionProvider;

      std::unique_ptr<const OpKernel> kernel;

      if (!node_on_cpu_ep) {
        // We need to copy the string here instead of taking a reference to it since node->SetExecutionProviderType
        // will change the value of the reference
        auto ep_type = node->GetExecutionProviderType();

        // override the EP assigned to the node so that it will use the CPU kernel for Compute.
        node->SetExecutionProviderType(kCpuExecutionProvider);

        kernel = info.CreateKernel(node, config_options_);

        // undo the EP change to the value that was assigned at graph partitioning time
        node->SetExecutionProviderType(ep_type);
      } else {
        kernel = info.CreateKernel(node, config_options_);
      }

      // We currently constant fold using the CPU EP only.
      // If we can't find a CPU kernel for this node, then we can't proceed with constant folding.
      //
      // TODO(adrianlizarraga): Support constant folding with other execution providers. For example, we may be able
      // to use a CUDA kernel to constant fold operators with data types not supported by the CPU EP kernel.
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

      // Wrap Compute in try/catch so that overflows (e.g., SafeInt) or other failures in a
      // single node don't abort the entire constant folding pass.
      try {
        ORT_RETURN_IF_ERROR(kernel->Compute(&op_kernel_context));
      } catch (const std::exception& ex) {
        LOGS(logger, WARNING) << "Exception during constant folding of " << node->OpType()
                              << " node '" << node->Name() << "': " << ex.what()
                              << ". Skipping constant folding for this node.";
        continue;
      }
#ifdef _WIN32
#pragma warning(pop)
#endif

      std::vector<OrtValue> fetches;
      ORT_RETURN_IF_ERROR(frame.GetOutputs(fetches));

      // Post-execution size check: verify actual output sizes don't exceed the limit.
      // This catches cases where pre-execution shape inference couldn't determine the output size.
      if (max_output_size > 0) {
        SafeInt<int64_t> actual_total_size = 0;
        bool size_exceeded = false;
        try {
          for (size_t fetch_idx = 0; fetch_idx < fetches.size(); ++fetch_idx) {
            if (fetches[fetch_idx].IsTensor()) {
              const auto& tensor = fetches[fetch_idx].Get<Tensor>();
              actual_total_size += tensor.SizeInBytes();
            }
          }
          size_exceeded = actual_total_size > max_output_size;
        } catch (const std::exception&) {
          // SafeInt overflow means total size is astronomically large
          size_exceeded = true;
        }

        if (size_exceeded) {
          LOGS(logger, WARNING) << "Skipping constant folding for " << node->OpType()
                                << " node '" << node->Name()
                                << "' because actual output size exceeds the limit ("
                                << max_output_size << " bytes).";
          continue;
        }
      }

      // Go over all output node args and substitute them with the newly computed tensors, which will be
      // added to the graph as initializers.
      ORT_ENFORCE(fetches.size() == fetch_to_output_idx.size());
      converted_to_constant = true;
      for (size_t fetch_idx = 0; fetch_idx < fetches.size(); ++fetch_idx) {
        const auto output_idx = fetch_to_output_idx[fetch_idx];
        const auto& constant_arg_out = *node->OutputDefs()[output_idx];
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
          const auto output_idx = fetch_to_output_idx[fetch_idx];
          // Build the TensorProto that corresponds to the computed OrtValue and add it as initializer to the graph.
          auto* constant_arg_out = node->MutableOutputDefs()[output_idx];
          const Tensor& out_tensor = ort_value.Get<Tensor>();
          constexpr const bool use_tensor_buffer_true = true;
          ONNX_NAMESPACE::TensorProto out_tensorproto = utils::TensorToTensorProto(
              out_tensor,
              constant_arg_out->Name(),
              use_tensor_buffer_true);

          ONNX_NAMESPACE::TensorShapeProto result_shape;
          for (auto& dim : out_tensor.Shape().GetDims()) {
            result_shape.add_dim()->set_dim_value(dim);
          }

          constant_arg_out->SetShape(result_shape);
          // The data is too small and has been inlined.
          if (!utils::HasExternalData(out_tensorproto)) {
            ORT_THROW_IF_ERROR(graph.AddInitializedOrtValue(out_tensorproto, OrtValue()));
          } else {
            ORT_THROW_IF_ERROR(graph.AddInitializedOrtValue(out_tensorproto, ort_value));
          }
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
