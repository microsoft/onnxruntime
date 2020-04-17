// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/constant_folding.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"

using namespace onnxruntime::common;

namespace onnxruntime {

Status ConstantFolding::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  while (true) {
    bool keep_going = false;

    // Resolve graph to force run shape and type inference.
    ORT_ENFORCE(graph.Resolve().IsOK());

    GraphViewer graph_viewer(graph);
    auto& order = graph_viewer.GetNodesInTopologicalOrder();

    for (NodeIndex i : order) {
      auto* node = graph.GetNode(i);
      if (!node) {
        continue;
      }

      ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

      bool convert_to_constant = false;
      if (node->OpType().compare("Shape") == 0) {
        auto shape = node->MutableInputDefs()[0]->Shape();
        bool is_concrete_shape = true;
        std::vector<int64_t> dim_values;
        if (shape != nullptr) {
          for (int i = 0; i < shape->dim_size(); i++) {
            auto dim = shape->dim(i);
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
          ONNX_NAMESPACE::TensorProto shape_constant;
          auto* constant_arg_out = node->MutableOutputDefs()[0];
          shape_constant.set_name(constant_arg_out->Name());
          shape_constant.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
          shape_constant.add_dims(dim_values.size());
          shape_constant.set_raw_data(dim_values.data(), dim_values.size() * sizeof(int64_t));
          ONNX_NAMESPACE::TensorShapeProto result_shape;
          result_shape.add_dim()->set_dim_value(dim_values.size());
          constant_arg_out->SetShape(result_shape);
          graph.AddInitializedTensor(shape_constant);
          convert_to_constant = true;
        }
      } else {
        InitializedTensorSet constant_inputs;

        // we currently constant fold using the CPU EP only.
        // if the node is assigned to a different EP we can run it if it's an ONNX op as we have CPU based implementations
        // for all ONNX ops. if it's from a different domain we can't.
        // NOTE: This is in addition to the IsSupportedProvider check below which will optionally do further filtering
        // on the EPs we constant fold for.
        auto ep_type = node->GetExecutionProviderType();
        bool cpu_ep = ep_type == kCpuExecutionProvider;
        if (!cpu_ep && node->Domain() != kOnnxDomain) {
          continue;
        }

        // Check if constant folding can be applied on this node.
        if (!graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders()) ||
            excluded_op_types_.find(node->OpType()) != excluded_op_types_.end() ||
            // constant folding does not support executing a node that includes subgraphs (control flow operators,
            // such as If/Loop/Scan, fall into this category). individual nodes in the subgraph will be processed
            // by the Recurse call above
            node->ContainsSubgraph() ||
            !graph_utils::AllNodeInputsAreConstant(graph, *node, constant_inputs, excluded_initializers_)) {
          continue;
        }

        // override the EP while setting up OptimizerExecutionFrame::Info so that it will use the CPU kernel for Compute.
        if (!cpu_ep) {
          node->SetExecutionProviderType(kCpuExecutionProvider);
        }

        // Create execution frame for executing constant nodes.
        OptimizerExecutionFrame::Info info({node}, constant_inputs);

        // undo the EP change in case something fails prior to node removal
        if (!cpu_ep) {
          node->SetExecutionProviderType(ep_type);
        }

        std::vector<int> fetch_mlvalue_idxs;
        for (const auto* node_out : node->OutputDefs()) {
          fetch_mlvalue_idxs.push_back(info.GetMLValueIndex(node_out->Name()));
        }

        OptimizerExecutionFrame frame(info, fetch_mlvalue_idxs);

        auto* kernel = info.GetKernel(node->Index());
        OpKernelContext op_kernel_context(&frame, kernel, nullptr, onnxruntime::logging::LoggingManager::DefaultLogger());

        ORT_RETURN_IF_ERROR(kernel->Compute(&op_kernel_context));

        std::vector<OrtValue> fetches;
        ORT_RETURN_IF_ERROR(frame.GetOutputs(fetches));

        // Go over all output node args and substitute them with the newly computed tensors, which will be
        // added to the graph as initializers.
        ORT_ENFORCE(fetches.size() == node->OutputDefs().size());
        convert_to_constant = true;
        for (size_t fetch_idx = 0; fetch_idx < fetches.size(); ++fetch_idx) {
          OrtValue& ort_value = fetches[fetch_idx];

          if (!ort_value.IsTensor()) {
            LOGS(logger, WARNING) << "Unsupported output type of " << ort_value.Type()
                                  << ". Can't constant fold " << node->OpType() << " node '" << node->Name() << "'";
            convert_to_constant = false;
            break;
          }
        }

        if (convert_to_constant) {
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
        }
      }

      if (convert_to_constant) {
        // Remove the output edges of the constant node and then remove the node itself.
        graph_utils::RemoveNodeOutputEdges(graph, *node);
        graph.RemoveNode(node->Index());
        modified = true;
        keep_going  = true;
      }
    }

    if (!keep_going) {
      break;
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
