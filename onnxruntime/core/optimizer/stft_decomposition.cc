// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "core/optimizer/stft_decomposition.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/optimizer/utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"

using namespace onnxruntime::common;

namespace onnxruntime {

STFTDecomposition::STFTDecomposition(const InlinedHashSet<std::string_view>& compatible_execution_providers) noexcept
    : GraphTransformer("STFTDecomposition", compatible_execution_providers) {
}

template <typename T>
constexpr static ONNX_NAMESPACE::TensorProto_DataType GetDataType() {
  if constexpr (std::is_same<T, float>::value) {
    return ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  } else if constexpr (std::is_same<T, MLFloat16>::value) {
    return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
  } else if constexpr (std::is_same<T, double>::value) {
    return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
  } else if constexpr (std::is_same<T, int64_t>::value) {
    return ONNX_NAMESPACE::TensorProto_DataType_INT64;
  } else {
    throw std::logic_error("Invalid data type requested for STFT decomposition");
  }
}

template <typename TDataType, size_t TDims>
NodeArg* AddInitializer(Graph& graph, const char* name, const int64_t (&shape)[TDims], const TDataType* begin) {
  ONNX_NAMESPACE::TensorProto proto;
  proto.set_name(graph.GenerateNodeArgName(name));
  proto.set_data_type(GetDataType<TDataType>());
  int64_t element_count = 1;
  for (size_t i = 0; i < TDims; i++) {
    element_count *= shape[i];
    proto.add_dims(shape[i]);
  }
  proto.set_raw_data(begin, element_count * sizeof(TDataType));
  return &graph_utils::AddInitializer(graph, proto);
}

template <size_t TDims>
NodeArg* AddShapeInitializer(Graph& graph, const char* name, const int64_t (&shape)[TDims]) {
  int64_t shape_shape[] = {TDims};
  return AddInitializer<int64_t>(graph, name, shape_shape, shape);
}

std::pair<Node*, NodeArg*> AddNode(Graph& graph,
                                   const char* op_type,
                                   ProviderType execution_provider_type,
                                   gsl::span<NodeArg*> inputs) {
  auto def_name = graph.GenerateNodeArgName(op_type);
  auto node_arg = &graph.GetOrCreateNodeArg(def_name, nullptr);
  Node& node = graph.AddNode(graph.GenerateNodeName(op_type),
                             op_type,
                             "",
                             inputs,
                             {node_arg});
  node.SetExecutionProviderType(execution_provider_type);
  return std::make_pair(&node, node_arg);
}

std::pair<Node*, NodeArg*> AddNodeCast(Graph& graph, NodeArg* in,
                                       ONNX_NAMESPACE::TensorProto_DataType data_type) {
  auto def_name = graph.GenerateNodeArgName("Cast");
  auto node_arg = &graph.GetOrCreateNodeArg(def_name, nullptr);
  Node& node = graph.AddNode(graph.GenerateNodeName("Cast"),
                             "Cast",
                             "",
                             {in},
                             {node_arg});
  node.AddAttribute("to", static_cast<int64_t>(data_type));
  node.SetExecutionProviderType(kCpuExecutionProvider);
  return std::make_pair(&node, node_arg);
}

#define CONTINUE_IF_NO_DIM_VALUE(dim) \
  if (!dim.has_dim_value()) {         \
    continue;                         \
  }
#define CONTINUE_IF_NULL(x) \
  if (x == nullptr) {       \
    continue;               \
  }

/*
    This function decomposes a STFT node into a subgraph.
    The decomposition requires that:
      1) The signal input is real valued and not complex valued!
      2) Both (frame_step) *and* either (window or frame_length) inputs must be constant.
    Otherwise the transform will not be applied.

    Subgraph pattern 1: STFT with optional Window parameter set
              [root]--(signal)--------------------+
              [root]--(frame_step)---------------+|
              [root]--(window)------------------+||
              [root]--(frame_length) ----------+|||
                                               ||||
                                               vvvv
                                              [STFT]--(output)-->
    After Fusion:
              [root]--(signal)-------------------------+
              [root]                                   |
              [root]--(window)--+                      |
              [root]            |                      |
                                v                      v
         (only for non-fp32) [Cast]             +--[Reshape]
                                |               |      |
                                v               |      v
                            [Reshape]-->[Mul]---|-->[Conv]-------+
                                |               |                |
                                |               +-----|          |
                                |                     v          v
                                +------>[Mul]------>[Conv]-->[Concat]-->[Reshape]-->[Transpose]--(output)-->


    Subgraph pattern 2: STFT without optional Window parameter set
              [root]--(signal)-------------------+
              [root]--(frame_step)--------------+|
              [root]                             |
              [root]--(frame_length) ----------+||
                                               |||
                                               vvv
                                              [STFT]--(output)-->
    After Fusion:
              [root]--(signal)-->[Reshape]-->[Conv]
              [root]                 |         |
              [root]                 |         v
              [root]                 +------>[Conv]-->[Concat]-->[Reshape]-->[Transpose]--(output)-->
*/
Status STFTDecomposition::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (NodeIndex i : order) {
    auto node = graph.GetNode(i);
    CONTINUE_IF_NULL(node);
    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    if (node->OpType() != "STFT") {
      continue;
    }

    Node& stft = *node;
    auto signal = stft.MutableInputDefs()[0];
    auto frame_step = stft.MutableInputDefs()[1];
    auto window = stft.MutableInputDefs()[2];
    auto frame_length = stft.MutableInputDefs()[3];

    // If the signal has free dimensions, do not transform...
    auto batch_size_dim = signal->Shape()->dim(0);
    auto signal_length_dim = signal->Shape()->dim(1);
    auto signal_components_dim = signal->Shape()->dim(2);
    CONTINUE_IF_NO_DIM_VALUE(signal_length_dim);
    CONTINUE_IF_NO_DIM_VALUE(signal_components_dim);

    auto batch_size = batch_size_dim.has_dim_value() ? batch_size_dim.dim_value() : static_cast<int64_t>(-1);
    auto signal_length = signal_length_dim.dim_value();
    auto is_real = signal_components_dim.dim_value() == 1;
    auto data_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(signal->TypeAsProto()->tensor_type().elem_type());

    auto frame_step_initializer = graph_utils::GetConstantInitializer(graph, frame_step->Name());
    auto window_initializer = graph_utils::GetConstantInitializer(graph, window->Name());
    auto frame_length_initializer = graph_utils::GetConstantInitializer(graph, frame_length->Name());
    CONTINUE_IF_NULL(frame_step_initializer);
    if (!frame_length_initializer && !window_initializer) {
      continue;
    }

    auto read_int64_initializer = [](Graph& graph, const ONNX_NAMESPACE::TensorProto* initializer) {
      return *Initializer(*initializer, graph.ModelPath()).data<int64_t>();
    };
    auto frame_step_value = read_int64_initializer(graph, frame_step_initializer);

    // Get DFT Size
    int64_t dft_size = 0;
    if (frame_length_initializer) {
      dft_size = read_int64_initializer(graph, frame_length_initializer);
    }
    if (dft_size == 0 && window_initializer) {
      auto window_length_dim = window->Shape()->dim(0);
      CONTINUE_IF_NO_DIM_VALUE(window_length_dim);
      dft_size = window_length_dim.dim_value();
    }

    bool is_onesided = true;
    auto& attrs = stft.GetAttributes();
    if (attrs.find("onesided") != attrs.end()) {
      auto& onesided_attr = attrs.at("onesided");
      if (utils::HasInt(onesided_attr)) {
        is_onesided = static_cast<bool>(onesided_attr.i());
      }
    }

    auto dft_unique_bins = is_onesided ? ((dft_size >> 1) + 1) : dft_size;

    Node* signal_recipient = nullptr;
    Node* window_recipient = nullptr;
    Node* stft_producer = nullptr;
    if (is_real) {
      auto output_num_frames = stft.MutableOutputDefs()[0]->Shape()->dim(1).dim_value();
      auto output_frame_length = stft.MutableOutputDefs()[0]->Shape()->dim(2).dim_value();
      auto weight_size = static_cast<size_t>(dft_unique_bins * dft_size);
      auto real_weights_data = std::vector<float>(weight_size);
      auto imag_weights_data = std::vector<float>(weight_size);

      // Populate weights
      for (size_t k = 0; k < static_cast<size_t>(dft_unique_bins); k++) {
        for (size_t n = 0; n < static_cast<size_t>(dft_size); n++) {
          auto index = static_cast<size_t>(k * dft_size + n);
          auto theta = -2 * M_PI * k * n / static_cast<float>(dft_size);
          real_weights_data[index] = static_cast<float>(cos(theta));
          imag_weights_data[index] = static_cast<float>(sin(theta));
        }
      }

      const int64_t weight_shape[] = {dft_unique_bins, 1, 1, dft_size};
      auto real_weights = AddInitializer<float>(graph, "stft_real_conv_weights", weight_shape, real_weights_data.data());
      auto imaginary_weights = AddInitializer<float>(graph, "stft_imaginary_conv_weights", weight_shape, imag_weights_data.data());

      const int64_t signal_reshaped[] = {batch_size, 1, 1, signal_length};
      auto signal_shape = AddShapeInitializer(graph, "stft_signal_shape", signal_reshaped);

      const int64_t unsqueezed_output_shape[] = {2, batch_size, output_frame_length, output_num_frames};
      auto unsqueezed_shape = AddShapeInitializer(graph, "stft_output_reshaped", unsqueezed_output_shape);

      NodeArg* signal_reshaped_inputs[] = {signal, signal_shape};
      Node* reshape_signal_node = nullptr;
      NodeArg* reshape_output = nullptr;
      std::tie(reshape_signal_node, reshape_output) =
          AddNode(graph, "Reshape", stft.GetExecutionProviderType(), signal_reshaped_inputs);

      NodeArg* real_weights_final = real_weights;
      NodeArg* imag_weights_final = imaginary_weights;
      if (!window->Exists()) {
        // When we are missing a window function
        if (real_weights_final->TypeAsProto()->tensor_type().elem_type() != data_type) {
          std::tie(std::ignore, real_weights_final) =
              AddNodeCast(graph, real_weights_final, data_type);
        }
        if (imag_weights_final->TypeAsProto()->tensor_type().elem_type() != data_type) {
          std::tie(std::ignore, imag_weights_final) =
              AddNodeCast(graph, imag_weights_final, data_type);
        }
      } else {
        // When we have a window function
        const int64_t window_reshaped_shape[] = {1, 1, 1, dft_size};
        auto window_shape = AddShapeInitializer(graph, "stft_window_shape", window_reshaped_shape);

        auto window_final = window;
        if (window->TypeAsProto()->tensor_type().elem_type() != GetDataType<float>()) {
          Node* window_cast_node = nullptr;
          std::tie(window_cast_node, window_final) =
              AddNodeCast(graph, window, GetDataType<float>());
          window_recipient = window_cast_node;
        }

        NodeArg* window_reshaped_inputs[] = {window_final, window_shape};
        Node* window_reshape_node;
        NodeArg* window_reshaped = nullptr;
        std::tie(window_reshape_node, window_reshaped) =
            AddNode(graph, "Reshape", kCpuExecutionProvider, window_reshaped_inputs);
        if (!window_recipient) {
          window_recipient = window_reshape_node;
        }

        NodeArg* scale_real_weights_inputs[] = {real_weights, window_reshaped};
        NodeArg* windowed_real_weights_output = nullptr;
        std::tie(std::ignore, windowed_real_weights_output) =
            AddNode(graph, "Mul", kCpuExecutionProvider, scale_real_weights_inputs);

        NodeArg* scale_imag_weights_inputs[] = {imaginary_weights, window_reshaped};
        NodeArg* windowed_imag_weights_output = nullptr;
        std::tie(std::ignore, windowed_imag_weights_output) =
            AddNode(graph, "Mul", kCpuExecutionProvider, scale_imag_weights_inputs);

        std::tie(std::ignore, real_weights_final) =
            AddNodeCast(graph, windowed_real_weights_output, data_type);
        std::tie(std::ignore, imag_weights_final) =
            AddNodeCast(graph, windowed_imag_weights_output, data_type);
      }

      // Add Convolution (reals)
      NodeArg* conv_real_inputs[] = {reshape_output, real_weights_final};
      Node* real_conv_node = nullptr;
      NodeArg* real_conv_output = nullptr;
      std::tie(real_conv_node, real_conv_output) =
          AddNode(graph, "Conv", stft.GetExecutionProviderType(), conv_real_inputs);
      real_conv_node->AddAttribute("strides", std::vector<int64_t>{1, frame_step_value});

      // Add Convolution (imaginary)
      NodeArg* conv_imag_inputs[] = {reshape_output, imag_weights_final};
      Node* imag_conv_node = nullptr;
      NodeArg* imag_conv_output = nullptr;
      std::tie(imag_conv_node, imag_conv_output) =
          AddNode(graph, "Conv", stft.GetExecutionProviderType(), conv_imag_inputs);
      imag_conv_node->AddAttribute("strides", std::vector<int64_t>{1, frame_step_value});

      // Concatenate
      NodeArg* concatenate_inputs[] = {real_conv_output, imag_conv_output};
      Node* concat_node = nullptr;
      NodeArg* concatenated_conv_output = nullptr;
      std::tie(concat_node, concatenated_conv_output) =
          AddNode(graph, "Concat", stft.GetExecutionProviderType(), concatenate_inputs);
      concat_node->AddAttribute("axis", static_cast<int64_t>(0));

      // Unsqueeze Reshape
      NodeArg* unsqueeze_reshape_inputs[] = {concatenated_conv_output, unsqueezed_shape};
      NodeArg* unsqueezed_output = nullptr;
      std::tie(std::ignore, unsqueezed_output) =
          AddNode(graph, "Reshape", stft.GetExecutionProviderType(), unsqueeze_reshape_inputs);

      // Transpose
      NodeArg* transpose_inputs[] = {unsqueezed_output};
      Node* transpose_node = nullptr;
      NodeArg* transpose_output = nullptr;
      std::tie(transpose_node, transpose_output) =
          AddNode(graph, "Transpose", stft.GetExecutionProviderType(), transpose_inputs);
      transpose_node->AddAttribute("perm", std::vector<int64_t>{1, 3, 2, 0});

      signal_recipient = reshape_signal_node;
      stft_producer = transpose_node;
    } else {
      continue;
    }

    auto input_edges = graph_utils::GraphEdge::GetNodeInputEdges(stft);
    auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(stft);

    // Copy inputs
    auto signal_target_idx = signal_recipient->Index();
    auto window_target_idx = window_recipient->Index();
    for (auto cur = input_edges.cbegin(), end = input_edges.cend(); cur != end; ++cur) {
      const graph_utils::GraphEdge& edge = *cur;
      NodeIndex target_idx = 0;
      Node* recipient = nullptr;
      switch (cur->dst_arg_index) {
        case 0:
          target_idx = signal_target_idx;
          recipient = signal_recipient;
          break;
        case 2:
          target_idx = window_target_idx;
          recipient = window_recipient;
          break;
      }

      if (!recipient) {
        continue;
      }

      auto arg_index = graph_utils::GetNodeInputIndexFromInputName(*recipient, edge.arg_name);
      graph.AddEdge(edge.src_node, target_idx, edge.src_arg_index, arg_index);
    }

    // Copy STFT outputs to stft_producer
    stft_producer->MutableOutputDefs() = stft.MutableOutputDefs();
    auto stft_producer_target_idx = stft_producer->Index();
    for (auto cur = output_edges.cbegin(), end = output_edges.cend(); cur != end; ++cur) {
      graph.AddEdge(stft_producer_target_idx, cur->dst_node, cur->src_arg_index, cur->dst_arg_index);
    }

    graph_utils::GraphEdge::RemoveGraphEdges(graph, input_edges);
    graph_utils::GraphEdge::RemoveGraphEdges(graph, output_edges);
    graph.RemoveNode(stft.Index());

    modified = true;
  }
  return Status::OK();
}
}  // namespace onnxruntime
