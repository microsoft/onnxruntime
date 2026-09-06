// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <optional>

#include "core/optimizer/stft_decomposition.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/optimizer/utils.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include <numbers>

using namespace onnxruntime::common;

namespace onnxruntime {
namespace {

constexpr size_t kMaxSTFTConvWeightSizeInBytes = 64 * 1024 * 1024;

std::optional<int64_t> ReadScalarIntegerInitializer(const Graph& graph,
                                                    const ONNX_NAMESPACE::TensorProto* initializer) {
  if (initializer == nullptr) {
    return std::nullopt;
  }

  const Initializer tensor(*initializer, graph.ModelPath());
  if (tensor.size() != 1) {
    return std::nullopt;
  }

  switch (initializer->data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return static_cast<int64_t>(*tensor.data<int32_t>());
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return *tensor.data<int64_t>();
    default:
      return std::nullopt;
  }
}

}  // namespace

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
  utils::SetRawDataInTensorProto(proto, begin, element_count * sizeof(TDataType));
  return &graph_utils::AddInitializerWithOrtValue(graph, proto);
}

template <size_t TDims>
NodeArg* AddShapeInitializer(Graph& graph, const char* name, const int64_t (&shape)[TDims]) {
  int64_t shape_shape[] = {TDims};
  return AddInitializer<int64_t>(graph, name, shape_shape, shape);
}

std::pair<Node*, NodeArg*> AddNode(Graph& graph,
                                   const char* op_type,
                                   ProviderType execution_provider_type,
                                   gsl::span<NodeArg*> inputs,
                                   const Node* annotation_source = nullptr) {
  auto def_name = graph.GenerateNodeArgName(op_type);
  auto node_arg = &graph.GetOrCreateNodeArg(def_name, nullptr);
  Node& node = annotation_source
                   ? graph.AddNode(graph.GenerateNodeName(op_type),
                                   op_type,
                                   "",
                                   inputs,
                                   {node_arg},
                                   *annotation_source)
                   : graph.AddNode(graph.GenerateNodeName(op_type),
                                   op_type,
                                   "",
                                   inputs,
                                   {node_arg});
  node.SetExecutionProviderType(execution_provider_type);
  return std::make_pair(&node, node_arg);
}

std::pair<Node*, NodeArg*> AddNodeCast(Graph& graph, NodeArg* in,
                                       ONNX_NAMESPACE::TensorProto_DataType data_type,
                                       const Node* annotation_source = nullptr) {
  auto def_name = graph.GenerateNodeArgName("Cast");
  auto node_arg = &graph.GetOrCreateNodeArg(def_name, nullptr);
  Node& node = annotation_source
                   ? graph.AddNode(graph.GenerateNodeName("Cast"),
                                   "Cast",
                                   "",
                                   {in},
                                   {node_arg},
                                   *annotation_source)
                   : graph.AddNode(graph.GenerateNodeName("Cast"),
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
    After Fusion when the window is folded into the Conv weights:
              [root]--(signal)-->[Reshape]-->[Conv]-->[Reshape]-->[Transpose]--(output)-->

    After Fusion when the window remains a graph input:
              [root]--(signal)------------------>[Reshape]------—----+
              [root]--(window)-->[optional Cast]-->[Reshape]-->[Mul]-+
                                                                     |
                                                                     v
                                                                    [Conv]-->[Reshape]-->[Transpose]--(output)-->


    Subgraph pattern 2: STFT without optional Window parameter set
              [root]--(signal)-------------------+
              [root]--(frame_step)--------------+|
              [root]                             |
              [root]--(frame_length) ----------+||
                                               |||
                                               vvv
                                              [STFT]--(output)-->
    After Fusion:
              [root]--(signal)-->[Reshape]-->[Conv]-->[Reshape]-->[Transpose]--(output)-->
*/
Status STFTDecomposition::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();
  const auto& compatible_eps = GetCompatibleExecutionProviders();
  const bool may_run_on_cpu = compatible_eps.empty() || compatible_eps.find(kCpuExecutionProvider) != compatible_eps.end();

  for (NodeIndex i : order) {
    auto node = graph.GetNode(i);
    CONTINUE_IF_NULL(node);
    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*node, "STFT", {17}) ||
        (!node->GetExecutionProviderType().empty() && !graph_utils::IsSupportedProvider(*node, compatible_eps))) {
      continue;
    }

    Node& stft = *node;
    if (stft.InputDefs().size() < 4 || stft.OutputDefs().empty()) {
      continue;
    }

    auto signal = stft.MutableInputDefs()[0];
    auto frame_step = stft.MutableInputDefs()[1];
    auto window = stft.MutableInputDefs()[2];
    auto frame_length = stft.MutableInputDefs()[3];

    const auto* signal_type = signal->TypeAsProto();
    if (signal_type == nullptr || !signal_type->has_tensor_type()) {
      continue;
    }

    const auto* signal_shape = signal->Shape();
    if (signal_shape == nullptr || signal_shape->dim_size() < 2 || signal_shape->dim_size() > 3) {
      continue;
    }

    auto batch_size_dim = signal_shape->dim(0);
    auto signal_length_dim = signal_shape->dim(1);
    CONTINUE_IF_NO_DIM_VALUE(signal_length_dim);

    auto batch_size = batch_size_dim.has_dim_value() ? batch_size_dim.dim_value() : static_cast<int64_t>(-1);
    auto signal_length = signal_length_dim.dim_value();
    auto is_real = signal_shape->dim_size() == 2 ||
                   (signal_shape->dim_size() == 3 &&
                    signal_shape->dim(2).has_dim_value() &&
                    signal_shape->dim(2).dim_value() == 1);
    auto data_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(signal_type->tensor_type().elem_type());
    if (!is_real || (may_run_on_cpu && data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT)) {
      continue;
    }

    auto frame_step_initializer = graph_utils::GetConstantInitializer(graph, frame_step->Name());
    auto window_initializer = window->Exists() ? graph_utils::GetConstantInitializer(graph, window->Name()) : nullptr;
    auto frame_length_initializer = frame_length->Exists() ? graph_utils::GetConstantInitializer(graph, frame_length->Name()) : nullptr;
    CONTINUE_IF_NULL(frame_step_initializer);
    const auto* window_type = window->Exists() ? window->TypeAsProto() : nullptr;
    if (window->Exists() && (window_type == nullptr || !window_type->has_tensor_type())) {
      continue;
    }
    if (!frame_length_initializer && !window_initializer) {
      continue;
    }

    auto frame_step_value = ReadScalarIntegerInitializer(graph, frame_step_initializer);
    if (!frame_step_value.has_value()) {
      continue;
    }

    // Get DFT Size
    int64_t dft_size = 0;
    if (frame_length_initializer) {
      auto frame_length_value = ReadScalarIntegerInitializer(graph, frame_length_initializer);
      if (!frame_length_value.has_value()) {
        continue;
      }
      dft_size = *frame_length_value;
    }
    if (!frame_length_initializer && window_initializer) {
      const auto* window_shape = window->Shape();
      if (window_shape == nullptr || window_shape->dim_size() != 1) {
        continue;
      }
      auto window_length_dim = window_shape->dim(0);
      CONTINUE_IF_NO_DIM_VALUE(window_length_dim);
      dft_size = window_length_dim.dim_value();
    }

    // Validate model-provided scalar values before using them in size calculations.
    // These come from untrusted model initializers/shapes and must be positive.
    if (dft_size <= 0 || *frame_step_value <= 0) {
      LOGS(logger, WARNING) << "STFT decomposition skipped: invalid dft_size (" << dft_size
                            << ") or frame_step_value (" << *frame_step_value << ")";
      continue;
    }

    if (dft_size > signal_length) {
      continue;
    }

    bool is_onesided = true;
    auto& attrs = stft.GetAttributes();
    if (attrs.find("onesided") != attrs.end()) {
      auto& onesided_attr = attrs.at("onesided");
      if (utils::HasInt(onesided_attr)) {
        is_onesided = static_cast<bool>(onesided_attr.i());
      }
    }

    const int64_t output_num_frames = ((signal_length - dft_size) / *frame_step_value) + 1;
    auto dft_unique_bins = is_onesided ? ((dft_size >> 1) + 1) : dft_size;

    Node* signal_recipient = nullptr;
    Node* window_recipient = nullptr;
    Node* stft_producer = nullptr;
    if (is_real) {
      size_t dft_size_sz, dft_unique_bins_sz, weight_size, conv_channels, weight_size_in_bytes;
      if (!SafeCast(dft_unique_bins, dft_unique_bins_sz) ||
          !SafeCast(dft_size, dft_size_sz) ||
          !SafeMultiply(dft_unique_bins_sz, static_cast<size_t>(2), conv_channels) ||
          !SafeMultiply(conv_channels, dft_size_sz, weight_size) ||
          !SafeMultiply(weight_size, sizeof(float), weight_size_in_bytes)) {
        LOGS(logger, WARNING) << "STFT decomposition skipped: weight size overflow";
        continue;
      }
      if (weight_size_in_bytes > kMaxSTFTConvWeightSizeInBytes) {
        LOGS(logger, VERBOSE) << "STFT decomposition skipped: generated Conv weights would require "
                              << weight_size_in_bytes << " bytes";
        continue;
      }

      auto weights_data = std::vector<float>(weight_size);
      const float* window_data = nullptr;
      std::unique_ptr<Initializer> window_tensor;
      if (window_initializer != nullptr && data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        if (window_initializer->data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
            window_initializer->dims_size() != 1 ||
            window_initializer->dims(0) != dft_size) {
          continue;
        }
        window_tensor = std::make_unique<Initializer>(*window_initializer, graph.ModelPath());
        window_data = window_tensor->data<float>();
      }

      // Populate weights
      for (size_t k = 0; k < dft_unique_bins_sz; k++) {
        for (size_t n = 0; n < dft_size_sz; n++) {
          auto real_index = k * dft_size_sz + n;
          auto imag_index = (dft_unique_bins_sz + k) * dft_size_sz + n;
          auto theta = -2 * std::numbers::pi_v<float> * k * n / static_cast<float>(dft_size);
          auto window_scale = window_data != nullptr ? window_data[n] : 1.0f;
          weights_data[real_index] = static_cast<float>(cos(theta)) * window_scale;
          weights_data[imag_index] = static_cast<float>(sin(theta)) * window_scale;
        }
      }

      const int64_t weight_shape[] = {2 * dft_unique_bins, 1, 1, dft_size};
      auto* weights = AddInitializer<float>(graph, "stft_conv_weights", weight_shape, weights_data.data());

      const int64_t signal_reshaped[] = {batch_size, 1, 1, signal_length};
      auto signal_shape = AddShapeInitializer(graph, "stft_signal_shape", signal_reshaped);

      const int64_t output_reshape_shape[] = {batch_size, 2, dft_unique_bins, output_num_frames};
      auto output_shape = AddShapeInitializer(graph, "stft_output_reshaped", output_reshape_shape);

      NodeArg* signal_reshaped_inputs[] = {signal, signal_shape};
      Node* reshape_signal_node = nullptr;
      NodeArg* reshape_output = nullptr;
      std::tie(reshape_signal_node, reshape_output) =
          AddNode(graph, "Reshape", stft.GetExecutionProviderType(), signal_reshaped_inputs, &stft);

      NodeArg* weights_final = weights;
      if (window->Exists() && window_data == nullptr) {
        const int64_t window_reshaped_shape[] = {1, 1, 1, dft_size};
        auto window_shape = AddShapeInitializer(graph, "stft_window_shape", window_reshaped_shape);

        auto window_final = window;
        if (window_type->tensor_type().elem_type() != GetDataType<float>()) {
          Node* window_cast_node = nullptr;
          std::tie(window_cast_node, window_final) =
              AddNodeCast(graph, window, GetDataType<float>(), &stft);
          window_recipient = window_cast_node;
        }

        NodeArg* window_reshaped_inputs[] = {window_final, window_shape};
        Node* window_reshape_node;
        NodeArg* window_reshaped = nullptr;
        std::tie(window_reshape_node, window_reshaped) =
            AddNode(graph, "Reshape", kCpuExecutionProvider, window_reshaped_inputs, &stft);
        if (!window_recipient) {
          window_recipient = window_reshape_node;
        }

        NodeArg* scale_weights_inputs[] = {weights, window_reshaped};
        NodeArg* windowed_weights_output = nullptr;
        std::tie(std::ignore, windowed_weights_output) =
            AddNode(graph, "Mul", kCpuExecutionProvider, scale_weights_inputs, &stft);

        weights_final = windowed_weights_output;
      }

      if (data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        std::tie(std::ignore, weights_final) =
            AddNodeCast(graph, weights_final, data_type, &stft);
      }

      NodeArg* conv_inputs[] = {reshape_output, weights_final};
      Node* conv_node = nullptr;
      NodeArg* conv_output = nullptr;
      std::tie(conv_node, conv_output) =
          AddNode(graph, "Conv", stft.GetExecutionProviderType(), conv_inputs, &stft);
      conv_node->AddAttribute("strides", std::vector<int64_t>{1, *frame_step_value});

      NodeArg* output_reshape_inputs[] = {conv_output, output_shape};
      NodeArg* reshaped_output = nullptr;
      std::tie(std::ignore, reshaped_output) =
          AddNode(graph, "Reshape", stft.GetExecutionProviderType(), output_reshape_inputs, &stft);

      NodeArg* transpose_inputs[] = {reshaped_output};
      Node* transpose_node = nullptr;
      NodeArg* transpose_output = nullptr;
      std::tie(transpose_node, transpose_output) =
          AddNode(graph, "Transpose", stft.GetExecutionProviderType(), transpose_inputs, &stft);
      transpose_node->AddAttribute("perm", std::vector<int64_t>{0, 3, 2, 1});

      signal_recipient = reshape_signal_node;
      stft_producer = transpose_node;
    } else {
      continue;
    }

    auto input_edges = graph_utils::GraphEdge::GetNodeInputEdges(stft);
    auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(stft);

    // Copy inputs
    auto signal_target_idx = signal_recipient->Index();
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
          if (window_recipient) {
            target_idx = window_recipient->Index();
            recipient = window_recipient;
          }
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
