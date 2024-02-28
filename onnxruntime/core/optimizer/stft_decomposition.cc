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
    : GraphTransformer("STFTDecomposition", compatible_execution_providers)
{
}

Status STFTDecomposition::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);
    if (!node) {
      continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    const ONNX_NAMESPACE::TensorProto* frame_step_initializer = nullptr;
    const ONNX_NAMESPACE::TensorProto* frame_length_initializer = nullptr;
    const ONNX_NAMESPACE::TensorProto* window_initializer = nullptr;

    const auto can_decompose = [&](const Node& n) {
      if (n.OpType() != "STFT" ||
          !graph_utils::IsSupportedProvider(n, GetCompatibleExecutionProviders()))
      {
        return false;
      }

      frame_step_initializer = graph_utils::GetConstantInitializer(graph, n.InputDefs()[1]->Name());
      window_initializer = graph_utils::GetConstantInitializer(graph, n.InputDefs()[2]->Name());
      frame_length_initializer = graph_utils::GetConstantInitializer(graph, n.InputDefs()[3]->Name());

      return frame_step_initializer &&
             (frame_length_initializer || window_initializer);
    };

    if (!can_decompose(*node)) {
      continue;
    }

    Node& stft = *node;

    Initializer frame_step_init_const{*frame_step_initializer, graph.ModelPath()};
    auto frame_step = *(frame_step_init_const.data<int64_t>());

    // Get DFT Size
    int64_t dft_size = 0;
    if (frame_length_initializer)
    {
      Initializer frame_length_init_const{*frame_length_initializer, graph.ModelPath()};
      dft_size = *(frame_length_init_const.data<int64_t>());
    }
    if (dft_size == 0 && window_initializer)
    {
      Initializer window_init_const{*window_initializer, graph.ModelPath()};
      dft_size = window_init_const.dims()[0];
    }

    bool is_onesided = true;
    auto& attrs = stft.GetAttributes();
    if (attrs.find("onesided") != attrs.end()) {
      auto& onesided_attr = attrs.at("onesided");
      if (utils::HasInt(onesided_attr))
      {
        is_onesided = static_cast<bool>(onesided_attr.i());
      }
    }

    auto dft_unique_bins =
      is_onesided ? ((dft_size >> 1) + 1) : dft_size;

    const NodeArg* signal = stft.InputDefs()[0];
    auto batch_size_dim = signal->Shape()->dim(0);
    auto signal_length_dim = signal->Shape()->dim(1);
    auto signal_real_vs_complex_dim = signal->Shape()->dim(2);
    if (!signal_length_dim.has_dim_value() ||
        !signal_real_vs_complex_dim.has_dim_value()) {
      continue;
    }

    auto batch_size = batch_size_dim.dim_value();
    auto signal_length = signal_length_dim.dim_value();
    auto is_real = (signal_real_vs_complex_dim.dim_value() == 1);
    auto data_type = signal->TypeAsProto()->tensor_type().elem_type();

    Node* first_replacement;
    Node* last_replacement;

    if (is_real) {
      auto weight_size = dft_unique_bins * dft_size;
      std::vector<float> real_weights(weight_size, 1);
      std::vector<float> imag_weights(weight_size, 1);

      // Populate weights
      for (size_t k = 0; k < static_cast<size_t>(dft_unique_bins); k++) {
        for (size_t n = 0; n < static_cast<size_t>(dft_size); n++) {
          auto index = k * dft_size + n;
          auto theta = -2 * 3.14159 * k * n / static_cast<float>(dft_size);
          real_weights[index] = static_cast<float>(cos(theta));
          imag_weights[index] = static_cast<float>(sin(theta));
        }
      }

      ONNX_NAMESPACE::TensorProto real_weights_initializer;
      real_weights_initializer.set_name(graph.GenerateNodeArgName("STFT_real_conv_weights"));
      real_weights_initializer.set_data_type(data_type);
      real_weights_initializer.add_dims(dft_unique_bins); // FILTER_OUT_CHANNEL
      real_weights_initializer.add_dims(1); // FILTER_IN_CHANNEL
      real_weights_initializer.add_dims(dft_size); // FILTER_SPATIAL
      real_weights_initializer.set_raw_data(real_weights.data(), real_weights.size() * sizeof(float));
      NodeArg& real_weights_node_arg = graph_utils::AddInitializer(graph, real_weights_initializer);

      ONNX_NAMESPACE::TensorProto imaginary_weights_initializer;
      imaginary_weights_initializer.set_name(graph.GenerateNodeArgName("STFT_imaginary_conv_weights"));
      imaginary_weights_initializer.set_data_type(data_type);
      imaginary_weights_initializer.add_dims(dft_unique_bins); // FILTER_OUT_CHANNEL
      imaginary_weights_initializer.add_dims(1); // FILTER_IN_CHANNEL
      imaginary_weights_initializer.add_dims(dft_size); // FILTER_SPATIAL
      imaginary_weights_initializer.set_raw_data(imag_weights.data(), imag_weights.size() * sizeof(float));
      NodeArg& imaginary_weights_node_arg = graph_utils::AddInitializer(graph, imaginary_weights_initializer);

      ONNX_NAMESPACE::TensorProto signal_shape_tensor_proto;
      signal_shape_tensor_proto.set_name(graph.GenerateNodeArgName("STFTSignalReshaped"));
      signal_shape_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
      signal_shape_tensor_proto.add_int64_data(batch_size);     // BATCH
      signal_shape_tensor_proto.add_int64_data(1);              // CHANNEL
      signal_shape_tensor_proto.add_int64_data(signal_length);  // SPATIAL
      signal_shape_tensor_proto.add_dims(3);
      NodeArg& signal_shape_node_arg = graph_utils::AddInitializer(graph, signal_shape_tensor_proto);

      ONNX_NAMESPACE::TensorProto output_shape_tensor_proto;
      output_shape_tensor_proto.set_name(graph.GenerateNodeArgName("STFTOutputReshaped"));
      output_shape_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
      output_shape_tensor_proto.add_int64_data(stft.MutableOutputDefs()[0]->Shape()->dim(0).dim_value());  // BATCH
      output_shape_tensor_proto.add_int64_data(stft.MutableOutputDefs()[0]->Shape()->dim(1).dim_value());  // FRAMES
      output_shape_tensor_proto.add_int64_data(stft.MutableOutputDefs()[0]->Shape()->dim(2).dim_value());  // FRAME_LENGTH
      output_shape_tensor_proto.add_int64_data(stft.MutableOutputDefs()[0]->Shape()->dim(3).dim_value());  // 2
      output_shape_tensor_proto.add_dims(4);
      NodeArg& output_shape_node_arg = graph_utils::AddInitializer(graph, output_shape_tensor_proto);

      ONNX_NAMESPACE::TensorProto unsqueezed_shape_tensor_proto;
      unsqueezed_shape_tensor_proto.set_name(graph.GenerateNodeArgName("STFTUnsqueezedOutputReshaped"));
      unsqueezed_shape_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
      unsqueezed_shape_tensor_proto.add_int64_data(2);                                                         // 2
      unsqueezed_shape_tensor_proto.add_int64_data(stft.MutableOutputDefs()[0]->Shape()->dim(0).dim_value());  // BATCH
      unsqueezed_shape_tensor_proto.add_int64_data(stft.MutableOutputDefs()[0]->Shape()->dim(2).dim_value());  // FRAME_LENGTH
      unsqueezed_shape_tensor_proto.add_int64_data(stft.MutableOutputDefs()[0]->Shape()->dim(1).dim_value());  // FRAMES
      unsqueezed_shape_tensor_proto.add_dims(4);
      NodeArg& unsqueezed_shape_node_arg = graph_utils::AddInitializer(graph, unsqueezed_shape_tensor_proto);

      std::string reshape_input_def_name = graph.GenerateNodeArgName("signal_reshaped");
      auto* reshape_output_node_arg = &graph.GetOrCreateNodeArg(reshape_input_def_name, nullptr);
      auto signal_node_arg = stft.MutableInputDefs()[0];
      Node& reshape_signal_node = graph.AddNode(graph.GenerateNodeName("STFT_reshape_signal"),
                                    "Reshape",
                                    "STFT Reshape Signal",
                                    {signal_node_arg, &signal_shape_node_arg},
                                    {reshape_output_node_arg});
      reshape_signal_node.SetExecutionProviderType(stft.GetExecutionProviderType());

      NodeArg* real_weights_final_node_arg = &real_weights_node_arg;
      NodeArg* imag_weights_final_node_arg = &imaginary_weights_node_arg;      
      auto window_node_arg = stft.MutableInputDefs().at(2);
      if (window_node_arg->Exists()) {
        ONNX_NAMESPACE::TensorProto window_shape_tensor_proto;
        window_shape_tensor_proto.set_name(graph.GenerateNodeArgName("STFTWindowReshaped"));
        window_shape_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
        window_shape_tensor_proto.add_int64_data(1);         // BATCH
        window_shape_tensor_proto.add_int64_data(1);         // CHANNEL
        window_shape_tensor_proto.add_int64_data(dft_size);  // SPATIAL
        window_shape_tensor_proto.add_dims(3);
        NodeArg& window_shape_node_arg = graph_utils::AddInitializer(graph, window_shape_tensor_proto);

        std::string window_reshaped_def_name = graph.GenerateNodeArgName("window_reshaped");
        auto* window_reshaped_node_arg = &graph.GetOrCreateNodeArg(window_reshaped_def_name, nullptr);
        Node& window_reshape_node = graph.AddNode(graph.GenerateNodeName("STFT_reshape_window"),
                                                  "Reshape",
                                                  "STFT Reshape Window",
                                                  {window_node_arg, &window_shape_node_arg},
                                                  {window_reshaped_node_arg});
        window_reshape_node.SetExecutionProviderType(stft.GetExecutionProviderType());

        std::string windowed_real_weights_output_def_name = graph.GenerateNodeArgName("windowed_real_weights_output");
        auto* windowed_real_weights_output_node_arg = &graph.GetOrCreateNodeArg(windowed_real_weights_output_def_name, nullptr);
        Node& window_real_weights_output_node = graph.AddNode(graph.GenerateNodeName("Windowed_real_weights_output"),
                                                      "Mul",
                                                      "STFT concat real and imaginary",
                                                      {&real_weights_node_arg, window_reshaped_node_arg},
                                                      {windowed_real_weights_output_node_arg});
        window_real_weights_output_node.SetExecutionProviderType(stft.GetExecutionProviderType());

        std::string windowed_imag_weights_output_def_name = graph.GenerateNodeArgName("windowed_imag_weights_output");
        auto* windowed_imag_weights_output_node_arg = &graph.GetOrCreateNodeArg(windowed_imag_weights_output_def_name, nullptr);
        Node& window_imag_weights_output_node = graph.AddNode(graph.GenerateNodeName("Windowed_imag_weights_output"),
                                                              "Mul",
                                                              "STFT concat real and imaginary",
                                                              {&imaginary_weights_node_arg, window_reshaped_node_arg},
                                                              {windowed_imag_weights_output_node_arg});
        window_imag_weights_output_node.SetExecutionProviderType(stft.GetExecutionProviderType());

        real_weights_final_node_arg = windowed_real_weights_output_node_arg;
        imag_weights_final_node_arg = windowed_imag_weights_output_node_arg;
      }

      // CONVOLUTION
      std::string real_conv_output_def_name = graph.GenerateNodeArgName("real_conv_output");
      auto* real_conv_output_node_arg = &graph.GetOrCreateNodeArg(real_conv_output_def_name, nullptr);
      Node& real_conv_node = graph.AddNode(graph.GenerateNodeName("RealConv"),
                                           "Conv",
                                           "STFT Real Conv Component",
                                           {reshape_output_node_arg, real_weights_final_node_arg},
                                           {real_conv_output_node_arg});
      real_conv_node.AddAttribute("strides", std::vector<int64_t>{frame_step});
      real_conv_node.SetExecutionProviderType(stft.GetExecutionProviderType());

      std::string imag_conv_output_def_name = graph.GenerateNodeArgName("imag_conv_output");
      auto* imag_conv_output_node_arg = &graph.GetOrCreateNodeArg(imag_conv_output_def_name, nullptr);
      Node& imaginary_conv_node = graph.AddNode(graph.GenerateNodeName("ImaginaryConv"),
                                           "Conv",
                                           "STFT Imaginary Conv Component",
                                           {reshape_output_node_arg, imag_weights_final_node_arg},
                                           {imag_conv_output_node_arg});
      imaginary_conv_node.AddAttribute("strides", std::vector<int64_t>{frame_step});
      imaginary_conv_node.SetExecutionProviderType(stft.GetExecutionProviderType());

      std::string concatenated_conv_output_def_name = graph.GenerateNodeArgName("concatenated_conv_output");
      auto* concatenated_conv_output_node_arg = &graph.GetOrCreateNodeArg(concatenated_conv_output_def_name, nullptr);
      Node& concat_node = graph.AddNode(graph.GenerateNodeName("Concat_real_and_imaginary"),
                                          "Concat",
                                          "STFT concat real and imaginary",
                                          {real_conv_output_node_arg, imag_conv_output_node_arg},
                                          {concatenated_conv_output_node_arg});
      concat_node.AddAttribute("axis", static_cast<int64_t>(0));
      concat_node.SetExecutionProviderType(stft.GetExecutionProviderType());

      std::string unsqueezed_output_def_name = graph.GenerateNodeArgName("output_unsqueezed");
      auto* unsqueezed_output_node_arg = &graph.GetOrCreateNodeArg(unsqueezed_output_def_name, nullptr);
      Node& unsqueezed_output_node = graph.AddNode(graph.GenerateNodeName("STFT_reshape_window"),
                                                "Reshape",
                                                "STFT Reshape Window",
                                                {concatenated_conv_output_node_arg, &unsqueezed_shape_node_arg},
                                                {unsqueezed_output_node_arg});
      unsqueezed_output_node.SetExecutionProviderType(stft.GetExecutionProviderType());

      std::string transpose_output_def_name = graph.GenerateNodeArgName("transpose_output");
      auto* transpose_output_node_arg = &graph.GetOrCreateNodeArg(transpose_output_def_name, nullptr);
      Node& transpose_node = graph.AddNode(graph.GenerateNodeName("Transpose"),
                                        "Transpose",
                                        "STFT Transpose",
                                        {unsqueezed_output_node_arg},
                                        {transpose_output_node_arg});
      transpose_node.AddAttribute("perm", std::vector<int64_t>{1, 3, 2, 0});
      transpose_node.SetExecutionProviderType(stft.GetExecutionProviderType());

      Node& reshape_final_node = graph.AddNode(graph.GenerateNodeName("STFTReshapeFinal"),
                                           "Reshape",
                                           "STFT Reshape Final",
                                           {transpose_output_node_arg, &output_shape_node_arg},
                                           {stft.MutableOutputDefs()[0]});
      reshape_final_node.SetExecutionProviderType(stft.GetExecutionProviderType());

      first_replacement = &reshape_signal_node;
      last_replacement = &reshape_final_node;
    } else {
      continue;
    }

    graph_utils::FinalizeNodeFusion(graph, {stft}, *first_replacement, *last_replacement);
    modified = true;
  }
  return Status::OK();
}
}  // namespace onnxruntime
