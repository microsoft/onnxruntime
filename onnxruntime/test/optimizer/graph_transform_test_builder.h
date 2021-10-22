// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <functional>
#include <type_traits>
#include <vector>

#include "core/common/type_utils.h"
#include "core/graph/graph.h"
#include "core/framework/framework_common.h"
#include "core/optimizer/graph_transformer_level.h"
#include "core/graph/onnx_protobuf.h"
#include "test/framework/test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime {
namespace test {
template <typename T>
struct IsTypeQuantLinearCompatible : utils::IsByteType<T> {};

template <typename T>
struct IsTypeDequantLinearCompatible : utils::IsByteType<T> {};

template <>
struct IsTypeDequantLinearCompatible<int32_t> : std::true_type {};

class ModelTestBuilder {
 public:
  ModelTestBuilder(Graph& graph) : graph_(graph) {
  }

  template <typename T>
  NodeArg* MakeInput(const std::vector<int64_t>& shape, T min, T max) {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());

    for (auto& dim : shape) {
      type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
    }

    OrtValue input_value;
    CreateMLValue<T>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault),
                     shape,
                     rand_gen_.Uniform<T>(shape, min, max),
                     &input_value);
    std::string name = graph_.GenerateNodeArgName("input");
    feeds_.insert(std::make_pair(name, input_value));

    return &graph_.GetOrCreateNodeArg(name, &type_proto);
  }

  NodeArg* MakeOutput() {
    std::string name = graph_.GenerateNodeArgName("output");
    output_names_.push_back(name);
    return &graph_.GetOrCreateNodeArg(name, nullptr);
  }

  NodeArg* MakeIntermediate() {
    std::string name = graph_.GenerateNodeArgName("node");
    return &graph_.GetOrCreateNodeArg(name, nullptr);
  }

  template <typename T>
  NodeArg* MakeInitializer(const std::vector<int64_t>& shape, const std::vector<T>& data) {
    std::string name = graph_.GenerateNodeArgName("constant");
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.set_name(name);
    tensor_proto.set_data_type(utils::ToTensorProtoElementType<T>());
    tensor_proto.set_raw_data(data.data(), data.size() * sizeof(T));

    for (auto& dim : shape) {
      tensor_proto.add_dims(dim);
    }

    graph_.AddInitializedTensor(tensor_proto);

    return &graph_.GetOrCreateNodeArg(name, nullptr);
  }

  template <typename T>
  NodeArg* MakeInitializer(const std::vector<int64_t>& shape, T min, T max) {
    return MakeInitializer<T>(shape, rand_gen_.Uniform<T>(shape, min, max));
  }

  template <typename T>
  NodeArg* MakeScalarInitializer(T data) {
    return MakeInitializer({}, std::vector<T>{data});
  }

  template <typename T>
  NodeArg* Make1DInitializer(const std::vector<T>& data) {
    return MakeInitializer({static_cast<int64_t>(data.size())}, data);
  }

  Node& AddNode(const std::string& op_type,
                const std::vector<NodeArg*>& input_args,
                const std::vector<NodeArg*>& output_args,
                const std::string& domain = "") {
    return graph_.AddNode(graph_.GenerateNodeName("node"),
                          op_type,
                          "description",
                          input_args,
                          output_args,
                          nullptr,
                          domain);
  }

  Node& AddConvNode(NodeArg* input_arg,
                    NodeArg* weights_arg,
                    NodeArg* output_arg) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(weights_arg);

    return AddNode("Conv", input_args, {output_arg});
  }

  template <typename T>
  typename std::enable_if<IsTypeQuantLinearCompatible<T>::value, Node&>::type
  AddQuantizeLinearNode(NodeArg* input_arg,
                        float input_scale,
                        T input_zero_point,
                        NodeArg* output_arg) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(MakeScalarInitializer<float>(input_scale));
    input_args.push_back(MakeScalarInitializer<T>(input_zero_point));

    return AddNode("QuantizeLinear", input_args, {output_arg});
  }

  Node& AddQuantizeLinearNode(NodeArg* input_arg,
                              float input_scale,
                              NodeArg* output_arg) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(MakeScalarInitializer<float>(input_scale));

    return AddNode("QuantizeLinear", input_args, {output_arg});
  }

  template <typename T>
  typename std::enable_if<IsTypeDequantLinearCompatible<T>::value, Node&>::type
  AddDequantizeLinearNode(NodeArg* input_arg,
                          float input_scale,
                          T input_zero_point,
                          NodeArg* output_arg) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(MakeScalarInitializer<float>(input_scale));
    input_args.push_back(MakeScalarInitializer<T>(input_zero_point));

    return AddNode("DequantizeLinear", input_args, {output_arg});
  }

  template <typename T>
  typename std::enable_if<IsTypeDequantLinearCompatible<T>::value, Node&>::type
  AddDequantizeLinearNode(NodeArg* input_arg,
                          float input_scale,
                          NodeArg* output_arg) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(MakeScalarInitializer<float>(input_scale));

    return AddNode("DequantizeLinear", input_args, {output_arg});
  }

  template <typename TWeight>
  Node& AddQLinearConvNode(NodeArg* input_arg,
                           float input_scale,
                           uint8_t input_zero_point,
                           NodeArg* weight_arg,
                           float weights_scale,
                           TWeight weights_zero_point,
                           NodeArg* output_arg,
                           float output_scale,
                           uint8_t output_zero_point) {
    std::vector<NodeArg*> input_args{input_arg};
    input_args.push_back(MakeScalarInitializer<float>(input_scale));
    input_args.push_back(MakeScalarInitializer<uint8_t>(input_zero_point));
    input_args.push_back(weight_arg);
    input_args.push_back(MakeScalarInitializer<float>(weights_scale));
    input_args.push_back(MakeScalarInitializer<TWeight>(weights_zero_point));
    input_args.push_back(MakeScalarInitializer<float>(output_scale));
    input_args.push_back(MakeScalarInitializer<TWeight>(output_zero_point));

    return AddNode("QLinearConv", input_args, {output_arg});
  }

  Node& AddQLinearBinaryNode(const std::string& op_type,
                             NodeArg* input1_arg,
                             float input1_scale,
                             uint8_t input1_zero_point,
                             NodeArg* input2_arg,
                             float input2_scale,
                             uint8_t input2_zero_point,
                             NodeArg* output_arg,
                             float output_scale,
                             uint8_t output_zero_point) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input1_arg);
    input_args.push_back(MakeScalarInitializer<float>(input1_scale));
    input_args.push_back(MakeScalarInitializer<uint8_t>(input1_zero_point));
    input_args.push_back(input2_arg);
    input_args.push_back(MakeScalarInitializer<float>(input2_scale));
    input_args.push_back(MakeScalarInitializer<uint8_t>(input2_zero_point));
    input_args.push_back(MakeScalarInitializer<float>(output_scale));
    input_args.push_back(MakeScalarInitializer<uint8_t>(output_zero_point));

    return AddNode(op_type, input_args, {output_arg}, kMSDomain);
  }

  Node& AddQLinearConcatLike(const std::string& op_type,
                             NodeArg* output_arg,
                             float output_scale,
                             uint8_t output_zero_point,
                             std::vector<std::tuple<NodeArg*, float, uint8_t>> quantized_inputs) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(MakeScalarInitializer<float>(output_scale));
    input_args.push_back(MakeScalarInitializer<uint8_t>(output_zero_point));
    for (size_t input_index = 0; input_index < quantized_inputs.size(); ++input_index) {
      input_args.push_back(std::get<0>(quantized_inputs[input_index]));
      input_args.push_back(MakeScalarInitializer<float>(std::get<1>(quantized_inputs[input_index])));
      input_args.push_back(MakeScalarInitializer<uint8_t>(std::get<2>(quantized_inputs[input_index])));
    }
    return AddNode(op_type, input_args, {output_arg}, kMSDomain);
  }

  Node& AddQLinearActivationNode(const std::string& op_type,
                                 NodeArg* input_arg,
                                 float input_scale,
                                 uint8_t input_zero_point,
                                 NodeArg* output_arg,
                                 float output_scale,
                                 uint8_t output_zero_point) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(MakeScalarInitializer<float>(input_scale));
    input_args.push_back(MakeScalarInitializer<uint8_t>(input_zero_point));
    input_args.push_back(MakeScalarInitializer<float>(output_scale));
    input_args.push_back(MakeScalarInitializer<uint8_t>(output_zero_point));

    return AddNode(op_type, input_args, {output_arg}, kMSDomain);
  }

  void SetGraphOutputs() {
    std::vector<const NodeArg*> outputs;
    for (auto& output_name : output_names_) {
      outputs.push_back(graph_.GetNodeArg(output_name));
    }
    graph_.SetOutputs(outputs);
  }

  Graph& graph_;
  NameMLValMap feeds_;
  std::vector<std::string> output_names_;
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
};

void TransformerTester(const std::function<void(ModelTestBuilder& helper)>& build_test_case,
                       const std::function<void(InferenceSessionWrapper& session)>& check_transformed_graph,
                       TransformerLevel baseline_level,
                       TransformerLevel target_level,
                       int opset_version = 12,
                       double per_sample_tolerance = 0.0,
                       double relative_per_sample_tolerance = 0.0,
                       std::unique_ptr<GraphTransformer> transformer = nullptr);

}  // namespace test
}  // namespace onnxruntime
