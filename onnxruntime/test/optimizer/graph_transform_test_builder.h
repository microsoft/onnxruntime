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

#define TEST_RETURN_IF(condition)                                               \
  do {                                                                          \
    if (condition) {                                                            \
      return ::onnxruntime::common::Status(::onnxruntime::common::ONNXRUNTIME,  \
                                           ::onnxruntime::common::FAIL,         \
                                           #condition " is evaluated to true"); \
    }                                                                           \
  } while (false)

#define TEST_RETURN_IF_NOT(condition)                                            \
  do {                                                                           \
    if (!(condition)) {                                                          \
      return ::onnxruntime::common::Status(::onnxruntime::common::ONNXRUNTIME,   \
                                           ::onnxruntime::common::FAIL,          \
                                           #condition " is evaluated to false"); \
    }                                                                            \
  } while (false)

namespace onnxruntime {
namespace test {
template <typename T>
struct IsTypeQuantLinearCompatible : utils::IsByteType<T> {};

template <>
struct IsTypeQuantLinearCompatible<int16_t> : std::true_type {};

template <>
struct IsTypeQuantLinearCompatible<uint16_t> : std::true_type {};

template <typename T>
struct IsTypeDequantLinearCompatible : utils::IsByteType<T> {};

template <>
struct IsTypeDequantLinearCompatible<int16_t> : std::true_type {};

template <>
struct IsTypeDequantLinearCompatible<uint16_t> : std::true_type {};

template <>
struct IsTypeDequantLinearCompatible<int32_t> : std::true_type {};

class ModelTestBuilder {
 public:
  ModelTestBuilder(Graph& graph) : graph_(graph) {
  }

  const std::unordered_map<std::string, int>& DomainToVersionMap() const noexcept {
    return graph_.DomainToVersionMap();
  }

  template <typename T>
  NodeArg* MakeInput(const std::vector<int64_t>& shape, const std::vector<T>& data) {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());

    // Set shape even if no dims (for scalar)
    type_proto.mutable_tensor_type()->mutable_shape();
    for (auto& dim : shape) {
      type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
    }

    OrtValue input_value;
    CreateMLValue<T>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                     shape,
                     data,
                     &input_value);
    std::string name = graph_.GenerateNodeArgName("input");
    feeds_.insert(std::make_pair(name, input_value));

    return &graph_.GetOrCreateNodeArg(name, &type_proto);
  }

  template <typename T>
  NodeArg* MakeInput(const std::vector<int64_t>& shape, T min, T max) {
    return MakeInput<T>(shape, rand_gen_.Uniform<T>(shape, min, max));
  }

  NodeArg* MakeInputBool(const std::vector<int64_t>& shape) {
    std::vector<uint8_t> data_uint8 = rand_gen_.Uniform<uint8_t>(shape, 0, 1);
    std::vector<bool> data;
    for (uint8_t x : data_uint8) {
      data.push_back(x != 0);
    }
    return MakeInput<bool>(shape, data);
  }

  template <typename T>
  NodeArg* MakeInput(const std::optional<std::vector<int64_t>>& shape,
                     std::optional<std::string> input_name = std::nullopt) {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());
    if (shape != std::nullopt) {
      type_proto.mutable_tensor_type()->mutable_shape();
      for (auto& d : *shape) {
        auto dim = type_proto.mutable_tensor_type()->mutable_shape()->add_dim();
        if (d != -1) {
          dim->set_dim_value(d);
        }
      }
    }

    if (input_name == std::nullopt) {
      std::string name = graph_.GenerateNodeArgName("input");
      return &graph_.GetOrCreateNodeArg(name, &type_proto);
    } else {
      ORT_ENFORCE(graph_.GetNodeArg(*input_name) == nullptr, "Input name already exists: ", *input_name);
      return &graph_.GetOrCreateNodeArg(*input_name, &type_proto);
    }
  }

  template <typename T>
  NodeArg* MakeSymbolicInput(const std::vector<std::variant<int64_t, std::string>>& shape) {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());
    type_proto.mutable_tensor_type()->mutable_shape();
    for (auto& d : shape) {
      auto dim = type_proto.mutable_tensor_type()->mutable_shape()->add_dim();
      std::visit([&dim](auto&& arg) -> void {
        using V = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<V, int64_t>) {
          ORT_ENFORCE(arg >= 0, "Negative dimension is not allowed in symbolic shape");
          dim->set_dim_value(arg);
        } else {
          dim->set_dim_param(arg);
        }
      },
                 d);
    }

    std::string name = graph_.GenerateNodeArgName("symbolic_input");
    return &graph_.GetOrCreateNodeArg(name, &type_proto);
  }

  NodeArg* MakeOutput() {
    std::string name = graph_.GenerateNodeArgName("output");
    output_names_.push_back(name);
    return &graph_.GetOrCreateNodeArg(name, nullptr);
  }

  template <typename T>
  NodeArg* MakeOutput(const std::optional<std::vector<int64_t>>& shape) {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());
    if (shape != std::nullopt) {
      ONNX_NAMESPACE::TensorShapeProto* shape_proto = type_proto.mutable_tensor_type()->mutable_shape();
      for (auto& d : *shape) {
        auto dim = shape_proto->add_dim();
        if (d != -1) {
          dim->set_dim_value(d);
        }
      }
    }
    std::string name = graph_.GenerateNodeArgName("output");
    output_names_.push_back(name);
    return &graph_.GetOrCreateNodeArg(name, &type_proto);
  }

  NodeArg* MakeIntermediate() {
    std::string name = graph_.GenerateNodeArgName("node");
    return &graph_.GetOrCreateNodeArg(name, nullptr);
  }

  template <typename T>
  NodeArg* MakeIntermediate(const std::optional<std::vector<int64_t>>& shape) {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());
    if (shape != std::nullopt) {
      type_proto.mutable_tensor_type()->mutable_shape();
      for (auto& d : *shape) {
        auto dim = type_proto.mutable_tensor_type()->mutable_shape()->add_dim();
        if (d != -1) {
          dim->set_dim_value(d);
        }
      }
    }
    std::string name = graph_.GenerateNodeArgName("node");
    return &graph_.GetOrCreateNodeArg(name, &type_proto);
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

  // Special handle for std::vector<bool>.
  NodeArg* MakeInitializerBool(const std::vector<int64_t>& shape, const std::vector<bool>& data) {
    std::string name = graph_.GenerateNodeArgName("constant");
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.set_name(name);
    tensor_proto.set_data_type(utils::ToTensorProtoElementType<bool>());
    std::unique_ptr<bool[]> data_buffer = std::make_unique<bool[]>(data.size());
    for (size_t i = 0; i < data.size(); ++i) data_buffer[i] = data[i];
    tensor_proto.set_raw_data(data_buffer.get(), data.size());

    for (auto& dim : shape) {
      tensor_proto.add_dims(dim);
    }

    graph_.AddInitializedTensor(tensor_proto);

    return &graph_.GetOrCreateNodeArg(name, nullptr);
  }

  NodeArg* MakeRandInitializerBool(const std::vector<int64_t>& shape) {
    std::vector<uint8_t> data_uint8 = rand_gen_.Uniform<uint8_t>(shape, 0, 1);
    std::vector<bool> data;
    for (uint8_t x : data_uint8) {
      data.push_back(x != 0);
    }
    return MakeInitializerBool(shape, data);
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

  NodeArg* MakeEmptyInput() {
    NodeArg* empty = &graph_.GetOrCreateNodeArg("", nullptr);
    return empty;
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
                        NodeArg* output_arg,
                        bool use_ms_domain = false) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(MakeScalarInitializer<float>(input_scale));
    input_args.push_back(MakeScalarInitializer<T>(input_zero_point));

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode("QuantizeLinear", input_args, {output_arg}, domain);
  }

  Node& AddQuantizeLinearNode(NodeArg* input_arg,
                              float input_scale,
                              NodeArg* output_arg,
                              bool use_ms_domain = false) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(MakeScalarInitializer<float>(input_scale));

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode("QuantizeLinear", input_args, {output_arg}, domain);
  }

  template <typename T>
  typename std::enable_if<IsTypeDequantLinearCompatible<T>::value, Node&>::type
  AddDequantizeLinearNode(NodeArg* input_arg,
                          float input_scale,
                          T input_zero_point,
                          NodeArg* output_arg,
                          bool use_ms_domain = false) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(MakeScalarInitializer<float>(input_scale));
    input_args.push_back(MakeScalarInitializer<T>(input_zero_point));

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode("DequantizeLinear", input_args, {output_arg}, domain);
  }

  Node& AddDequantizeLinearNode(NodeArg* input_arg,
                                float input_scale,
                                NodeArg* output_arg,
                                bool use_ms_domain = false) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(MakeScalarInitializer<float>(input_scale));

    std::string domain = use_ms_domain ? kMSDomain : "";
    return AddNode("DequantizeLinear", input_args, {output_arg}, domain);
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
                       std::unique_ptr<GraphTransformer> transformer = nullptr,
                       const std::function<void(SessionOptions&)>& add_session_options = {},
                       const InlinedHashSet<std::string>& disabled_optimizers = {});

void TransformerTester(const std::function<void(ModelTestBuilder& helper)>& build_test_case,
                       const std::function<void(InferenceSessionWrapper& session)>& check_transformed_graph,
                       TransformerLevel baseline_level,
                       TransformerLevel target_level,
                       const std::vector<int>& opset_versions,
                       double per_sample_tolerance = 0.0,
                       double relative_per_sample_tolerance = 0.0,
                       std::unique_ptr<GraphTransformer> transformer = nullptr,  // must be null in this case.
                       const std::function<void(SessionOptions&)>& add_session_options = {},
                       const InlinedHashSet<std::string>& disabled_optimizers = {});

/**
 * @brief Apply a GraphTransformer to a graph, and run graph checkers before and after applying the transformer.
 *
 * @param build_test_case The function to build a graph for testing
 * @param opset_version The OpSet version of the graph
 * @param logger The logger
 * @param transformer The GraphTransformer to be applied
 * @param level The transformer level on which the transformer will be applied
 * @param steps The step count of the GraphTransformerManager
 * @param pre_graph_checker The graph checker function before applying the transformer
 * @param post_graph_checker The graph checker function after applying the transformer
 */
Status TestGraphTransformer(const std::function<void(ModelTestBuilder& helper)>& build_test_case, int opset_version,
                            const logging::Logger& logger, std::unique_ptr<GraphTransformer> transformer,
                            TransformerLevel level, unsigned steps, const std::function<Status(Graph&)>& pre_graph_checker,
                            const std::function<Status(Graph&)>& post_graph_checker);

/**
 * @brief Apply a GraphTransformer to a graph, and run graph checkers before and after applying the transformer.
 *
 * @param build_test_case The function to build a graph for testing
 * @param opset_versions A graph is created and tested for every opset in this set
 * @param logger The logger
 * @param transformer The GraphTransformer to be applied
 * @param level The transformer level on which the transformer will be applied
 * @param steps The step count of the GraphTransformerManager
 * @param pre_graph_checker The graph checker function before applying the transformer
 * @param post_graph_checker The graph checker function after applying the transformer
 */
Status TestGraphTransformer(const std::function<void(ModelTestBuilder& helper)>& build_test_case,
                            const std::vector<int>& opset_versions,
                            const logging::Logger& logger, std::unique_ptr<GraphTransformer> transformer,
                            TransformerLevel level, unsigned steps, const std::function<Status(Graph&)>& pre_graph_checker,
                            const std::function<Status(Graph&)>& post_graph_checker);
}  // namespace test
}  // namespace onnxruntime
