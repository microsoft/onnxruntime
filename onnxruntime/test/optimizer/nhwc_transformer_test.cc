// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

template <typename T>
struct NhwcWeightsRange {
  static constexpr T min_value = std::numeric_limits<T>::min();
  static constexpr T max_value = std::numeric_limits<T>::max();
};

template <>
struct NhwcWeightsRange<int8_t> {
  // Avoid saturation from u8s8 math.
  static constexpr int8_t min_value = -63;
  static constexpr int8_t max_value = +63;
};

struct NhwcTestHelper {
  NhwcTestHelper(Graph& graph) : graph_(graph) {
  }

  template <typename T>
  NodeArg* MakeInput(const std::vector<int64_t>& shape, const ONNX_NAMESPACE::TypeProto& type_proto) {
    OrtValue input_value;
    CreateMLValue<T>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), shape,
                     FillRandomData<T>(shape, 0, 31), &input_value);
    std::string name = graph_.GenerateNodeArgName("input");
    feeds_.insert(std::make_pair(name, input_value));

    return &graph_.GetOrCreateNodeArg(name, &type_proto);
  }

  template <typename T>
  NodeArg* MakeInput(const std::vector<int64_t>& shape) {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());

    for (auto& dim : shape) {
      type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
    }

    return MakeInput<T>(shape, type_proto);
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
  NodeArg* MakeInitializer(const std::vector<int64_t>& shape, int32_t min_value, int32_t max_value) {
    return MakeInitializer<T>(shape, FillRandomData<T>(shape, min_value, max_value));
  }

  template <typename T>
  NodeArg* MakeScalarInitializer(T data) {
    return MakeInitializer({}, std::vector<T>{data});
  }

  template <typename T>
  NodeArg* Make1DInitializer(const std::vector<T>& data) {
    return MakeInitializer({static_cast<int64_t>(data.size())}, data);
  }

  template <typename T>
  NodeArg* MakeWeightsInitializer(const std::vector<int64_t>& shape) {
    return MakeInitializer<T>(shape, NhwcWeightsRange<T>::min_value, NhwcWeightsRange<T>::max_value);
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

  Node& AddQLinearConvNode(NodeArg* input_arg,
                           NodeArg* input_scale_arg,
                           NodeArg* input_zero_point_arg,
                           NodeArg* weights_arg,
                           NodeArg* weights_scale_arg,
                           NodeArg* weights_zero_point_arg,
                           NodeArg* output_arg,
                           NodeArg* output_scale,
                           NodeArg* output_zero_point) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(input_scale_arg);
    input_args.push_back(input_zero_point_arg);
    input_args.push_back(weights_arg);
    input_args.push_back(weights_scale_arg);
    input_args.push_back(weights_zero_point_arg);
    input_args.push_back(output_scale);
    input_args.push_back(output_zero_point);

    return AddNode("QLinearConv", input_args, {output_arg});
  }

  template <typename TWeight>
  Node& AddQLinearConvNode(NodeArg* input_arg,
                           float input_scale,
                           uint8_t input_zero_point,
                           const std::vector<int64_t>& weights_shape,
                           float weights_scale,
                           TWeight weights_zero_point,
                           NodeArg* output_arg,
                           float output_scale,
                           uint8_t output_zero_point) {
    return AddQLinearConvNode(input_arg,
                              MakeScalarInitializer<float>(input_scale),
                              MakeScalarInitializer<uint8_t>(input_zero_point),
                              MakeWeightsInitializer<TWeight>(weights_shape),
                              MakeScalarInitializer<float>(weights_scale),
                              MakeScalarInitializer<TWeight>(weights_zero_point),
                              output_arg,
                              MakeScalarInitializer<float>(output_scale),
                              MakeScalarInitializer<TWeight>(output_zero_point));
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

  Node& AddDequantizeLinearNode(NodeArg* input_arg,
                                float input_scale,
                                uint8_t input_zero_point,
                                NodeArg* output_arg) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(MakeScalarInitializer<float>(input_scale));
    input_args.push_back(MakeScalarInitializer<uint8_t>(input_zero_point));

    return AddNode("DequantizeLinear", input_args, {output_arg});
  }

  template <typename T>
  std::vector<T> FillRandomData(size_t count, int32_t min_value, int32_t max_value) {
    std::vector<T> random_data;
    random_data.resize(count);
    std::uniform_int_distribution<int32_t> distribution(min_value, max_value);
    for (size_t n = 0; n < count; n++) {
      random_data[n] = static_cast<T>(distribution(generator_));
    }
    return random_data;
  }

  template <typename T>
  std::vector<T> FillRandomData(const std::vector<int64_t>& shape, int32_t min_value, int32_t max_value) {
    int64_t num_elements = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>{});
    return FillRandomData<T>(static_cast<size_t>(num_elements), min_value, max_value);
  }

  Graph& graph_;
  NameMLValMap feeds_;
  std::vector<std::string> output_names_;
  std::default_random_engine generator_{2345};
};

void NhwcTransformerTester(const std::function<void(NhwcTestHelper& helper)>& build_test_case,
                           const std::function<void(InferenceSessionWrapper& session)>& check_nhwc_graph,
                           int opset_version = 12) {
  // Build the model for this test.
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = opset_version;
  domain_to_version[kMSDomain] = 1;
  Model model("nhwc", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, {}, DefaultLoggingManager().DefaultLogger());
  NhwcTestHelper helper(model.MainGraph());
  build_test_case(helper);
  ASSERT_TRUE(model.MainGraph().Resolve().IsOK());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  auto run_model = [&](TransformerLevel level, std::vector<OrtValue>& fetches) {
    SessionOptions session_options;
    session_options.graph_optimization_level = level;
    session_options.session_logid = "NhwcTransformerTests";
    InferenceSessionWrapper session{session_options, GetEnvironment()};
    ASSERT_TRUE(session.Load(model_data.data(), static_cast<int>(model_data.size())).IsOK());
    ASSERT_TRUE(session.Initialize().IsOK());

    RunOptions run_options;
    auto status = session.Run(run_options, helper.feeds_, helper.output_names_, &fetches);
    if (!status.IsOK()) {
      std::cout << "Run failed with status message: " << status.ErrorMessage() << std::endl;
    }
    ASSERT_TRUE(status.IsOK());

    if (level == TransformerLevel::Level3) {
      check_nhwc_graph(session);
    }
  };

  std::vector<OrtValue> level2_fetches;
  run_model(TransformerLevel::Level2, level2_fetches);

  std::vector<OrtValue> level3_fetches;
  run_model(TransformerLevel::Level3, level3_fetches);

  size_t num_outputs = level2_fetches.size();
  ASSERT_TRUE(num_outputs == level3_fetches.size());

  for (size_t i = 0; i < num_outputs; i++) {
    double per_sample_tolerance = 0.0;
    double relative_per_sample_tolerance = 0.0;
    std::pair<COMPARE_RESULT, std::string> ret =
        CompareOrtValue(level3_fetches[i],
                        level2_fetches[i],
                        per_sample_tolerance,
                        relative_per_sample_tolerance,
                        false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
  }
}

#ifndef DISABLE_CONTRIB_OPS

#if defined(MLAS_TARGET_AMD64_IX86)

TEST(NhwcTransformerTests, Conv) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
    auto build_test_case = [&](NhwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<uint8_t>(input_shape);
      auto* output_arg = helper.MakeOutput();

      helper.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                         weights_shape, .02f, 126,
                                         output_arg, .37f, 131);
    };

    auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 2);
    };

    NhwcTransformerTester(build_test_case, check_nhwc_graph);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
}

TEST(NhwcTransformerTests, ConvDequantizeLinear) {
  auto build_test_case = [&](NhwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput<uint8_t>({1, 12, 37});
    auto* conv_output_arg = helper.MakeIntermediate();
    auto* output_arg = helper.MakeOutput();

    helper.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                       {32, 12, 5}, .02f, 126,
                                       conv_output_arg, .37f, 131);
    helper.AddDequantizeLinearNode(conv_output_arg, .37f, 131, output_arg);
  };

  auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 0);
    EXPECT_EQ(op_to_count["Transpose"], 0);
  };

  // QLinearConv followed by only DequantizeLinear will remain as the ONNX
  // version of the operator to avoid adding unnecessary Transpose nodes to
  // the graph.
  NhwcTransformerTester(build_test_case, check_nhwc_graph);
}

TEST(NhwcTransformerTests, ConvBlockBinary) {
  auto test_case = [&](const std::string& binary_op_type) {
    auto build_test_case = [&](NhwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<uint8_t>({1, 23, 13, 13});
      auto* conv1_output_arg = helper.MakeIntermediate();
      auto* conv2_output_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      Node& conv1_node = helper.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                                            {30, 23, 3, 3}, .02f, 126,
                                                            conv1_output_arg, .37f, 131);
      conv1_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
      helper.AddQLinearConvNode<uint8_t>(input_arg, .01f, 135,
                                         {30, 23, 1, 1}, .015f, 129,
                                         conv2_output_arg, .37f, 131);
      helper.AddQLinearBinaryNode(binary_op_type,
                                  conv1_output_arg, .37f, 131,
                                  conv2_output_arg, .37f, 131,
                                  output_arg, .43f, 126);
    };

    auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 2);
      EXPECT_EQ(op_to_count["Transpose"], 2);
    };

    NhwcTransformerTester(build_test_case, check_nhwc_graph);
  };

  std::vector<std::string> activation_op_types{"QLinearAdd", "QLinearMul"};
  for (auto& activation_op_type : activation_op_types) {
    test_case(activation_op_type);
  }
}

TEST(NhwcTransformerTests, ConvBlockActivation) {
  auto test_case = [&](uint32_t extra_edges) {
    auto build_test_case = [&](NhwcTestHelper& helper) {
      auto* input1_arg = helper.MakeInput<uint8_t>({1, 10, 13, 13});
      auto* input2_arg = helper.MakeInput<uint8_t>({1, 13, 13, 13});
      auto* concat_arg = helper.MakeIntermediate();
      auto* conv1_output_arg = helper.MakeIntermediate();
      auto* conv2_output_arg = helper.MakeIntermediate();
      auto* act1_output_arg = helper.MakeIntermediate();
      auto* act2_output_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      // Create a convolution input that isn't directly a graph input.
      Node& concat_node = helper.AddNode("Concat", {input1_arg, input2_arg}, {concat_arg});
      concat_node.AddAttribute("axis", static_cast<int64_t>(1));

      Node& conv1_node = helper.AddQLinearConvNode<uint8_t>(concat_arg, .01f, 135,
                                                            {30, 23, 3, 3}, .02f, 126,
                                                            conv1_output_arg, .37f, 131);
      conv1_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
      helper.AddQLinearActivationNode("QLinearSigmoid",
                                      conv1_output_arg, .37f, 131,
                                      act1_output_arg, .37f, 131);
      helper.AddQLinearConvNode<uint8_t>(concat_arg, .01f, 135,
                                         {30, 23, 1, 1}, .015f, 129,
                                         conv2_output_arg, .37f, 131);
      helper.AddQLinearActivationNode("QLinearLeakyRelu",
                                      conv2_output_arg, .37f, 131,
                                      act2_output_arg, .37f, 131);
      helper.AddQLinearBinaryNode("QLinearAdd",
                                  act1_output_arg, .37f, 131,
                                  act2_output_arg, .37f, 131,
                                  output_arg, .39f, 126);

      // Create extra uses of the various NodeArgs to exercise the transformer.
      if ((extra_edges & 1) != 0) {
        helper.AddDequantizeLinearNode(concat_arg, .01f, 135, helper.MakeOutput());
      }
      if ((extra_edges & 2) != 0) {
        helper.AddDequantizeLinearNode(conv1_output_arg, .37f, 131, helper.MakeOutput());
      }
      if ((extra_edges & 4) != 0) {
        helper.AddDequantizeLinearNode(conv2_output_arg, .37f, 131, helper.MakeOutput());
      }
      if ((extra_edges & 8) != 0) {
        helper.AddDequantizeLinearNode(act1_output_arg, .37f, 131, helper.MakeOutput());
      }
      if ((extra_edges & 16) != 0) {
        helper.AddDequantizeLinearNode(act2_output_arg, .37f, 131, helper.MakeOutput());
      }
    };

    auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 2);
    };

    NhwcTransformerTester(build_test_case, check_nhwc_graph);
  };

  // Add extra uses of the edges that cause the transformer to insert additional
  // Transpose operations.
  for (uint32_t extra_edges = 0; extra_edges < 32; extra_edges++) {
    test_case(extra_edges);
  }
}

TEST(NhwcTransformerTests, ConvMixTensorRanks) {
  auto build_test_case = [&](NhwcTestHelper& helper) {
    auto* input1_arg = helper.MakeInput<uint8_t>({1, 10, 7});
    auto* input2_arg = helper.MakeInput<uint8_t>({1, 12, 7, 7});
    auto* conv1_output_arg = helper.MakeIntermediate();
    auto* conv2_output_arg = helper.MakeIntermediate();
    auto* output_arg = helper.MakeOutput();

    helper.AddQLinearConvNode<uint8_t>(input1_arg, .01f, 135,
                                       {1, 10, 3}, .02f, 126,
                                       conv1_output_arg, .37f, 131);
    helper.AddQLinearConvNode<uint8_t>(input2_arg, .01f, 135,
                                       {1, 12, 3, 3}, .02f, 126,
                                       conv2_output_arg, .37f, 131);
    // Broadcast add {1, 1, 5} to {1, 1, 5, 5}.
    helper.AddQLinearBinaryNode("QLinearAdd",
                                conv1_output_arg, .37f, 131,
                                conv2_output_arg, .37f, 131,
                                output_arg, .39f, 126);
  };

  auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.QLinearConv"], 2);
    EXPECT_EQ(op_to_count["Transpose"], 4);
  };

  // Generate a graph with QLinearAdd that broadcasts adds a 1D tensor to a
  // 2D tensor and verify that the transformer handles the mixed tensor ranks.
  NhwcTransformerTester(build_test_case, check_nhwc_graph);
}

#endif

#endif // DISABLE_CONTRIB_OPS

}  // namespace test
}  // namespace onnxruntime
