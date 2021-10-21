// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "test/util/include/inference_session_wrapper.h"
#include <cmath>

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

struct NchwcTestHelper {
  NchwcTestHelper(Graph& graph) : graph_(graph), fill_value_(0), per_sample_tolerance_(0.0) {
  }

  template <typename T>
  NodeArg* MakeInput(const std::vector<int64_t>& shape, const ONNX_NAMESPACE::TypeProto& type_proto) {
    OrtValue input_value;
    CreateMLValue<T>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), shape,
                     FillRandomData<T>(shape), &input_value);
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

  NodeArg* MakeInitializer(const std::vector<int64_t>& shape) {
    return MakeInitializer<float>(shape, FillRandomData<float>(shape));
  }

  template <typename T>
  NodeArg* Make1DInitializer(const std::vector<T>& data) {
    return MakeInitializer({static_cast<int64_t>(data.size())}, data);
  }

  Node& AddNode(const std::string& op_type,
                const std::vector<NodeArg*>& input_args,
                const std::vector<NodeArg*>& output_args) {
    return graph_.AddNode(graph_.GenerateNodeName("node"),
                          op_type,
                          "description",
                          input_args,
                          output_args);
  }

  Node& AddConvNode(NodeArg* input_arg, NodeArg* output_arg, const std::vector<int64_t>& weights_shape, bool no_bias = false) {
    auto* weights_arg = MakeInitializer(weights_shape);
    std::vector<NodeArg*> input_args{input_arg, weights_arg};
    if (!no_bias) {
      auto* biases_arg = MakeInitializer({weights_shape[0]});
      input_args.push_back(biases_arg);
    }
    return AddNode("Conv", input_args, {output_arg});
  }

  Node& AddClipNode(NodeArg* input_arg, NodeArg* output_arg, float min, float max) {
    int opset_version = graph_.DomainToVersionMap().find(kOnnxDomain)->second;
    std::vector<NodeArg*> input_args{input_arg};
    if (opset_version >= 11) {
      input_args.push_back(Make1DInitializer<float>({min}));
      input_args.push_back(Make1DInitializer<float>({max}));
    }
    auto& node = AddNode("Clip", input_args, {output_arg});
    if (opset_version < 11) {
      node.AddAttribute("min", min);
      node.AddAttribute("max", max);
    }
    return node;
  }

  Node& AddTransposeNode(NodeArg* input_arg, NodeArg* output_arg, const std::vector<int64_t>& perm) {
    auto& node = AddNode("Transpose", {input_arg}, {output_arg});
    node.AddAttribute("perm", perm);
    return node;
  }

  Node& AddTransposeToNchwNode(NodeArg* input_arg, NodeArg* output_arg) {
    return AddTransposeNode(input_arg, output_arg, {0, 3, 1, 2});
  }

  Node& AddTransposeToNhwcNode(NodeArg* input_arg, NodeArg* output_arg) {
    return AddTransposeNode(input_arg, output_arg, {0, 2, 3, 1});
  }

  Node& AddTransposeToCnhwNode(NodeArg* input_arg, NodeArg* output_arg) {
    return AddTransposeNode(input_arg, output_arg, {1, 0, 2, 3});
  }

  template <typename T>
  std::vector<T> FillRandomData(size_t count) {
    constexpr int min_fill_value = -23;
    constexpr int max_fill_value = 23;

    std::vector<T> random_data;
    random_data.resize(count);
    for (size_t n = 0; n < count; n++) {
      random_data[n] = static_cast<T>(fill_value_);
      fill_value_++;
      if (fill_value_ == max_fill_value) {
        fill_value_ = min_fill_value;
      }
    }
    return random_data;
  }

  template <typename T>
  std::vector<T> FillRandomData(const std::vector<int64_t>& shape) {
    int64_t num_elements = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>{});
    return FillRandomData<T>(static_cast<size_t>(num_elements));
  }

  Graph& graph_;
  NameMLValMap feeds_;
  std::vector<std::string> output_names_;
  int fill_value_;
  double per_sample_tolerance_;
};

void NchwcOptimizerTester(const std::function<void(NchwcTestHelper& helper)>& build_test_case,
                          const std::function<void(InferenceSessionWrapper& session)>& check_nchwc_graph,
                          int opset_version = 13) {
  // Ignore the test if NCHWc is not supported by the platform.
  if (MlasNchwcGetBlockSize() <= 1) {
    return;
  }

  // Build the model for this test.
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = opset_version;
  Model model("nchwc", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, {}, DefaultLoggingManager().DefaultLogger());
  NchwcTestHelper helper(model.MainGraph());
  build_test_case(helper);
  ASSERT_TRUE(model.MainGraph().Resolve().IsOK());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  auto run_model = [&](TransformerLevel level, std::vector<OrtValue>& fetches) {
    SessionOptions session_options;
    session_options.graph_optimization_level = level;
    session_options.session_logid = "NchwcOptimizerTests";
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
      check_nchwc_graph(session);
    }
  };

  std::vector<OrtValue> level2_fetches;
  run_model(TransformerLevel::Level2, level2_fetches);

  std::vector<OrtValue> level3_fetches;
  run_model(TransformerLevel::Level3, level3_fetches);

  size_t num_outputs = level2_fetches.size();
  ASSERT_TRUE(num_outputs == level3_fetches.size());

  for (size_t i = 0; i < num_outputs; i++) {
    double relative_per_sample_tolerance = 0.0;
    std::pair<COMPARE_RESULT, std::string> ret =
        CompareOrtValue(level3_fetches[i],
                        level2_fetches[i],
                        helper.per_sample_tolerance_,
                        relative_per_sample_tolerance,
                        false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
  }
}

#ifndef DISABLE_CONTRIB_OPS

TEST(NchwcOptimizerTests, ConvNchw) {
  auto test_case = [&](const std::string& activation_op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({16, 3, 112, 112});
      auto* output_arg = helper.MakeOutput();

      auto* conv_output_arg = output_arg;
      if (!activation_op_type.empty()) {
        conv_output_arg = helper.MakeIntermediate();
        if (activation_op_type == "Clip") {
          helper.AddClipNode(conv_output_arg, output_arg, 0.f, 6.f);
        } else {
          helper.AddNode(activation_op_type, {conv_output_arg}, {output_arg});
        }
      }

      auto& conv_node = helper.AddConvNode(input_arg, conv_output_arg, {130, 3, 3, 3});
      conv_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
      conv_node.AddAttribute("strides", std::vector<int64_t>{2, 2});
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 0);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
      if (!activation_op_type.empty()) {
        EXPECT_EQ(op_to_count[activation_op_type], 0);
      }
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  std::vector<std::string> activation_op_types{"", "Relu", "LeakyRelu", "Clip"};
  for (auto& activation_op_type : activation_op_types) {
    test_case(activation_op_type);
  }
}

TEST(NchwcOptimizerTests, ConvNchwc) {
  auto test_case = [&](const std::string& activation_op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({16, 64, 28, 28});
      auto* output_arg = helper.MakeOutput();

      auto* conv_output_arg = output_arg;
      if (!activation_op_type.empty()) {
        conv_output_arg = helper.MakeIntermediate();
        if (activation_op_type == "Clip") {
          helper.AddClipNode(conv_output_arg, output_arg, -6.f, 6.f);
        } else {
          helper.AddNode(activation_op_type, {conv_output_arg}, {output_arg});
        }
      }

      helper.AddConvNode(input_arg, conv_output_arg, {127, 64, 3, 3});
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
      if (!activation_op_type.empty()) {
        EXPECT_EQ(op_to_count[activation_op_type], 0);
      }
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  std::vector<std::string> activation_op_types{"", "Relu", "LeakyRelu", "Clip"};
  for (auto& activation_op_type : activation_op_types) {
    test_case(activation_op_type);
  }
}

TEST(NchwcOptimizerTests, ConvNchwcGrouped) {
  auto test_case = [&](const std::string& activation_op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({16, 48, 28, 28});
      auto* output_arg = helper.MakeOutput();

      auto* conv_output_arg = output_arg;
      if (!activation_op_type.empty()) {
        conv_output_arg = helper.MakeIntermediate();
        helper.AddNode(activation_op_type, {conv_output_arg}, {output_arg});
      }

      auto& conv_node = helper.AddConvNode(input_arg, conv_output_arg, {192, 16, 3, 3});
      conv_node.AddAttribute("group", static_cast<int64_t>(3));
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
      if (!activation_op_type.empty()) {
        EXPECT_EQ(op_to_count[activation_op_type], 0);
      }
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  std::vector<std::string> activation_op_types{"", "Relu", "LeakyRelu"};
  for (auto& activation_op_type : activation_op_types) {
    test_case(activation_op_type);
  }
}

TEST(NchwcOptimizerTests, ConvDepthwise) {
  auto test_case = [&](const std::string& activation_op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({16, 96, 28, 28});
      auto* output_arg = helper.MakeOutput();

      auto* conv_output_arg = output_arg;
      if (!activation_op_type.empty()) {
        conv_output_arg = helper.MakeIntermediate();
        helper.AddNode(activation_op_type, {conv_output_arg}, {output_arg});
      }

      auto& conv_node = helper.AddConvNode(input_arg, conv_output_arg, {96, 1, 3, 3});
      conv_node.AddAttribute("group", static_cast<int64_t>(96));
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
      if (!activation_op_type.empty()) {
        EXPECT_EQ(op_to_count[activation_op_type], 0);
      }
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  std::vector<std::string> activation_op_types{"", "Relu", "LeakyRelu"};
  for (auto& activation_op_type : activation_op_types) {
    test_case(activation_op_type);
  }
}

TEST(NchwcOptimizerTests, ConvPointwise) {
  auto test_case = [&](const std::string& activation_op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({16, 64, 28, 42});
      auto* output_arg = helper.MakeOutput();

      auto* conv_output_arg = output_arg;
      if (!activation_op_type.empty()) {
        conv_output_arg = helper.MakeIntermediate();
        helper.AddNode(activation_op_type, {conv_output_arg}, {output_arg});
      }

      helper.AddConvNode(input_arg, conv_output_arg, {128, 64, 1, 1});
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
      if (!activation_op_type.empty()) {
        EXPECT_EQ(op_to_count[activation_op_type], 0);
      }
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  std::vector<std::string> activation_op_types{"", "Relu", "LeakyRelu"};
  for (auto& activation_op_type : activation_op_types) {
    test_case(activation_op_type);
  }
}

TEST(NchwcOptimizerTests, ConvMaxPool) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput<float>({1, 48, 34, 34});
    auto* conv_output_arg = helper.MakeIntermediate();
    auto* output_arg = helper.MakeOutput();

    helper.AddConvNode(input_arg, conv_output_arg, {160, 48, 5, 5});

    auto& pool_node = helper.AddNode("MaxPool", {conv_output_arg}, {output_arg});
    pool_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
    pool_node.AddAttribute("kernel_shape", std::vector<int64_t>{5, 5});
  };

  auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.MaxPool"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
  };

  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, ConvMaxPoolDilations) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput<float>({1, 48, 66, 77});
    auto* conv_output_arg = helper.MakeIntermediate();
    auto* output_arg = helper.MakeOutput();

    helper.AddConvNode(input_arg, conv_output_arg, {160, 48, 5, 5});

    auto& pool_node = helper.AddNode("MaxPool", {conv_output_arg}, {output_arg});
    pool_node.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
    pool_node.AddAttribute("dilations", std::vector<int64_t>{2, 2});
  };

  auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.MaxPool"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
  };

  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, ConvAveragePool) {
  auto test_case = [&](bool count_include_pad) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({1, 48, 34, 34});
      auto* conv_output_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      helper.AddConvNode(input_arg, conv_output_arg, {128, 48, 5, 5});

      auto& pool_node = helper.AddNode("AveragePool", {conv_output_arg}, {output_arg});
      pool_node.AddAttribute("auto_pad", "SAME_UPPER");
      pool_node.AddAttribute("kernel_shape", std::vector<int64_t>{4, 4});
      if (count_include_pad) {
        pool_node.AddAttribute("count_include_pad", static_cast<int64_t>(1));
      }
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.AveragePool"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  test_case(false);
  test_case(true);
}

TEST(NchwcOptimizerTests, ConvGlobalPool) {
  auto test_case = [&](const std::string& op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({1, 96, 54, 54});
      auto* conv_output_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      auto& conv_node = helper.AddConvNode(input_arg, conv_output_arg, {160, 96, 3, 3});
      conv_node.AddAttribute("dilations", std::vector<int64_t>{2, 2});

      helper.AddNode(op_type, {conv_output_arg}, {output_arg});
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc." + op_type], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  std::vector<std::string> op_types{"GlobalMaxPool", "GlobalAveragePool"};
  for (auto& op_type : op_types) {
    test_case(op_type);
  }
}

TEST(NchwcOptimizerTests, ConvAddFusion) {
  auto test_case = [&](const std::string& op_type, int opset_version, bool do_relu) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({1, 32, 28, 28});
      auto* conv1_output_arg = helper.MakeIntermediate();
      auto* conv2_output_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      helper.AddConvNode(input_arg, conv1_output_arg, {32, 32, 3, 3});
      helper.AddConvNode(input_arg, conv2_output_arg, {32, 32, 3, 3});

      if (do_relu) {
        auto* add_output_arg = helper.MakeIntermediate();
        helper.AddNode(op_type, {conv1_output_arg, conv2_output_arg}, {add_output_arg});
        helper.AddNode("Relu", {add_output_arg}, {output_arg});
      } else {
        helper.AddNode(op_type, {conv1_output_arg, conv2_output_arg}, {output_arg});
      }
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 2);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
      EXPECT_EQ(op_to_count[op_type], 0);
      EXPECT_EQ(op_to_count["Relu"], 0);
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph, opset_version);
  };

  // Verify that Add or Sum can be fused into a preceding NCHWc Conv node,
  // with an optional Relu node following.
  std::vector<std::string> op_types{"Add", "Sum"};
  static const int opset_versions[] = {7, 10, 11, 12};
  for (auto& op_type : op_types) {
    for (auto opset_version : opset_versions) {
      test_case(op_type, opset_version, false);
      test_case(op_type, opset_version, true);
    }
  }
}

TEST(NchwcOptimizerTests, ConvNoBiasAddFusion) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput<float>({1, 32, 28, 28});
    auto* conv1_output_arg = helper.MakeIntermediate();
    auto* conv2_output_arg = helper.MakeIntermediate();
    auto* output_arg = helper.MakeOutput();

    helper.AddConvNode(input_arg, conv1_output_arg, {32, 32, 3, 3}, true);
    helper.AddConvNode(input_arg, conv2_output_arg, {32, 32, 3, 3}, true);
    helper.AddNode("Add", {conv1_output_arg, conv2_output_arg}, {output_arg});
  };

  auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 2);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
    EXPECT_EQ(op_to_count["Add"], 0);
  };

  // Verify that the optimizer can do the Conv/Add fusion when the Conv nodes
  // are missing the optional bias tensor.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, FusedConvAddFusion) {
  auto test_case = [&](bool do_relu1, bool do_relu2, int add_count) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({1, 32, 28, 28});
      auto* add1_input_arg = helper.MakeIntermediate();
      auto* add2_input_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      helper.AddConvNode(input_arg, add1_input_arg, {32, 32, 3, 3});
      if (do_relu1) {
        auto* relu_output_arg = helper.MakeIntermediate();
        helper.AddNode("Relu", {add1_input_arg}, {relu_output_arg});
        add1_input_arg = relu_output_arg;
      }

      helper.AddConvNode(input_arg, add2_input_arg, {32, 32, 3, 3});
      if (do_relu2) {
        auto* relu_output_arg = helper.MakeIntermediate();
        helper.AddNode("Relu", {add2_input_arg}, {relu_output_arg});
        add2_input_arg = relu_output_arg;
      }

      helper.AddNode("Add", {add1_input_arg, add2_input_arg}, {output_arg});
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 2);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
      EXPECT_EQ(op_to_count["Add"], add_count);
      EXPECT_EQ(op_to_count["Relu"], 0);
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  // More variations of Conv/Add fusion: one or more of the inputs to the Add
  // may already have a fused activation and cannot take the place of the Add
  // node, but can be an input to another Conv node that doesn't have a fused
  // activation.
  test_case(false, false, 0);
  test_case(false, true, 0);
  test_case(true, false, 0);
  test_case(true, true, 1);
}

TEST(NchwcOptimizerTests, ConvBinary) {
  auto test_case = [&](const std::string& op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({1, 32, 23, 23});
      auto* conv1_output_arg = helper.MakeIntermediate();
      auto* conv2_output_arg = helper.MakeIntermediate();
      auto* relu1_output_arg = helper.MakeIntermediate();
      auto* relu2_output_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      helper.AddConvNode(input_arg, conv1_output_arg, {32, 32, 3, 3});
      helper.AddNode("Relu", {conv1_output_arg}, {relu1_output_arg});
      helper.AddConvNode(input_arg, conv2_output_arg, {32, 32, 3, 3});
      helper.AddNode("Relu", {conv2_output_arg}, {relu2_output_arg});

      helper.AddNode(op_type, {relu1_output_arg, relu2_output_arg}, {output_arg});
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 2);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
      EXPECT_EQ(op_to_count[op_type], 1);
      EXPECT_EQ(op_to_count["Relu"], 0);
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  // Verify that the optimizer keeps the inputs to the binary operator as NCHWc
  // and only reorders the output of the binary operator.
  std::vector<std::string> op_types{"Add", "Sum", "Mul"};
  for (auto& op_type : op_types) {
    test_case(op_type);
  }
}

TEST(NchwcOptimizerTests, ConvBinaryBroadcast) {
  auto test_case = [&](const std::string& op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({1, 32, 25, 21});
      auto* conv_output_arg = helper.MakeIntermediate();
      auto* pool_output_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      helper.AddConvNode(input_arg, conv_output_arg, {32, 32, 3, 3});
      helper.AddNode("GlobalAveragePool", {input_arg}, {pool_output_arg});
      helper.AddNode(op_type, {conv_output_arg, pool_output_arg}, {output_arg});
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.GlobalAveragePool"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
      EXPECT_EQ(op_to_count["Reshape"], 3);
      EXPECT_EQ(op_to_count[op_type], 1);
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  // Verify that the optimizer keeps the inputs to the binary operator as NCHWc
  // and only reorders the output of the binary operator.
  std::vector<std::string> op_types{"Add", "Sum"};
  for (auto& op_type : op_types) {
    test_case(op_type);
  }
}

TEST(NchwcOptimizerTests, ConvConcat) {
  auto test_case = [&](int axis, int channel_count, int reorder_output_count) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({1, 48, 17, 34});
      auto* conv1_output_arg = helper.MakeIntermediate();
      auto* conv2_output_arg = helper.MakeIntermediate();
      auto* conv3_output_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      helper.AddConvNode(input_arg, conv1_output_arg, {64, 48, 5, 5});
      helper.AddConvNode(input_arg, conv2_output_arg, {channel_count, 48, 5, 5});
      helper.AddConvNode(input_arg, conv3_output_arg, {64, 48, 5, 5});

      auto& concat_node = helper.AddNode("Concat", {conv1_output_arg, conv2_output_arg, conv3_output_arg}, {output_arg});
      concat_node.AddAttribute("axis", static_cast<int64_t>(axis));
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 3);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], reorder_output_count);
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  // Concat along channel axis with aligned channel counts (stays in NCHWc format).
  test_case(1, 96, 1);

  // Concat along channel axis with unaligned channel counts (reorders back to NCHW).
  test_case(1, 98, 3);

  // Concat along non-channel axis (reorders back to NCHW).
  test_case(0, 64, 3);
}

TEST(NchwcOptimizerTests, ConvReuseWeightsOIHWBiBo) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput<float>({1, 64, 7, 7});
    auto* output1_arg = helper.MakeOutput();
    auto* output2_arg = helper.MakeOutput();
    auto* output3_arg = helper.MakeOutput();

    std::vector<int64_t> weights_shape{60, 64, 3, 3};
    auto* weights_arg = helper.MakeInitializer(weights_shape);
    auto* biases_arg = helper.MakeInitializer({weights_shape[0]});

    helper.AddNode("Conv", {input_arg, weights_arg, biases_arg}, {output1_arg});
    helper.AddNode("Conv", {input_arg, weights_arg, biases_arg}, {output2_arg});
    helper.AddNode("Conv", {input_arg, weights_arg, biases_arg}, {output3_arg});
  };

  auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 3);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 3);

    // Verify that the weights and biases were converted once and reused.
    std::unordered_set<const NodeArg*> weight_args;
    std::unordered_set<const NodeArg*> bias_args;
    const auto& graph = session.GetGraph();
    for (auto& node : graph.Nodes()) {
      if (node.Domain() == kMSNchwcDomain && node.OpType() == "Conv") {
        EXPECT_EQ(node.InputDefs().size(), 3u);
        weight_args.emplace(node.InputDefs()[1]);
        bias_args.emplace(node.InputDefs()[2]);
      }
    }
    EXPECT_EQ(weight_args.size(), 1u);
    EXPECT_EQ(bias_args.size(), 1u);
  };

  // Verify that a single weight tensor is reordered once.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, ConvReuseWeightsOIHWBo) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input1_arg = helper.MakeInput<float>({1, 64, 7, 7});
    auto* input2_arg = helper.MakeInput<float>({1, 64, 7, 7});
    auto* input3_arg = helper.MakeInput<float>({1, 1, 7, 7});
    auto* input4_arg = helper.MakeInput<float>({1, 1, 7, 7});
    auto* output1_arg = helper.MakeOutput();
    auto* output2_arg = helper.MakeOutput();
    auto* output3_arg = helper.MakeOutput();
    auto* output4_arg = helper.MakeOutput();

    std::vector<int64_t> weights_shape{64, 1, 3, 3};
    auto* weights_arg = helper.MakeInitializer(weights_shape);
    auto* biases_arg = helper.MakeInitializer({weights_shape[0]});

    auto& conv1_node = helper.AddNode("Conv", {input1_arg, weights_arg, biases_arg}, {output1_arg});
    conv1_node.AddAttribute("group", static_cast<int64_t>(64));

    auto& conv2_node = helper.AddNode("Conv", {input2_arg, weights_arg, biases_arg}, {output2_arg});
    conv2_node.AddAttribute("group", static_cast<int64_t>(64));

    helper.AddNode("Conv", {input3_arg, weights_arg, biases_arg}, {output3_arg});
    helper.AddNode("Conv", {input4_arg, weights_arg, biases_arg}, {output4_arg});
  };

  auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 4);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 2);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 4);

    // Verify that the weights and biases were converted once and reused.
    std::unordered_set<const NodeArg*> weight_args;
    std::unordered_set<const NodeArg*> bias_args;
    const auto& graph = session.GetGraph();
    for (auto& node : graph.Nodes()) {
      if (node.Domain() == kMSNchwcDomain && node.OpType() == "Conv") {
        EXPECT_EQ(node.InputDefs().size(), 3u);
        weight_args.emplace(node.InputDefs()[1]);
        bias_args.emplace(node.InputDefs()[2]);
      }
    }
    EXPECT_EQ(weight_args.size(), 1u);
    EXPECT_EQ(bias_args.size(), 1u);
  };

  // Verify that a single weight tensor is reordered once.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, ShapeInferencing) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
    type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("input_height");
    type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("input_width");

    auto* input_arg = helper.MakeInput<float>({1, 3, 50, 100}, type_proto);
    auto* output_arg = helper.MakeOutput();

    // With these padding and kernel arguments, the shape along each spatial
    // dimension is unchanged.
    auto* conv1_output_arg = helper.MakeIntermediate();
    auto& conv1_node = helper.AddConvNode(input_arg, conv1_output_arg, {48, 3, 3, 3});
    conv1_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});

    auto* pool2a_output_arg = helper.MakeIntermediate();
    auto& pool2a_node = helper.AddNode("MaxPool", {conv1_output_arg}, {pool2a_output_arg});
    pool2a_node.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
    pool2a_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});

    auto* pool2b_output_arg = helper.MakeIntermediate();
    auto& pool2b_node = helper.AddNode("MaxPool", {conv1_output_arg}, {pool2b_output_arg});
    pool2b_node.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
    pool2b_node.AddAttribute("auto_pad", "SAME_LOWER");

    auto* conv3a_output_arg = helper.MakeIntermediate();
    auto& conv3a_node = helper.AddConvNode(pool2a_output_arg, conv3a_output_arg, {64, 48, 3, 3});
    conv3a_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});

    auto* conv3b_output_arg = helper.MakeIntermediate();
    auto& conv3b_node = helper.AddConvNode(pool2b_output_arg, conv3b_output_arg, {64, 48, 3, 3});
    conv3b_node.AddAttribute("auto_pad", "SAME_UPPER");

    helper.AddNode("Add", {conv3a_output_arg, conv3b_output_arg}, {output_arg});
  };

  auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 3);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.MaxPool"], 2);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
    EXPECT_EQ(op_to_count["Add"], 0);
  };

  // The NCHWc optimizer does a limited amount of symbolic shape inferencing to
  // handle models such as YoloV3 which can have variable height/width. Without
  // shape inferencing, the transformer would be unable to detect that the inputs
  // to the Add node have identical shapes and thus is eligble for Conv/Add
  // fusion.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, ShapeInferencing2) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("input_height");
    type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("input_width");

    auto* input_arg = helper.MakeInput<float>({1, 1, 49, 98}, type_proto);
    auto* output_arg = helper.MakeOutput();

    auto* conv1_output_arg = helper.MakeIntermediate();
    helper.AddConvNode(input_arg, conv1_output_arg, {16, 1, 1, 1});

    auto* conv2a1_output_arg = helper.MakeIntermediate();
    auto& conv2a1_node = helper.AddConvNode(conv1_output_arg, conv2a1_output_arg, {16, 16, 2, 2});
    conv2a1_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 0, 0});
    conv2a1_node.AddAttribute("strides", std::vector<int64_t>{2, 2});

    auto* conv2a_output_arg = helper.MakeIntermediate();
    auto& conv2a2_node = helper.AddConvNode(conv2a1_output_arg, conv2a_output_arg, {16, 16, 2, 2});
    conv2a2_node.AddAttribute("auto_pad", "SAME_UPPER");

    auto* conv2b_output_arg = helper.MakeIntermediate();
    auto& conv2b_node = helper.AddConvNode(conv1_output_arg, conv2b_output_arg, {16, 16, 1, 1});
    conv2b_node.AddAttribute("strides", std::vector<int64_t>{2, 2});

    helper.AddNode("Add", {conv2a_output_arg, conv2b_output_arg}, {output_arg});
  };

  auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 4);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
    EXPECT_EQ(op_to_count["Add"], 0);
  };

  // Verify that convolutions using strides of 2 and variable height/width are
  // recognized as eligible for Conv/Add fusion. This pattern occurs in models
  // such as Faster-RCNN.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, MixedOutputUsage) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput<float>({6, 5, 11, 11});
    auto* output_arg = helper.MakeOutput();

    auto* conv1_output_arg = helper.MakeIntermediate();
    helper.AddConvNode(input_arg, conv1_output_arg, {96, 5, 2, 2});

    // Use conv1_output_arg as NCHWc.
    auto* conv2_output_arg = helper.MakeIntermediate();
    auto& conv2_node = helper.AddConvNode(conv1_output_arg, conv2_output_arg, {96, 96, 3, 3});
    conv2_node.AddAttribute("auto_pad", "SAME_LOWER");

    // Use conv1_output_arg as NCHW.
    auto* neg_output_arg = helper.MakeIntermediate();
    helper.AddNode("Neg", {conv1_output_arg}, {neg_output_arg});

    helper.AddNode("Add", {conv2_output_arg, neg_output_arg}, {output_arg});
  };

  auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 2);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 2);
  };

  // Verify that mixed NCHWc/NCHW usages of NCHWc nodes.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, TensorAlignment) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    // Input channel count must currently be a multiple of 4.
    auto* input1_arg = helper.MakeInput<float>({1, 62, 28, 42});
    auto* output1_arg = helper.MakeOutput();
    helper.AddConvNode(input1_arg, output1_arg, {128, 62, 1, 1});

    // Grouped input channel count must be a multiple of the NCHWc block size.
    auto* input2_arg = helper.MakeInput<float>({1, 48, 28, 42});
    auto* output2_arg = helper.MakeOutput();
    auto& conv2_node = helper.AddConvNode(input2_arg, output2_arg, {128, 12, 3, 3});
    conv2_node.AddAttribute("group", static_cast<int64_t>(4));

    // Grouped output channel count must be a multiple of the NCHWc block size.
    auto* input3_arg = helper.MakeInput<float>({1, 64, 28, 42});
    auto* output3_arg = helper.MakeOutput();
    auto& conv3_node = helper.AddConvNode(input3_arg, output3_arg, {48, 16, 3, 3});
    conv3_node.AddAttribute("group", static_cast<int64_t>(4));

    // Channel count must currently be a multiple of the NCHWc block size.
    auto* input4_arg = helper.MakeInput<float>({1, 60, 12, 12});
    auto* output4_arg = helper.MakeOutput();
    auto& pool_node = helper.AddNode("MaxPool", {input4_arg}, {output4_arg});
    pool_node.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  };

  auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["Conv"], 3);
    EXPECT_EQ(op_to_count["MaxPool"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.MaxPool"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 0);
  };

  // Verify that convolutions with unaligned inputs are not transformed.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, IntermediatesAsGraphOutputs) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput<float>({1, 48, 34, 34});
    auto* conv_output_arg = helper.MakeOutput();
    auto* output_arg = helper.MakeOutput();

    helper.AddConvNode(input_arg, conv_output_arg, {112, 48, 4, 4});

    auto& pool_node = helper.AddNode("MaxPool", {conv_output_arg}, {output_arg});
    pool_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 3, 3});
    pool_node.AddAttribute("kernel_shape", std::vector<int64_t>{4, 4});

    // conv_output_arg is not marked as an output by default because the node
    // argument is used as an input to another node, so the graph outputs must
    // be set explicitly.
    helper.graph_.SetOutputs({output_arg, conv_output_arg});
  };

  auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.MaxPool"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 2);
  };

  // Verify that intermediates used inside the graph but that are also graph
  // outputs result in the expected number of ReorderOutput nodes.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, BatchNormalization) {
  auto test_case = [&](bool training_outputs) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({1, 1, 23, 21});
      auto* conv1_output_arg = helper.MakeIntermediate();
      auto* conv2_output_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      // Using a channel count not aligned to the block size to verify handling
      // of unaligned data.
      helper.AddConvNode(input_arg, conv1_output_arg, {34, 1, 3, 3});
      helper.AddConvNode(input_arg, conv2_output_arg, {34, 1, 3, 3});

      auto* add_output_arg = helper.MakeIntermediate();
      helper.AddNode("Add", {conv1_output_arg, conv2_output_arg}, {add_output_arg});

      std::vector<float> bn_scale(34);
      std::vector<float> bn_bias(34);
      std::vector<float> bn_mean(34);
      std::vector<float> bn_var(34);

      for (int i = 0; i < 34; i++) {
        bn_scale[i] = static_cast<float>((i % 5) + 1) * 0.01f;
        bn_bias[i] = static_cast<float>(i - 17) * 0.25f;
        bn_mean[i] = static_cast<float>(i % 7) * 0.001f;
        bn_var[i] = static_cast<float>((i % 9) + 1) * 0.001f;
      }

      auto* bn_scale_arg = helper.Make1DInitializer(bn_scale);
      auto* bn_bias_arg = helper.Make1DInitializer(bn_bias);
      auto* bn_mean_arg = helper.Make1DInitializer(bn_mean);
      auto* bn_var_arg = helper.Make1DInitializer(bn_var);

      auto* bn_output_arg = helper.MakeIntermediate();
      std::vector<NodeArg*> bn_output_args{bn_output_arg};
      if (training_outputs) {
        bn_output_args.push_back(helper.MakeIntermediate());
        bn_output_args.push_back(helper.MakeIntermediate());
        bn_output_args.push_back(helper.MakeIntermediate());
        bn_output_args.push_back(helper.MakeIntermediate());
      }
      helper.AddNode("BatchNormalization", {add_output_arg, bn_scale_arg, bn_bias_arg, bn_mean_arg, bn_var_arg}, bn_output_args);
      helper.AddNode("Relu", {bn_output_arg}, {output_arg});

      // Override the sample tolerance for this test. By default, the NCHWc
      // tests generate bit identical results when run with and without
      // optimizations, but the BatchNormalization transform does introduce
      // small bit differences.
      helper.per_sample_tolerance_ = .00025;
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      if (training_outputs) {
        EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 2);
        EXPECT_EQ(op_to_count["BatchNormalization"], 1);
        EXPECT_EQ(op_to_count["Relu"], 1);
      } else {
        EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 3);
        EXPECT_EQ(op_to_count["BatchNormalization"], 0);
        EXPECT_EQ(op_to_count["Relu"], 0);
      }
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 0);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  // Verify that a batch normalization node can be converted to a convolution
  // if the input tensor is already in NCHWc format. However, this transform
  // should be skipped if the batch normalization node has the optional training
  // outputs supplied.
  test_case(false);
#if defined(ENABLE_TRAINING)
  test_case(true);
#endif
}

TEST(NchwcOptimizerTests, ConvReorderInputNhwc) {
  auto test_case = [&](int64_t channels) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({5, 27, 29, channels});
      auto* transpose_output_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      helper.AddTransposeToNchwNode(input_arg, transpose_output_arg);
      helper.AddConvNode(transpose_output_arg, output_arg, {34, channels, 1, 1});
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
      EXPECT_EQ(op_to_count["Transpose"], 0);
    };

    // Verify that a NHWC->NCHW transpose is fused into ReorderInput.
    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  for (int64_t channels = 16; channels <= 32; channels += 4) {
    test_case(channels);
  }
}

TEST(NchwcOptimizerTests, ConvReorderOutputNhwc) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput<float>({1, 64, 28, 32});
    auto* conv_output_arg = helper.MakeIntermediate();
    auto* nhwc_output_arg = helper.MakeOutput();

    helper.AddConvNode(input_arg, conv_output_arg, {130, 64, 1, 1});
    helper.AddTransposeToNhwcNode(conv_output_arg, nhwc_output_arg);
  };

  auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
    EXPECT_EQ(op_to_count["Transpose"], 0);
  };

  // Verify that a NHWC transpose is fused into ReorderOutput.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, ConvReorderOutputBoth) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput<float>({5, 64, 33, 37});
    auto* conv_output_arg = helper.MakeIntermediate();
    auto* nchw_output_arg = helper.MakeOutput();
    auto* nhwc_output_arg = helper.MakeOutput();

    helper.AddConvNode(input_arg, conv_output_arg, {7, 64, 1, 1});
    helper.AddTransposeToNhwcNode(conv_output_arg, nhwc_output_arg);
    helper.AddNode("Neg", {conv_output_arg}, {nchw_output_arg});
  };

  auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 2);
    EXPECT_EQ(op_to_count["Transpose"], 0);
  };

  // Verify that if an output argument is used as both NCHW and NHWC, then
  // two ReorderOutput nodes are inserted.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, ConvReorderOutputCnhw) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput<float>({1, 64, 28, 32});
    auto* conv_output_arg = helper.MakeIntermediate();
    auto* nhwc_output_arg = helper.MakeOutput();

    helper.AddConvNode(input_arg, conv_output_arg, {130, 64, 1, 1});
    helper.AddTransposeToCnhwNode(conv_output_arg, nhwc_output_arg);
  };

  auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
    EXPECT_EQ(op_to_count["Transpose"], 1);
  };

  // Verify that a CNHW transpose is not fused into ReorderOutput.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, UpsampleNearest) {
  auto test_case = [&](int opset_version, float scale_h, float scale_w, bool use_sizes_arg) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({3, 16, 27, 15});
      auto* conv_output_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      helper.AddConvNode(input_arg, conv_output_arg, {42, 16, 1, 1});

      std::string op_name = opset_version >= 10 ? "Resize" : "Upsample";
      std::vector<NodeArg*> input_args;
      input_args.push_back(conv_output_arg);
      if (opset_version >= 11) {
        input_args.push_back(helper.Make1DInitializer<float>({0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f}));
      }
      if (use_sizes_arg) {
        std::vector<int64_t> sizes_shape(4);
        sizes_shape[0] = 3;
        sizes_shape[1] = 42;
        sizes_shape[2] = static_cast<int64_t>(scale_h * 27);
        sizes_shape[3] = static_cast<int64_t>(scale_w * 15);
        input_args.push_back(helper.Make1DInitializer<float>({}));
        input_args.push_back(helper.Make1DInitializer<int64_t>(sizes_shape));
      } else {
        input_args.push_back(helper.Make1DInitializer<float>({1.f, 1.f, scale_h, scale_w}));
      }
      Node& resize_node = helper.AddNode(op_name, input_args, {output_arg});
      if (opset_version >= 11) {
        resize_node.AddAttribute("coordinate_transformation_mode", "asymmetric");
        resize_node.AddAttribute("nearest_mode", "floor");
      } else if (opset_version == 10) {
        // Explicitly set the mode to nearest as an extra test.
        resize_node.AddAttribute("mode", "nearest");
      }
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
      if (scale_h == std::round(scale_h) && scale_w == std::round(scale_w)) {
        EXPECT_EQ(op_to_count["com.microsoft.nchwc.Upsample"], 1);
        EXPECT_EQ(op_to_count["Resize"] + op_to_count["Upsample"], 0);
      } else {
        EXPECT_EQ(op_to_count["com.microsoft.nchwc.Upsample"], 0);
        EXPECT_EQ(op_to_count["Resize"] + op_to_count["Upsample"], 1);
      }
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph, opset_version);
  };

  // Verify that upsample nodes can be converted to the NCHWc format for
  // various versions of the operator.
  static const int opset_versions[] = {9, 10, 11, 13};
  for (auto opset_version : opset_versions) {
    test_case(opset_version, 1.f, 1.f, false);
    test_case(opset_version, 2.f, 2.f, false);
    test_case(opset_version, 3.f, 5.f, false);
    if (opset_version >= 11) {
      test_case(opset_version, 2.f, 2.f, true);
      test_case(opset_version, 5.f, 3.f, true);
    }
  }
  // Verify that non-integral scales are not converted to the NCHWc format.
  test_case(13, 2.2f, 2.8f, false);
  test_case(13, 2.2f, 2.8f, true);
}

TEST(NchwcOptimizerTests, UpsampleLinear) {
  auto test_case = [&](int opset_version, float scale_h, float scale_w, const std::string& transformation_mode) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({3, 16, 21, 25});
      auto* conv_output_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      helper.AddConvNode(input_arg, conv_output_arg, {28, 16, 1, 1});

      std::string op_name = opset_version >= 10 ? "Resize" : "Upsample";
      std::vector<NodeArg*> input_args;
      input_args.push_back(conv_output_arg);
      if (opset_version >= 11) {
        input_args.push_back(helper.Make1DInitializer<float>({0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f}));
      }
      input_args.push_back(helper.Make1DInitializer<float>({1.f, 1.f, scale_h, scale_w}));
      Node& resize_node = helper.AddNode(op_name, input_args, {output_arg});
      resize_node.AddAttribute("mode", "linear");
      if (opset_version >= 11) {
        resize_node.AddAttribute("coordinate_transformation_mode", transformation_mode);
      }

      helper.per_sample_tolerance_ = .001f;
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Upsample"], 1);
      EXPECT_EQ(op_to_count["Resize"] + op_to_count["Upsample"], 0);
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph, opset_version);
  };

  // Verify that upsample nodes can be converted to the NCHWc format for
  // various versions of the operator.
  std::vector<std::string> transformation_modes{"asymmetric", "align_corners", "half_pixel"};
  for (auto& transformation_mode : transformation_modes) {
    static const int opset_versions[] = {9, 10, 11, 13};
    for (auto opset_version : opset_versions) {
      // Older versions of the operator do not support transformation modes.
      if (opset_version < 11 && transformation_mode == "asymmetric") {
        continue;
      }
      test_case(opset_version, 1.f, 1.f, transformation_mode);
      test_case(opset_version, 2.f, 2.f, transformation_mode);
      test_case(opset_version, 3.f, 5.f, transformation_mode);
      test_case(opset_version, 9.f, 7.f, transformation_mode);
    }
  }
}

TEST(NchwcOptimizerTests, Activation) {
  auto test_case = [&](const std::string& activation_op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput<float>({1, 48, 11, 15});
      auto* conv1_output_arg = helper.MakeIntermediate();
      auto* activation_output_arg = helper.MakeIntermediate();
      auto* mul_output_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      helper.AddConvNode(input_arg, conv1_output_arg, {32, 48, 3, 3});
      helper.AddNode(activation_op_type, {conv1_output_arg}, {activation_output_arg});
      helper.AddNode("Add", {conv1_output_arg, activation_output_arg}, {mul_output_arg});
      helper.AddConvNode(mul_output_arg, output_arg, {16, 32, 1, 1});
    };

    auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.Conv"], 2);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
      EXPECT_EQ(op_to_count[activation_op_type], 1);
      EXPECT_EQ(op_to_count["Add"], 1);
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  // Verify that the optimizer doesn't add reorders for these activations that
  // cannot be fused with a convolution.
  std::vector<std::string> activation_op_types{"Relu", "Sigmoid", "Tanh"};
  for (auto& activation_op_type : activation_op_types) {
    test_case(activation_op_type);
  }
}

TEST(NchwcOptimizerTests, MaxPoolTypeCheck) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto add_pool_node = [&](NchwcTestHelper& helper, NodeArg* input_arg) {
      auto* output_arg = helper.MakeOutput();
      auto& pool_node = helper.AddNode("MaxPool", {input_arg}, {output_arg});
      pool_node.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
      pool_node.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
    };

    const std::vector<int64_t> input_shape{1, 32, 13, 13};
    add_pool_node(helper, helper.MakeInput<float>(input_shape));
    add_pool_node(helper, helper.MakeInput<uint8_t>(input_shape));
  };

  auto check_nchwc_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["MaxPool"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.MaxPool"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderInput"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.nchwc.ReorderOutput"], 1);
  };

  // Verify that the optimizer checks the type of the MaxPool node.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph, 12);
}

#endif

}  // namespace test
}  // namespace onnxruntime
