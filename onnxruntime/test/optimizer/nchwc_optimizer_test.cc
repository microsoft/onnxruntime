// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "test/compare_ortvalue.h"
#include "gtest/gtest.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace test {

// InferenceSession wrapper in order to gain access to the loaded graph.
class NchwcInferenceSession : public InferenceSession {
 public:
  explicit NchwcInferenceSession(const SessionOptions& session_options,
                                 logging::LoggingManager* logging_manager) : InferenceSession(session_options, logging_manager) {
  }

  std::unordered_map<std::string, int> CountOpsInGraph() {
    std::unordered_map<std::string, int> op_to_count;
    if (model_.get() != nullptr) {
      for (auto& node : model_->MainGraph().Nodes()) {
        std::string key = node.OpType();
        if (node.Domain() == kMSNchwcDomain) {
          key = "nchwc." + key;
        }
        op_to_count[key] = op_to_count[key] + 1;
      }
    }
    return op_to_count;
  }

  const Graph& GetGraph() {
    return model_->MainGraph();
  }
};

struct NchwcTestHelper {
  NchwcTestHelper(Graph& graph) : graph_(graph), fill_value_(0) {
  }

  NodeArg* MakeInput(const std::vector<int64_t>& shape, const ONNX_NAMESPACE::TypeProto& type_proto) {
    int64_t num_elements = 1;
    for (auto& dim : shape) {
      num_elements *= dim;
    }

    OrtValue input_value;
    CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), shape,
                         FillRandomData(static_cast<size_t>(num_elements)), &input_value);
    std::string name = graph_.GenerateNodeArgName("input");
    feeds_.insert(std::make_pair(name, input_value));

    return &graph_.GetOrCreateNodeArg(name, &type_proto);
  }

  NodeArg* MakeInput(const std::vector<int64_t>& shape) {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

    for (auto& dim : shape) {
      type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
    }

    return MakeInput(shape, type_proto);
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

  NodeArg* MakeInitializer(const std::vector<int64_t>& shape) {
    std::string name = graph_.GenerateNodeArgName("constant");
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.set_name(name);
    tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

    int64_t num_elements = 1;
    for (auto& dim : shape) {
      tensor_proto.add_dims(dim);
      num_elements *= dim;
    }

    auto random_data = FillRandomData(static_cast<size_t>(num_elements));
    tensor_proto.mutable_float_data()->Resize(static_cast<int>(num_elements), 0.0f);
    memcpy(tensor_proto.mutable_float_data()->mutable_data(), random_data.data(), random_data.size() * sizeof(float));

    graph_.AddInitializedTensor(tensor_proto);

    return &graph_.GetOrCreateNodeArg(name, nullptr);
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
    std::vector<NodeArg*> input_args = {input_arg, weights_arg};
    if (!no_bias) {
      auto* biases_arg = MakeInitializer({weights_shape[0]});
      input_args.push_back(biases_arg);
    }
    return AddNode("Conv", input_args, {output_arg});
  }

  std::vector<float> FillRandomData(size_t count) {
    constexpr int min_fill_value = -23;
    constexpr int max_fill_value = 23;

    std::vector<float> random_data;
    random_data.resize(count);
    for (size_t n = 0; n < count; n++) {
      random_data[n] = static_cast<float>(fill_value_);
      fill_value_++;
      if (fill_value_ == max_fill_value) {
        fill_value_ = min_fill_value;
      }
    }
    return random_data;
  }

  Graph& graph_;
  NameMLValMap feeds_;
  std::vector<std::string> output_names_;
  int fill_value_;
};

void NchwcOptimizerTester(const std::function<void(NchwcTestHelper& helper)>& build_test_case,
                          const std::function<void(NchwcInferenceSession& session)>& check_nchwc_graph,
                          int opset_version = 10) {
  // Ignore the test if NCHWc is not supported by the platform.
  if (MlasNchwcGetBlockSize() <= 1) {
    return;
  }

  // Build the model for this test.
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = opset_version;
  Model model("nchwc", false, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), domain_to_version);
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
    NchwcInferenceSession session{session_options, &DefaultLoggingManager()};
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
    double per_sample_tolerance = 0.0;
    double relative_per_sample_tolerance = 0.0;
    std::pair<COMPARE_RESULT, std::string> ret =
        CompareOrtValue(level3_fetches[i],
                        level2_fetches[i],
                        per_sample_tolerance,
                        relative_per_sample_tolerance,
                        false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS);
  }
}

#ifndef DISABLE_CONTRIB_OPS

TEST(NchwcOptimizerTests, ConvNchw) {
  auto test_case = [&](const std::string& activation_op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput({16, 3, 112, 112});
      auto* output_arg = helper.MakeOutput();

      auto* conv_output_arg = output_arg;
      if (!activation_op_type.empty()) {
        conv_output_arg = helper.MakeIntermediate();
        auto& act_node = helper.AddNode(activation_op_type, {conv_output_arg}, {output_arg});
        if (activation_op_type == "Clip") {
          act_node.AddAttribute("min", 0.0f);
          act_node.AddAttribute("max", 6.0f);
        }
      }

      auto& conv_node = helper.AddConvNode(input_arg, conv_output_arg, {130, 3, 3, 3});
      conv_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
      conv_node.AddAttribute("strides", std::vector<int64_t>{2, 2});
    };

    auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
      auto op_to_count = session.CountOpsInGraph();
      EXPECT_EQ(op_to_count["nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 0);
      EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 1);
      if (!activation_op_type.empty()) {
        EXPECT_EQ(op_to_count[activation_op_type], 0);
      }
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  std::vector<std::string> activation_op_types = {"", "Relu", "LeakyRelu", "Clip"};
  for (auto& activation_op_type : activation_op_types) {
    test_case(activation_op_type);
  }
}

TEST(NchwcOptimizerTests, ConvNchwc) {
  auto test_case = [&](const std::string& activation_op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput({16, 64, 28, 28});
      auto* output_arg = helper.MakeOutput();

      auto* conv_output_arg = output_arg;
      if (!activation_op_type.empty()) {
        conv_output_arg = helper.MakeIntermediate();
        auto& act_node = helper.AddNode(activation_op_type, {conv_output_arg}, {output_arg});
        if (activation_op_type == "Clip") {
          act_node.AddAttribute("min", -6.0f);
          act_node.AddAttribute("max", 6.0f);
        }
      }

      helper.AddConvNode(input_arg, conv_output_arg, {127, 64, 3, 3});
    };

    auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
      auto op_to_count = session.CountOpsInGraph();
      EXPECT_EQ(op_to_count["nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 1);
      if (!activation_op_type.empty()) {
        EXPECT_EQ(op_to_count[activation_op_type], 0);
      }
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  std::vector<std::string> activation_op_types = {"", "Relu", "LeakyRelu", "Clip"};
  for (auto& activation_op_type : activation_op_types) {
    test_case(activation_op_type);
  }
}

TEST(NchwcOptimizerTests, ConvNchwcGrouped) {
  auto test_case = [&](const std::string& activation_op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput({16, 48, 28, 28});
      auto* output_arg = helper.MakeOutput();

      auto* conv_output_arg = output_arg;
      if (!activation_op_type.empty()) {
        conv_output_arg = helper.MakeIntermediate();
        helper.AddNode(activation_op_type, {conv_output_arg}, {output_arg});
      }

      auto& conv_node = helper.AddConvNode(input_arg, conv_output_arg, {192, 16, 3, 3});
      conv_node.AddAttribute("group", static_cast<int64_t>(3));
    };

    auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
      auto op_to_count = session.CountOpsInGraph();
      EXPECT_EQ(op_to_count["nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 1);
      if (!activation_op_type.empty()) {
        EXPECT_EQ(op_to_count[activation_op_type], 0);
      }
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  std::vector<std::string> activation_op_types = {"", "Relu", "LeakyRelu"};
  for (auto& activation_op_type : activation_op_types) {
    test_case(activation_op_type);
  }
}

TEST(NchwcOptimizerTests, ConvDepthwise) {
  auto test_case = [&](const std::string& activation_op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput({16, 96, 28, 28});
      auto* output_arg = helper.MakeOutput();

      auto* conv_output_arg = output_arg;
      if (!activation_op_type.empty()) {
        conv_output_arg = helper.MakeIntermediate();
        helper.AddNode(activation_op_type, {conv_output_arg}, {output_arg});
      }

      auto& conv_node = helper.AddConvNode(input_arg, conv_output_arg, {96, 1, 3, 3});
      conv_node.AddAttribute("group", static_cast<int64_t>(96));
    };

    auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
      auto op_to_count = session.CountOpsInGraph();
      EXPECT_EQ(op_to_count["nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 1);
      if (!activation_op_type.empty()) {
        EXPECT_EQ(op_to_count[activation_op_type], 0);
      }
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  std::vector<std::string> activation_op_types = {"", "Relu", "LeakyRelu"};
  for (auto& activation_op_type : activation_op_types) {
    test_case(activation_op_type);
  }
}

TEST(NchwcOptimizerTests, ConvPointwise) {
  auto test_case = [&](const std::string& activation_op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput({16, 64, 28, 42});
      auto* output_arg = helper.MakeOutput();

      auto* conv_output_arg = output_arg;
      if (!activation_op_type.empty()) {
        conv_output_arg = helper.MakeIntermediate();
        helper.AddNode(activation_op_type, {conv_output_arg}, {output_arg});
      }

      helper.AddConvNode(input_arg, conv_output_arg, {128, 64, 1, 1});
    };

    auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
      auto op_to_count = session.CountOpsInGraph();
      EXPECT_EQ(op_to_count["nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 1);
      if (!activation_op_type.empty()) {
        EXPECT_EQ(op_to_count[activation_op_type], 0);
      }
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  std::vector<std::string> activation_op_types = {"", "Relu", "LeakyRelu"};
  for (auto& activation_op_type : activation_op_types) {
    test_case(activation_op_type);
  }
}

TEST(NchwcOptimizerTests, ConvMaxPool) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput({1, 48, 34, 34});
    auto* conv_output_arg = helper.MakeIntermediate();
    auto* output_arg = helper.MakeOutput();

    helper.AddConvNode(input_arg, conv_output_arg, {160, 48, 5, 5});

    auto& pool_node = helper.AddNode("MaxPool", {conv_output_arg}, {output_arg});
    pool_node.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
    pool_node.AddAttribute("kernel_shape", std::vector<int64_t>{5, 5});
  };

  auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
    auto op_to_count = session.CountOpsInGraph();
    EXPECT_EQ(op_to_count["nchwc.Conv"], 1);
    EXPECT_EQ(op_to_count["nchwc.MaxPool"], 1);
    EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 1);
    EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 1);
  };

  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, ConvMaxPoolDilations) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput({1, 48, 66, 77});
    auto* conv_output_arg = helper.MakeIntermediate();
    auto* output_arg = helper.MakeOutput();

    helper.AddConvNode(input_arg, conv_output_arg, {160, 48, 5, 5});

    auto& pool_node = helper.AddNode("MaxPool", {conv_output_arg}, {output_arg});
    pool_node.AddAttribute("kernel_shape", std::vector<int64_t>{3, 3});
    pool_node.AddAttribute("dilations", std::vector<int64_t>{2, 2});
  };

  auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
    auto op_to_count = session.CountOpsInGraph();
    EXPECT_EQ(op_to_count["nchwc.Conv"], 1);
    EXPECT_EQ(op_to_count["nchwc.MaxPool"], 1);
    EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 1);
    EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 1);
  };

  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, ConvAveragePool) {
  auto test_case = [&](bool count_include_pad) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput({1, 48, 34, 34});
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

    auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
      auto op_to_count = session.CountOpsInGraph();
      EXPECT_EQ(op_to_count["nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["nchwc.AveragePool"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 1);
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  test_case(false);
  test_case(true);
}

TEST(NchwcOptimizerTests, ConvGlobalPool) {
  auto test_case = [&](const std::string& op_type) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput({1, 96, 54, 54});
      auto* conv_output_arg = helper.MakeIntermediate();
      auto* output_arg = helper.MakeOutput();

      auto& conv_node = helper.AddConvNode(input_arg, conv_output_arg, {160, 96, 3, 3});
      conv_node.AddAttribute("dilations", std::vector<int64_t>{2, 2});

      helper.AddNode(op_type, {conv_output_arg}, {output_arg});
    };

    auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
      auto op_to_count = session.CountOpsInGraph();
      EXPECT_EQ(op_to_count["nchwc.Conv"], 1);
      EXPECT_EQ(op_to_count["nchwc." + op_type], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 1);
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph);
  };

  std::vector<std::string> op_types = {"GlobalMaxPool", "GlobalAveragePool"};
  for (auto& op_type : op_types) {
    test_case(op_type);
  }
}

TEST(NchwcOptimizerTests, ConvAddFusion) {
  auto test_case = [&](const std::string& op_type, int opset_version, bool do_relu) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput({1, 32, 28, 28});
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

    auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
      auto op_to_count = session.CountOpsInGraph();
      EXPECT_EQ(op_to_count["nchwc.Conv"], 2);
      EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 1);
      EXPECT_EQ(op_to_count[op_type], 0);
      EXPECT_EQ(op_to_count["Relu"], 0);
    };

    NchwcOptimizerTester(build_test_case, check_nchwc_graph, opset_version);
  };

  // Verify that Add or Sum can be fused into a preceding NCHWc Conv node,
  // with an optional Relu node following.
  std::vector<std::string> op_types = {"Add", "Sum"};
  static const int opset_versions[] = {7, 10};
  for (auto& op_type : op_types) {
    for (auto opset_version : opset_versions) {
      test_case(op_type, opset_version, false);
      test_case(op_type, opset_version, true);
    }
  }
}

TEST(NchwcOptimizerTests, ConvNoBiasAddFusion) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput({1, 32, 28, 28});
    auto* conv1_output_arg = helper.MakeIntermediate();
    auto* conv2_output_arg = helper.MakeIntermediate();
    auto* output_arg = helper.MakeOutput();

    helper.AddConvNode(input_arg, conv1_output_arg, {32, 32, 3, 3}, true);
    helper.AddConvNode(input_arg, conv2_output_arg, {32, 32, 3, 3}, true);
    helper.AddNode("Add", {conv1_output_arg, conv2_output_arg}, {output_arg});
  };

  auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
    auto op_to_count = session.CountOpsInGraph();
    EXPECT_EQ(op_to_count["nchwc.Conv"], 2);
    EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 1);
    EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 1);
    EXPECT_EQ(op_to_count["Add"], 0);
  };

  // Verify that the optimizer can do the Conv/Add fusion when the Conv nodes
  // are missing the optional bias tensor.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, FusedConvAddFusion) {
  auto test_case = [&](bool do_relu1, bool do_relu2, int add_count) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput({1, 32, 28, 28});
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

    auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
      auto op_to_count = session.CountOpsInGraph();
      EXPECT_EQ(op_to_count["nchwc.Conv"], 2);
      EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 1);
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

TEST(NchwcOptimizerTests, ConvConcat) {
  auto test_case = [&](int axis, int channel_count, int reorder_output_count) {
    auto build_test_case = [&](NchwcTestHelper& helper) {
      auto* input_arg = helper.MakeInput({1, 48, 17, 34});
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

    auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
      auto op_to_count = session.CountOpsInGraph();
      EXPECT_EQ(op_to_count["nchwc.Conv"], 3);
      EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 1);
      EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], reorder_output_count);
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
    auto* input_arg = helper.MakeInput({1, 64, 7, 7});
    auto* output1_arg = helper.MakeOutput();
    auto* output2_arg = helper.MakeOutput();
    auto* output3_arg = helper.MakeOutput();

    std::vector<int64_t> weights_shape = {60, 64, 3, 3};
    auto* weights_arg = helper.MakeInitializer(weights_shape);
    auto* biases_arg = helper.MakeInitializer({weights_shape[0]});

    helper.AddNode("Conv", {input_arg, weights_arg, biases_arg}, {output1_arg});
    helper.AddNode("Conv", {input_arg, weights_arg, biases_arg}, {output2_arg});
    helper.AddNode("Conv", {input_arg, weights_arg, biases_arg}, {output3_arg});
  };

  auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
    auto op_to_count = session.CountOpsInGraph();
    EXPECT_EQ(op_to_count["nchwc.Conv"], 3);
    EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 1);
    EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 3);

    // Verify that the weights and biases were converted once and reused.
    std::unordered_set<const NodeArg*> weight_args;
    std::unordered_set<const NodeArg*> bias_args;
    const auto& graph = session.GetGraph();
    for (auto& node : graph.Nodes()) {
      if (node.Domain() == kMSNchwcDomain && node.OpType() == "Conv") {
        EXPECT_EQ(node.InputDefs().size(), 3);
        weight_args.emplace(node.InputDefs()[1]);
        bias_args.emplace(node.InputDefs()[2]);
      }
    }
    EXPECT_EQ(weight_args.size(), 1);
    EXPECT_EQ(bias_args.size(), 1);
  };

  // Verify that a single weight tensor is reordered once.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, ConvReuseWeightsOIHWBo) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input1_arg = helper.MakeInput({1, 64, 7, 7});
    auto* input2_arg = helper.MakeInput({1, 64, 7, 7});
    auto* input3_arg = helper.MakeInput({1, 1, 7, 7});
    auto* input4_arg = helper.MakeInput({1, 1, 7, 7});
    auto* output1_arg = helper.MakeOutput();
    auto* output2_arg = helper.MakeOutput();
    auto* output3_arg = helper.MakeOutput();
    auto* output4_arg = helper.MakeOutput();

    std::vector<int64_t> weights_shape = {64, 1, 3, 3};
    auto* weights_arg = helper.MakeInitializer(weights_shape);
    auto* biases_arg = helper.MakeInitializer({weights_shape[0]});

    auto& conv1_node = helper.AddNode("Conv", {input1_arg, weights_arg, biases_arg}, {output1_arg});
    conv1_node.AddAttribute("group", static_cast<int64_t>(64));

    auto& conv2_node = helper.AddNode("Conv", {input2_arg, weights_arg, biases_arg}, {output2_arg});
    conv2_node.AddAttribute("group", static_cast<int64_t>(64));

    helper.AddNode("Conv", {input3_arg, weights_arg, biases_arg}, {output3_arg});
    helper.AddNode("Conv", {input4_arg, weights_arg, biases_arg}, {output4_arg});
  };

  auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
    auto op_to_count = session.CountOpsInGraph();
    EXPECT_EQ(op_to_count["nchwc.Conv"], 4);
    EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 2);
    EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 4);

    // Verify that the weights and biases were converted once and reused.
    std::unordered_set<const NodeArg*> weight_args;
    std::unordered_set<const NodeArg*> bias_args;
    const auto& graph = session.GetGraph();
    for (auto& node : graph.Nodes()) {
      if (node.Domain() == kMSNchwcDomain && node.OpType() == "Conv") {
        EXPECT_EQ(node.InputDefs().size(), 3);
        weight_args.emplace(node.InputDefs()[1]);
        bias_args.emplace(node.InputDefs()[2]);
      }
    }
    EXPECT_EQ(weight_args.size(), 1);
    EXPECT_EQ(bias_args.size(), 1);
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

    auto* input_arg = helper.MakeInput({1, 3, 50, 100}, type_proto);
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

  auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
    auto op_to_count = session.CountOpsInGraph();
    EXPECT_EQ(op_to_count["nchwc.Conv"], 3);
    EXPECT_EQ(op_to_count["nchwc.MaxPool"], 2);
    EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 0);
    EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 1);
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

    auto* input_arg = helper.MakeInput({1, 1, 49, 98}, type_proto);
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

  auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
    auto op_to_count = session.CountOpsInGraph();
    EXPECT_EQ(op_to_count["nchwc.Conv"], 4);
    EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 0);
    EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 1);
    EXPECT_EQ(op_to_count["Add"], 0);
  };

  // Verify that convolutions using strides of 2 and variable height/width are
  // recognized as eligible for Conv/Add fusion. This pattern occurs in models
  // such as Faster-RCNN.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, MixedOutputUsage) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput({6, 5, 11, 11});
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

  auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
    auto op_to_count = session.CountOpsInGraph();
    EXPECT_EQ(op_to_count["nchwc.Conv"], 2);
    EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 0);
    EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 2);
  };

  // Verify that mixed NCHWc/NCHW usages of NCHWc nodes.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

TEST(NchwcOptimizerTests, TensorAlignment) {
  auto build_test_case = [&](NchwcTestHelper& helper) {
    // Input channel count must currently be a multiple of the NCHWc block size.
    auto* input1_arg = helper.MakeInput({1, 60, 28, 42});
    auto* output1_arg = helper.MakeOutput();
    helper.AddConvNode(input1_arg, output1_arg, {128, 60, 1, 1});

    // Grouped input channel count must be a multiple of the NCHWc block size.
    auto* input2_arg = helper.MakeInput({1, 48, 28, 42});
    auto* output2_arg = helper.MakeOutput();
    auto& conv2_node = helper.AddConvNode(input2_arg, output2_arg, {128, 12, 3, 3});
    conv2_node.AddAttribute("group", static_cast<int64_t>(4));

    // Grouped output channel count must be a multiple of the NCHWc block size.
    auto* input3_arg = helper.MakeInput({1, 64, 28, 42});
    auto* output3_arg = helper.MakeOutput();
    auto& conv3_node = helper.AddConvNode(input3_arg, output3_arg, {48, 16, 3, 3});
    conv3_node.AddAttribute("group", static_cast<int64_t>(4));

    // Channel count must currently be a multiple of the NCHWc block size.
    auto* input4_arg = helper.MakeInput({1, 60, 12, 12});
    auto* output4_arg = helper.MakeOutput();
    auto& pool_node = helper.AddNode("MaxPool", {input4_arg}, {output4_arg});
    pool_node.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  };

  auto check_nchwc_graph = [&](NchwcInferenceSession& session) {
    auto op_to_count = session.CountOpsInGraph();
    EXPECT_EQ(op_to_count["Conv"], 3);
    EXPECT_EQ(op_to_count["MaxPool"], 1);
    EXPECT_EQ(op_to_count["nchwc.Conv"], 0);
    EXPECT_EQ(op_to_count["nchwc.MaxPool"], 0);
    EXPECT_EQ(op_to_count["nchwc.ReorderInput"], 0);
    EXPECT_EQ(op_to_count["nchwc.ReorderOutput"], 0);
  };

  // Verify that convolutions with unaligned inputs are not transformed.
  NchwcOptimizerTester(build_test_case, check_nchwc_graph);
}

#endif

}  // namespace test
}  // namespace onnxruntime
