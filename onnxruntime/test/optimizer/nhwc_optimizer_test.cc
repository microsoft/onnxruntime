// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_ACL)

#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "test/compare_ortvalue.h"
#include "gtest/gtest.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// InferenceSession wrapper in order to gain access to the loaded graph.
class NhwcInferenceSession : public InferenceSession {
 public:
  explicit NhwcInferenceSession(const SessionOptions& session_options,
                                 const Environment& env) : InferenceSession(session_options, env) {
  }

  std::unordered_map<std::string, int> CountOpsInGraph() {
    std::unordered_map<std::string, int> op_to_count;
    if (model_.get() != nullptr) {
      for (auto& node : model_->MainGraph().Nodes()) {
        std::string key = node.OpType();
        if (node.Domain() == kMSNhwcDomain) {
          key = "nhwc." + key;
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

struct NhwcTestHelper {
  NhwcTestHelper(Graph& graph) : graph_(graph), fill_value_(0) {
  }

  NodeArg* MakeInput(const std::vector<int64_t>& shape, const ONNX_NAMESPACE::TypeProto& type_proto) {
    int64_t num_elements = 1;
    for (auto& dim : shape) {
      num_elements *= dim;
    }

    OrtValue input_value;
    CreateMLValue<float>(TestACLExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), shape,
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

  NodeArg* MakeInitializer(const std::vector<int64_t>& shape, const std::vector<float>& data) {
    std::string name = graph_.GenerateNodeArgName("constant");
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.set_name(name);
    tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

    for (auto& dim : shape) {
      tensor_proto.add_dims(dim);
    }

    tensor_proto.mutable_float_data()->Resize(static_cast<int>(data.size()), 0.0f);
    memcpy(tensor_proto.mutable_float_data()->mutable_data(), data.data(), data.size() * sizeof(float));

    graph_.AddInitializedTensor(tensor_proto);

    return &graph_.GetOrCreateNodeArg(name, nullptr);
  }

  NodeArg* MakeInitializer(const std::vector<int64_t>& shape) {
    int64_t num_elements = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>{});
    return MakeInitializer(shape, FillRandomData(static_cast<size_t>(num_elements)));
  }

  Node& AddNode(const std::string& op_type,
                const std::vector<NodeArg*>& input_args,
                const std::vector<NodeArg*>& output_args) {
   Node& node = graph_.AddNode(graph_.GenerateNodeName("node"),
                          op_type,
                          "description",
                          input_args,
                          output_args,
                          nullptr,
                          kOnnxDomain);
   node.SetExecutionProviderType(kAclExecutionProvider);
   return node;
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

    std::vector<float> random_data {};
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

void NhwcOptimizerTester(const std::function<void(NhwcTestHelper& helper)>& build_test_case,
                          const std::function<void(NhwcInferenceSession& session)>& check_nhwc_graph,
                          int opset_version = 10) {

#ifndef USE_ACL
  return;
#endif

  // Build the model for this test.
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = opset_version;
  Model model("nhwc", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, {}, DefaultLoggingManager().DefaultLogger());

  NhwcTestHelper helper(model.MainGraph());
  build_test_case(helper);
  ASSERT_TRUE(model.MainGraph().Resolve().IsOK());

  // assign all nodes to ACL. the constant folding should override this to perform the constant folding on cpu
  for (auto& node : model.MainGraph().Nodes())
    node.SetExecutionProviderType(kAclExecutionProvider);

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  auto run_model = [&](TransformerLevel level, std::vector<OrtValue>& fetches) {
    SessionOptions session_options;
    session_options.graph_optimization_level = level;
    session_options.session_logid = "NhwcOptimizerTests";

    NhwcInferenceSession session{session_options, GetEnvironment()};
    ASSERT_TRUE(session.RegisterExecutionProvider(std::move(DefaultAclExecutionProvider())).IsOK());
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
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS);
  }
}

#ifndef DISABLE_CONTRIB_OPS
TEST(NhwcOptimizerTests, Test1Nhwc) {
  auto build_test_case = [&](NhwcTestHelper& helper) {
    auto* input_arg = helper.MakeInput({1, 4, 28, 28});
    auto* output_arg = helper.MakeOutput();

    helper.AddConvNode(input_arg, output_arg, {2, 4, 3, 3});
  };

  auto check_nhwc_graph = [&](NhwcInferenceSession& session) {
    auto op_to_count = session.CountOpsInGraph();

    EXPECT_EQ(op_to_count["nhwc.ReorderInput"], 1);
    EXPECT_EQ(op_to_count["nhwc.Conv"], 1);
    EXPECT_EQ(op_to_count["nhwc.ReorderOutput"], 1);
  };

  NhwcOptimizerTester(build_test_case, check_nhwc_graph);
}
#endif

}  // namespace test
}  // namespace onnxruntime
#endif
