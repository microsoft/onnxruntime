// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <unordered_map>

#include "gtest/gtest.h"
#include "core/common/span_utils.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/framework/model_builder_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_utils.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/test_environment.h"

using namespace ONNX_NAMESPACE;

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

using namespace modelbuilder;

class ShapeInferenceTest : public ::testing::Test {
 protected:
  onnxruntime::Model model_;
  int node_count_;
  std::unordered_map<std::string, std::unique_ptr<onnxruntime::NodeArg>> name_to_arg_;

 public:
  ShapeInferenceTest() : model_("Test", false, DefaultLoggingManager().DefaultLogger()), node_count_(0) {}

  void Input(const std::string& name, const Type& type) {
    name_to_arg_[name] = std::make_unique<onnxruntime::NodeArg>(name, &type.value);
  }

  onnxruntime::NodeArg* Arg(const std::string& name) {
    if (name_to_arg_.count(name) == 0)
      name_to_arg_[name] = std::make_unique<onnxruntime::NodeArg>(name, nullptr);
    return name_to_arg_[name].get();
  }

  onnxruntime::Node& Node(const std::string& op, const std::string& input, const std::string& output) {
    std::vector<onnxruntime::NodeArg*> input_args({Arg(input)});
    std::vector<onnxruntime::NodeArg*> output_args({Arg(output)});
    int num = node_count_++;
    return model_.MainGraph().AddNode("node" + std::to_string(num), op, "test op", input_args, output_args);
  }

  void DoShapeInference() {
    auto status = model_.MainGraph().Resolve();
    EXPECT_TRUE(status.IsOK()) << "Graph resolve failed: " << status.ErrorMessage();
  }

  const TensorShapeProto* InputShape(onnxruntime::Node& node, int arg_num = 0) {
    return node.InputDefs()[arg_num]->Shape();
  }

  const TensorShapeProto* OutputShape(onnxruntime::Node& node, int arg_num = 0) {
    return node.OutputDefs()[arg_num]->Shape();
  }

};  // namespace test

TEST_F(ShapeInferenceTest, BasicTest) {
  Type type1({1, 50, 100});
  Input("X1", type1);

  auto& node = Node("Cast", "X1", "Y1");
  node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_INT32});

  DoShapeInference();
  // check inferred shapes
  Shape expected_shape({1, 50, 100});
  CheckShapeEquality(OutputShape(node), &expected_shape.value);
  CheckShapeEquality(InputShape(node), OutputShape(node));
}

TEST(ShapeInferenceV2Test, PartialDataPropagationTest) {
  {
    // Model #1
    // This model contains "Shape" and "Reshape" operators.
    auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_with_shape_related_nodes.onnx");

    Ort::SessionOptions session_options{};
    session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    session_options.AddFreeDimensionOverrideByName("batch", 1);
    session_options.AddFreeDimensionOverrideByName("width", 64);
    session_options.AddFreeDimensionOverrideByName("height", 64);

    // Even though all graph optimizations are disabled, the free dimension override is still enabled by default.
    // The shape of graph's output should be correctly inferred by shape inference and data propagation.
    Ort::Session session(*ort_env, model_path, session_options);

    // This graph only has one output
    ORT_ENFORCE(session.GetOutputCount() == 1);

    Ort::TypeInfo type_info = session.GetOutputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = tensor_info.GetShape();
    EXPECT_TRUE(output_shape.size() == 4) << "The output shape should have 4 dimensions";
    EXPECT_TRUE(output_shape[0] == 1) << "The first dimension should have 1 as value";
    EXPECT_TRUE(output_shape[1] == 3) << "The second dimension should have 3 as value";
    EXPECT_TRUE(output_shape[2] == 64) << "The second dimension should have 64 as value";
    EXPECT_TRUE(output_shape[3] == 64) << "The second dimension should have 64 as value";
  }

  {
    // Model #2
    // This model contains "Shape", "Reshape", "Gather" and "Unsqueeze" operators.
    auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_with_shape_related_nodes_v2.onnx");

    Ort::SessionOptions session_options{};
    session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    session_options.AddFreeDimensionOverrideByName("batch", 1);
    session_options.AddFreeDimensionOverrideByName("width", 64);
    session_options.AddFreeDimensionOverrideByName("height", 64);

    // Even though all graph optimizations are disabled, the free dimension override is still enabled by default.
    // The shape of graph's output should be correctly inferred by shape inference and data propagation.
    Ort::Session session(*ort_env, model_path, session_options);

    // This graph only has one output
    ORT_ENFORCE(session.GetOutputCount() == 1);

    Ort::TypeInfo type_info = session.GetOutputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = tensor_info.GetShape();
    EXPECT_TRUE(output_shape.size() == 3) << "The output shape should have 3 dimensions";
    EXPECT_TRUE(output_shape[0] == 1) << "The first dimension should have 1 as value";
    EXPECT_TRUE(output_shape[1] == 3) << "The second dimension should have 3 as value";
    EXPECT_TRUE(output_shape[2] == 4096) << "The second dimension should have 4096 as value";
  }

  {
    // Model #3
    // This model extends model #2 and appends Unsqueeze -> Unsqueeze -> Squeeze -> Squeeze -> Reshape to the end.
    auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_with_shape_related_nodes_v3.onnx");

    Ort::SessionOptions session_options{};
    session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    session_options.AddFreeDimensionOverrideByName("batch", 1);
    session_options.AddFreeDimensionOverrideByName("width", 64);
    session_options.AddFreeDimensionOverrideByName("height", 64);

    // Even though all graph optimizations are disabled, the free dimension override is still enabled by default.
    // The shape of graph's output should be correctly inferred by shape inference and data propagation.
    Ort::Session session(*ort_env, model_path, session_options);

    // This graph only has one output
    ORT_ENFORCE(session.GetOutputCount() == 1);

    Ort::TypeInfo type_info = session.GetOutputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = tensor_info.GetShape();
    EXPECT_TRUE(output_shape.size() == 3) << "The output shape should have 3 dimensions";
    EXPECT_TRUE(output_shape[0] == 1) << "The first dimension should have 1 as value";
    EXPECT_TRUE(output_shape[1] == 3) << "The second dimension should have 3 as value";
    EXPECT_TRUE(output_shape[2] == 4096) << "The second dimension should have 4096 as value";
  }

  {
    // Model #4
    // This model contains Shape, Reshape, Squeeze, Range, ReduceSum.
    // It's from SoftmaxGrad_DefaultAxis test.
    auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_with_shape_related_nodes_v4.onnx");

    Ort::SessionOptions session_options{};
    session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);

    // Make sure it can load the model and run shape inference without errors.
    Ort::Session session(*ort_env, model_path, session_options);
  }
}

namespace {
struct MyCustomKernelWithOptionalInput {
  MyCustomKernelWithOptionalInput(const OrtKernelInfo* info) {
    Ort::ConstKernelInfo k_info(info);

    Ort::KeyValuePairs kvp = k_info.GetConfigEntries();

    EXPECT_NE(nullptr, kvp.GetValue("session.inter_op.allow_spinning"));
    EXPECT_STREQ("0", kvp.GetValue("session.inter_op.allow_spinning"));

    EXPECT_NE(nullptr, kvp.GetValue("session.intra_op.allow_spinning"));
    EXPECT_STREQ("0", kvp.GetValue("session.intra_op.allow_spinning"));

    EXPECT_EQ(nullptr, kvp.GetValue("__not__exist__"));
  }

  OrtStatusPtr ComputeV2(OrtKernelContext* /* context */) const {
    return nullptr;
  }
};

struct MyCustomOpWithOptionalInput : Ort::CustomOpBase<MyCustomOpWithOptionalInput,
                                                       MyCustomKernelWithOptionalInput,
                                                       true> {
  explicit MyCustomOpWithOptionalInput(const char* provider) : provider_(provider) {}

  OrtStatusPtr CreateKernelV2(const OrtApi& /* api */, const OrtKernelInfo* info, void** kernel) const {
    *kernel = new MyCustomKernelWithOptionalInput(info);
    return nullptr;
  };

  const char* GetName() const { return "FooBar"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 3; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
    // The second input (index == 1) is optional
    if (index == 1)
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;

    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }

 private:
  const char* provider_;
};

const ORTCHAR_T* const OPTIONAL_INPUT_CUSTOM_OP_MODEL_URI_2 = ORT_TSTR("testdata/foo_bar_2.onnx");

}  // namespace

// CustomOps Output type inference function quits if it
// encounters the an output that is optional and absent.
// It quits without any errors or logging. We want to make sure
// that inference proceeds for all of the outputs when absent optional inputs are present
TEST(ShapeInferenceCustomOpTest, custom_op_optional_input_inference_test) {
  MyCustomOpWithOptionalInput custom_op{onnxruntime::kCpuExecutionProvider};
  custom_op.InferOutputShapeFn = [](const OrtCustomOp* /*op*/, OrtShapeInferContext* /*ctx*/) -> OrtStatusPtr {
    return nullptr;
  };

  const auto& env = GetEnvironment();

  Ort::CustomOpDomain op_domain("test");
  op_domain.Add(&custom_op);

  std::initializer_list<OrtCustomOpDomain*> op_domains = {static_cast<OrtCustomOpDomain*>(op_domain)};

  SessionOptions sess_opts;
  sess_opts.inter_op_param.thread_pool_size = 1;
  sess_opts.intra_op_param.thread_pool_size = 1;
  ASSERT_STATUS_OK(sess_opts.config_options.AddConfigEntry("session.inter_op.allow_spinning", "0"));
  ASSERT_STATUS_OK(sess_opts.config_options.AddConfigEntry("session.intra_op.allow_spinning", "0"));

  InferenceSessionWrapper session{sess_opts, env, OPTIONAL_INPUT_CUSTOM_OP_MODEL_URI_2};
  ASSERT_STATUS_OK(session.AddCustomOpDomains(AsSpan(op_domains)));

  ASSERT_STATUS_OK(session.Load());
  ASSERT_STATUS_OK(session.Initialize());

  const onnxruntime::Model& model = session.GetModel();
  const auto& graph = model.MainGraph();
  const auto& nodes = graph.Nodes();
  for (const auto& node : nodes) {
    if (node.OpType() == "FooBar") {
      // check inferred shapes
      const auto* node_arg = node.OutputDefs()[0];
      const auto* type_proto = node_arg->TypeAsProto();
      ASSERT_NE(nullptr, type_proto);
      ASSERT_EQ(ONNX_NAMESPACE::TypeProto::ValueCase::kTensorType, type_proto->value_case());
      ASSERT_EQ(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, type_proto->tensor_type().elem_type());
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
