// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/providers/coreml/coreml_execution_provider.h"
#include "core/providers/coreml/coreml_provider_factory.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/test_utils.h"

#if !defined(ORT_MINIMAL_BUILD)
// if this is a full build we need the provider test utils
#include "test/providers/provider_test_utils.h"
#endif  // !(ORT_MINIMAL_BUILD)

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {
namespace test {

// We want to run UT on CPU only to get output value without losing precision to pass the verification
static constexpr uint32_t s_coreml_flags = COREML_FLAG_USE_CPU_ONLY;

#if !defined(ORT_MINIMAL_BUILD)

TEST(CoreMLExecutionProviderTest, FunctionTest) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("coreml_execution_provider_test_graph.onnx");

  {  // Create the model with 2 add nodes
    onnxruntime::Model model("graph_1", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    std::vector<onnxruntime::NodeArg*> inputs;
    std::vector<onnxruntime::NodeArg*> outputs;

    // FLOAT tensor.
    ONNX_NAMESPACE::TypeProto float_tensor;
    float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

    auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
    auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
    inputs.push_back(&input_arg_1);
    inputs.push_back(&input_arg_2);
    auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
    outputs.push_back(&output_arg);
    graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

    auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
    inputs.clear();
    inputs.push_back(&output_arg);
    inputs.push_back(&input_arg_3);
    auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
    outputs.clear();
    outputs.push_back(&output_arg_2);
    graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

    ASSERT_STATUS_OK(graph.Resolve());
    ASSERT_STATUS_OK(onnxruntime::Model::Save(model, model_file_name));
  }

  std::vector<int64_t> dims_mul_x = {1, 1, 3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;

  CreateMLValue<float>(TestCoreMLExecutionProvider(s_coreml_flags)->GetAllocator(0, OrtMemTypeDefault),
                       dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(TestCoreMLExecutionProvider(s_coreml_flags)->GetAllocator(0, OrtMemTypeDefault),
                       dims_mul_x, values_mul_x, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(TestCoreMLExecutionProvider(s_coreml_flags)->GetAllocator(0, OrtMemTypeDefault),
                       dims_mul_x, values_mul_x, &ml_value_z);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  RunAndVerifyOutputsWithEP(model_file_name, "CoreMLExecutionProviderTest.FunctionTest",
                            std::make_unique<CoreMLExecutionProvider>(s_coreml_flags),
                            feeds);
}

// CoreML EP currently handles a special case for supporting ArgMax op:
// An ArgMax followed by a Cast to int32 type.
// Please see in <repo_root>/onnxruntime/core/providers/coreml/builders/impl/argmax_op_builder.cc
// and /cast_op_builder.cc. We have the following UT test here for this special case
// This test case can also be shared later if we want to support similar cases in NNAPI
TEST(CoreMLExecutionProviderTest, ArgMax_Cast) {
  SessionOptions so;

  InferenceSession session_object{so, GetEnvironment()};
  Status st;
  ASSERT_TRUE((st = session_object.Load("testdata/coreml_argmax_cast_test.onnx")).IsOK()) << st;
  ASSERT_TRUE((st = session_object.Initialize()).IsOK()) << st;

  // prepare inputs
  std::vector<int64_t> dims_mul_x = {3, 2, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  OrtValue ml_value_x;

  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault),
                       dims_mul_x, values_mul_x, &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<OrtValue> fetches;

  // prepare expected outputs
  std::vector<int64_t> expected_dims_mul_y = {3, 1, 2};
  std::vector<int32_t> expected_values_mul_y = {1, 1, 1, 1, 1, 1};

  // Now run
  onnxruntime::RunOptions run_options;
  st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  ASSERT_EQ(1u, fetches.size());

  auto& y_out = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims_mul_y);
  //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&expected_shape), *reinterpret_cast<const std::vector<int64_t>*>(&y_out.Shape()));
  const std::vector<int32_t> found(y_out.template Data<int32_t>(),
                                   y_out.template Data<int32_t>() + expected_values_mul_y.size());
  ASSERT_EQ(expected_values_mul_y, found);
}

#endif  // !(ORT_MINIMAL_BUILD)

TEST(CoreMLExecutionProviderTest, TestOrtFormatModel) {
  // mnist model that has only had basic optimizations applied. nnapi should be able to take at least some of the nodes
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/mnist.level1_opt.ort");

  RandomValueGenerator random{};
  const std::vector<int64_t> dims = {1, 1, 28, 28};
  std::vector<float> data = random.Gaussian<float>(dims, 0.0f, 1.f);

  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims, data, &ml_value);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("Input3", ml_value));

  RunAndVerifyOutputsWithEP(model_file_name, "CoreMLExecutionProviderTest.TestOrtFormatModel",
                            std::make_unique<CoreMLExecutionProvider>(s_coreml_flags),
                            feeds);
}

}  // namespace test
}  // namespace onnxruntime
