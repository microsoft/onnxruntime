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

#include "test/providers/provider_test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {
namespace test {

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

  // We want to run UT on CPU only to get output value without losing precision
  uint32_t coreml_flags = 0;
  coreml_flags |= COREML_FLAG_USE_CPU_ONLY;

  CreateMLValue<float>(TestCoreMLExecutionProvider(coreml_flags)->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(TestCoreMLExecutionProvider(coreml_flags)->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(TestCoreMLExecutionProvider(coreml_flags)->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  RunAndVerifyOutputsWithEP(model_file_name, "CoreMLExecutionProviderTest.FunctionTest",
                            onnxruntime::make_unique<CoreMLExecutionProvider>(coreml_flags),
                            feeds);
}

}  // namespace test
}  // namespace onnxruntime
