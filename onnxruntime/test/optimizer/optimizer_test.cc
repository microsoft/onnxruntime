// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/framework/data_types.h"
#include "core/framework/ml_value.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/platform/env.h"
#include "test/framework/test_utils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "gtest/gtest.h"

using namespace std;
using namespace ONNX_NAMESPACE;

using namespace onnx;

namespace onnxruntime {
namespace test {

static const std::string MODEL_FOLDER = "testdata/transform/";

TEST(OptimizerTest, Basic) {
  Model model("OptimizerBasic");
  auto& graph = model.MainGraph();

  const int tensor_dim = 10;
  const int input_num = 2;
  TensorProto initializer_tensor[input_num];
  std::vector<std::unique_ptr<NodeArg>> inputs(input_num);
  std::vector<std::unique_ptr<NodeArg>> outputs(1);
  InitializedTensorSet initialized_tensor_set;

  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  tensor_int32.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(tensor_dim);

  for (int i = 0; i < input_num; i++) {
    string name("input_" + std::to_string(i));
    inputs[i] = std::make_unique<NodeArg>(name, &tensor_int32);

    initializer_tensor[i].set_name(inputs[i]->Name());
    initializer_tensor[i].add_dims(tensor_dim);
    initializer_tensor[i].set_data_type(onnx::TensorProto_DataType_INT32);
    for (int j = 0; j < tensor_dim; j++) {
      initializer_tensor[i].add_int32_data((i + 1) * j);
    }
    initialized_tensor_set[name] = &initializer_tensor[i];
  }
  outputs[0] = std::make_unique<NodeArg>("out", &tensor_int32);

  std::vector<NodeArg*> tmp_inputs{inputs[0].get(), inputs[1].get()};
  std::vector<NodeArg*> tmp_outputs{outputs[0].get()};
  graph.AddNode("a", "Add", "a", tmp_inputs, tmp_outputs);
  graph.Resolve();

  std::vector<const Node*> nodes;
  for (auto& node : graph.Nodes()) {
    nodes.push_back(&node);
  }

  OptimizerExecutionFrame::Info info(nodes, initialized_tensor_set);
  std::vector<int> fetch_mlvalue_idxs{info.GetMLValueIndex("out")};
  OptimizerExecutionFrame frame(info, fetch_mlvalue_idxs);
  const logging::Logger& logger = ::onnxruntime::test::DefaultLoggingManager().DefaultLogger();

  for (auto& node : graph.Nodes()) {
    auto* kernel = info.GetKernel(node.Index());

    OpKernelContext op_kernel_context(&frame, kernel, logger);

    kernel->Compute(&op_kernel_context);

    std::vector<MLValue> fetches;
    frame.GetOutputs(fetches);
    auto& tensor = fetches[0].Get<Tensor>();
    const std::vector<int32_t> found(tensor.template Data<int32_t>(), tensor.template Data<int32_t>() + tensor_dim);
    std::vector<int32_t> expected;
    for (int j = 0; j < tensor_dim; j++) {
      expected.push_back(3 * j);
    }
    ASSERT_EQ(expected, found);
  }
}

}  // namespace test
}  // namespace onnxruntime
