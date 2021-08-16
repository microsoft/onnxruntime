// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/util/include/test_utils.h"

#include "core/framework/ort_value.h"
#include "core/session/inference_session.h"

#include "test/util/include/asserts.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gmock/gmock.h"

namespace onnxruntime {
namespace test {
static void VerifyOutputs(const std::vector<std::string>& output_names,
                          const std::vector<OrtValue>& expected_fetches,
                          const std::vector<OrtValue>& fetches) {
  ASSERT_EQ(expected_fetches.size(), fetches.size());

  for (size_t i = 0, end = expected_fetches.size(); i < end; ++i) {
    auto& ltensor = expected_fetches[i].Get<Tensor>();
    auto& rtensor = fetches[i].Get<Tensor>();
    ASSERT_EQ(ltensor.Shape().GetDims(), rtensor.Shape().GetDims());
    auto element_type = ltensor.GetElementType();
    switch (element_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        EXPECT_THAT(ltensor.DataAsSpan<int32_t>(), ::testing::ContainerEq(rtensor.DataAsSpan<int32_t>()))
            << " mismatch for " << output_names[i];
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        EXPECT_THAT(ltensor.DataAsSpan<int64_t>(), ::testing::ContainerEq(rtensor.DataAsSpan<int64_t>()))
            << " mismatch for " << output_names[i];
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        const float abs_err = float(1e-5);

        EXPECT_THAT(ltensor.DataAsSpan<float>(),
                    ::testing::Pointwise(::testing::FloatNear(abs_err), rtensor.DataAsSpan<float>()));
        break;
      }
      default:
        ORT_THROW("Unhandled data type. Please add 'case' statement for ", element_type);
    }
  }
}

int CountAssignedNodes(const Graph& current_graph, const std::string& ep_type) {
  int count = 0;

  for (const auto& node : current_graph.Nodes()) {
    if (node.GetExecutionProviderType() == ep_type) {
      ++count;
    }

    if (node.ContainsSubgraph()) {
      for (const auto& entry : node.GetSubgraphs()) {
        count += CountAssignedNodes(*entry, ep_type);
      }
    }
  }

  return count;
}

void RunAndVerifyOutputsWithEP(const ORTCHAR_T* model_path, const char* log_id,
                               std::unique_ptr<IExecutionProvider> execution_provider,
                               const NameMLValMap& feeds) {
  SessionOptions so;
  so.session_logid = log_id;
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  //
  // get expected output from CPU EP
  //
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_path));
  ASSERT_STATUS_OK(session_object.Initialize());

  const auto& graph = session_object.GetGraph();
  const auto& outputs = graph.GetOutputs();

  // fetch all outputs
  std::vector<std::string> output_names;
  output_names.reserve(outputs.size());
  for (const auto* node_arg : outputs) {
    if (node_arg->Exists()) {
      output_names.push_back(node_arg->Name());
    }
  }

  std::vector<OrtValue> expected_fetches;
  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &expected_fetches));

  auto provider_type = execution_provider->Type();  // copy string so the std::move doesn't affect us

  //
  // get output with EP enabled
  //
  InferenceSessionWrapper session_object2{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object2.RegisterExecutionProvider(std::move(execution_provider)));
  ASSERT_STATUS_OK(session_object2.Load(model_path));
  ASSERT_STATUS_OK(session_object2.Initialize());

  // make sure that some nodes are assigned to the EP, otherwise this test is pointless...
  const auto& graph2 = session_object2.GetGraph();
  auto ep_nodes = CountAssignedNodes(graph2, provider_type);
  ASSERT_GT(ep_nodes, 0) << "No nodes were assigned to " << provider_type << " for " << model_path;

  // Run with EP and verify the result
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session_object2.Run(run_options, feeds, output_names, &fetches));
  VerifyOutputs(output_names, expected_fetches, fetches);
}

}  // namespace test
}  // namespace onnxruntime
