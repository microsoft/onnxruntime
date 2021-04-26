// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <string>
#include <vector>

#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/util/include/inference_session_wrapper.h"

#include "graph_transform_test_builder.h"

namespace onnxruntime {
namespace test {

void TransformerTester(const std::function<void(ModelTestBuilder& helper)>& build_test_case,
                       const std::function<void(InferenceSessionWrapper& session)>& check_transformed_graph,
                       TransformerLevel baseline_level,
                       TransformerLevel target_level,
                       int opset_version,
                       double per_sample_tolerance,
                       double relative_per_sample_tolerance) {
  // Build the model for this test.
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = opset_version;
  domain_to_version[kMSDomain] = 1;
  Model model("TransformerTester", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, {}, DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  build_test_case(helper);
  ASSERT_TRUE(model.MainGraph().Resolve().IsOK());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  auto run_model = [&](TransformerLevel level, std::vector<OrtValue>& fetches) {
    SessionOptions session_options;
    session_options.graph_optimization_level = level;
    InferenceSessionWrapper session{session_options, GetEnvironment()};
    ASSERT_TRUE(session.Load(model_data.data(), static_cast<int>(model_data.size())).IsOK());
    auto status = session.Initialize();
    if (!status.IsOK()) {
      std::cout << "Model initialized failed with status message: " << status.ErrorMessage() << std::endl;
    }
    ASSERT_TRUE(status.IsOK());

    RunOptions run_options;
    status = session.Run(run_options,
                         helper.feeds_,
                         helper.output_names_,
                         &fetches);
    if (!status.IsOK()) {
      std::cout << "Run failed with status message: " << status.ErrorMessage() << std::endl;
    }
    ASSERT_TRUE(status.IsOK());

    if (level == target_level) {
      check_transformed_graph(session);
    }
  };

  std::vector<OrtValue> baseline_fetches;
  run_model(baseline_level, baseline_fetches);

  std::vector<OrtValue> target_fetches;
  run_model(target_level, target_fetches);

  size_t num_outputs = baseline_fetches.size();
  ASSERT_TRUE(num_outputs == target_fetches.size());

  for (size_t i = 0; i < num_outputs; i++) {
    std::pair<COMPARE_RESULT, std::string> ret =
        CompareOrtValue(target_fetches[i],
                        baseline_fetches[i],
                        per_sample_tolerance,
                        relative_per_sample_tolerance,
                        false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
  }
}

}  // namespace test
}  // namespace onnxruntime
