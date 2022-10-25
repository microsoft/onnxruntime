// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/optimizer/graph_transform_test_builder.h"

#include <functional>
#include <string>
#include <vector>

#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime {
namespace test {

void TransformerTester(const std::function<void(ModelTestBuilder& helper)>& build_test_case,
                       const std::function<void(InferenceSessionWrapper& session)>& check_transformed_graph,
                       TransformerLevel baseline_level,
                       TransformerLevel target_level,
                       int opset_version,
                       double per_sample_tolerance,
                       double relative_per_sample_tolerance,
                       std::unique_ptr<GraphTransformer> transformer,
                       const std::function<void(SessionOptions&)>& add_session_options,
                       const InlinedHashSet<std::string>& disabled_optimizers) {
  // Build the model for this test.
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = opset_version;
  domain_to_version[kMSDomain] = 1;
  Model model("TransformerTester", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, {}, DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  ASSERT_TRUE(build_test_case);
  build_test_case(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  auto run_model = [&](TransformerLevel level, std::vector<OrtValue>& fetches,
                       std::unique_ptr<GraphTransformer> transformer = nullptr) {
    SessionOptions session_options;
    session_options.graph_optimization_level = transformer ? baseline_level : level;
#if 0  // enable to dump model for debugging
    session_options.optimized_model_filepath =
        ToPathString("model" + std::to_string(static_cast<int>(level)) + ".onnx");
#endif
    if (add_session_options) {
      add_session_options(session_options);
    }
    InferenceSessionWrapper session{session_options, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(model_data.data(), static_cast<int>(model_data.size())));
    if (transformer) {
      ASSERT_STATUS_OK(session.RegisterGraphTransformer(std::move(transformer), level));
    } else if (!disabled_optimizers.empty()) {
      ASSERT_STATUS_OK(session.FilterEnabledOptimizers(InlinedHashSet<std::string>{disabled_optimizers}));
    }

    ASSERT_STATUS_OK(session.Initialize());

    RunOptions run_options;
    ASSERT_STATUS_OK(session.Run(run_options,
                                 helper.feeds_,
                                 helper.output_names_,
                                 &fetches));

    if (level == target_level) {
      ASSERT_TRUE(check_transformed_graph);
      check_transformed_graph(session);
    }
  };

  std::vector<OrtValue> baseline_fetches;
  ASSERT_NO_FATAL_FAILURE(run_model(baseline_level, baseline_fetches));

  std::vector<OrtValue> target_fetches;
  ASSERT_NO_FATAL_FAILURE(run_model(target_level, target_fetches, std::move(transformer)));

  size_t num_outputs = baseline_fetches.size();
  ASSERT_EQ(num_outputs, target_fetches.size());

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

void TestGraphTransformer(const std::function<void(ModelTestBuilder& helper)>& build_test_case, int opset_version,
                          const logging::Logger& logger, std::unique_ptr<GraphTransformer> transformer,
                          TransformerLevel level, unsigned steps, const std::function<void(Graph&)>& pre_graph_checker,
                          const std::function<void(Graph&)>& post_graph_checker) {
  // Build the model for this test.
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = opset_version;
  domain_to_version[kMSDomain] = 1;
  Model model("TransformerTester", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, {}, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  ASSERT_TRUE(build_test_case);
  build_test_case(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());
  pre_graph_checker(graph);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{steps};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(transformer), level));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, level, logger));
  post_graph_checker(graph);
}

}  // namespace test
}  // namespace onnxruntime
