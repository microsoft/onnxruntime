// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/execution_provider.h"
#include "core/framework/session_options.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/optimizer/graph_transformer.h"
#include "core/session/inference_session.h"
#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime {
namespace test {

// Variant of TransformerTester for WebGPU fusion tests that creates a fresh execution provider
// per session via the provided factory, instead of sharing one EP across the baseline and target
// sessions. Sharing a single WebGPU EP across multiple InferenceSessions in series can leave the
// EP holding a dangling pointer to a destroyed session-level profiler; a separate fix to the EP
// addresses that, but using a fresh EP per session also avoids the issue and keeps the fusion PR
// independent of profiler-lifetime changes.
inline void RunWebGpuFusionTransformerTest(
    const std::function<void(ModelTestBuilder& helper)>& build_test_case,
    const std::function<void(InferenceSessionWrapper& session)>& check_transformed_graph,
    TransformerLevel baseline_level,
    TransformerLevel target_level,
    int opset_version,
    double per_sample_tolerance,
    double relative_per_sample_tolerance,
    std::unique_ptr<GraphTransformer> transformer,
    const std::function<std::unique_ptr<IExecutionProvider>()>& ep_factory,
    const std::function<void(SessionOptions&)>& add_session_options = {}) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = opset_version;
  domain_to_version[kMSDomain] = 1;
  Model model("WebGpuFusionTester", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, {}, DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  ASSERT_TRUE(build_test_case);
  build_test_case(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  auto run_model = [&](TransformerLevel level, std::vector<OrtValue>& fetches,
                       std::unique_ptr<GraphTransformer> level_transformer) {
    SessionOptions session_options;
    session_options.graph_optimization_level = level_transformer ? baseline_level : level;
    if (add_session_options) {
      add_session_options(session_options);
    }

    InferenceSessionWrapper session{session_options, GetEnvironment()};
    auto ep = ep_factory();
    ASSERT_TRUE(ep != nullptr) << "ep_factory() returned nullptr";
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(ep)));

    ASSERT_STATUS_OK(session.Load(model_data.data(), static_cast<int>(model_data.size())));
    if (level_transformer) {
      ASSERT_STATUS_OK(session.RegisterGraphTransformer(std::move(level_transformer), level));
    }

    ASSERT_STATUS_OK(session.Initialize());

    RunOptions run_options;
    ASSERT_STATUS_OK(session.Run(run_options, helper.feeds_, helper.output_names_, &fetches));

    if (level == target_level && check_transformed_graph) {
      check_transformed_graph(session);
    }
  };

  std::vector<OrtValue> baseline_fetches;
  ASSERT_NO_FATAL_FAILURE(run_model(baseline_level, baseline_fetches, /*level_transformer=*/nullptr));

  std::vector<OrtValue> target_fetches;
  ASSERT_NO_FATAL_FAILURE(run_model(target_level, target_fetches, std::move(transformer)));

  const size_t num_outputs = baseline_fetches.size();
  ASSERT_EQ(num_outputs, target_fetches.size());
  for (size_t i = 0; i < num_outputs; ++i) {
    auto ret = CompareOrtValue(target_fetches[i], baseline_fetches[i],
                               per_sample_tolerance, relative_per_sample_tolerance, false);
    EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
  }
}

}  // namespace test
}  // namespace onnxruntime
