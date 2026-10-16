// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
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

// Deterministic packed INT4 data and host math used as an oracle for fused decode shaders.
struct FusedNBitsTestProjection {
  std::vector<uint8_t> weights;
  std::vector<MLFloat16> scales;
};

inline std::vector<MLFloat16> MakeFusedNBitsTestInput(int64_t k_size) {
  std::vector<MLFloat16> values(static_cast<size_t>(k_size));
  for (int64_t k = 0; k < k_size; ++k) {
    values[static_cast<size_t>(k)] = MLFloat16(static_cast<float>((k * 13) % 29 - 14) / 16.0f);
  }
  return values;
}

inline std::vector<MLFloat16> MakeFusedNBitsTestNormScale(int64_t k_size) {
  std::vector<MLFloat16> values(static_cast<size_t>(k_size));
  for (int64_t k = 0; k < k_size; ++k) {
    values[static_cast<size_t>(k)] = MLFloat16(0.75f + static_cast<float>(k % 5) * 0.0625f);
  }
  return values;
}

inline FusedNBitsTestProjection MakeFusedNBitsTestProjection(int64_t n, int64_t k_size,
                                                             int64_t block_size, int seed) {
  const int64_t blocks = k_size / block_size;
  const int64_t blob_size = block_size / 2;
  FusedNBitsTestProjection projection;
  projection.weights.resize(static_cast<size_t>(n * blocks * blob_size));
  projection.scales.resize(static_cast<size_t>(n * blocks));
  for (int64_t row = 0; row < n; ++row) {
    for (int64_t block = 0; block < blocks; ++block) {
      const uint8_t quantized = static_cast<uint8_t>(3 + (row * 5 + block * 3 + seed) % 11);
      const uint8_t packed = static_cast<uint8_t>(quantized | (quantized << 4));
      const size_t block_offset = static_cast<size_t>((row * blocks + block) * blob_size);
      std::fill_n(projection.weights.begin() + block_offset, blob_size, packed);
      projection.scales[static_cast<size_t>(row * blocks + block)] =
          MLFloat16(0.015625f * static_cast<float>(1 + (row + block + seed) % 4));
    }
  }
  return projection;
}

inline std::vector<float> MakeFusedNBitsTestNormalizedInput(int64_t k_size, float epsilon) {
  const auto input = MakeFusedNBitsTestInput(k_size);
  const auto norm_scale = MakeFusedNBitsTestNormScale(k_size);
  float sum_squared = 0.0f;
  for (MLFloat16 value : input) {
    const float x = value.ToFloat();
    sum_squared += x * x;
  }
  const float inverse_rms = MLFloat16(1.0f / std::sqrt(sum_squared / k_size + epsilon)).ToFloat();
  std::vector<float> normalized(static_cast<size_t>(k_size));
  for (int64_t k = 0; k < k_size; ++k) {
    normalized[static_cast<size_t>(k)] =
        MLFloat16(input[static_cast<size_t>(k)].ToFloat() * inverse_rms *
                  norm_scale[static_cast<size_t>(k)].ToFloat())
            .ToFloat();
  }
  return normalized;
}

inline std::vector<float> MakeFusedNBitsTestMatMulReference(int64_t n, int64_t k_size,
                                                           int64_t block_size, float epsilon, int seed) {
  const int64_t blocks = k_size / block_size;
  const int64_t blob_size = block_size / 2;
  const auto normalized = MakeFusedNBitsTestNormalizedInput(k_size, epsilon);
  const auto projection = MakeFusedNBitsTestProjection(n, k_size, block_size, seed);
  std::vector<float> output(static_cast<size_t>(n), 0.0f);
  for (int64_t row = 0; row < n; ++row) {
    float sum = 0.0f;
    for (int64_t k = 0; k < k_size; ++k) {
      const int64_t block = k / block_size;
      const size_t block_offset = static_cast<size_t>((row * blocks + block) * blob_size);
      const int shift = (k & 1) * 4;
      const int quantized =
          (projection.weights[block_offset + static_cast<size_t>((k % block_size) / 2)] >> shift) & 0x0f;
      const float weight =
          MLFloat16(static_cast<float>(quantized - 8) *
                    projection.scales[static_cast<size_t>(row * blocks + block)].ToFloat())
              .ToFloat();
      sum += normalized[static_cast<size_t>(k)] * weight;
    }
    output[static_cast<size_t>(row)] = sum;
  }
  return output;
}

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
    const std::function<void(SessionOptions&)>& add_session_options = {},
    const std::function<void(const std::vector<OrtValue>&)>& check_target_fetches = {}) {
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

  // ASSERT_NO_FATAL_FAILURE is load-bearing, even though run_model returns void.
  // It propagates fatal assertion failures from run_model and its nested calls,
  // preventing the test from continuing after a failure.
  std::vector<OrtValue> baseline_fetches;
  ASSERT_NO_FATAL_FAILURE(run_model(baseline_level, baseline_fetches, /*level_transformer=*/nullptr));

  std::vector<OrtValue> target_fetches;
  ASSERT_NO_FATAL_FAILURE(run_model(target_level, target_fetches, std::move(transformer)));

  if (check_target_fetches) {
    check_target_fetches(target_fetches);
  }

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
