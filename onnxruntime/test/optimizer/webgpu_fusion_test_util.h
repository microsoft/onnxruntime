// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <utility>

#include "test/optimizer/fusion_test_util.h"

namespace onnxruntime {
namespace test {

// Retain the existing entry point for WebGPU fusion tests.
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
  RunFusionTransformerTest(build_test_case, check_transformed_graph, baseline_level, target_level,
                           opset_version, per_sample_tolerance, relative_per_sample_tolerance,
                           std::move(transformer), ep_factory, add_session_options);
}

}  // namespace test
}  // namespace onnxruntime
