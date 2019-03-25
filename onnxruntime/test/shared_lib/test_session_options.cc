// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/optimizer/graph_transformer_level.h"

#include "test_fixture.h"
using namespace onnxruntime;

TEST_F(CApiTest, session_options) {
  std::unique_ptr<OrtSessionOptions> options(OrtCreateSessionOptions());
  ASSERT_NE(options, nullptr);
}

TEST_F(CApiTest, session_options_graph_optimization_level) {
  // Test set optimization level succeeds when valid level is provided.
  uint32_t valid_optimization_level = static_cast<uint32_t>(TransformerLevel::Level2);
  std::unique_ptr<OrtSessionOptions> options(OrtCreateSessionOptions());
  ASSERT_EQ(OrtSetSessionGraphOptimizationLevel(options.get(), valid_optimization_level), 0);

  // Test set optimization level fails when invalid level is provided.
  uint32_t invalid_level = static_cast<uint32_t>(TransformerLevel::MaxTransformerLevel);
  ASSERT_EQ(OrtSetSessionGraphOptimizationLevel(options.get(), invalid_level), -1);
}