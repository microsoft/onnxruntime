// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/optimizer/graph_transformer_level.h"
#include <gtest/gtest.h>

using namespace onnxruntime;

TEST(CApiTest, session_options_graph_optimization_level) {
  // Test set optimization level succeeds when valid level is provided.
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
}
