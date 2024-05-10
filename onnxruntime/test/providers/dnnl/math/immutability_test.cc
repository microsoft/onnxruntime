// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/model_tester.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/current_test_name.h"

namespace onnxruntime {
namespace test {

#ifdef USE_DNNL
// Need to ensure DNNL op implementations respect immutability principle of graphs (i.e. no side effects on inputs)

TEST(ImmutabilityTest, DNNL_LayerNorm) {
  ModelTester tester(CurrentTestName(), ORT_TSTR("testdata/layernorm_and_add.onnx"));
  tester.ConfigEp(DefaultDnnlExecutionProvider());
  tester.AddInput("input", {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
  tester.AddOutput("output_add", {3, 3}, {-0.22473586f, 2.0f, 4.2247357f, 2.7752638f, 5.0f, 7.224736f, 5.775262f, 8.0f, 10.224738f});
  tester.RunWithConfig();
}

#endif  // USE_DNNL

}  // namespace test
}  // namespace onnxruntime
