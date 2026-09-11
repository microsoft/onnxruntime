// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

// Build float test: Add -> Reshape(rank-6) -> Transpose -> Reshape -> Add
// Uses smaller dimensions for testing
GetTestModelFn BuildRank6ToRank5FloatTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    auto input_def = TestInputDef<float>({256, 64}, false, -10.0f, 10.0f);
    NodeArg* input = MakeTestInput<float>(builder, input_def);

    NodeArg* add_const1 = builder.MakeScalarInitializer<float>(1.0f);
    NodeArg* add1_out = builder.MakeIntermediate();
    builder.AddNode("Add", {input, add_const1}, {add1_out});

    // Reshape: (256, 64) -> (1, 4, 4, 4, 4, 64)
    NodeArg* reshape1_shape = builder.Make1DInitializer<int64_t>({1, 4, 4, 4, 4, 64});
    NodeArg* reshape1_out = builder.MakeIntermediate();
    builder.AddNode("Reshape", {add1_out, reshape1_shape}, {reshape1_out});

    // Transpose: perm [0, 2, 1, 3, 4, 5]
    NodeArg* transpose_out = builder.MakeIntermediate();
    Node& transpose = builder.AddNode("Transpose", {reshape1_out}, {transpose_out});
    transpose.AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3, 4, 5});

    // Reshape: (1, 4, 4, 4, 4, 64) -> (1, 256, 64)
    NodeArg* reshape2_shape = builder.Make1DInitializer<int64_t>({1, 256, 64});
    NodeArg* reshape2_out = builder.MakeIntermediate();
    builder.AddNode("Reshape", {transpose_out, reshape2_shape}, {reshape2_out});

    NodeArg* add_const2 = builder.MakeScalarInitializer<float>(1.0f);
    NodeArg* output = builder.MakeOutput();
    builder.AddNode("Add", {reshape2_out, add_const2}, {output});
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  return provider_options;
}

}  // namespace

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

TEST_F(QnnHTPBackendTests, Rank6ToRank5Fusion_Float) {
  RunQnnModelTest(BuildRank6ToRank5FloatTestCase(),
                  GetProviderOptions(),
                  13,
                  ExpectedEPNodeAssignment::All,
                  1e-2f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
