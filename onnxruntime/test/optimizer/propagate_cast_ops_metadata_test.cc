// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/propagate_cast_ops_metadata.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
namespace {

using propagate_cast_ops_internal::FindRelevantOpArgs;
using propagate_cast_ops_internal::kRelevantOpArgs;

struct ExpectedRelevantOpArgs {
  std::string_view op_type;
  std::array<bool, 4> relevant_inputs;
  std::array<bool, 2> relevant_outputs;
};

constexpr std::array<ExpectedRelevantOpArgs, 7> kExpectedRelevantOpArgs{{
    {"Dropout", {true, false, false, false}, {true, false}},
    {"Expand", {true, false, false, false}, {true, false}},
    {"Gather", {true, false, false, false}, {true, false}},
    {"LayerNormalization", {true, true, true, false}, {true, false}},
    {"Reshape", {true, false, false, false}, {true, false}},
    {"Squeeze", {true, false, false, false}, {true, false}},
    {"Unsqueeze", {true, false, false, false}, {true, false}},
}};

consteval bool PropagateCastOpsMetadataMatchesExpected() {
  if (kRelevantOpArgs.size() != kExpectedRelevantOpArgs.size()) {
    return false;
  }

  for (const auto& expected : kExpectedRelevantOpArgs) {
    const auto* metadata = FindRelevantOpArgs(expected.op_type);
    if (metadata == nullptr) {
      return false;
    }

    for (size_t index = 0; index < expected.relevant_inputs.size(); ++index) {
      if (metadata->IsRelevantInput(static_cast<int>(index)) != expected.relevant_inputs[index]) {
        return false;
      }
    }

    for (size_t index = 0; index < expected.relevant_outputs.size(); ++index) {
      if (metadata->IsRelevantOutput(static_cast<int>(index)) != expected.relevant_outputs[index]) {
        return false;
      }
    }
  }

  return FindRelevantOpArgs("MatMul") == nullptr;
}

static_assert(PropagateCastOpsMetadataMatchesExpected());

TEST(PropagateCastOpsMetadataTest, FindsRuntimeOperatorMetadata) {
  const std::string_view op_type = "Reshape";
  const auto* metadata = FindRelevantOpArgs(op_type);

  ASSERT_NE(metadata, nullptr);
  EXPECT_TRUE(metadata->IsRelevantInput(0));
  EXPECT_FALSE(metadata->IsRelevantInput(1));
  EXPECT_TRUE(metadata->IsRelevantOutput(0));
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime