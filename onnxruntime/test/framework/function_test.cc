// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "onnx/defs/parser.h"

#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"

#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "test/common/tensor_op_test_utils.h"

// Unit tests to check the implementation of functions, model-local functions,
// function-inlining etc.

namespace onnxruntime {
namespace test {

static void Check(const char* source, const NameMLValMap& feeds,
                  const char* output_name, std::vector<float> output_values) {
  // Convert source-representation of model to ModelProto:
  ONNX_NAMESPACE::OnnxParser parser(source);
  ONNX_NAMESPACE::ModelProto model;
  auto parse_status = parser.Parse(model);
  EXPECT_TRUE(parse_status.IsOK()) << parse_status.ErrorMessage();
  EXPECT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  // Serialize and then load model:
  std::string serialized_model;
  const bool serialization_status = model.SerializeToString(&serialized_model);
  EXPECT_TRUE(serialization_status) << "Failed to serialize proto to string";

  SessionOptions session_options;
  InferenceSession session_object{session_options, GetEnvironment()};

  std::stringstream sstr(serialized_model);
  auto status = session_object.Load(sstr);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  status = session_object.Initialize();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  RunOptions run_options;
  run_options.run_tag = session_options.session_logid;

  std::vector<OrtValue> fetches;

  status = session_object.Run(run_options, feeds, {output_name}, &fetches);
  EXPECT_TRUE(status.IsOK()) << "Session Run failed.";

  auto& tensor = fetches[0].Get<Tensor>();
  size_t size = static_cast<size_t>(tensor.Shape().Size());
  EXPECT_EQ(size, output_values.size());

  auto* data = tensor.template Data<float>();
  float threshold = 0.001f;

  for (size_t i = 0; i < size; ++i) {
    ASSERT_NEAR(data[i], output_values[i], threshold) << "at position i:" << i;
  }
}

NameMLValMap input(const char* name, std::vector<float> values) {
  NameMLValMap input_value_map;

  auto* provider = new CPUExecutionProvider(CPUExecutionProviderInfo());
  OrtValue ort_value;
  CreateMLValue<float>(provider->GetAllocator(0, OrtMemTypeDefault), {int64_t(values.size())}, values, &ort_value);

  input_value_map.insert(std::make_pair(std::string(name), ort_value));
  return std::move(input_value_map);
}

TEST(FunctionTest, Basic) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.myfun (x)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun (lx) => (ly) {
            two = Constant <value = float[1] {2.0}> ()
            ly = Mul (lx, two)
        }
        )";

  Check(code, input("x", {1.0, 2.0, 3.0}), "y", {2.0, 4.0, 6.0});
}

// Check that variables are renamed to avoid conflicts when multiple
// calls are inlined.
TEST(FunctionTest, Renaming) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y1 = local.myfun (x)
            y = local.myfun (y1)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun (lx) => (ly) {
            two = Constant <value = float[1] {2.0}> ()
            ly = Mul (lx, two)
        }
        )";

  Check(code, input("x", {1.0, 2.0, 3.0}), "y", {4.0, 8.0, 12.0});
}

// Check variable renaming in subgraphs.
// Scenario: input lx is used within subgraphs, but not in main graph.
// Both must be renamed to match the actual parameter name.
TEST(FunctionTest, InputInSubgraph) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            f = Constant <value = bool {0}> ()
            t = Constant <value = bool {1}> ()
            y1 = local.myfun (f, x)
            y = local.myfun (t, y1)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun (b, lx) => (ly) {
            ly = If (b) <
                then_branch = g1 () => (float[N] z_then)
                {
                    two = Constant <value = float[1] {2.0}> ()
                    z_then =  Mul (lx, two)
                },
                else_branch = g2 () => (float[N] z_else)
                {
                    three = Constant <value = float[1] {3.0}> ()
                    z_else =  Mul (lx, three)
                }
                >
        }
        )";

  Check(code, input("x", {1.0, 2.0, 3.0}), "y", {6.0, 12.0, 18.0});
}

// Check variable renaming in subgraphs.
// Scenario: intermediate temp is used within subgraphs, defined in main graph.
// Both must be renamed with a unique temporary name.
TEST(FunctionTest, TempInSubgraph) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            f = Constant <value = bool {0}> ()
            t = Constant <value = bool {1}> ()
            y1 = local.myfun (f, x)
            y = local.myfun (t, y1)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun (b, lx) => (ly) {
            temp = Identity (lx)
            ly = If (b) <
                then_branch = g1 () => (float[N] z_then)
                {
                    two = Constant <value = float[1] {2.0}> ()
                    z_then =  Mul (temp, two)
                },
                else_branch = g2 () => (float[N] z_else)
                {
                    three = Constant <value = float[1] {3.0}> ()
                    z_else =  Mul (temp, three)
                }
                >
        }
        )";

  Check(code, input("x", {1.0, 2.0, 3.0}), "y", {6.0, 12.0, 18.0});
}

}  // namespace test
}  // namespace onnxruntime
