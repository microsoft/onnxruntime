// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "onnx/defs/parser.h"

#include "core/common/span_utils.h"
#include "core/framework/customregistry.h"
#include "core/framework/op_kernel.h"
#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"

#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "inference_session_wrapper.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/asserts.h"

#include "test/providers/internal_testing/internal_testing_execution_provider.h"

// Unit tests to check the implementation of functions, model-local functions,
// function-inlining etc.

namespace onnxruntime {
namespace test {

// Convert source-representation of model to ModelProto:
static void ParseOnnxSource(const char* source, std::string& result) {
  ONNX_NAMESPACE::OnnxParser parser(source);
  ONNX_NAMESPACE::ModelProto model;
  auto parse_status = parser.Parse(model);
  ASSERT_TRUE(parse_status.IsOK()) << parse_status.ErrorMessage();
  ASSERT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  // Serialize
  std::string serialized_model;
  const bool serialization_status = model.SerializeToString(&serialized_model);
  ASSERT_TRUE(serialization_status) << "Failed to serialize proto to string";
  result = std::move(serialized_model);
}

static void Check(const char* source,
                  const char* input_name, std::vector<float> input_values,
                  const char* output_name, std::vector<float> output_values) {
  // Serialize and then load model:
  std::string serialized_model;
  ParseOnnxSource(source, serialized_model);

  SessionOptions session_options;
  InferenceSession session_object{session_options, GetEnvironment()};

  std::stringstream sstr(serialized_model);
  auto status = session_object.Load(sstr);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  RunOptions run_options;
  run_options.run_tag = session_options.session_logid;

  NameMLValMap feeds;

  std::unique_ptr<CPUExecutionProvider> provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  OrtValue ort_value;
  CreateMLValue<float>(provider->CreatePreferredAllocators()[0], {int64_t(input_values.size())}, input_values, &ort_value);

  feeds.insert(std::make_pair(std::string(input_name), ort_value));

  std::vector<OrtValue> fetches;

  status = session_object.Run(run_options, feeds, AsSpan({std::string(output_name)}), &fetches);
  ASSERT_TRUE(status.IsOK()) << "Session Run failed: " << status.ErrorMessage() << std::endl;

  auto& tensor = fetches[0].Get<Tensor>();
  size_t size = static_cast<size_t>(tensor.Shape().Size());
  EXPECT_EQ(size, output_values.size());

  auto* data = tensor.Data<float>();
  float threshold = 0.001f;

  for (size_t i = 0; i < size; ++i) {
    if (!std::isnan(data[i]) && !std::isnan(output_values[i])) {
      ASSERT_NEAR(data[i], output_values[i], threshold) << "at position i:" << i;
    }
  }
}

namespace {
const char* basic_code = R"(
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
}

TEST(FunctionTest, Basic) {
  Check(basic_code, "x", {1.0, 2.0, 3.0}, "y", {2.0, 4.0, 6.0});
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

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {4.0, 8.0, 12.0});
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

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {6.0, 12.0, 18.0});
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

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {6.0, 12.0, 18.0});
}

// Test a function body that calls another function.
TEST(FunctionTest, NestedCall) {
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
        opset_import: [ "" : 16, "local" : 1],
        domain: "local"
        >
        myfun (lx) => (ly) {
            one = Constant <value = float[1] {1.0}> ()
            tmp = local.twice (lx)
            ly = Add (tmp, one)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        twice (lx) => (ly) {
            two = Constant <value = float[1] {2.0}> ()
            ly = Mul (lx, two)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {3.0, 5.0, 7.0});
}

// Nested call inside a conditional statement.
TEST(FunctionTest, CallInConditional) {
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
        opset_import: [ "" : 16, "local" : 1],
        domain: "local"
        >
        myfun (b, lx) => (ly) {
            temp = Identity (lx)
            ly = If (b) <
                then_branch = g1 () => (float[N] z_then)
                {
                    two = Constant <value = float[1] {2.0}> ()
                    z_then =  local.MulFun (temp, two)
                },
                else_branch = g2 () => (float[N] z_else)
                {
                    three = Constant <value = float[1] {3.0}> ()
                    z_else =  local.MulFun (temp, three)
                }
                >
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        MulFun (ax, bx) => (cx) {
            cx = Mul (ax, bx)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {6.0, 12.0, 18.0});
}

// Test use of attibute references, especially where source/target attribute
// names are not the same. In this example, the "start : int = @s" attribute-reference
// binds the attribute named "start" of the Shape op to the attribute named "s"
// of the containing function myfun.
TEST(FunctionTest, AttrName) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.myfun <s = 0> (x)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun <s> (lx) => (ly) {
            d = Shape <start : int = @s> (lx)
            df = Cast <to = 1> (d)
            ly = Mul (lx, df)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {3.0, 6.0, 9.0});
}

// Test function with attribute that has default value.
TEST(FunctionTest, AttrWithDefault) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y0 = local.myfun <a = 2.0> (x)
            y1 = local.myfun (x)
            y = Add (y0, y1)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun <a: float=1.0> (x) => (y) {
            x2 = Constant <value_float: float=@a>()
            x3 = CastLike (x2, x)
            y = Add (x, x3)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {5.0, 7.0, 9.0});
}

#if !defined(DISABLE_FLOAT8_TYPES)

// Attribute 'saturate' was introduced in opset 19, ir_version=9.
// The test checks the parser gets it right and returns the expected results.
TEST(FunctionTest, AttrSaturate) {
  const char* code = R"(
        <
        ir_version: 9,
        opset_import: [ "" : 19, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y0 = local.myfun <a = 2.0> (x)
            y1 = local.myfun (x)
            y = Add (y0, y1)
        }

        <
        opset_import: [ "" : 19 ],
        domain: "local"
        >
        myfun <a: float=1.0> (x) => (y) {
            x2 = Constant <value_float: float=@a>()
            x2_ = Cast<to=17>(x2)
            x3 = CastLike<saturate=0>(x2, x2_)
            x3_ = Cast<to=1>(x3)
            y = Add (x, x3_)
        }
        )";

  Check(code, "x", {1.0, 2.0, 1e6}, "y", {5.0, 7.0, 2000003.0});
}

// Attribute 'saturate' was introduced in opset 19, ir_version=9.
// The test checks the model does not saturate a value out of float 8 boundary.
// TODO: change the expected value when this PR is merged in onnx:
// https://github.com/onnx/onnx/pull/5246
TEST(FunctionTest, AttrSaturateNan) {
  const char* code = R"(
        <
        ir_version: 9,
        opset_import: [ "" : 19, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            x_E4M3FNUZ = Cast<to=18>(x)
            x_E4M3FNUZ_2 = CastLike<saturate=0>(x, x_E4M3FNUZ)  # NaN when OOR
            y = Cast<to=1>(x_E4M3FNUZ_2)
        }
        )";

  Check(code, "x", {1.0, 2.0, 1e6}, "y", {1.0, 2.0, std::numeric_limits<float>::quiet_NaN()});
}

#endif

// Test use of constants inside sub-graphs, which are promoted to initializers by ORT.
TEST(FunctionTest, NestedConstant) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 17 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            xseq = SequenceConstruct (x)
            yseq = SequenceMap (xseq) <body =
              zeropad (float[3] lx) => (float[6] ly) {
                zeros = Constant <value = float[3] {0.0, 0.0, 0.0}> ()
                ly = Concat <axis = 0> (lx, zeros)
              }>
            zero = Constant <value = int64{0}> ()
            y = SequenceAt (yseq, zero)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {1.0, 2.0, 3.0, 0.0, 0.0, 0.0});
}

// GH13121. Model with function body that has variadic inputs (or outputs) was not loading.
// Add handling for variadics to IOTypeConstraintHelper. Test model has a Concat and Split to test both variadic
// inputs and outputs.
TEST(FunctionTest, Variadics) {
  Status status;
  auto model_uri = ORT_TSTR("testdata/function_with_variadics.onnx");

  SessionOptions so;
  so.session_logid = "FunctionTest.Variadics";
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_uri));
  ASSERT_STATUS_OK(session_object.Initialize());
}

// A variation of the variadics issue above, where the first input/output of the
// variadic list is NOT an input/output of the function.
TEST(FunctionTest, VariadicsNonInputOutput) {
  const char* code = R"(
    <ir_version: 8, opset_import: ["" : 17, "local" : 1]>
    mymodel (float[2] x) => (float[3] y) {
      y = local.func (x)
    }

    <opset_import: ["" : 17 ],  domain: "local">
    func (a) => (y) {
      b = Identity(a)
      z = Concat <axis = 0> (b, a, b)
      y, w = Split (z)
    }
  )";

  Check(code, "x", {1.0, 2.0}, "y", {1.0, 2.0, 1.0});
}

// Test use of outer-scope names inside sub-graphs in functions that are inlined.
TEST(FunctionTest, OuterScopeName) {
  const char* code = R"(
        <ir_version: 8, opset_import: [ "" : 17 ]>
        agraph (float[N] x) => (float[N] y)
        {
            xseq = SequenceConstruct (x)
            zeros = Constant <value = float[3] {0.0, 0.0, 0.0}> ()
            yseq = SequenceMap (xseq) <body =
              zeropad (float[3] lx) => (float[6] ly) {
                ly = Concat <axis = 0> (lx, zeros)
              }>
            zero = Constant <value = int64{0}> ()
            y = SequenceAt (yseq, zero)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {1.0, 2.0, 3.0, 0.0, 0.0, 0.0});
}

// Test use of functions with unused inputs:
TEST(FunctionTest, UnusedFunctionInputs) {
  const char* code = R"(
    <ir_version: 8, opset_import: ["" : 17, "local" : 1]>
    mymodel (float[3] x) => (float[3] y) {
      y = local.func (x, x, x)
    }

    <opset_import: ["" : 17 ],  domain: "local">
    func (a, b, c) => (y) {
      y = Mul (a, b)
    }
  )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {1.0, 4.0, 9.0});
}

// Test constant-folding inside a sub-graph is handled correctly
// for functions that are inlined.
TEST(FunctionTest, ConstantFoldingInSubGraph) {
  const char* code = R"(
    <ir_version: 8, opset_import: [ "" : 17 ]>
    agraph (float[N] X) => (float[M] Y)  {
        seq1 = SequenceConstruct(X, X, X)
        seq2 = SequenceMap (seq1) <body =
            add1 (float[K] Z) => (float[K] W) {
                C1 = Constant <value = float {1.0}> ()
                C2 = Constant <value = float {1.0}> ()
                # C is a constant, which will be constant-folded into an initializer out of the sub-graph.
                C = Add (C1, C2)
                # After optimization, only following Add will be left in this sub-graph.
                W = Add (Z, C)
            }
        >
        Y = ConcatFromSequence <axis=0> (seq2)
    }
  )";

  Check(code, "X", {1.0, 2.0, 3.0}, "Y", {3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0});
}

TEST(FunctionTest, TestInlinedLocalFunctionRemoved) {
  std::string serialized_model;
  ParseOnnxSource(basic_code, serialized_model);

  // Default is to do AOT Function inlining
  SessionOptions session_options;
  InferenceSessionWrapper session_object{session_options, GetEnvironment()};

  std::stringstream sstr(serialized_model);
  auto status = session_object.Load(sstr);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  auto model_proto = session_object.GetModel().ToProto();
  ASSERT_EQ(1, model_proto.functions_size());

  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  // All functions removed
  model_proto = session_object.GetModel().ToProto();
  ASSERT_EQ(0, model_proto.functions_size());
}

TEST(FunctionTest, TestInlinedLocalFunctionNotRemoved) {
  std::string serialized_model;
  ParseOnnxSource(basic_code, serialized_model);

  // Default is to do AOT Function inlining
  SessionOptions session_options;
  InferenceSessionWrapper session_object{session_options, GetEnvironment()};

  using InternalTestingEP = onnxruntime::internal_testing_ep::InternalTestingExecutionProvider;
  const std::unordered_set<std::string> empty_set;
  auto internal_testing_ep = std::make_unique<InternalTestingEP>(empty_set, empty_set, DataLayout::NCHW);
  internal_testing_ep->EnableStaticKernels().TakeAllNodes();

  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(internal_testing_ep)));

  std::stringstream sstr(serialized_model);
  ASSERT_STATUS_OK(session_object.Load(sstr));

  auto model_proto = session_object.GetModel().ToProto();
  ASSERT_EQ(1, model_proto.functions_size());

  ASSERT_STATUS_OK(session_object.Initialize());

  // myfun is not removed because it was claimed by InternalTestingEP
  model_proto = session_object.GetModel().ToProto();
#ifdef USE_TVM
  // TVM EP takes the whole graph and optimizes it within its own framework.
  // It does not retain the original graph.
  ASSERT_EQ(0, model_proto.functions_size());
#else
  ASSERT_EQ(1, model_proto.functions_size());
#endif
}

TEST(FunctionTest, TestInlinedFunctionDoesNotReserrectNonExistingArgs) {
  // Verify this runs
  constexpr const ORTCHAR_T* model_uri = ORT_TSTR("testdata/transform/gh_issue_18338.onnx");

  SessionOptions session_options;
  InferenceSessionWrapper session_object{session_options, GetEnvironment()};

  ASSERT_STATUS_OK(session_object.Load(model_uri));
  ASSERT_STATUS_OK(session_object.Initialize());

  // Scalar shape for input_0 and output
  const std::string input_names[] = {"input_0"};
  const std::string output_names[] = {"_val_3"};
  TensorShape input_shape;
  MLFloat16 input_0_data{684.f};

  OrtValue input_0;
  Tensor::InitOrtValue(DataTypeImpl::GetType<MLFloat16>(), input_shape, &input_0_data, OrtMemoryInfo(), input_0);

  std::vector<OrtValue> fetches(1);
  RunOptions run_options;
  ASSERT_STATUS_OK(session_object.Run(run_options, AsSpan(input_names), AsSpan({input_0}),
                                      AsSpan(output_names), &fetches, 0));
}

}  // namespace test
}  // namespace onnxruntime
