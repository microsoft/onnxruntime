// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/graph/onnx_protobuf.h"
#include "onnx/defs/parser.h"

#include "core/common/span_utils.h"
#include "core/framework/customregistry.h"
#include "core/framework/op_kernel.h"
#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/internal_testing_ep/internal_testing_execution_provider.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

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
  ASSERT_STATUS_OK(session_object.Load(sstr));

  auto model_proto = session_object.GetModel().ToProto();
  ASSERT_EQ(1, model_proto.functions_size());

  ASSERT_STATUS_OK(session_object.Initialize());

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
  ASSERT_EQ(1, model_proto.functions_size());
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

/// <summary>
/// This test covers the issues:
/// https://github.com/microsoft/onnxruntime/issues/16438
/// https://github.com/microsoft/onnxruntime/issues/18781
/// </summary>
TEST(FunctionTest, Test_GH_issue_16438) {
  const char* code = R"(
    <
       ir_version: 8,
       opset_import: ["pkg.onnxscript.torch_lib" : 1, "" : 18],
       producer_name: "pytorch",
       producer_version: "2.1.0"
    >
    torch_jit (float16[5,10,5] input_0) => (double[5,10,5] _val_1) {
       _val_1 = pkg.onnxscript.torch_lib.aten_special_log_softmax <dim: int = 2, dtype: int = 11> (input_0)
    }
    <
      domain: "pkg.onnxscript.torch_lib",
      opset_import: ["" : 18]
    >
    aten_special_log_softmax <dim, dtype>(self) => (result_8)
    {
      tmp = Shape(self)
      tmp_0 = Size(tmp)
      int64_0 = Constant<value : tensor = int64 int64_0{0}> ()
      int64_0_cast = CastLike(int64_0, tmp_0)
      self_is_scalar = Equal(tmp_0, int64_0_cast)
      self_4 = If(self_is_scalar) <then_branch : graph = thenGraph_8() => (self_2) {
        tmp_1 = Constant<value_ints : ints = [0]> ()
        self_2 = Unsqueeze(self, tmp_1)
      }, else_branch : graph = elseGraph_8() => (self_3) {
        self_3 = Identity(self)
      }>
      result = LogSoftmax<axis : int = @dim>(self_4)
      result_5 = Cast<to : int = @dtype>(result)
      result_8 = If(self_is_scalar) <then_branch : graph = thenGraph_12() => (result_6) {
       result_6 = Squeeze(result_5)
      }, else_branch : graph = elseGraph_12() => (result_7) {
        result_7 = Identity(result_5)
      }>
    }
  )";

  std::string serialized_model;
  ParseOnnxSource(code, serialized_model);
  SessionOptions session_options;
  InferenceSession session_object{session_options, GetEnvironment()};

  std::stringstream sstr(serialized_model);
  auto status = session_object.Load(sstr);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
}

// Verify that when a function node with a layering annotation is inlined,
// the inlined nodes inherit the parent function node's annotation.
TEST(FunctionTest, InlinedNodesInheritLayeringAnnotation) {
  // Parse and build a Model with a local function (multi-node body: Constant + Mul).
  ONNX_NAMESPACE::OnnxParser parser(basic_code);
  ONNX_NAMESPACE::ModelProto model_proto;
  auto parse_status = parser.Parse(model_proto);
  ASSERT_TRUE(parse_status.IsOK()) << parse_status.ErrorMessage();
  ASSERT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  auto& logger = DefaultLoggingManager().DefaultLogger();
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, nullptr, logger));

  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  // Find the function call node (local.myfun) and annotate it.
  Node* func_node = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "myfun") {
      func_node = &node;
      break;
    }
  }
  ASSERT_NE(func_node, nullptr) << "Could not find function call node 'myfun'";
  ASSERT_TRUE(func_node->CanBeInlined());

  const std::string annotation = "TestLayerAnnotation";
  func_node->SetLayeringAnnotation(annotation);

  // Inline the function node.
  ASSERT_STATUS_OK(graph.InlineFunction(*func_node));
  ASSERT_STATUS_OK(graph.Resolve());

  // After inlining, the original function call node is removed and replaced
  // by the function body nodes (a Mul node; the Constant becomes an initializer).
  // Verify every remaining node inherited the annotation.
  int node_count = 0;
  for (const auto& node : graph.Nodes()) {
    ++node_count;
    EXPECT_EQ(node.GetLayeringAnnotation(), annotation)
        << "Node '" << node.Name() << "' (op: " << node.OpType()
        << ") did not inherit the parent function's layering annotation.";
  }
  EXPECT_GT(node_count, 0) << "Expected at least one inlined node in the graph.";
}

// Verify that when a function node with no layering annotation is inlined,
// the inlined nodes remain unannotated.
TEST(FunctionTest, InlinedNodesNoAnnotationWhenParentUnannotated) {
  ONNX_NAMESPACE::OnnxParser parser(basic_code);
  ONNX_NAMESPACE::ModelProto model_proto;
  auto parse_status = parser.Parse(model_proto);
  ASSERT_TRUE(parse_status.IsOK()) << parse_status.ErrorMessage();
  ASSERT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  auto& logger = DefaultLoggingManager().DefaultLogger();
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, nullptr, logger));

  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  Node* func_node = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "myfun") {
      func_node = &node;
      break;
    }
  }
  ASSERT_NE(func_node, nullptr);
  // Do NOT set any annotation on the function node.
  ASSERT_TRUE(func_node->GetLayeringAnnotation().empty());

  ASSERT_STATUS_OK(graph.InlineFunction(*func_node));
  ASSERT_STATUS_OK(graph.Resolve());

  for (const auto& node : graph.Nodes()) {
    EXPECT_TRUE(node.GetLayeringAnnotation().empty())
        << "Node '" << node.Name() << "' should not have a layering annotation "
        << "when the parent function node was unannotated.";
  }
}

// Verify annotation inheritance with two calls to the same function,
// where each call has a different annotation.
TEST(FunctionTest, InlinedNodesInheritDistinctAnnotationsPerCallSite) {
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

  ONNX_NAMESPACE::OnnxParser parser(code);
  ONNX_NAMESPACE::ModelProto model_proto;
  auto parse_status = parser.Parse(model_proto);
  ASSERT_TRUE(parse_status.IsOK()) << parse_status.ErrorMessage();
  ASSERT_TRUE(parser.EndOfInput());

  auto& logger = DefaultLoggingManager().DefaultLogger();
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(std::move(model_proto), model, nullptr, logger));

  Graph& graph = model->MainGraph();
  ASSERT_STATUS_OK(graph.Resolve());

  // Collect the two function call nodes in graph order.
  std::vector<Node*> func_nodes;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "myfun") {
      func_nodes.push_back(&node);
    }
  }
  ASSERT_EQ(func_nodes.size(), 2u);

  // Annotate each call site differently.
  func_nodes[0]->SetLayeringAnnotation("AnnotationA");
  func_nodes[1]->SetLayeringAnnotation("AnnotationB");

  // Inline the first call, then the second.
  ASSERT_STATUS_OK(graph.InlineFunction(*func_nodes[0]));
  ASSERT_STATUS_OK(graph.InlineFunction(*func_nodes[1]));
  ASSERT_STATUS_OK(graph.Resolve());

  // After inlining both calls, the graph should have nodes from both expansions.
  // Each group should carry its respective annotation.
  bool found_a = false;
  bool found_b = false;
  for (const auto& node : graph.Nodes()) {
    const auto& ann = node.GetLayeringAnnotation();
    EXPECT_TRUE(ann == "AnnotationA" || ann == "AnnotationB")
        << "Node '" << node.Name() << "' has unexpected annotation: '" << ann << "'";
    if (ann == "AnnotationA") found_a = true;
    if (ann == "AnnotationB") found_b = true;
  }
  EXPECT_TRUE(found_a) << "No node found with AnnotationA";
  EXPECT_TRUE(found_b) << "No node found with AnnotationB";
}

// Test that overloaded functions (IR version 10+) are resolved correctly.
// Two functions with the same domain and name but different overload identifiers.
TEST(FunctionTest, OverloadedFunctions) {
  const char* code = R"(
        <
        ir_version: 10,
        opset_import: [ "" : 17, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y, float[N] z)
        {
            y = local.myfun:double_it (x)
            z = local.myfun:triple_it (x)
        }

        <
        opset_import: [ "" : 17 ],
        domain: "local",
        overload: "double_it"
        >
        myfun (lx) => (ly) {
            two = Constant <value = float[1] {2.0}> ()
            ly = Mul (lx, two)
        }

        <
        opset_import: [ "" : 17 ],
        domain: "local",
        overload: "triple_it"
        >
        myfun (lx) => (ly) {
            three = Constant <value = float[1] {3.0}> ()
            ly = Mul (lx, three)
        }
        )";

  // Serialize and then load model:
  std::string serialized_model;
  ParseOnnxSource(code, serialized_model);

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
  std::vector<float> input_values = {1.0f, 2.0f, 3.0f};
  OrtValue ort_value;
  CreateMLValue<float>(provider->CreatePreferredAllocators()[0], {int64_t(input_values.size())}, input_values, &ort_value);
  feeds.insert(std::make_pair(std::string("x"), ort_value));

  std::vector<OrtValue> fetches;
  status = session_object.Run(run_options, feeds, AsSpan({std::string("y"), std::string("z")}), &fetches);
  ASSERT_TRUE(status.IsOK()) << "Session Run failed: " << status.ErrorMessage() << std::endl;

  // Check "y" output (doubled)
  auto& tensor_y = fetches[0].Get<Tensor>();
  auto* data_y = tensor_y.Data<float>();
  EXPECT_NEAR(data_y[0], 2.0f, 0.001f);
  EXPECT_NEAR(data_y[1], 4.0f, 0.001f);
  EXPECT_NEAR(data_y[2], 6.0f, 0.001f);

  // Check "z" output (tripled)
  auto& tensor_z = fetches[1].Get<Tensor>();
  auto* data_z = tensor_z.Data<float>();
  EXPECT_NEAR(data_z[0], 3.0f, 0.001f);
  EXPECT_NEAR(data_z[1], 6.0f, 0.001f);
  EXPECT_NEAR(data_z[2], 9.0f, 0.001f);
}

// Test that non-overloaded functions (empty overload) still work as before.
TEST(FunctionTest, OverloadedFunctionBackwardCompat) {
  // Same as basic_code but with ir_version: 10 to verify backward compatibility
  const char* code = R"(
        <
        ir_version: 10,
        opset_import: [ "" : 17, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.myfun (x)
        }

        <
        opset_import: [ "" : 17 ],
        domain: "local"
        >
        myfun (lx) => (ly) {
            two = Constant <value = float[1] {2.0}> ()
            ly = Mul (lx, two)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {2.0, 4.0, 6.0});
}

}  // namespace test
}  // namespace onnxruntime
