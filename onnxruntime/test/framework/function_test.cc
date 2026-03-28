// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/graph/onnx_protobuf.h"
#include "onnx/defs/parser.h"

#include "core/common/span_utils.h"
#include "core/framework/customregistry.h"
#include "core/framework/op_kernel.h"
#include "core/graph/function_utils.h"
#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/internal_testing_ep/internal_testing_execution_provider.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#endif

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
// Verify that function_utils::Specialize correctly renames value_info entries
// in Scan body subgraphs (fix for ORT issue #27887).
//
// When a local function containing a Scan op is inlined, the Inliner renames
// all node outputs in the Scan body with a unique prefix. Before the fix,
// value_info entries in the body (intermediate value shape annotations) were
// NOT renamed, leaving them orphaned. CUDA EP's memory planner reads these
// annotations to pre-allocate carry state buffers; orphaned entries cause it to
// fall back to dim_value=0 (treated as 1), producing undersized buffers.
//
// The test builds a FunctionProto whose Scan body has an intermediate value
// "temp" annotated in value_info, calls Specialize, and asserts that the
// value_info name was updated to match the renamed node output.
TEST(FunctionTest, ScanBodyValueInfoRenamedAfterSpecialize) {
  using namespace ONNX_NAMESPACE;

  // Helper: build a ValueInfoProto with symbolic-dim shape.
  auto make_vi = [](const std::string& name, int elem_type,
                    const std::vector<std::string>& dims) -> ValueInfoProto {
    ValueInfoProto vi;
    vi.set_name(name);
    auto* t = vi.mutable_type()->mutable_tensor_type();
    t->set_elem_type(elem_type);
    for (const auto& d : dims)
      t->mutable_shape()->add_dim()->set_dim_param(d);
    return vi;
  };

  // Build the Scan body GraphProto.
  // Inputs:  carry_in [B,H], x_in [B,H]
  // Outputs: carry_out [B,H], x_out [B,H]
  // Nodes:   temp = Add(carry_in, x_in)
  //          carry_out = Identity(temp)
  //          x_out     = Identity(temp)
  // value_info: temp [B,H]  ← this must be renamed after Specialize
  GraphProto body;
  body.set_name("scan_body");
  *body.add_input() = make_vi("carry_in", TensorProto_DataType_FLOAT, {"B", "H"});
  *body.add_input() = make_vi("x_in", TensorProto_DataType_FLOAT, {"B", "H"});
  *body.add_output() = make_vi("carry_out", TensorProto_DataType_FLOAT, {"B", "H"});
  *body.add_output() = make_vi("x_out", TensorProto_DataType_FLOAT, {"B", "H"});
  *body.add_value_info() = make_vi("temp", TensorProto_DataType_FLOAT, {"B", "H"});

  {
    auto* n = body.add_node();
    n->set_op_type("Add");
    n->add_input("carry_in");
    n->add_input("x_in");
    n->add_output("temp");
  }
  {
    auto* n = body.add_node();
    n->set_op_type("Identity");
    n->add_input("temp");
    n->add_output("carry_out");
  }
  {
    auto* n = body.add_node();
    n->set_op_type("Identity");
    n->add_input("temp");
    n->add_output("x_out");
  }

  // Build a FunctionProto: RunScan(fn_init, fn_seq) => (fn_final, fn_out)
  // containing the Scan node.
  FunctionProto func;
  func.set_domain("local");
  func.set_name("RunScan");
  func.add_input("fn_init");
  func.add_input("fn_seq");
  func.add_output("fn_final");
  func.add_output("fn_out");
  func.add_opset_import()->set_version(18);

  {
    auto* scan = func.add_node();
    scan->set_op_type("Scan");
    scan->add_input("fn_init");
    scan->add_input("fn_seq");
    scan->add_output("fn_final");
    scan->add_output("fn_out");

    auto* num_scan = scan->add_attribute();
    num_scan->set_name("num_scan_inputs");
    num_scan->set_type(AttributeProto_AttributeType_INT);
    num_scan->set_i(1);

    auto* body_attr = scan->add_attribute();
    body_attr->set_name("body");
    body_attr->set_type(AttributeProto_AttributeType_GRAPH);
    *body_attr->mutable_g() = body;
  }

  // Build the calling NodeProto: result_final, result_out = local.RunScan(init, seq)
  NodeProto call_node;
  call_node.set_op_type("RunScan");
  call_node.set_domain("local");
  call_node.add_input("init");
  call_node.add_input("seq");
  call_node.add_output("result_final");
  call_node.add_output("result_out");

  // Call Specialize — this simulates what Graph::InlineFunction does.
  const std::string prefix = "inl";
  onnxruntime::NodeAttributes empty_attrs;
  function_utils::Specialize(func, call_node, empty_attrs, prefix);

  // Find the Scan node in the specialized function and inspect its body.
  for (const auto& node : func.node()) {
    if (node.op_type() != "Scan") continue;

    for (const auto& attr : node.attribute()) {
      if (attr.name() != "body" || !attr.has_g()) continue;

      const GraphProto& spec_body = attr.g();

      // Collect all node output names (these have been renamed, e.g. "inl_temp").
      std::unordered_set<std::string> node_outputs;
      for (const auto& n : spec_body.node())
        for (const auto& out : n.output())
          node_outputs.insert(out);

      // There must be at least one value_info entry (the one we added for "temp").
      ASSERT_GT(spec_body.value_info_size(), 0)
          << "value_info entries in the Scan body were lost after Specialize.";

      // Every value_info name must match a node output name.
      // Before the fix: "temp" is not renamed → not in node_outputs → test fails.
      // After the fix:  "inl_temp" is in node_outputs → test passes.
      for (const auto& vi : spec_body.value_info()) {
        EXPECT_NE(node_outputs.find(vi.name()), node_outputs.end())
            << "value_info '" << vi.name() << "' is orphaned after Specialize: "
            << "its name does not match any renamed node output. "
            << "Fix: rename value_info entries in Inliner::transform(GraphProto&). "
            << "See https://github.com/microsoft/onnxruntime/issues/27887";
      }
      return;
    }
  }

  FAIL() << "Scan node with body attribute not found in specialized FunctionProto";
}

// Verifies that when a local function containing a Scan is inlined at a call site with
// concrete input shapes (e.g. init: float[4]), the inliner propagates those concrete
// shapes into the Scan body's value_info entries (replacing symbolic dim_params with
// concrete dim_values).
//
// This is important for CUDA EP: its memory planner reads the Scan body's value_info
// shape annotations to pre-allocate carry-state buffers.  If the annotations still use
// dim_param="S" after inlining, the planner falls back to dim_value=0 (treated as 1),
// causing undersized buffers and wrong results.
//
// Model: testdata/scan_in_local_function.onnx (generated by scan_in_local_function.py)
//   Main graph: init: float[4], seq: float[3,4]  (concrete dims)
//   Function local.RunAccum contains a Scan whose body value_info uses dim_param="S"
//   After inlining with concrete init:float[4], "S" must become dim_value=4.
TEST(FunctionTest, ScanBodyValueInfoConcreteShapeAfterInline) {
  constexpr const ORTCHAR_T* model_uri = ORT_TSTR("testdata/scan_in_local_function.onnx");

  // Use InferenceSessionWrapper so we can inspect the model proto after Initialize()
  // triggers AOT function inlining via Graph::InlineFunction (the Node& variant that
  // has access to concrete input TypeProtos).
  SessionOptions so;
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_uri));
  ASSERT_STATUS_OK(session_object.Initialize());

  // After initialization all functions are inlined; find the Scan node in the flat graph.
  const auto model_proto = session_object.GetModel().ToProto();
  ASSERT_EQ(model_proto.functions_size(), 0) << "Functions should all be inlined after Initialize()";

  bool found_scan = false;
  for (const auto& node : model_proto.graph().node()) {
    if (node.op_type() != "Scan") continue;
    found_scan = true;

    for (const auto& attr : node.attribute()) {
      if (attr.name() != "body" || !attr.has_g()) continue;

      const GraphProto& body = attr.g();

      // Every value_info entry that has a shape should have only concrete dim_values —
      // no symbolic dim_params should remain after inlining with concrete call-site shapes.
      for (const auto& vi : body.value_info()) {
        if (!vi.has_type() || !vi.type().has_tensor_type()) continue;
        const auto& tensor_type = vi.type().tensor_type();
        if (!tensor_type.has_shape()) continue;

        for (const auto& dim : tensor_type.shape().dim()) {
          EXPECT_TRUE(dim.dim_param().empty())
              << "value_info '" << vi.name() << "' still has symbolic dim_param='"
              << dim.dim_param() << "' after inlining with concrete call-site shapes. "
              << "Expected dim_param to be replaced with a concrete dim_value. "
              << "Fix: Inliner::Specialize(Node&) must build outer_var_types and pass "
              << "concrete shapes to build_scan_dim_bindings. "
              << "See https://github.com/microsoft/onnxruntime/issues/27887";
          if (!dim.dim_param().empty()) continue;
          // The carry-state dim should be 4 (from init: float[4]).
          EXPECT_GT(dim.dim_value(), 0)
              << "value_info '" << vi.name() << "' has dim_value=0 after inlining; "
              << "expected a positive concrete value (e.g. 4 from init: float[4]).";
        }
      }
      break;
    }
    break;
  }

  EXPECT_TRUE(found_scan) << "No Scan node found in the inlined graph — the function was not inlined.";
}
//
// Verifies that after function inlining ORT can successfully run a model
// whose local function body contains a Scan with symbolic dim annotations in
// value_info.  CUDA EP's memory planner pre-allocates Scan carry-state buffers
// using the shape annotations; without the value_info renaming fix the names
// are orphaned after inlining and CUDA falls back to dim_value=0 (→ size 1),
// producing undersized buffers and wrong results.
//
// Model: testdata/scan_in_local_function.onnx  (generated by scan_in_local_function.py)
//   Function local.RunAccum(fn_init: float[S], fn_seq: float[N,S])
//     Scan body: temp = Add(carry_in, x_in); value_info: temp: float[S]
//   Main graph (concrete dims): init: float[4], seq: float[3,4]
//
// Input:  init=[0,0,0,0], seq=[[1,1,1,1],[2,2,2,2],[3,3,3,3]]
// Output: result=[6,6,6,6], out_seq=[[1,1,1,1],[3,3,3,3],[6,6,6,6]]
static void RunScanInLocalFunctionTest(InferenceSession& session) {
  std::unique_ptr<CPUExecutionProvider> cpu_provider =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  auto alloc = cpu_provider->CreatePreferredAllocators()[0];

  // init: float[4] = [0, 0, 0, 0]
  OrtValue init_val;
  CreateMLValue<float>(alloc, {4}, {0.f, 0.f, 0.f, 0.f}, &init_val);

  // seq: float[3, 4] = [[1,1,1,1],[2,2,2,2],[3,3,3,3]]
  OrtValue seq_val;
  CreateMLValue<float>(alloc, {3, 4},
                       {1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 3.f},
                       &seq_val);

  NameMLValMap feeds{{"init", init_val}, {"seq", seq_val}};
  std::vector<OrtValue> fetches;
  RunOptions run_options;

  const std::vector<std::string> output_names = {"result", "out_seq"};
  ASSERT_STATUS_OK(session.Run(run_options, feeds, AsSpan(output_names), &fetches));
  ASSERT_EQ(fetches.size(), 2u);

  // result: float[4] = [6, 6, 6, 6]
  const auto& result_tensor = fetches[0].Get<Tensor>();
  ASSERT_EQ(result_tensor.Shape().GetDims(), (std::vector<int64_t>{4}));
  for (float v : result_tensor.DataAsSpan<float>()) {
    EXPECT_FLOAT_EQ(v, 6.f);
  }

  // out_seq: float[3, 4] = [[1,1,1,1],[3,3,3,3],[6,6,6,6]]
  const auto& out_seq_tensor = fetches[1].Get<Tensor>();
  ASSERT_EQ(out_seq_tensor.Shape().GetDims(), (std::vector<int64_t>{3, 4}));
  const float* out_data = out_seq_tensor.Data<float>();
  const float expected_out_seq[12] = {1, 1, 1, 1, 3, 3, 3, 3, 6, 6, 6, 6};
  for (int i = 0; i < 12; ++i) {
    EXPECT_FLOAT_EQ(out_data[i], expected_out_seq[i]) << "at index " << i;
  }
}

TEST(FunctionTest, ScanInLocalFunctionEndToEnd_CPU) {
  // Model generated by testdata/scan_in_local_function.py
  constexpr const ORTCHAR_T* model_uri = ORT_TSTR("testdata/scan_in_local_function.onnx");

  SessionOptions so;
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_uri));
  ASSERT_STATUS_OK(session_object.Initialize());

  RunScanInLocalFunctionTest(session_object);
}

#ifdef USE_CUDA
TEST(FunctionTest, ScanInLocalFunctionEndToEnd_CUDA) {
  // Exercises the CUDA EP memory planner path that reads value_info shape
  // annotations from the inlined Scan body to pre-allocate carry state
  // buffers. Without the fix, orphaned value_info names cause the planner to
  // fall back to dim_value=0 (treated as 1), producing undersized buffers.
  constexpr const ORTCHAR_T* model_uri = ORT_TSTR("testdata/scan_in_local_function.onnx");

  SessionOptions so;
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
  ASSERT_STATUS_OK(session_object.Load(model_uri));
  ASSERT_STATUS_OK(session_object.Initialize());

  RunScanInLocalFunctionTest(session_object);
}
#endif  // USE_CUDA

}  // namespace test
}  // namespace onnxruntime
