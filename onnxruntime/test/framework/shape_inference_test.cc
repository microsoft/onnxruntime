// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <unordered_map>

#include "gtest/gtest.h"
#include "core/common/span_utils.h"
#include "core/graph/model.h"
#include "core/graph/data_propagation/data_propagation_value_utils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/framework/model_builder_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_utils.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/test_environment.h"

using namespace ONNX_NAMESPACE;

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

using namespace modelbuilder;

class ShapeInferenceTest : public ::testing::Test {
 protected:
  onnxruntime::Model model_;
  int node_count_;
  std::unordered_map<std::string, std::unique_ptr<onnxruntime::NodeArg>> name_to_arg_;

 public:
  ShapeInferenceTest() : model_("Test", false, DefaultLoggingManager().DefaultLogger()), node_count_(0) {}

  void Input(const std::string& name, const Type& type) {
    name_to_arg_[name] = std::make_unique<onnxruntime::NodeArg>(name, &type.value);
  }

  onnxruntime::NodeArg* Arg(const std::string& name) {
    if (name_to_arg_.count(name) == 0)
      name_to_arg_[name] = std::make_unique<onnxruntime::NodeArg>(name, nullptr);
    return name_to_arg_[name].get();
  }

  onnxruntime::Node& Node(const std::string& op, const std::string& input, const std::string& output) {
    std::vector<onnxruntime::NodeArg*> input_args({Arg(input)});
    std::vector<onnxruntime::NodeArg*> output_args({Arg(output)});
    int num = node_count_++;
    return model_.MainGraph().AddNode("node" + std::to_string(num), op, "test op", input_args, output_args);
  }

  void DoShapeInference() {
    auto status = model_.MainGraph().Resolve();
    EXPECT_TRUE(status.IsOK()) << "Graph resolve failed: " << status.ErrorMessage();
  }

  const TensorShapeProto* InputShape(onnxruntime::Node& node, int arg_num = 0) {
    return node.InputDefs()[arg_num]->Shape();
  }

  const TensorShapeProto* OutputShape(onnxruntime::Node& node, int arg_num = 0) {
    return node.OutputDefs()[arg_num]->Shape();
  }

};  // namespace test

TEST_F(ShapeInferenceTest, BasicTest) {
  Type type1({1, 50, 100});
  Input("X1", type1);

  auto& node = Node("Cast", "X1", "Y1");
  node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_INT32});

  DoShapeInference();
  // check inferred shapes
  Shape expected_shape({1, 50, 100});
  CheckShapeEquality(OutputShape(node), &expected_shape.value);
  CheckShapeEquality(InputShape(node), OutputShape(node));
}

TEST(ShapeInferenceV2Test, PartialDataPropagationTest) {
  {
    // Model #1
    // This model contains "Shape" and "Reshape" operators.
    auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_with_shape_related_nodes.onnx");

    Ort::SessionOptions session_options{};
    session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    session_options.AddFreeDimensionOverrideByName("batch", 1);
    session_options.AddFreeDimensionOverrideByName("width", 64);
    session_options.AddFreeDimensionOverrideByName("height", 64);

    // Even though all graph optimizations are disabled, the free dimension override is still enabled by default.
    // The shape of graph's output should be correctly inferred by shape inference and data propagation.
    Ort::Session session(*ort_env, model_path, session_options);

    // This graph only has one output
    ORT_ENFORCE(session.GetOutputCount() == 1);

    Ort::TypeInfo type_info = session.GetOutputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = tensor_info.GetShape();
    EXPECT_TRUE(output_shape.size() == 4) << "The output shape should have 4 dimensions";
    EXPECT_TRUE(output_shape[0] == 1) << "The first dimension should have 1 as value";
    EXPECT_TRUE(output_shape[1] == 3) << "The second dimension should have 3 as value";
    EXPECT_TRUE(output_shape[2] == 64) << "The second dimension should have 64 as value";
    EXPECT_TRUE(output_shape[3] == 64) << "The second dimension should have 64 as value";
  }

  {
    // Model #2
    // This model contains "Shape", "Reshape", "Gather" and "Unsqueeze" operators.
    auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_with_shape_related_nodes_v2.onnx");

    Ort::SessionOptions session_options{};
    session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    session_options.AddFreeDimensionOverrideByName("batch", 1);
    session_options.AddFreeDimensionOverrideByName("width", 64);
    session_options.AddFreeDimensionOverrideByName("height", 64);

    // Even though all graph optimizations are disabled, the free dimension override is still enabled by default.
    // The shape of graph's output should be correctly inferred by shape inference and data propagation.
    Ort::Session session(*ort_env, model_path, session_options);

    // This graph only has one output
    ORT_ENFORCE(session.GetOutputCount() == 1);

    Ort::TypeInfo type_info = session.GetOutputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = tensor_info.GetShape();
    EXPECT_TRUE(output_shape.size() == 3) << "The output shape should have 3 dimensions";
    EXPECT_TRUE(output_shape[0] == 1) << "The first dimension should have 1 as value";
    EXPECT_TRUE(output_shape[1] == 3) << "The second dimension should have 3 as value";
    EXPECT_TRUE(output_shape[2] == 4096) << "The second dimension should have 4096 as value";
  }

  {
    // Model #3
    // This model extends model #2 and appends Unsqueeze -> Unsqueeze -> Squeeze -> Squeeze -> Reshape to the end.
    auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_with_shape_related_nodes_v3.onnx");

    Ort::SessionOptions session_options{};
    session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    session_options.AddFreeDimensionOverrideByName("batch", 1);
    session_options.AddFreeDimensionOverrideByName("width", 64);
    session_options.AddFreeDimensionOverrideByName("height", 64);

    // Even though all graph optimizations are disabled, the free dimension override is still enabled by default.
    // The shape of graph's output should be correctly inferred by shape inference and data propagation.
    Ort::Session session(*ort_env, model_path, session_options);

    // This graph only has one output
    ORT_ENFORCE(session.GetOutputCount() == 1);

    Ort::TypeInfo type_info = session.GetOutputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = tensor_info.GetShape();
    EXPECT_TRUE(output_shape.size() == 3) << "The output shape should have 3 dimensions";
    EXPECT_TRUE(output_shape[0] == 1) << "The first dimension should have 1 as value";
    EXPECT_TRUE(output_shape[1] == 3) << "The second dimension should have 3 as value";
    EXPECT_TRUE(output_shape[2] == 4096) << "The second dimension should have 4096 as value";
  }

  {
    // Model #4
    // This model contains Shape, Reshape, Squeeze, Range, ReduceSum.
    // It's from SoftmaxGrad_DefaultAxis test.
    auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_with_shape_related_nodes_v4.onnx");

    Ort::SessionOptions session_options{};
    session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);

    // Make sure it can load the model and run shape inference without errors.
    Ort::Session session(*ort_env, model_path, session_options);
  }

  {
    // Model #5
    // Regression test for the Shape -> Identity -> Unsqueeze path. ORT_ENABLE_BASIC removes the Identity,
    // then graph resolution re-runs data propagation after large initializers have been converted to
    // in-memory external OrtValues. The Unsqueeze axes initializer exercises the INT64 in-memory external path in
    // Graph::SaveShapeValuesFromDataPropagation.
    auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_unsqueeze_inmemory_int64.onnx");

    Ort::SessionOptions session_options{};
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    Ort::Session session(*ort_env, model_path, session_options);

    ORT_ENFORCE(session.GetOutputCount() == 1);
  }
}

// Regression test for the Shape -> Gather(1-D index) -> TopK rank-drop.
//
// The Gather index is the 1-D constant [-1] (rank 1), so per ONNX Gather semantics the Gather
// output is a rank-1, single-element tensor -- exactly what TopK's K input requires. Previously
// the Gather custom data propagation scalarized that single element (dropping the rank),
// producing a 0-D K initializer that ONNX TopK shape inference rejected with
// "K input must be a one-dimensional tensor of size 1." at model load time.
//
// The failure reproduces even at ORT_DISABLE_ALL (constant folding never runs there), which
// proves the cause is shape-inference data propagation, not constant folding. This test loads
// the model at the disabled, basic, and all-optimization levels, which bracket the data-propagation
// path: data propagation runs in the pre-optimization Graph::Resolve pass, so it is independent of
// the graph-optimization level (ORT_ENABLE_ALL already applies the extended and layout transformers,
// so they need not be enumerated separately). The test asserts the rank-1 K shape is preserved so the
// model loads and TopK output shapes are inferred correctly.
TEST(ShapeInferenceV2Test, GatherToTopKRankPreservationTest) {
  auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_gather_topk.onnx");

  for (auto opt_level : {ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_ALL}) {
    Ort::SessionOptions session_options{};
    session_options.SetGraphOptimizationLevel(opt_level);

    // Loading must succeed at each level; before the fix this threw a
    // ShapeInferenceError at ORT_DISABLE_ALL (and above).
    Ort::Session session(*ort_env, model_path, session_options);

    ORT_ENFORCE(session.GetOutputCount() == 2);

    // K is data-propagated as the last dimension of X (2000), so both TopK outputs are 1-D
    // tensors of length 2000.
    for (size_t output_index = 0; output_index < 2; ++output_index) {
      Ort::TypeInfo type_info = session.GetOutputTypeInfo(output_index);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> output_shape = tensor_info.GetShape();
      EXPECT_TRUE(output_shape.size() == 1) << "TopK output should be a 1-D tensor";
      EXPECT_TRUE(output_shape[0] == 2000) << "TopK output length should be the inferred K value (2000)";
    }
  }
}

// Regression test for the Shape -> Gather(1-D index) -> Mul -> TopK chain.
//
// This covers the elementwise data-propagation consumers (Add/Sub/Mul/Div) with rank-1
// single-element operands. Both Gather indices are 1-D constants, so each Gather output is a
// rank-1 single-element value; Mul must keep propagating that (as a rank-1 [1] value) so the
// downstream TopK still receives a valid 1-D K. Previously the elementwise ops only handled
// rank-0 scalar operands, so once the Gather producer was fixed to emit rank-1 values the chain
// would have silently stopped propagating at Mul (degrading TopK's K to a symbolic dim).
//
// Asserting that both TopK outputs resolve to the concrete length 2000 (== 50 * 40) proves the
// value propagated end to end through Mul rather than being lost.
TEST(ShapeInferenceV2Test, GatherMulToTopKRankPreservationTest) {
  auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_gather_mul_topk.onnx");

  for (auto opt_level : {ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_ALL}) {
    Ort::SessionOptions session_options{};
    session_options.SetGraphOptimizationLevel(opt_level);

    // Loading must succeed at each level; before the fix this threw a
    // ShapeInferenceError at ORT_DISABLE_ALL (and above).
    Ort::Session session(*ort_env, model_path, session_options);

    ORT_ENFORCE(session.GetOutputCount() == 2);

    // K = 50 * 40 = 2000 is data-propagated through the two Gathers and the Mul, so both TopK
    // outputs are 1-D tensors of length 2000.
    for (size_t output_index = 0; output_index < 2; ++output_index) {
      Ort::TypeInfo type_info = session.GetOutputTypeInfo(output_index);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> output_shape = tensor_info.GetShape();
      EXPECT_TRUE(output_shape.size() == 1) << "TopK output should be a 1-D tensor";
      EXPECT_TRUE(output_shape[0] == 2000) << "TopK output length should be the propagated K value (2000)";
    }
  }
}

// Negative regression test for the single-element guard in the elementwise data-propagation
// consumers (Add/Sub/Mul/Div), exercised end to end.
//
// Shape(X[3, 4]) -> S is a rank-1 MULTI-element value [3, 4]. Mul(S, S) -> M must stay the
// multi-element value [9, 16] so ConstantOfShape(M) resolves to the rank-2 shape [9, 16]. For a
// multi-element data-propagation output ONNX's own Mul data propagation supplies the value and
// ORT's custom elementwise propagation is intentionally NOT engaged (it is dispatched only for a
// rank-0 result, see CreateCustomDataPropagation). This is an end-to-end non-regression guard that
// the rank-routing changes did not disturb the multi-element chain; the single-element guard inside
// the shared helper is exercised directly by SinglePropagatedShapeValueGuardTest below.
TEST(ShapeInferenceV2Test, ShapeMulMultiElementNoScalarCollapseTest) {
  auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_shape_mul_constantofshape.onnx");

  for (auto opt_level : {ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_ALL}) {
    Ort::SessionOptions session_options{};
    session_options.SetGraphOptimizationLevel(opt_level);

    Ort::Session session(*ort_env, model_path, session_options);

    ORT_ENFORCE(session.GetOutputCount() == 1);

    Ort::TypeInfo type_info = session.GetOutputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = tensor_info.GetShape();

    // ConstantOfShape(Mul(Shape(X), Shape(X))) == ConstantOfShape([9, 16]) -> rank-2 [9, 16].
    // A collapse of the multi-element value to a scalar would yield rank 1 here.
    EXPECT_TRUE(output_shape.size() == 2) << "Output should stay rank 2; the multi-element value must not collapse";
    EXPECT_TRUE(output_shape.size() == 2 && output_shape[0] == 9 && output_shape[1] == 16)
        << "ConstantOfShape output should be the propagated multi-element shape [9, 16]";
  }
}

// Regression for microsoft/onnxruntime#29072.
// End-to-end lock for Gather's custom data-propagation DECLINE on a rank-2 index.
//
// The Gather index is the constant [[-1]] of shape [1, 1] (rank 2), so the Gather output rank is
// data_rank - 1 + index_rank = 1 - 1 + 2 = 2. The single-value data-propagation channel can carry
// only rank-0 (scalar) and rank-1 [1] values, so Gather's custom propagation must DECLINE rather
// than fabricate a rank-1 value. The index is a raw graph initializer (not surfaced through the
// data-propagation getInputData channel), so ONNX's own Gather data propagator bails and control
// reaches our custom decline branch.
//
// The decline is made OBSERVABLE through a rank-lowering Squeeze:
//   Shape(X) -> Gather([[ -1 ]]) -> Squeeze -> Range(0, K, 1)
// With the correct decline, nothing is propagated, Squeeze has no value to lower, and Range's limit
// stays symbolic -- so the Range output dimension is UNKNOWN (it carries no concrete dim_value). If
// the decline were relaxed to emit a rank-1 [1] value, Squeeze would lower it to the scalar 2000 and
// Range would resolve to a concrete length 2000. Asserting the dimension has no concrete value
// discriminates the correct decline from the rank-fabricating bug -- real end-to-end coverage of the
// decline branch (no shared-helper abstraction, no tautological unit test).
//
// This drives the decline through onnxruntime::Model::Load (which runs Graph::Resolve) and inspects
// the resulting output NodeArg shape directly -- no InferenceSession, so none of the session's
// arena / execution-provider / kernel-registry allocations are created (microsoft/onnxruntime#29139:
// the single-process onnxruntime_test_all run sits near the AddressSanitizer size-class ceiling, so
// each added test must stay lean). Data propagation runs inside Graph::Resolve
// (Graph::InferAndVerifyTypeMatch -> data-propagation RunInferencing); constant folding is a separate
// session-level optimizer pass that never runs here, so Resolve alone exercises the decline branch in
// isolation with no folding to mask it.
TEST(ShapeInferenceV2Test, GatherRank2IndexDeclineTest) {
  auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_gather_rank2_decline.onnx");

  std::shared_ptr<onnxruntime::Model> model;
  // Model::Load runs Graph::Resolve, which performs data propagation. A correct decline keeps the
  // Range length symbolic, so the resolve (and therefore the load) succeeds.
  ASSERT_STATUS_OK(onnxruntime::Model::Load(model_path, model, nullptr,
                                            DefaultLoggingManager().DefaultLogger()));

  const auto& graph_outputs = model->MainGraph().GetOutputs();
  ASSERT_EQ(graph_outputs.size(), static_cast<size_t>(1));
  const TensorShapeProto* output_shape = graph_outputs[0]->Shape();
  ASSERT_NE(output_shape, nullptr) << "Range output should have an inferred (rank-1) shape";

  // Range output is 1-D; its single dimension must stay SYMBOLIC (no concrete dim_value) because the
  // rank-2 Gather index declines rather than propagating a (rank-fabricated) concrete K. A relaxed
  // decline would fabricate a rank-1 value that Squeeze lowers to scalar 2000, giving the dimension a
  // concrete value 2000 -- which these assertions reject.
  ASSERT_EQ(output_shape->dim_size(), 1) << "Range output should be a 1-D tensor";
  EXPECT_FALSE(output_shape->dim(0).has_dim_value())
      << "Range length must stay symbolic; the rank-2 Gather index must decline rather than "
         "fabricate a concrete K (a relaxed decline concretizes it to 2000)";
}

// Regression for microsoft/onnxruntime#29072.
// End-to-end lock for Unsqueeze's custom data-propagation DECLINE on a single-element value.
//
// Shape(X) -> Gather([-1]) produces a rank-1 single-element value (the last dimension of X, 2000).
// Unsqueezing that single-element (scalar-like, rank-1 [1]) value would yield a rank >= 2 result
// ([1, 2000]) that the single-value channel cannot faithfully represent, so Unsqueeze's custom
// propagation must DECLINE rather than fabricate the misleading [1, value].
//
// The decline is made OBSERVABLE through a rank-lowering Squeeze:
//   Shape(X) -> Gather([-1]) -> Unsqueeze([0]) -> Squeeze -> Range(0, K, 1)
// With the correct decline, no value is propagated past Unsqueeze, Squeeze has nothing to lower, and
// Range's limit stays symbolic, so the graph resolves with an unknown Range length. A relaxed decline
// that fabricated [1, 2000] would propagate a non-scalar value to Range, which ONNX Range shape
// inference rejects ("Input to 'Range' op should be scalars") -- Graph::Resolve then returns a
// non-OK status and the load fails. So the decline IS observable: correct -> loads with a symbolic
// length; relaxed -> load error. This test asserts the model loads and the length stays symbolic.
//
// Like GatherRank2IndexDeclineTest this drives the decline through onnxruntime::Model::Load (which
// runs Graph::Resolve) and inspects the output NodeArg shape directly -- no InferenceSession, so the
// session arena / EP / kernel-registry allocations are never created (microsoft/onnxruntime#29139).
// Data propagation runs inside Graph::Resolve; constant folding is a separate session-level optimizer
// pass that never runs here, so Resolve alone exercises the decline branch with no folding to mask it.
TEST(ShapeInferenceV2Test, UnsqueezeSingleValueDeclineTest) {
  auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_unsqueeze_decline.onnx");

  std::shared_ptr<onnxruntime::Model> model;
  // Model::Load runs Graph::Resolve. A correct decline keeps Range's input scalar so the resolve (and
  // therefore the load) succeeds; a relaxed decline makes Range's input non-scalar, which Range shape
  // inference rejects, so Resolve returns non-OK and this assertion fails on the load itself.
  ASSERT_STATUS_OK(onnxruntime::Model::Load(model_path, model, nullptr,
                                            DefaultLoggingManager().DefaultLogger()));

  const auto& graph_outputs = model->MainGraph().GetOutputs();
  ASSERT_EQ(graph_outputs.size(), static_cast<size_t>(1));
  const TensorShapeProto* output_shape = graph_outputs[0]->Shape();
  ASSERT_NE(output_shape, nullptr) << "Range output should have an inferred (rank-1) shape";

  // Range output is 1-D; its single dimension must stay SYMBOLIC because Unsqueeze declines rather
  // than fabricating a rank >= 2 value. A relaxed decline fabricates [1, 2000], which makes Range's
  // input non-scalar and the load fails above -- so any regression here is caught either as a load
  // failure or as a concrete dim_value below.
  ASSERT_EQ(output_shape->dim_size(), 1) << "Range output should be a 1-D tensor";
  EXPECT_FALSE(output_shape->dim(0).has_dim_value())
      << "Range length must stay symbolic; unsqueezing a single-element value must decline "
         "(a relaxed decline fabricates [1, value] and the model fails to load)";
}

// Lock test for the Shape -> Gather(1-D index) -> Squeeze -> Range chain.
//
// Squeeze of a rank-1 single-element value [1] removes the size-1 dimension and yields a 0-D
// scalar -- the strict-improvement behavior of Squeeze's custom data propagation. This test locks
// that improvement end to end: the Gather produces a rank-1 single element (the last dimension of
// X, 2000); Squeeze must propagate it as the scalar 2000; and Range(0, K, 1) must then resolve to
// a concrete 1-D tensor of length 2000. If Squeeze dropped, corrupted, or failed to scalarize the
// value, Range's length would stay symbolic (size 0 here) instead of the concrete 2000.
TEST(ShapeInferenceV2Test, GatherSqueezeRangeRankPreservationTest) {
  auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_gather_squeeze_range.onnx");

  for (auto opt_level : {ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_ALL}) {
    Ort::SessionOptions session_options{};
    session_options.SetGraphOptimizationLevel(opt_level);

    Ort::Session session(*ort_env, model_path, session_options);

    ORT_ENFORCE(session.GetOutputCount() == 1);

    Ort::TypeInfo type_info = session.GetOutputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = tensor_info.GetShape();

    // Range(0, Squeeze(Gather(Shape(X), [-1])), 1) == Range(0, 2000, 1) -> 1-D length 2000.
    EXPECT_TRUE(output_shape.size() == 1) << "Range output should be a 1-D tensor";
    EXPECT_TRUE(output_shape.size() == 1 && output_shape[0] == 2000)
        << "Range length should be the scalar K (2000) propagated through Squeeze";
  }
}

// Unit test for the single-element guard in TryGetSinglePropagatedShapeValue, the helper shared by
// the Add/Sub/Mul/Div data-propagation consumers.
//
// A NodeArg can carry a propagated shape value in one of two non-interchangeable channels: a rank-0
// scalar (inferred_scalar_value_) or a rank>=1 list of values (inferred_shape_values_). The helper
// must surface a single value from either a scalar source or a rank-1 SINGLE-element source (and
// report its rank), and must DECLINE for a rank-1 MULTI-element source. Declining is load-bearing:
// silently using element[0] of a multi-element value would let an elementwise op emit a bogus
// scalar product that getInputData() would then hand to a rank-sensitive consumer. This test
// exercises the guard directly -- a relaxed guard that accepted element[0] fails the multi-element
// case below.
TEST(ShapeInferenceV2Test, SinglePropagatedShapeValueGuardTest) {
  // Scalar channel -> accepted, reported as rank-0.
  {
    NodeArg arg("scalar_arg", nullptr);
    arg.SetInferredShapeScalarValue(5);
    int64_t value = 0;
    bool is_rank1 = true;
    EXPECT_TRUE(TryGetSinglePropagatedShapeValue(arg, value, is_rank1));
    EXPECT_EQ(value, 5);
    EXPECT_FALSE(is_rank1);
  }

  // Rank-1 single-element value -> accepted, reported as rank-1.
  {
    NodeArg arg("rank1_single_arg", nullptr);
    auto& values = arg.GetMutableInferredShapeValues();
    values.emplace();
    values->add_dim()->set_dim_value(7);
    int64_t value = 0;
    bool is_rank1 = false;
    EXPECT_TRUE(TryGetSinglePropagatedShapeValue(arg, value, is_rank1));
    EXPECT_EQ(value, 7);
    EXPECT_TRUE(is_rank1);
  }

  // Rank-1 MULTI-element value -> DECLINED (the guard); element[0] must never be used.
  {
    NodeArg arg("rank1_multi_arg", nullptr);
    auto& values = arg.GetMutableInferredShapeValues();
    values.emplace();
    values->add_dim()->set_dim_value(3);
    values->add_dim()->set_dim_value(4);
    int64_t value = -1;
    bool is_rank1 = false;
    EXPECT_FALSE(TryGetSinglePropagatedShapeValue(arg, value, is_rank1))
        << "A rank-1 multi-element value must be declined, not collapsed to element[0]";
  }

  // Rank-1 single element with a SYMBOLIC (no concrete value) dim -> declined.
  {
    NodeArg arg("rank1_symbolic_arg", nullptr);
    auto& values = arg.GetMutableInferredShapeValues();
    values.emplace();
    values->add_dim()->set_dim_param("N");
    int64_t value = -1;
    bool is_rank1 = false;
    EXPECT_FALSE(TryGetSinglePropagatedShapeValue(arg, value, is_rank1))
        << "A symbolic single-element value has no concrete value and must be declined";
  }

  // No propagated value on either channel -> declined.
  {
    NodeArg arg("empty_arg", nullptr);
    int64_t value = -1;
    bool is_rank1 = false;
    EXPECT_FALSE(TryGetSinglePropagatedShapeValue(arg, value, is_rank1));
  }
}

// Unit test that SetSinglePropagatedShapeValue is correct-by-construction: it populates exactly one
// channel and clears the other. This pins the invariant the elementwise/Gather consumers rely on --
// the scalar-first reader (TryGetSinglePropagatedShapeValue) and the values-first getInputData() can
// only agree on rank if no NodeArg ever carries both channels. A setter that left the opposite
// channel populated would let a stale value win (reintroducing a rank mismatch), so each case below
// pre-populates the opposite channel and asserts the setter cleared it.
TEST(ShapeInferenceV2Test, SetSinglePropagatedShapeValueKeepsSingleChannelTest) {
  // Scalar write must clear a stale values channel.
  {
    NodeArg arg("scalar_over_stale_values", nullptr);
    auto& stale = arg.GetMutableInferredShapeValues();
    stale.emplace();
    stale->add_dim()->set_dim_value(99);

    SetSinglePropagatedShapeValue(arg, 5, /*is_rank1=*/false);

    ASSERT_TRUE(arg.GetInferredShapeScalarValue().has_value());
    EXPECT_EQ(arg.GetInferredShapeScalarValue().value(), 5);
    EXPECT_FALSE(arg.GetInferredShapeValues().has_value())
        << "Scalar write must clear the values channel that getInputData() would otherwise prefer";

    int64_t value = 0;
    bool is_rank1 = true;
    EXPECT_TRUE(TryGetSinglePropagatedShapeValue(arg, value, is_rank1));
    EXPECT_EQ(value, 5);
    EXPECT_FALSE(is_rank1);
  }

  // Rank-1 write must clear a stale scalar channel.
  {
    NodeArg arg("values_over_stale_scalar", nullptr);
    arg.SetInferredShapeScalarValue(99);

    SetSinglePropagatedShapeValue(arg, 7, /*is_rank1=*/true);

    EXPECT_FALSE(arg.GetInferredShapeScalarValue().has_value())
        << "Rank-1 write must clear the scalar channel that the scalar-first reader would otherwise return";
    ASSERT_TRUE(arg.GetInferredShapeValues().has_value());
    ASSERT_EQ(arg.GetInferredShapeValues()->dim_size(), 1);
    EXPECT_EQ(arg.GetInferredShapeValues()->dim(0).dim_value(), 7);

    int64_t value = 0;
    bool is_rank1 = false;
    EXPECT_TRUE(TryGetSinglePropagatedShapeValue(arg, value, is_rank1));
    EXPECT_EQ(value, 7);
    EXPECT_TRUE(is_rank1);
  }
}

namespace {
struct MyCustomKernelWithOptionalInput {
  MyCustomKernelWithOptionalInput(const OrtKernelInfo* info) {
    Ort::ConstKernelInfo k_info(info);

    Ort::KeyValuePairs kvp = k_info.GetConfigEntries();

    EXPECT_NE(nullptr, kvp.GetValue("session.inter_op.allow_spinning"));
    EXPECT_STREQ("0", kvp.GetValue("session.inter_op.allow_spinning"));

    EXPECT_NE(nullptr, kvp.GetValue("session.intra_op.allow_spinning"));
    EXPECT_STREQ("0", kvp.GetValue("session.intra_op.allow_spinning"));

    EXPECT_EQ(nullptr, kvp.GetValue("__not__exist__"));
  }

  OrtStatusPtr ComputeV2(OrtKernelContext* /* context */) const {
    return nullptr;
  }
};

struct MyCustomOpWithOptionalInput : Ort::CustomOpBase<MyCustomOpWithOptionalInput,
                                                       MyCustomKernelWithOptionalInput,
                                                       true> {
  explicit MyCustomOpWithOptionalInput(const char* provider) : provider_(provider) {}

  OrtStatusPtr CreateKernelV2(const OrtApi& /* api */, const OrtKernelInfo* info, void** kernel) const {
    *kernel = new MyCustomKernelWithOptionalInput(info);
    return nullptr;
  };

  const char* GetName() const { return "FooBar"; };
  const char* GetExecutionProviderType() const { return provider_; };

  size_t GetInputTypeCount() const { return 3; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
    // The second input (index == 1) is optional
    if (index == 1)
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;

    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }

 private:
  const char* provider_;
};

const ORTCHAR_T* const OPTIONAL_INPUT_CUSTOM_OP_MODEL_URI_2 = ORT_TSTR("testdata/foo_bar_2.onnx");

}  // namespace

// CustomOps Output type inference function quits if it
// encounters the an output that is optional and absent.
// It quits without any errors or logging. We want to make sure
// that inference proceeds for all of the outputs when absent optional inputs are present
TEST(ShapeInferenceCustomOpTest, custom_op_optional_input_inference_test) {
  MyCustomOpWithOptionalInput custom_op{onnxruntime::kCpuExecutionProvider};
  custom_op.InferOutputShapeFn = [](const OrtCustomOp* /*op*/, OrtShapeInferContext* /*ctx*/) -> OrtStatusPtr {
    return nullptr;
  };

  const auto& env = GetEnvironment();

  Ort::CustomOpDomain op_domain("test");
  op_domain.Add(&custom_op);

  std::initializer_list<OrtCustomOpDomain*> op_domains = {static_cast<OrtCustomOpDomain*>(op_domain)};

  SessionOptions sess_opts;
  sess_opts.inter_op_param.thread_pool_size = 1;
  sess_opts.intra_op_param.thread_pool_size = 1;
  ASSERT_STATUS_OK(sess_opts.config_options.AddConfigEntry("session.inter_op.allow_spinning", "0"));
  ASSERT_STATUS_OK(sess_opts.config_options.AddConfigEntry("session.intra_op.allow_spinning", "0"));

  InferenceSessionWrapper session{sess_opts, env, OPTIONAL_INPUT_CUSTOM_OP_MODEL_URI_2};
  ASSERT_STATUS_OK(session.AddCustomOpDomains(AsSpan(op_domains)));

  ASSERT_STATUS_OK(session.Load());
  ASSERT_STATUS_OK(session.Initialize());

  const onnxruntime::Model& model = session.GetModel();
  const auto& graph = model.MainGraph();
  const auto& nodes = graph.Nodes();
  for (const auto& node : nodes) {
    if (node.OpType() == "FooBar") {
      // check inferred shapes
      const auto* node_arg = node.OutputDefs()[0];
      const auto* type_proto = node_arg->TypeAsProto();
      ASSERT_NE(nullptr, type_proto);
      ASSERT_EQ(ONNX_NAMESPACE::TypeProto::ValueCase::kTensorType, type_proto->value_case());
      ASSERT_EQ(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, type_proto->tensor_type().elem_type());
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
