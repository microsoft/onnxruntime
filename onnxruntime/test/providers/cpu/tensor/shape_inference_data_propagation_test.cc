// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Shape-inference data-propagation regression tests for the custom Gather / Unsqueeze / Squeeze
// and elementwise (Add/Sub/Mul/Div) propagators. These lock the rank-preservation fix for
// microsoft/onnxruntime#29072 (a 1-D single-element Gather output must keep rank 1 so a downstream
// TopK/Range/ConstantOfShape sees a valid shape, while a rank>=2 Gather/Unsqueeze result must
// decline rather than fabricate a misleading rank).
//
// Why these tests live under test/providers/ rather than test/framework/:
// The unit tests are split across two ctest binaries by source path. partition_provider_test_srcs()
// in cmake/onnxruntime_unittests.cmake routes sources under test/providers/ (and test/contrib_ops/)
// into the onnxruntime_provider_test binary (shard 2); everything else compiles into
// onnxruntime_test_all (shard 1). Under AddressSanitizer, shard 1 sits close to an architectural
// allocator ceiling (microsoft/onnxruntime#29139): its cumulative live high-water can exhaust the
// 8 KB size class during a later baseline test (SessionState prepacking), independent of these
// tests. These cases are not the allocation site, but every test compiled into shard 1 adds to that
// cumulative footprint, so they are filed here -- in shard 2, which has headroom -- to keep shard 1
// at its known-good baseline. They exercise only the public C++ API plus Model::Load/Graph::Resolve,
// so they have no framework-internal dependency and relocate cleanly. The TEST suite name
// (ShapeInferenceV2Test) is preserved so history stays traceable.

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "core/graph/model.h"
#include "core/graph/data_propagation/data_propagation_value_utils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/asserts.h"
#include "test/test_environment.h"

using namespace ONNX_NAMESPACE;

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

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
// End-to-end lock for BOTH custom data-propagation DECLINE paths -- Gather on a rank-2 index and
// Unsqueeze on a single-element value -- driven through a SINGLE shared model so the coverage costs
// one onnxruntime::Model::Load instead of two (microsoft/onnxruntime#29139: the single-process
// onnxruntime_test_all run sits near the AddressSanitizer size-class ceiling, so the two decline
// fixtures are folded into one graph to minimize the live Model/Graph footprint).
//
// One rank-3 input X drives a shared Shape(X) -> S (the 1-D vector [3, 4, 2000]); two independent
// branches gather from S and route into a rank-lowering Squeeze -> Range so each decline is
// observable end to end:
//
//   Branch A (Gather rank-2 index -> output RA): S -> Gather([[ -1 ]]) -> Squeeze -> Range(0, K, 1)
//     The index [[-1]] has shape [1, 1] (rank 2), so the Gather output rank is
//     data_rank - 1 + index_rank = 1 - 1 + 2 = 2 -- a rank-2 single-element value the single-value
//     channel cannot represent, so Gather's custom propagation must DECLINE. With the correct
//     decline nothing propagates, Squeeze has no value to lower, and RA's length stays SYMBOLIC. A
//     relaxed decline would emit a rank-1 [1] value that Squeeze lowers to the scalar 2000,
//     concretizing RA to length 2000 -- so asserting RA stays symbolic discriminates the correct
//     decline from the rank-fabricating bug.
//
//   Branch B (Unsqueeze single value -> output RB): S -> Gather([-1]) -> Unsqueeze([0]) -> Squeeze ->
//     Range(0, K, 1). Gather([-1]) is a rank-1 single-element value (the last dimension of X, 2000);
//     unsqueezing it would yield a rank >= 2 result ([1, 2000]) the single-value channel cannot
//     represent, so Unsqueeze's custom propagation must DECLINE. With the correct decline nothing
//     propagates past Unsqueeze, RB's length stays SYMBOLIC, and the model loads. A relaxed decline
//     would fabricate [1, 2000], which Squeeze lowers into a non-scalar Range limit that ONNX Range
//     shape inference rejects ("Input to 'Range' op should be scalars"), so Graph::Resolve (and
//     therefore Model::Load) FAILS. So Branch B is observable either as a load failure or, were that
//     to pass, as a concrete RB length.
//
// Both Gather indices are raw graph initializers (not surfaced through the data-propagation
// getInputData channel), so ONNX's own Gather data propagator bails and control reaches our custom
// decline branches. The branches are independent: a regression in either is caught (Branch A as a
// concrete RA dimension, Branch B as a failed load), so this one combined model keeps the full
// discriminating power of the two original per-path fixtures.
//
// The decline is driven through onnxruntime::Model::Load (which runs Graph::Resolve) and the output
// NodeArg shapes are inspected directly -- no InferenceSession, so none of the session's arena /
// execution-provider / kernel-registry allocations are created. Data propagation runs inside
// Graph::Resolve (Graph::InferAndVerifyTypeMatch -> data-propagation RunInferencing); constant
// folding is a separate session-level optimizer pass that never runs here, so Resolve alone
// exercises the decline branches in isolation with no folding to mask them.
TEST(ShapeInferenceV2Test, GatherUnsqueezeDeclineTest) {
  auto model_path = ORT_TSTR("testdata/test_shape_data_propagation_decline_combined.onnx");

  std::shared_ptr<onnxruntime::Model> model;
  // Model::Load runs Graph::Resolve, which performs data propagation. A correct decline on BOTH
  // branches keeps each Range length symbolic, so the resolve (and therefore the load) succeeds. A
  // relaxed Unsqueeze decline (Branch B) makes Range's input non-scalar, which Range shape inference
  // rejects, so Resolve returns non-OK and this assertion fails on the load itself.
  ASSERT_STATUS_OK(onnxruntime::Model::Load(model_path, model, nullptr,
                                            DefaultLoggingManager().DefaultLogger()));

  const auto& graph_outputs = model->MainGraph().GetOutputs();
  ASSERT_EQ(graph_outputs.size(), static_cast<size_t>(2));

  // Both Range outputs are 1-D; each single dimension must stay SYMBOLIC (no concrete dim_value)
  // because its branch declines rather than propagating a (rank-fabricated) concrete K. RA[0] would
  // concretize to 2000 if the rank-2 Gather index stopped declining; RB[0] would concretize to 2000
  // (or the load would fail) if the single-value Unsqueeze stopped declining.
  for (const NodeArg* output : graph_outputs) {
    const TensorShapeProto* output_shape = output->Shape();
    ASSERT_NE(output_shape, nullptr) << "Range output '" << output->Name()
                                     << "' should have an inferred (rank-1) shape";
    ASSERT_EQ(output_shape->dim_size(), 1) << "Range output '" << output->Name()
                                           << "' should be a 1-D tensor";
    EXPECT_FALSE(output_shape->dim(0).has_dim_value())
        << "Range length '" << output->Name() << "' must stay symbolic; the decline must not "
                                                 "fabricate a concrete K (a relaxed decline concretizes it to 2000)";
  }
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

}  // namespace test
}  // namespace onnxruntime
