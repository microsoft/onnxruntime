// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <random>
#include "core/providers/ngraph/ngraph_execution_provider.h"
#include "test/providers/provider_test_utils.h"
#include "default_providers.h"
#include "gtest/gtest.h"
#include "core/session/inference_session.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "core/util/math.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

//This is a dummy op that just increments the tensor values by one.
class UnSupportedOp final : public OpKernel {
 public:
  UnSupportedOp(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* p_context) const {
    const auto* X = p_context->Input<Tensor>(0);

    ORT_ENFORCE(X);
    auto X_Data = X->Data<float>();

    auto& shape = X->Shape().GetDims();
    auto* Y = p_context->Output(0, shape);
    auto* Y_Data = Y->MutableData<float>();

    size_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      size *= shape[i];
    }

    for (size_t i = 0; i < size; i++) {
      Y_Data[i] = X_Data[i] + 1;
    }

    return Status::OK();
  }
};

KernelDefBuilder UnSupportedOpDef() {
  KernelDefBuilder def;
  def.SetName("UnSupportedOp")
      .SetDomain(onnxruntime::kOnnxDomain)
      .SinceVersion(7)
      .Provider(onnxruntime::kCpuExecutionProvider)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  return def;
}

ONNX_NAMESPACE::OpSchema GetUnSupportedOpSchema() {
  ONNX_NAMESPACE::OpSchema schema("UnSupportedOp", "unknown", 0);
  schema.Input(0, "A", "Data.", "T");
  schema.Output(0, "C", "Result, has same dimensions and type as A", "T");
  schema.TypeConstraint("T", {"tensor(float)"}, "Constrain input and output types to float tensors.");
  schema.SinceVersion(7);
  return schema;
}

void add_feeds(NameMLValMap& feeds, std::string name, std::vector<int64_t> dims, std::vector<float> value) {
  MLValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims, value, &ml_value);
  feeds.insert(std::make_pair(name, ml_value));
}

//TODO:(nivas) Refractor to use existing code
void RunTest(const std::string& model_path, const NameMLValMap& feeds, const std::vector<std::string>& output_names, const std::vector<std::vector<int64_t>>& expected_shapes, const std::vector<std::vector<float>>& expected_values) {
  SessionOptions so;
  InferenceSession session_object(so, &DefaultLoggingManager());

  EXPECT_TRUE(session_object.RegisterExecutionProvider(DefaultNGraphExecutionProvider()).IsOK());

  std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();
  EXPECT_TRUE(session_object.RegisterCustomRegistry(registry).IsOK());
  auto unsupported_schema = GetUnSupportedOpSchema();
  std::vector<OpSchema> schemas = {unsupported_schema};
  EXPECT_TRUE(registry->RegisterOpSet(schemas, onnxruntime::kOnnxDomain, 7, 8).IsOK());

  auto def = UnSupportedOpDef();
  KernelCreateFn kernel_create_fn = [](const OpKernelInfo& info) -> OpKernel* { return new UnSupportedOp(info); };
  EXPECT_TRUE(registry->RegisterCustomKernel(def, kernel_create_fn).IsOK());

  auto status = session_object.Load(model_path);

  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Load failed with status: " << status.ErrorMessage();
    return;
  }

  status = session_object.Initialize();
  //TODO : Count number of nodes to ensure fusion and also verify that, fused node is assigned to nGraph_EP.
  // Currently there is no api to get above info from here.

  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  RunOptions run_options{};
  run_options.run_tag = "nGraph EP test tag";
  run_options.run_log_verbosity_level = 1;

  std::vector<MLValue> fetches;
  status = session_object.Run(run_options, feeds, output_names, &fetches);

  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Run failed with status: " << status.ErrorMessage();
    return;
  }

  for (size_t idx = 0; idx < expected_values.size(); ++idx) {
    auto& got_tensor = fetches[idx].Get<Tensor>();
    auto* got = got_tensor.Data<float>();
    auto& expected = expected_values[idx];
    TensorShape expected_shape(expected_shapes[idx]);
    ASSERT_EQ(got_tensor.Shape(), expected_shape);
    for (size_t i = 0; i < expected.size(); i++) {
      ASSERT_EQ(got[i], expected[i]);
    }
  }
}

/*
Basic test: To ensure fusion. TODO: Count number of nodes after model initializaton.
     (A)   (A)
       \   /
        Add
    (B)  |
      \  |
        Mul
         |
         |
        (Z)

*/
TEST(NGraphExecutionProviderTest, Basic_Test) {
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 2.0f, 2.0f});

  std::vector<std::vector<float>> expected_values = {
      {4.0f, 8.0f, 12.0f, 16.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/ngraph/Basic_Test.onnx", feeds, {"Z"}, expected_shapes, expected_values);
}

/*
    (A)    (A)
      \    /
        Add
    (B)  |
      \  |
        Mul
         |
         |
    UnSupportedOp
         |
         |
        (Z)
Simple test-case for a graph with UnSupportedOp
*/
TEST(NGraphExecutionProviderTest, Graph_with_UnSupportedOp) {
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 2.0f, 2.0f});

  std::vector<std::vector<float>> expected_values = {
      {5.0f, 9.0f, 13.0f, 17.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/ngraph/Graph_with_UnSupportedOp.onnx", feeds, {"Z"}, expected_shapes, expected_values);
}

/*
    (A)    (A)
      \    /
        Add
    (B)  |
      \  |
        Mul
         |
         |
    UnSupportedOp
         |  (C)
         |  /
        Add
         |
         |
        (Z)
Same as above, here are we have two sub-graphs that are run by nGraph execution provider
*/
TEST(NGraphExecutionProviderTest, Two_Subgraphs) {
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 2.0f, 2.0f});
  add_feeds(feeds, "C", {4}, {1.0f, 1.0f, 1.0f, 1.0f});

  std::vector<std::vector<float>> expected_values = {
      {6.0f, 10.0f, 14.0f, 18.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/ngraph/Two_Subgraphs.onnx", feeds, {"Z"}, expected_shapes, expected_values);
}

/*
    (A)    (A)
      \    /
        Add
    (B)  |
      \  |
        Mul ----------------\
         |                   |
         |                   |
    UnSupportedOp            |
         |  (C)              |
         |  /                |
        Add                  |
         |                   |
         |                   |
        (Z)                 (Y)

Output of the sub-graph is also graph output
*/
TEST(NGraphExecutionProviderTest, ClusterOut_isAlso_GraphOut) {
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 2.0f, 2.0f});
  add_feeds(feeds, "C", {4}, {1.0f, 1.0f, 1.0f, 1.0f});

  std::vector<std::vector<float>> expected_values = {
      {2.0f, 4.0f, 6.0f, 8.0f},
      {6.0f, 10.0f, 14.0f, 18.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4},
      {4}};

  RunTest("testdata/ngraph/ClusterOut_isAlso_GraphOut.onnx", feeds, {"Y", "Z"}, expected_shapes, expected_values);
}

/*
    (A)    (A)
      \    /
        Add
    (B)  |  \
      \  |   \
        Mul   \
         |     \
         |      \
    UnSupportedOp\
    (C)  |        \
      \  |         \
        Add         \
         |           \
         |            \
        [Z]           [Y]

Sub-graph in-out is also graph output
*/
TEST(NGraphExecutionProviderTest, InOut_isAlso_GraphOut) {
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 2.0f, 2.0f});
  add_feeds(feeds, "C", {4}, {1.0f, 1.0f, 1.0f, 1.0f});

  std::vector<std::vector<float>> expected_values = {
      {4.0f, 8.0f, 12.0f, 16.0f},
      {6.0f, 10.0f, 14.0f, 18.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4},
      {4}};

  RunTest("testdata/ngraph/InOut_isAlso_GraphOut.onnx", feeds, {"Y", "Z"}, expected_shapes, expected_values);
}


/*
    (A)    (A)
      \    /
        Add
         |
         |
      Dropout
         |   \ Mask (Unused output)
         |
    UnSupportedOp
    (C)  |
      \  |
        Add
         |
         |
        [Z]
Test-case to esnure ep working in case of unused or optional input that is NOT graph output.
*/
TEST(NGraphExecutionProviderTest, Op_with_Optional_or_Unused_Outputs) {
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "C", {4}, {1.0f, 1.0f, 1.0f, 1.0f});

  std::vector<std::vector<float>> expected_values = {
      {4.0f, 6.0f, 8.0f, 10.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/ngraph/Op_with_Optional_or_Unused_Outputs.onnx", feeds, {"Z"}, expected_shapes, expected_values);
}

/*
           (A)   (A)
             \   /
              Add(1)
        _______|_______
        |             |
        |            Add(2)
        |             |
  UnSupportedOp(4)    |
        |            Add(3)
        |             |
        |_____________|
               |
               |
              Add(5)
               |
               |
              [Z]

This test is to ensure, we do not have cyclic dependent sub-graphs.
Example: Sub-Graph-1{1,2,3,5} is invalid because the output of this cluster is input to UnSupportedOp whose output is again input to the same cluster.

*/
TEST(NGraphExecutionProviderTest, Independent_SubGraphs) {
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});

  std::vector<std::vector<float>> expected_values = {
      {7.0f, 11.0f, 15.0f, 19.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/ngraph/Independent_SubGraphs.onnx", feeds, {"Z"}, expected_shapes, expected_values);
}

}  // namespace test
}  // namespace onnxruntime
