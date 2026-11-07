// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

#include "core/graph/graph.h"
#include "core/graph/onnx_protobuf.h"
#include "core/optimizer/mul_reducesum_matmul_fusion.h"
#include "core/session/inference_session.h"

#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

struct Case {
  std::vector<int64_t> x_shape;
  std::vector<int64_t> v_shape;
  int64_t unsqueeze_axis = -1;
  int64_t reduce_axis = -2;
  int64_t keepdims = 0;
  bool column_first = false;  // Mul(Unsqueeze(v), X) instead of Mul(X, Unsqueeze(v))
  bool product_used = false;  // something else reads the Mul's output
  bool column_used = false;   // something else reads the Unsqueeze's output
};

// Adds `op_type` (Unsqueeze or ReduceSum) of `input` over `axis`, as its opset takes it.
Node& AddAxisNode(ModelTestBuilder& builder, const std::string& op_type, NodeArg* input, NodeArg* output,
                  int64_t axis, int opset) {
  if (opset < 13) {
    Node& node = builder.AddNode(op_type, {input}, {output});
    node.AddAttribute("axes", std::vector<int64_t>{axis});
    return node;
  }
  return builder.AddNode(op_type, {input, builder.Make1DInitializer<int64_t>({axis})}, {output});
}

std::function<void(ModelTestBuilder&)> Build(const Case& c, int opset) {
  return [c, opset](ModelTestBuilder& builder) {
    NodeArg* x = builder.MakeInput<float>(c.x_shape, -1.f, 1.f);
    NodeArg* v = builder.MakeInput<float>(c.v_shape, -1.f, 1.f);
    NodeArg* column = builder.MakeIntermediate();
    NodeArg* product = builder.MakeIntermediate();
    NodeArg* sum = builder.MakeOutput();

    AddAxisNode(builder, "Unsqueeze", v, column, c.unsqueeze_axis, opset);
    if (c.column_first) {
      builder.AddNode("Mul", {column, x}, {product});
    } else {
      builder.AddNode("Mul", {x, column}, {product});
    }
    // ReduceSum takes its axes as an input from opset 13.
    Node& reduce = AddAxisNode(builder, "ReduceSum", product, sum, c.reduce_axis, opset);
    reduce.AddAttribute("keepdims", c.keepdims);

    if (c.product_used) {
      builder.AddNode("Identity", {product}, {builder.MakeOutput()});
    }
    if (c.column_used) {
      builder.AddNode("Identity", {column}, {builder.MakeOutput()});
    }
  };
}

// Runs the case before and after the fusion, compares the outputs, and checks the fused graph's ops.
void Check(const Case& c, bool expect_fused) {
  for (int opset : {12, 13, 18}) {
    SCOPED_TRACE("opset " + std::to_string(opset));
    auto check_graph = [&](InferenceSessionWrapper& session) {
      const auto ops = CountOpsInGraph(session.GetGraph());
      const auto count = [&](const std::string& op) { return ops.count(op) ? ops.at(op) : 0; };
      if (expect_fused) {
        EXPECT_EQ(count("ReduceSum"), 0);
        EXPECT_EQ(count("Mul"), 0);
        EXPECT_EQ(count("MatMul"), 1);
        EXPECT_EQ(count("Squeeze"), c.keepdims ? 0 : 1);
        // The new row Unsqueeze, plus the column one if something else still reads it.
        EXPECT_EQ(count("Unsqueeze"), c.column_used ? 2 : 1);
      } else {
        EXPECT_EQ(count("ReduceSum"), 1);
        EXPECT_EQ(count("Mul"), 1);
        EXPECT_EQ(count("MatMul"), 0);
      }
    };
    TransformerTester(Build(c, opset), check_graph, TransformerLevel::Level1, TransformerLevel::Level2, opset,
                      1e-5, 1e-5, std::make_unique<MulReduceSumToMatMulFusion>());
  }
}

}  // namespace

TEST(MulReduceSumToMatMulFusionTest, Fuses) {
  Check({{2, 3, 4, 5}, {2, 3, 4}}, true);
}

TEST(MulReduceSumToMatMulFusionTest, FusesWithKeepDims) {
  Check({{2, 3, 4, 5}, {2, 3, 4}, -1, -2, 1}, true);
}

TEST(MulReduceSumToMatMulFusionTest, FusesWithColumnFirst) {
  Check({{2, 3, 4, 5}, {2, 3, 4}, -1, -2, 0, true}, true);
}

TEST(MulReduceSumToMatMulFusionTest, FusesWithPositiveAxes) {
  Check({{2, 3, 4, 5}, {2, 3, 4}, 3, 2}, true);
}

TEST(MulReduceSumToMatMulFusionTest, FusesMatrixAndVector) {
  Check({{4, 5}, {4}}, true);
}

TEST(MulReduceSumToMatMulFusionTest, FusesWithBroadcastBatch) {
  // v broadcasts over X's first axis, as Unsqueeze(v, -2) does in the MatMul.
  Check({{2, 3, 4, 5}, {3, 4}}, true);
  // X broadcasts over v's first axis.
  Check({{3, 4, 5}, {2, 3, 4}}, true);
}

TEST(MulReduceSumToMatMulFusionTest, KeepsColumnReadElsewhere) {
  Check({{2, 3, 4, 5}, {2, 3, 4}, -1, -2, 0, false, false, true}, true);
}

TEST(MulReduceSumToMatMulFusionTest, SkipsProductReadElsewhere) {
  Check({{2, 3, 4, 5}, {2, 3, 4}, -1, -2, 0, false, true}, false);
}

TEST(MulReduceSumToMatMulFusionTest, SkipsOtherReduceAxis) {
  // A sum over the last axis is not a vector-matrix product.
  Check({{2, 3, 4, 5}, {2, 3, 4}, -1, -1}, false);
  Check({{2, 3, 4, 5}, {2, 3, 4}, -1, 1}, false);
}

TEST(MulReduceSumToMatMulFusionTest, SkipsBroadcastAlongSummedAxis) {
  // X has 1 where v has 4: the Mul broadcasts X along the summed axis, which a MatMul can't.
  Check({{2, 3, 1, 5}, {2, 3, 4}}, false);
  // v has 1 where X has 4: the Mul broadcasts v along the summed axis.
  Check({{2, 3, 4, 5}, {2, 3, 1}}, false);
}

TEST(MulReduceSumToMatMulFusionTest, SkipsOtherUnsqueezeAxis) {
  Check({{2, 3, 4, 5}, {2, 3, 5}, -2, -2}, false);
}

namespace {

void AddValueInfo(ONNX_NAMESPACE::ValueInfoProto& info, const std::string& name, const std::vector<int64_t>& dims) {
  info.set_name(name);
  auto* tensor_type = info.mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  if (!dims.empty()) {
    auto* shape = tensor_type->mutable_shape();
    for (int64_t dim : dims) {
      shape->add_dim()->set_dim_value(dim);
    }
  }
}

void AddNodeProto(ONNX_NAMESPACE::GraphProto& graph, const std::string& op_type,
                  const std::vector<std::string>& inputs, const std::string& output) {
  auto& node = *graph.add_node();
  node.set_op_type(op_type);
  node.set_name(output);
  for (const auto& input : inputs) {
    node.add_input(input);
  }
  node.add_output(output);
}

}  // namespace

// The fusion applies inside a Scan body, whose inputs take their shapes from the Scan's inputs: here a linear
// attention style recurrence that reads its state S with a key k each step, y = sum_i S[i, :] * k[i].
TEST(MulReduceSumToMatMulFusionTest, FusesInScanBody) {
  constexpr int64_t kSteps = 3, kRows = 4, kCols = 5;

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  ONNX_NAMESPACE::GraphProto body;
  body.set_name("body");
  // No shapes in the body: they come from the Scan.
  AddValueInfo(*body.add_input(), "s", {});
  AddValueInfo(*body.add_input(), "k", {});
  auto* axes = body.add_initializer();
  axes->set_name("minus_one");
  axes->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  axes->add_dims(1);
  axes->add_int64_data(-1);
  auto* reduce_axes = body.add_initializer();
  reduce_axes->set_name("minus_two");
  reduce_axes->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  reduce_axes->add_dims(1);
  reduce_axes->add_int64_data(-2);
  AddNodeProto(body, "Unsqueeze", {"k", "minus_one"}, "k_column");
  AddNodeProto(body, "Mul", {"s", "k_column"}, "s_k");
  AddNodeProto(body, "ReduceSum", {"s_k", "minus_two"}, "y");
  auto* keepdims = body.mutable_node(2)->add_attribute();
  keepdims->set_name("keepdims");
  keepdims->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  keepdims->set_i(0);
  AddNodeProto(body, "Identity", {"s"}, "s_out");
  AddValueInfo(*body.add_output(), "s_out", {});
  AddValueInfo(*body.add_output(), "y", {});

  auto& graph = *model.mutable_graph();
  graph.set_name("main");
  AddValueInfo(*graph.add_input(), "S", {kRows, kCols});
  AddValueInfo(*graph.add_input(), "K", {kSteps, kRows});
  auto& scan = *graph.add_node();
  scan.set_op_type("Scan");
  scan.set_name("scan");
  scan.add_input("S");
  scan.add_input("K");
  scan.add_output("S_final");
  scan.add_output("Y");
  auto* num_scan_inputs = scan.add_attribute();
  num_scan_inputs->set_name("num_scan_inputs");
  num_scan_inputs->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  num_scan_inputs->set_i(1);
  auto* body_attr = scan.add_attribute();
  body_attr->set_name("body");
  body_attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  *body_attr->mutable_g() = body;
  AddValueInfo(*graph.add_output(), "S_final", {kRows, kCols});
  AddValueInfo(*graph.add_output(), "Y", {kSteps, kCols});

  std::string model_bytes;
  ASSERT_TRUE(model.SerializeToString(&model_bytes));

  const std::vector<float> s_values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  const std::vector<float> k_values{1, 0, 0, 0, 0.5f, 0.25f, 0, 0, -1, 2, -3, 4};
  std::vector<float> expected;
  for (int64_t t = 0; t < kSteps; ++t) {
    for (int64_t j = 0; j < kCols; ++j) {
      float sum = 0;
      for (int64_t i = 0; i < kRows; ++i) {
        sum += s_values[i * kCols + j] * k_values[t * kRows + i];
      }
      expected.push_back(sum);
    }
  }

  for (auto level : {TransformerLevel::Level1, TransformerLevel::Level2}) {
    SessionOptions so;
    so.graph_optimization_level = level;
    InferenceSessionWrapper session{so, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
    ASSERT_STATUS_OK(session.Initialize());

    const auto ops = CountOpsInGraph(session.GetGraph(), true);
    const auto count = [&](const std::string& op) { return ops.count(op) ? ops.at(op) : 0; };
    if (level == TransformerLevel::Level2) {
      EXPECT_EQ(count("ReduceSum"), 0);
      EXPECT_EQ(count("MatMul"), 1);
    } else {
      EXPECT_EQ(count("ReduceSum"), 1);
      EXPECT_EQ(count("MatMul"), 0);
    }

    auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
    OrtValue s;
    OrtValue k;
    CreateMLValue<float>(allocator, {kRows, kCols}, s_values, &s);
    CreateMLValue<float>(allocator, {kSteps, kRows}, k_values, &k);
    std::vector<OrtValue> fetches;
    const std::vector<std::string> output_names{"Y"};
    ASSERT_STATUS_OK(session.Run(RunOptions{}, NameMLValMap{{"S", s}, {"K", k}}, output_names, &fetches));
    const auto y = fetches[0].Get<Tensor>().DataAsSpan<float>();
    ASSERT_EQ(y.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(y[i], expected[i], 1e-5f);
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
