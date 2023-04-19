// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "core/framework/session_state.h"
#include "core/providers/cpu/controlflow/if.h"
#include "test/providers/provider_test_utils.h"
#include "core/session/inference_session.h"

#include "test/util/include/default_providers.h"

/*
 * The tests validate that if a fusion occures the expected output matches
 * the output of each graph if they had not be done separatly.
 *
 * Unfortantly there is no hook to actually check that the fussion occured
 * other than inspecting debug logs.
 *
 * The 8 tests use patterns that we have seen in actual models durring testing.
 * Other tests validate that non-associative ops work as expected. We are able
 * to fuse the output of matmul divided by another value but we can not fuse
 * the a value divided by the output of matmul. Similar with Subtraction.
 *
 * A few tests are there simply to validate the limits of the MatMul + post op
 * fusion. The max number of ops fusable are 32 post ops so we exced that number
 * and make sure the generated fusion is not a broken graph.
 *
 * A current implementation limitation is that we can only support a single instance
 * ops that use the 'alpha' attribute. We purposly test models that have more than
 * one instance of LeakRelu or Elu to make sure the graph generated is not broken.
 *
 * Most numbers for the tests were randomly generated and calculated using
 * python numpy library.
 *
 *  // fusions seen in most bert models
 *  matmul_add
 *  matmul_add_add
 *  // fusions seen in mobilebert
 *  matmul_add_add_mul_add
 *  matmul_add_relu
 *  matmul_add_mul_add_add_mul_add
 *  // fusions seen in bertsquade
 *  matmul_mul
 *  matmul_mul_add
 *  // fusions seen in bidif
 *  matmul_add_sigmoid_mul_add
 *
 *  // testing other possible combinations that are not seen in models
 *  // Non-associative ops
 *  // not all layouts can be fused
 *  matmul_div_add_0
 *  matmul_div_add_1
 *  matmul_div_sub_0
 *  matmul_div_sub_1
 *
 *  // Max number of post ops supported by OneDNN is 32.
 *  // Test that the post-op fusion does not fail when that value is exceded
 *  matmul_36_post_ops
 *
 *  // test fusion of remaining eltwise ops
 *  matmul_add_abs_mul
 *  matmul_add_exp_mul
 *  matmul_add_abs_log_mul
 *  matmul_add_round_mul
 *  matmul_add_softplus_mul
 *  matmul_add_abs_sqrt_mul
 *  matmul_add_tanh_mul
 *
 *  //element wise functions that take alpha attribute
 *  matmul_add_leakyrelu_mul
 *  matmul_add_leakyrelu_mul_leakyrelu
 *  matmul_add_elu_mul_leakyrelu
 */

namespace onnxruntime {
namespace test {
// Although these tests should not fail when run on other EPs there
// is not much gained by running these on other EPs
#ifdef USE_DNNL
class Dnnl_MatMul_Add_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMul_Add_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 3u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a = graph_input_defs[2];
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a}, {y});
  }
};

TEST(DnnlMatMulFusion, MatMul_Add) {
  Dnnl_MatMul_Add_PostOpTester test;
  test.AddInput<float>("x", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("w", {3, 2}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f});
  test.AddInput<float>("a", {2}, {0.2f, 0.5f});
  test.AddOutput<float>("y", {2, 2}, {1.2f, -8.5f, 7.2f, -14.5f});

  test.Run();
}

class Dnnl_MatMul_Add_Add_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMul_Add_Add_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* a2 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", a2->TypeAsProto());
    auto& add1_out = graph.GetOrCreateNodeArg("add1_out", a2->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&add1_out});
    graph.AddNode("add2", "Add", "", {&add1_out, a2}, {y});
  }
};

TEST(DnnlMatMulFusion, MatMul_Add_Add) {
  Dnnl_MatMul_Add_Add_PostOpTester test;
  test.AddInput<float>("x", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("w", {3, 2}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f});
  test.AddInput<float>("a1", {2}, {0.2f, 0.5f});
  test.AddInput<float>("a2", {2, 2}, {0.01f, 0.02f, 0.03f, 0.04f});
  test.AddOutput<float>("y", {2, 2}, {1.21f, -8.48f, 7.23f, -14.46f});

  test.Run();
}

class Dnnl_matmul_add_add_mul_add_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_add_mul_add_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 6u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* a2 = graph_input_defs[3];
    NodeArg* m1 = graph_input_defs[4];
    NodeArg* a3 = graph_input_defs[5];

    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& a2_out = graph.GetOrCreateNodeArg("a2_out", y->TypeAsProto());
    auto& m1_out = graph.GetOrCreateNodeArg("m1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("add2", "Add", "", {&a1_out, a2}, {&a2_out});
    graph.AddNode("mul1", "Mul", "", {&a2_out, m1}, {&m1_out});
    graph.AddNode("add3", "Add", "", {&m1_out, a3}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_add_add_mul_add) {
  Dnnl_matmul_add_add_mul_add_PostOpTester test;
  test.AddInput<float>("x", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("w", {3, 2}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f});
  test.AddInput<float>("a1", {2}, {0.2f, 0.5f});
  test.AddInput<float>("a2", {2, 2}, {0.01f, 0.02f, 0.03f, 0.04f});
  test.AddInput<float>("m1", {2, 2}, {1.4f, -3.5f, 2.001f, -4.0f});
  test.AddInput<float>("a3", {2, 2}, {-0.001f, 0.002f, -0.003f, 0.004f});

  test.AddOutput<float>("y", {2, 2}, {1.693f, 29.682f, 14.46423f, 57.844f});

  test.Run();
}

class Dnnl_matmul_add_relu_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_relu_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 3u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("relu1", "Relu", "", {&a1_out}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_add_relu) {
  Dnnl_matmul_add_relu_PostOpTester test;
  test.AddInput<float>("x", {2, 3}, {1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f});
  test.AddInput<float>("w", {3, 2}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f});
  test.AddInput<float>("a1", {2}, {0.2f, 0.5f});

  test.AddOutput<float>("y", {2, 2}, {1.2f, 0.0f, 0.0f, 15.5f});

  test.Run();
}

class Dnnl_matmul_add_mul_add_add_mul_add_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_mul_add_add_mul_add_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 8u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* m1 = graph_input_defs[3];
    NodeArg* a2 = graph_input_defs[4];
    NodeArg* a3 = graph_input_defs[5];
    NodeArg* m2 = graph_input_defs[6];
    NodeArg* a4 = graph_input_defs[7];

    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& m1_out = graph.GetOrCreateNodeArg("m1_out", y->TypeAsProto());
    auto& a2_out = graph.GetOrCreateNodeArg("a2_out", y->TypeAsProto());
    auto& a3_out = graph.GetOrCreateNodeArg("a3_out", y->TypeAsProto());
    auto& m2_out = graph.GetOrCreateNodeArg("m2_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("mul1", "Mul", "", {&a1_out, m1}, {&m1_out});
    graph.AddNode("add2", "Add", "", {&m1_out, a2}, {&a2_out});
    graph.AddNode("add3", "Add", "", {&a2_out, a3}, {&a3_out});
    graph.AddNode("mul2", "Mul", "", {&a3_out, m2}, {&m2_out});
    graph.AddNode("add4", "Add", "", {&m2_out, a4}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_add_mul_add_add_mul_add) {
  Dnnl_matmul_add_mul_add_add_mul_add_PostOpTester test;
  test.AddInput<float>("x", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<float>("w", {3, 2}, {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f});
  test.AddInput<float>("a1", {2}, {0.2f, 0.5f});
  test.AddInput<float>("m1", {2, 2}, {1.4f, -3.5f, 2.001f, -4.0f});
  test.AddInput<float>("a2", {2, 2}, {0.01f, 0.02f, 0.03f, 0.04f});
  test.AddInput<float>("a3", {2, 2}, {-0.001f, 0.002f, -0.003f, 0.004f});
  test.AddInput<float>("m2", {2}, {-0.8208212f, 0.33592367f});
  test.AddInput<float>("a4", {2, 2}, {0.7998259f, 0.12788436f, 0.7284704f, -0.2239327f});

  test.AddOutput<float>("y", {2, 2}, {-0.5865412f, 10.129004f, -11.119426f, 19.274422f});

  test.Run();
}

class Dnnl_matmul_mul_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_mul_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 3u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* m1 = graph_input_defs[2];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgss
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("mul1", "Mul", "", {&matmul_out, m1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_mul) {
  Dnnl_matmul_mul_PostOpTester test;
  test.AddInput<float>("x", {2, 3}, {-3.938475f, 6.618832f, 3.595865f, -1.7349255f, 8.948036f, 0.1722578f});
  test.AddInput<float>("w", {3, 2}, {2.6024234f, -9.086209f, -4.964996f, -7.153241f, -4.9535093f, -3.7669854f});
  test.AddInput<float>("m1", {2}, {-0.7146569f, 0.055316295f});

  test.AddOutput<float>("y", {2, 2}, {43.5399f, -1.3887634f, 35.58652f, -2.7045496f});

  test.Run();
}

class Dnnl_matmul_mul_add_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_mul_add_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* m1 = graph_input_defs[2];
    NodeArg* a1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& m1_out = graph.GetOrCreateNodeArg("m1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("mul1", "Mul", "", {&matmul_out, m1}, {&m1_out});
    graph.AddNode("add1", "Add", "", {&m1_out, a1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_mul_add) {
  Dnnl_matmul_mul_add_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {8.342882f, -3.7048159f, -6.253444f, 2.4014788f, -6.429128f, -7.743094f});
  test.AddInput<float>("w", {3, 2}, {3.35541f, 8.924917f, 9.76079f, -4.250105f, -4.5255866f, 9.858987f});
  test.AddInput<float>("m1", {2}, {0.06467651f, -0.779639f});
  test.AddInput<float>("a1", {2, 2}, {0.8491748f, 0.67852724f, 0.3233882f, -0.21742907f});

  test.AddOutput<float>("y", {2, 2}, {2.1512659f, -21.582323f, -0.94772387f, 21.286257f});

  test.Run();
}

class Dnnl_matmul_add_sigmoid_mul_add_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_sigmoid_mul_add_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 5u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* m1 = graph_input_defs[3];
    NodeArg* a2 = graph_input_defs[4];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& sigmoid1_out = graph.GetOrCreateNodeArg("sigmoid1_out", y->TypeAsProto());
    auto& m1_out = graph.GetOrCreateNodeArg("m1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("sigmoid1", "Sigmoid", "", {&a1_out}, {&sigmoid1_out});
    graph.AddNode("mul1", "Mul", "", {&sigmoid1_out, m1}, {&m1_out});
    graph.AddNode("add2", "Add", "", {&m1_out, a2}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_add_sigmoid_mul_add) {
  Dnnl_matmul_add_sigmoid_mul_add_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {9.6508f, -0.20282122f, 4.2003503f, 2.656795f, 9.703601f, -5.107499f});
  test.AddInput<float>("w", {3, 2}, {3.0818954f, -6.0421305f, 0.13558745f, -5.1593084f, 0.54915303f, -5.227745f});
  test.AddInput<float>("a1", {2, 2}, {-0.15068504f, -0.005190207f, 0.2361607f, 0.6988064f});
  test.AddInput<float>("m1", {2}, {-0.9981574f, 0.5388084f});
  test.AddInput<float>("a2", {1, 2}, {-0.5595895f, -0.0069904923f});

  test.AddOutput<float>("y", {2, 2}, {-1.5577469f, -0.0069904923f, -1.5567765f, -0.0069904923f});

  test.Run();
}

class Dnnl_matmul_add_relu_mul_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_relu_mul_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* m1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& relu1_out = graph.GetOrCreateNodeArg("relu1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("relu1", "Relu", "", {&a1_out}, {&relu1_out});
    graph.AddNode("mul1", "Mul", "", {&relu1_out, m1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_add_relu_mul) {
  Dnnl_matmul_add_relu_mul_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {1.1126449f, 6.8456435f, -4.495065f, 1.8816451f, 9.040679f, -0.21254322f});
  test.AddInput<float>("w", {3, 2}, {-2.3927493f, 6.8880315f, -4.3973136f, 5.9552994f, 3.502647f, -1.5567348f});
  test.AddInput<float>("a1", {2}, {-6.0162272f, 7.687855f});
  test.AddInput<float>("m1", {2}, {-1.0859109f, 6.10515f});
  test.AddOutput<float>("y", {2, 2}, {-0.0f, 385.3404f, -0.0f, 456.7843f});

  test.Run();
}

class Dnnl_matmul_div_add_0_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_div_add_0_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* d1 = graph_input_defs[2];
    NodeArg* a1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& d1_out = graph.GetOrCreateNodeArg("d1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("div1", "Div", "", {&matmul_out, d1}, {&d1_out});
    graph.AddNode("add1", "Add", "", {&d1_out, a1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_div_add_0) {
  Dnnl_matmul_div_add_0_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {4.369459f, -6.2142677f, 4.656685f, -5.3218303f, 6.780895f, -6.7362905f});
  test.AddInput<float>("w", {3, 2}, {-1.0897748f, 1.759465f, 8.868668f, -0.30992413f, -6.19089f, 0.742947f});
  test.AddInput<float>("d1", {2}, {-0.027203506f, -1.4872944f});
  test.AddInput<float>("a1", {2, 2}, {-0.23250997f, 1.9224997f, -0.3070893f, 0.14720583f});

  test.AddOutput<float>("y", {2, 2}, {3260.488f, -6.8676443f, -3957.177f, 11.2209015f});

  test.Run();
}

// We can do (matmul / div) as a post op but (div / matmul) can not be done as a post op
// this will verify that the fusion code done not identify this as a possible post op
class Dnnl_matmul_div_add_1_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_div_add_1_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* d1 = graph_input_defs[2];
    NodeArg* a1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& d1_out = graph.GetOrCreateNodeArg("d1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("div1", "Div", "", {d1, &matmul_out}, {&d1_out});
    graph.AddNode("add1", "Add", "", {&d1_out, a1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_div_add_1) {
  Dnnl_matmul_div_add_1_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {-4.740553f, 3.725597f, -2.9754384f, 3.3195355f, -1.228481f, -5.1616836f});
  test.AddInput<float>("w", {3, 2}, {8.33237f, -1.3398589f, -0.95578283f, 3.7844374f, -2.2136214f, 0.11026443f});
  test.AddInput<float>("d1", {2}, {0.14418359f, 0.4324585f});
  test.AddInput<float>("a1", {2, 2}, {0.6258448f, -0.86796135f, -0.18080634f, -0.70554227f});

  test.AddOutput<float>("y", {2, 2}, {0.6218918f, -0.8464705f, -0.17722501f, -0.7502826f});

  test.Run();
}

// this will produce a single matmul->div->sub op.
class Dnnl_matmul_div_sub_0_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_div_sub_0_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* d1 = graph_input_defs[2];
    NodeArg* s1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& d1_out = graph.GetOrCreateNodeArg("d1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("div1", "Div", "", {&matmul_out, d1}, {&d1_out});
    graph.AddNode("sub1", "Sub", "", {&d1_out, s1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_div_sub_0) {
  Dnnl_matmul_div_sub_0_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {-2.1785734f, -0.6487803f, 2.3332274f, -8.685032f, 5.256502f, -5.2456903f});
  test.AddInput<float>("w", {3, 2}, {-5.553054f, 2.9051464f, 8.103106f, 8.064656f, 5.6170983f, -4.5699472f});
  test.AddInput<float>("d1", {2}, {-0.06625523f, -0.49858263f});
  test.AddInput<float>("s1", {2, 2}, {1.7707943f, -0.67333674f, 1.680975f, 1.5092063f});

  test.AddOutput<float>("y", {2, 2}, {-302.8273f, 45.247673f, -927.7496f, -84.009315f});

  test.Run();
}

// this will produce a single matmul->div op. The sub should not be supported
class Dnnl_matmul_div_sub_1_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_div_sub_1_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* d1 = graph_input_defs[2];
    NodeArg* s1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& d1_out = graph.GetOrCreateNodeArg("d1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("div1", "Div", "", {&matmul_out, d1}, {&d1_out});
    graph.AddNode("sub1", "Sub", "", {s1, &d1_out}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_div_sub_1) {
  Dnnl_matmul_div_sub_1_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {-7.7071023f, -9.326053f, 5.493533f, -0.52132344f, -4.76854f, -7.6051574f});
  test.AddInput<float>("w", {3, 2}, {-2.5150526f, -1.3449488f, -9.300374f, 8.302707f, -2.1621704f, -6.8621817f});
  test.AddInput<float>("d1", {2}, {1.4335649f, -1.431184f});
  test.AddInput<float>("s1", {2, 2}, {-1.4626782f, -1.5727872f, 1.8534334f, 0.22876069f});

  test.AddOutput<float>("y", {2, 2}, {-67.202f, -74.77332f, -41.467945f, 9.519903f});

  test.Run();
}

// The matmul post op fusion is limited to 32 post ops.
// This test will put a string of 36 ops that could be placed
// in the matmul post op fusion to check that the 32 post op
// limit is not exceded.
// to do this we just run the matmul->[add->mul->sub-div] 9 times
// input params are shared accross multiple ops
class Dnnl_matmul_36_post_ops_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_36_post_ops_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 6u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* m1 = graph_input_defs[3];
    NodeArg* s1 = graph_input_defs[4];
    NodeArg* d1 = graph_input_defs[5];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& m1_out = graph.GetOrCreateNodeArg("m1_out", y->TypeAsProto());
    auto& s1_out = graph.GetOrCreateNodeArg("s1_out", y->TypeAsProto());
    auto& d1_out = graph.GetOrCreateNodeArg("d1_out", y->TypeAsProto());
    auto& a2_out = graph.GetOrCreateNodeArg("a2_out", y->TypeAsProto());
    auto& m2_out = graph.GetOrCreateNodeArg("m2_out", y->TypeAsProto());
    auto& s2_out = graph.GetOrCreateNodeArg("s2_out", y->TypeAsProto());
    auto& d2_out = graph.GetOrCreateNodeArg("d2_out", y->TypeAsProto());
    auto& a3_out = graph.GetOrCreateNodeArg("a3_out", y->TypeAsProto());
    auto& m3_out = graph.GetOrCreateNodeArg("m3_out", y->TypeAsProto());
    auto& s3_out = graph.GetOrCreateNodeArg("s3_out", y->TypeAsProto());
    auto& d3_out = graph.GetOrCreateNodeArg("d3_out", y->TypeAsProto());
    auto& a4_out = graph.GetOrCreateNodeArg("a4_out", y->TypeAsProto());
    auto& m4_out = graph.GetOrCreateNodeArg("m4_out", y->TypeAsProto());
    auto& s4_out = graph.GetOrCreateNodeArg("s4_out", y->TypeAsProto());
    auto& d4_out = graph.GetOrCreateNodeArg("d4_out", y->TypeAsProto());
    auto& a5_out = graph.GetOrCreateNodeArg("a5_out", y->TypeAsProto());
    auto& m5_out = graph.GetOrCreateNodeArg("m5_out", y->TypeAsProto());
    auto& s5_out = graph.GetOrCreateNodeArg("s5_out", y->TypeAsProto());
    auto& d5_out = graph.GetOrCreateNodeArg("d5_out", y->TypeAsProto());
    auto& a6_out = graph.GetOrCreateNodeArg("a6_out", y->TypeAsProto());
    auto& m6_out = graph.GetOrCreateNodeArg("m6_out", y->TypeAsProto());
    auto& s6_out = graph.GetOrCreateNodeArg("s6_out", y->TypeAsProto());
    auto& d6_out = graph.GetOrCreateNodeArg("d6_out", y->TypeAsProto());
    auto& a7_out = graph.GetOrCreateNodeArg("a7_out", y->TypeAsProto());
    auto& m7_out = graph.GetOrCreateNodeArg("m7_out", y->TypeAsProto());
    auto& s7_out = graph.GetOrCreateNodeArg("s7_out", y->TypeAsProto());
    auto& d7_out = graph.GetOrCreateNodeArg("d7_out", y->TypeAsProto());
    auto& a8_out = graph.GetOrCreateNodeArg("a8_out", y->TypeAsProto());
    auto& m8_out = graph.GetOrCreateNodeArg("m8_out", y->TypeAsProto());
    auto& s8_out = graph.GetOrCreateNodeArg("s8_out", y->TypeAsProto());
    auto& d8_out = graph.GetOrCreateNodeArg("d8_out", y->TypeAsProto());
    auto& a9_out = graph.GetOrCreateNodeArg("a9_out", y->TypeAsProto());
    auto& m9_out = graph.GetOrCreateNodeArg("m9_out", y->TypeAsProto());
    auto& s9_out = graph.GetOrCreateNodeArg("s9_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("mul1", "Mul", "", {&a1_out, m1}, {&m1_out});
    graph.AddNode("sub1", "Sub", "", {&m1_out, s1}, {&s1_out});
    graph.AddNode("div1", "Div", "", {&s1_out, d1}, {&d1_out});

    graph.AddNode("add2", "Add", "", {&d1_out, a1}, {&a2_out});
    graph.AddNode("mul2", "Mul", "", {&a2_out, m1}, {&m2_out});
    graph.AddNode("sub2", "Sub", "", {&m2_out, s1}, {&s2_out});
    graph.AddNode("div2", "Div", "", {&s2_out, d1}, {&d2_out});

    graph.AddNode("add3", "Add", "", {&d2_out, a1}, {&a3_out});
    graph.AddNode("mul3", "Mul", "", {&a3_out, m1}, {&m3_out});
    graph.AddNode("sub3", "Sub", "", {&m3_out, s1}, {&s3_out});
    graph.AddNode("div3", "Div", "", {&s3_out, d1}, {&d3_out});

    graph.AddNode("add4", "Add", "", {&d3_out, a1}, {&a4_out});
    graph.AddNode("mul4", "Mul", "", {&a4_out, m1}, {&m4_out});
    graph.AddNode("sub4", "Sub", "", {&m4_out, s1}, {&s4_out});
    graph.AddNode("div4", "Div", "", {&s4_out, d1}, {&d4_out});

    graph.AddNode("add5", "Add", "", {&d4_out, a1}, {&a5_out});
    graph.AddNode("mul5", "Mul", "", {&a5_out, m1}, {&m5_out});
    graph.AddNode("sub5", "Sub", "", {&m5_out, s1}, {&s5_out});
    graph.AddNode("div5", "Div", "", {&s5_out, d1}, {&d5_out});

    graph.AddNode("add6", "Add", "", {&d5_out, a1}, {&a6_out});
    graph.AddNode("mul6", "Mul", "", {&a6_out, m1}, {&m6_out});
    graph.AddNode("sub6", "Sub", "", {&m6_out, s1}, {&s6_out});
    graph.AddNode("div6", "Div", "", {&s6_out, d1}, {&d6_out});

    graph.AddNode("add7", "Add", "", {&d6_out, a1}, {&a7_out});
    graph.AddNode("mul7", "Mul", "", {&a7_out, m1}, {&m7_out});
    graph.AddNode("sub7", "Sub", "", {&m7_out, s1}, {&s7_out});
    graph.AddNode("div7", "Div", "", {&s7_out, d1}, {&d7_out});

    graph.AddNode("add8", "Add", "", {&d7_out, a1}, {&a8_out});
    graph.AddNode("mul8", "Mul", "", {&a8_out, m1}, {&m8_out});
    graph.AddNode("sub8", "Sub", "", {&m8_out, s1}, {&s8_out});
    graph.AddNode("div8", "Div", "", {&s8_out, d1}, {&d8_out});

    graph.AddNode("add9", "Add", "", {&d8_out, a1}, {&a9_out});
    graph.AddNode("mul9", "Mul", "", {&a9_out, m1}, {&m9_out});
    graph.AddNode("sub9", "Sub", "", {&m9_out, s1}, {&s9_out});
    graph.AddNode("div9", "Div", "", {&s9_out, d1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_36_post_ops) {
  {
    Dnnl_matmul_36_post_ops_PostOpTester test;

    test.AddInput<float>("x", {2, 3}, {0.52502286f, -0.037000213f, 1.9962887f, -0.6692533f, 0.20213701f, 0.99108446f});
    test.AddInput<float>("w", {3, 2}, {0.2297944f, 0.15045366f, -1.1718011f, 1.4384725f, -0.2647171f, -0.6923775f});
    test.AddInput<float>("a1", {2, 2}, {0.11038251f, -1.5043938f, 0.56713784f, 0.22446638f});
    test.AddInput<float>("m1", {2, 2}, {1.6751015f, -0.81349576f, 0.6223372f, 0.87491375f});
    test.AddInput<float>("s1", {2, 2}, {1.4681175f, 1.3843504f, -1.967671f, 1.1759177f});
    test.AddInput<float>("d1", {2, 2}, {1.9126604f, 1.2817254f, 0.6508832f, -1.8487604f});

    test.AddOutput<float>("y", {2, 2}, {-3.8747377f, -0.055229627f, 26.562538f, 0.36065403f});

    test.Run();
  }
  {
    Dnnl_matmul_36_post_ops_PostOpTester test;

    test.AddInput<float>("x", {2, 3}, {0.3476627f, -0.21223389f, -0.9447632f, -1.8456933f, 0.4254643f, -0.6302209f});
    test.AddInput<float>("w", {3, 2}, {1.5235442f, 1.3257335f, 0.3717919f, -0.7343129f, -0.6779793f, -0.40215403f});
    test.AddInput<float>("a1", {2, 2}, {1.0222285f, -1.1521666f, 1.6293323f, 1.1395578f});
    test.AddInput<float>("m1", {2, 2}, {0.44675776f, 1.0777473f, 0.8223334f, 0.07545234f});
    test.AddInput<float>("s1", {2, 2}, {-0.016137114f, -1.442459f, 1.3463489f, 1.5103843f});
    test.AddInput<float>("d1", {2, 2}, {-0.40666148f, -0.89990497f, 0.35486937f, -0.86016685f});

    test.AddOutput<float>("y", {2, 2}, {-4.389406f, -5.66769f, -4316.6245f, 1.5224165f});

    test.Run();
  }
}

// verfy the Abs ops is being added to the post ops.
class Dnnl_matmul_add_abs_mul_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_abs_mul_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* m1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& abs1_out = graph.GetOrCreateNodeArg("abs1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("abs1", "Abs", "", {&a1_out}, {&abs1_out});
    graph.AddNode("mul1", "Mul", "", {&abs1_out, m1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_add_abs_mul) {
  Dnnl_matmul_add_abs_mul_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {-0.6285251f, -1.7329292f, -0.7914873f, 1.1629477f, 0.12759529f, -0.39523655f});
  test.AddInput<float>("w", {3, 2}, {1.7596923f, -1.5206126f, 1.3103367f, -0.76566243f, -0.26033303f, 0.60094094f});
  test.AddInput<float>("a1", {2}, {1.7229141f, 6.997456f});
  test.AddInput<float>("m1", {2, 2}, {-9.631676f, 2.2766902f, -4.828093f, -9.138716f});

  test.AddOutput<float>("y", {2, 2}, {-13.944424f, 20.044895f, -19.502745f, -44.723545f});

  test.Run();
}

// verfy the Exp ops is being added to the post ops.
class Dnnl_matmul_add_exp_mul_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_exp_mul_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* m1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& exp1_out = graph.GetOrCreateNodeArg("exp1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("exp1", "Exp", "", {&a1_out}, {&exp1_out});
    graph.AddNode("mul1", "Mul", "", {&exp1_out, m1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_add_exp_mul) {
  Dnnl_matmul_add_exp_mul_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {-0.45230612f, 0.62779653f, -0.11451248f, -0.735042f, 0.721971f, -0.38090658f});
  test.AddInput<float>("w", {3, 2}, {0.7029298f, -0.5394501f, 0.99953383f, 0.2477564f, -0.84789735f, -0.40307873f});
  test.AddInput<float>("a1", {2}, {-9.132685f, 1.5531663f});
  test.AddInput<float>("m1", {2, 2}, {4.269247f, -5.7232485f, 2.173664f, 2.670998f});

  test.AddOutput<float>("y", {2, 2}, {0.0006929254f, -42.24127f, 0.0003982826f, 26.16821f});

  test.Run();
}

// verfy the Log op is being added to the post ops. Abs is called befor log to prevent NaN.
class Dnnl_matmul_add_abs_log_mul_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_abs_log_mul_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* m1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& abs1_out = graph.GetOrCreateNodeArg("abs1_out", y->TypeAsProto());
    auto& log1_out = graph.GetOrCreateNodeArg("log1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("abs1", "Abs", "", {&a1_out}, {&abs1_out});
    graph.AddNode("log1", "Log", "", {&abs1_out}, {&log1_out});
    graph.AddNode("mul1", "Mul", "", {&log1_out, m1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_add_abs_log_mul) {
  Dnnl_matmul_add_abs_log_mul_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {0.1732719f, 1.011866f, 0.34941992f, 1.412382f, -1.1263472f, 0.2608548f});
  test.AddInput<float>("a", {3, 2}, {-1.7700142f, -1.6810288f, -1.1070849f, -0.51743853f, -0.76337415f, -1.3527646f});
  test.AddInput<float>("a1", {2}, {2.0407753f, 7.3168464f});
  test.AddInput<float>("m1", {2, 2}, {-9.335384f, 0.38610986f, -6.241185f, 6.3155437f});

  test.AddOutput<float>("y", {2, 2}, {9.877575f, 0.6936976f, 3.3071246f, 10.378726f});

  test.Run();
}

// verfy the Round op is being added to the post ops.
class Dnnl_matmul_add_round_mul_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_round_mul_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* m1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& round1_out = graph.GetOrCreateNodeArg("round1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("round1", "Log", "", {&a1_out}, {&round1_out});
    graph.AddNode("mul1", "Mul", "", {&round1_out, m1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_add_round_mul) {
  Dnnl_matmul_add_round_mul_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {-1.9386154f, -0.2794915f, -0.4832877f, -0.622788f, -0.9714142f, 1.4512274f});
  test.AddInput<float>("w", {3, 2}, {0.81588864f, 1.1876478f, -1.0020647f, 0.5448352f, -0.7245603f, -0.17671005f});
  test.AddInput<float>("a1", {2}, {5.420931f, 5.271474f});
  test.AddInput<float>("m1", {2, 2}, {-9.678547f, 1.9440855f, 1.6208986f, 9.2071495f});

  test.AddOutput<float>("y", {2, 2}, {-14.491409f, 2.0713673f, 2.5542507f, 12.160057f});

  test.Run();
}

// verfy the Softplus op is being added to the post ops.
class Dnnl_matmul_add_softplus_mul_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_softplus_mul_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* m1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& softplus1_out = graph.GetOrCreateNodeArg("softplus1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("softplus1", "Softplus", "", {&a1_out}, {&softplus1_out});
    graph.AddNode("mul1", "Mul", "", {&softplus1_out, m1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_add_softplus_mul) {
  Dnnl_matmul_add_softplus_mul_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {-0.048156887f, 0.19971251f, -1.4158537f, -0.026562484f, 1.5599211f, -1.1806089f});
  test.AddInput<float>("w", {3, 2}, {-0.5598413f, -1.0820478f, -1.6692561f, -1.7626349f, 0.72114486f, 1.2662032f});
  test.AddInput<float>("a1", {2}, {5.058426f, -8.577537f});
  test.AddInput<float>("m1", {2, 2}, {9.214308f, -8.086882f, -7.9873796f, -1.5645525f});

  test.AddOutput<float>("y", {2, 2}, {34.596645f, -0.00018798397f, -14.3684845f, -4.289706e-06f});

  test.Run();
}

// verfy the Sqrt op is being added to the post ops add Abs to prevent Sqrt of negative numbers.
class Dnnl_matmul_add_abs_sqrt_mul_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_abs_sqrt_mul_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* m1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& abs1_out = graph.GetOrCreateNodeArg("abs1_out", y->TypeAsProto());
    auto& sqrt1_out = graph.GetOrCreateNodeArg("sqrt1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("abs1", "Abs", "", {&a1_out}, {&abs1_out});
    graph.AddNode("sqrt1", "Sqrt", "", {&abs1_out}, {&sqrt1_out});
    graph.AddNode("mul1", "Mul", "", {&sqrt1_out, m1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_add_abs_sqrt_mul) {
  Dnnl_matmul_add_abs_sqrt_mul_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {0.51311743f, -0.41487834f, -1.6790653f, 1.178804f, 0.0047367346f, 0.56704f});
  test.AddInput<float>("w", {3, 2}, {0.1990868f, 0.6381897f, -1.085156f, 1.3352108f, 1.3156614f, 0.43205962f});
  test.AddInput<float>("a1", {2}, {3.0645058f, -4.0001354f});
  test.AddInput<float>("m1", {2, 2}, {-0.81868356f, -2.9432542f, -1.7674304f, 8.759958f});

  test.AddOutput<float>("y", {2, 2}, {-0.9713697f, -6.5497f, -3.5525277f, 15.1638775f});

  test.Run();
}

// verfy the Tanh op is being added to the post ops.
class Dnnl_matmul_add_tanh_mul_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_tanh_mul_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* m1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& tanh1_out = graph.GetOrCreateNodeArg("tanh1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("tanh1", "Tanh", "", {&a1_out}, {&tanh1_out});
    graph.AddNode("mul1", "Mul", "", {&tanh1_out, m1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_add_tanh_mul) {
  Dnnl_matmul_add_tanh_mul_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {1.3596787f, -1.0537782f, -0.33225316f, 1.1453571f, -0.019471083f, 1.7967123f});
  test.AddInput<float>("w", {3, 2}, {0.2384665f, 1.3449599f, -1.7269896f, -1.2526665f, -1.6400307f, -1.128408f});
  test.AddInput<float>("a1", {2}, {-9.934079f, 9.043955f});
  test.AddInput<float>("m1", {2, 2}, {-7.4799314f, 2.6152263f, -2.160944f, -6.7666516f});

  test.AddOutput<float>("y", {2, 2}, {7.4799237f, 2.6152263f, 2.160944f, -6.766651f});

  test.Run();
}

// verfy the LeakyRelu op is being added to the post ops.
class Dnnl_matmul_add_leakyrelu_mul_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_leakyrelu_mul_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* m1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& leakyrelu1_out = graph.GetOrCreateNodeArg("leakyrelu1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("leakyrelu1", "LeakyRelu", "", {&a1_out}, {&leakyrelu1_out}).AddAttribute("alpha", 0.01f);
    graph.AddNode("mul1", "Mul", "", {&leakyrelu1_out, m1}, {y});
  }
};

TEST(DnnlMatMulFusion, matmul_add_leakyrelu_mul) {
  Dnnl_matmul_add_leakyrelu_mul_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {0.0693956f, 0.98840743f, 1.6092012f, 1.5831419f, -1.3505223f, -0.27673742f});
  test.AddInput<float>("w", {3, 2}, {1.0791942f, -1.9235526f, -0.76987845f, -1.0750697f, -1.1833277f, 0.6738822f});
  test.AddInput<float>("a1", {2}, {1.8111061f, -9.293419f});
  test.AddInput<float>("m1", {2, 2}, {-5.0943666f, -7.922687f, 3.5371988f, -3.3592103f});

  test.AddOutput<float>("y", {2, 2}, {0.0396937f, 0.7451366f, 17.285698f, 0.37197402f});

  test.Run();
}

// verfy the second LeakyRelu op is NOT being added to the post ops.
// This is testing a limitation in the current implementation that
// we can only fuse one instance of each attribute.
// its possible that possible future work could avoid this limitation
class Dnnl_matmul_add_leakyrelu_mul_leakyrelu_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_leakyrelu_mul_leakyrelu_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* m1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& leakyrelu1_out = graph.GetOrCreateNodeArg("leakyrelu1_out", y->TypeAsProto());
    auto& m1_out = graph.GetOrCreateNodeArg("m1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("leakyrelu1", "LeakyRelu", "", {&a1_out}, {&leakyrelu1_out}).AddAttribute("alpha", 0.01f);
    graph.AddNode("mul1", "Mul", "", {&leakyrelu1_out, m1}, {&m1_out});
    graph.AddNode("leakyrelu2", "LeakyRelu", "", {&m1_out}, {y}).AddAttribute("alpha", 0.1f);
  }
};

TEST(DnnlMatMulFusion, matmul_add_leakyrelu_mul_leakyrelu) {
  Dnnl_matmul_add_leakyrelu_mul_leakyrelu_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {1.765603f, -1.2213482f, -1.1872864f, 0.45118868f, -1.133612f, 1.6618071f});
  test.AddInput<float>("w", {3, 2}, {-1.1658593f, 1.9765378f, -1.241956f, -0.9877922f, 0.38090283f, -1.6604117f});
  test.AddInput<float>("a1", {2}, {6.7648396f, -9.334238f});
  test.AddInput<float>("m1", {2, 2}, {-8.733212f, 6.1143303f, -1.6509179f, -6.8720984f});

  test.AddOutput<float>("y", {2, 2}, {-5.03995f, -0.016304689f, -1.3669106f, 0.692842f});

  test.Run();
}

// verfy the second LeakyRelu op is NOT being added to the post ops.
// This is testing a limitation in the current implementation that
// we can only fuse one instance of each attribute.
// its possible that possible future work could avoid this limitation
class Dnnl_matmul_add_elu_mul_leakyrelu_PostOpTester : public OpTester {
 public:
  explicit Dnnl_matmul_add_elu_mul_leakyrelu_PostOpTester(int opset_version = 7)
      : OpTester("MatMul", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);
    // inputs
    NodeArg* x = graph_input_defs[0];
    NodeArg* w = graph_input_defs[1];
    NodeArg* a1 = graph_input_defs[2];
    NodeArg* m1 = graph_input_defs[3];
    // outputs
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("a1_out", y->TypeAsProto());
    auto& elu1_out = graph.GetOrCreateNodeArg("elu1_out", y->TypeAsProto());
    auto& m1_out = graph.GetOrCreateNodeArg("m1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMul", "", {x, w}, {&matmul_out});
    graph.AddNode("add1", "Add", "", {&matmul_out, a1}, {&a1_out});
    graph.AddNode("elu1", "Elu", "", {&a1_out}, {&elu1_out}).AddAttribute("alpha", 1.5f);
    graph.AddNode("mul1", "Mul", "", {&elu1_out, m1}, {&m1_out});
    graph.AddNode("leakyrelu1", "LeakyRelu", "", {&m1_out}, {y}).AddAttribute("alpha", 0.1f);
  }
};

// verfy the Elu works.
// we can only fuse one instance of each attribute. So the leakyrelu
// will not be fused.
// its possible that possible future work could avoid this limitation
TEST(DnnlMatMulFusion, matmul_add_elu_mul_leakyrelu) {
  Dnnl_matmul_add_elu_mul_leakyrelu_PostOpTester test;

  test.AddInput<float>("x", {2, 3}, {-0.75100183f, 0.4477339f, 0.13396399f, -0.16658644f, 0.38382065f, -0.02262967f});
  test.AddInput<float>("w", {3, 2}, {0.99857223f, -0.8490593f, 0.06453693f, 0.86028147f, 0.7728725f, -0.35186958f});
  test.AddInput<float>("a1", {2}, {-1.2312771f, -3.4552245f});
  test.AddInput<float>("m1", {2, 2}, {4.063479f, 6.9212875f, -3.374929f, 9.077627f});

  test.AddOutput<float>("y", {2, 2}, {-0.5135648f, -0.9512116f, 3.801911f, -1.2921791f});

  test.Run();
}
#endif  // USE_DNNL
}  // namespace test
}  // namespace onnxruntime
