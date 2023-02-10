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
* Unfortantly there is no hook to actually check that the fussion occurred
* other than inspecting debug logs.
*
* The tests use patterns that we have seen in actual models during testing.
*
* A few tests are there simply to validate the limits of the MatMulInteger +
* post op fusion. The max number of ops fusable are 32 post ops so we exced
* that number and make sure the generated fusion is not a broken graph.
*
* A current implementation limitation is that we can only support a single instance
* ops that use the 'alpha' attribute. We purposly test models that have more than
* one instance of LeakRelu or Elu to make sure the graph generated is not broken.
*
* Most numbers for the tests were randomly generated and calculated using
* python numpy library.
*
*  // fusions seen in most bert models (bert_base_int8, DistilBert_int8, MobileBert_int8)
*  MatMulInteger_Cast
*  MatMulInteger_Cast_Mul
*  MatMulInteger_Cast_Mul_Add
*  MatMulInteger_Cast_Mul_Add_Add
*  MatMulInteger_Cast_Mul_Add_Mul_Add
*  MatMulInteger_Cast_Mul_Add_Add_Mul_Add
*  MatMulInteger_Cast_Mul_Add_Relu
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
class Dnnl_MatMulInteger_to_float_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_to_float_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];

    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    // The output from MatMulInteger is int32 but this is only an internal Node
    // so we make the int32 TypeProto manually to match the same shape as the
    // output shape.
    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {y});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_to_float) {
  Dnnl_MatMulInteger_to_float_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddOutput<float>("Y", {2, 4},
                        {-55.0f, 16.0f, 89.0f, -44.0f,
                         122.0f, 154.0f, 68.0f, -39.0f});

  test.Run();
}

class Dnnl_MatMulInteger_Mul_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Mul_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 5u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];

    NodeArg* c1 = graph_input_defs[4];
    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    // The output from MatMulInteger is int32 but this is only an internal Node
    // so we make the int32 TypeProto manually to match the same shape as the
    // output shape.
    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());


    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("mul1", "Mul", "", {&cast_out, c1}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Mul) {
  Dnnl_MatMulInteger_Mul_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddInput<float>("c1", {2, 4},
                       {0.5f, 1.5f, 2.0f, 2.5f,
                       -0.5f, -1.5f, -2.0f, -2.5f});
  test.AddOutput<float>("Y", {2, 4},
                        {-27.5f, 24.0f, 178.0f, -110.0f,
                         -61.0f, -231.0f, -136.0f, 97.5f});

  test.Run();
}

// MatMulInteger_Cast_Mul_Add
class Dnnl_MatMulInteger_Cast_Mul_Add_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Cast_Mul_Add_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 6u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];
    NodeArg* c1 = graph_input_defs[4];
    NodeArg* c2 = graph_input_defs[5];

    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    // The output from MatMulInteger is int32 but this is only an internal Node
    // so we make the int32 TypeProto manually to match the same shape as the
    // output shape.
    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());
    auto& mul1_out = graph.GetOrCreateNodeArg("mul1_out", c2->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("mul1", "Mul", "", {&cast_out, c1}, {&mul1_out});
    graph.AddNode("add1", "Add", "", {&mul1_out, c2}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Mul_Add) {
  Dnnl_MatMulInteger_Cast_Mul_Add_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddInput<float>("c1", {2, 4},
                       {0.5f, 1.5f, 2.0f, 2.5f,
                        -0.5f, -1.5f, -2.0f, -2.5f});
  test.AddInput<float>("c2", {2, 4},
                       {0.2f, 1.4f, 2.6f, 2.8f,
                        -0.2f, -1.4f, -2.6f, -2.8f});
  test.AddOutput<float>("Y", {2, 4},
                        {-27.3f, 25.4f, 180.6f, -107.2f,
                         -61.2f, -232.4f, -138.6f, 94.7f});

  test.Run();
}
// MatMulInteger_Cast_Mul_Add_Add
// MatMulInteger_Cast_Mul_Add_Mul_Add
// MatMulInteger_Cast_Mul_Add_Add_Mul_Add
// MatMulInteger_Cast_Mul_Add_Relu
class Dnnl_MatMulInteger_Cast_Mul_Add_Relu_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Cast_Mul_Add_Relu_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 6u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];
    NodeArg* c1 = graph_input_defs[4];
    NodeArg* c2 = graph_input_defs[5];

    NodeArg* y = graph_output_defs[0];

    // internal NodeArgs
    // The output from MatMulInteger is int32 but this is only an internal Node
    // so we make the int32 TypeProto manually to match the same shape as the
    // output shape.
    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());
    auto& mul1_out = graph.GetOrCreateNodeArg("mul1_out", c2->TypeAsProto());
    auto& add1_out = graph.GetOrCreateNodeArg("add1_out", c2->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("mul1", "Mul", "", {&cast_out, c1}, {&mul1_out});
    graph.AddNode("add1", "Add", "", {&mul1_out, c2}, {&add1_out});
    graph.AddNode("relu1", "Relu", "", {&add1_out}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Mul_Add_Relu) {
  Dnnl_MatMulInteger_Cast_Mul_Add_Relu_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddInput<float>("c1", {2, 4},
                       {0.5f, 1.5f, 2.0f, 2.5f,
                        -0.5f, -1.5f, -2.0f, -2.5f});
  test.AddInput<float>("c2", {2, 4},
                       {0.2f, 1.4f, 2.6f, 2.8f,
                        -0.2f, -1.4f, -2.6f, -2.8f});
  test.AddOutput<float>("Y", {2, 4},
                        {0.0f, 25.4f, 180.6f, 0.0f,
                         0.0f, 0.0f, 0.0f, 94.7f});

  test.Run();
}

class Dnnl_MatMulInteger_Cast_Div1_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Cast_Div1_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 5u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];

    NodeArg* c1 = graph_input_defs[4];
    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("div1", "Div", "", {&cast_out, c1}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Div1) {
  Dnnl_MatMulInteger_Cast_Div1_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddInput<float>("c1", {2, 4},
                       {0.5f, 1.5f, 2.0f, 2.5f,
                        -0.5f, -1.5f, -2.0f, -2.5f});
  test.AddOutput<float>("Y", {2, 4},
                        {-110.0f, 10.666667f, 44.5f, -17.6f,
                         -244.0f, -102.666664f, -34.0f, 15.6f});

  test.Run();
}

class Dnnl_MatMulInteger_Cast_Div2_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Cast_Div2_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 5u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];

    NodeArg* c1 = graph_input_defs[4];
    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("div1", "Div", "", {c1, &cast_out}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Div2) {
  Dnnl_MatMulInteger_Cast_Div2_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddInput<float>("c1", {2, 4},
                       {0.5f, 1.5f, 2.0f, 2.5f,
                        -0.5f, -1.5f, -2.0f, -2.5f});
  test.AddOutput<float>("Y", {2, 4},
                        {-0.009090909f, 0.09375f, 0.02247191f, -0.056818184f,
                         -0.0040983604f, -0.0097402595f, -0.029411765f, 0.06410257f});

  test.Run();
}

class Dnnl_MatMulInteger_Cast_Sub1_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Cast_Sub1_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 5u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];

    NodeArg* c1 = graph_input_defs[4];
    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("div1", "Sub", "", {&cast_out, c1}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Sub1) {
  Dnnl_MatMulInteger_Cast_Sub1_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddInput<float>("c1", {2, 4},
                       {0.5f, 1.5f, 2.0f, 2.5f,
                        -0.5f, -1.5f, -2.0f, -2.5f});
  test.AddOutput<float>("Y", {2, 4},
                        {-55.5f, 14.5f, 87.0f, -46.5f,
                         122.5f, 155.5f, 70.0f, -36.5f});

  test.Run();
}

// Fusion not possible with this but it should still run and pass
class Dnnl_MatMulInteger_Cast_Sub2_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Cast_Sub2_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 5u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];

    NodeArg* c1 = graph_input_defs[4];
    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("div1", "Sub", "", {c1, &cast_out}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Sub2) {
  Dnnl_MatMulInteger_Cast_Sub2_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddInput<float>("c1", {2, 4},
                       {0.5f, 1.5f, 2.0f, 2.5f,
                        -0.5f, -1.5f, -2.0f, -2.5f});
  test.AddOutput<float>("Y", {2, 4},
                        {55.5f, -14.5f, -87.0f, 46.5f,
                         -122.5f, -155.5f, -70.0f, 36.5f});

  test.Run();
}

class Dnnl_MatMulInteger_Cast_Abs_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Cast_Abs_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];

    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("abs1", "Abs", "", {&cast_out}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Abs) {
  Dnnl_MatMulInteger_Cast_Abs_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddOutput<float>("Y", {2, 4},
                        {55.0f, 16.0f, 89.0f, 44.0f,
                         122.0f, 154.0f, 68.0f, 39.0f});

  test.Run();
}

class Dnnl_MatMulInteger_Cast_Elu_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Cast_Elu_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];

    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("abs1", "Elu", "", {&cast_out}, {y}).AddAttribute("alpha", 1.5f);
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Elu) {
  Dnnl_MatMulInteger_Cast_Elu_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddOutput<float>("Y", {2, 4},
                        {-1.5f, 16.0f, 89.0f, -1.5f,
                         122.0f, 154.0f, 68.0f, -1.5f});

  test.Run();
}

class Dnnl_MatMulInteger_Cast_Mul_Exp_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Cast_Mul_Exp_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 5u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];
    NodeArg* c1 = graph_input_defs[4];

    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());
    auto& mul1_out = graph.GetOrCreateNodeArg("mul1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("mul1", "Mul", "", {&cast_out, c1}, {&mul1_out});
    graph.AddNode("exp1", "Exp", "", {&mul1_out}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Mul_Exp) {
  Dnnl_MatMulInteger_Cast_Mul_Exp_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddInput<float>("c1", {}, {0.01f});
  test.AddOutput<float>("Y", {2, 4},
                        {0.5769498103804866f, 1.1735108709918103f,
                         2.4351296512898744f, 0.6440364210831414f,
                         3.3871877336213347f, 4.664590270988126f,
                         1.9738777322304477f, 0.6770568744981647f});

  test.Run();
}

class Dnnl_MatMulInteger_Cast_LeakyRelu_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Cast_LeakyRelu_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];

    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("leakyrelu1", "LeakyRelu", "", {&cast_out}, {y}).AddAttribute("alpha", 0.015f);
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_LeakyRelu) {
  Dnnl_MatMulInteger_Cast_LeakyRelu_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddOutput<float>("Y", {2, 4},
                        {-0.825f, 16.0f, 89.0f, -0.65999997f,
                         122.0f, 154.0f, 68.0f, -0.585f});

  test.Run();
}

class Dnnl_MatMulInteger_Cast_Log_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Cast_Log_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];

    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());
    auto& abs_out = graph.GetOrCreateNodeArg("abs_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("abs1", "Abs", "", {&cast_out}, {&abs_out});
    graph.AddNode("log1", "Log", "", {&abs_out}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Abs_Log) {
  Dnnl_MatMulInteger_Cast_Log_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddOutput<float>("Y", {2, 4},
                        {4.0073333f, 2.7725887f, 4.4886365f, 3.7841897f,
                         4.804021f, 5.0369525f, 4.2195077f, 3.6635616f});

  test.Run();
}

class Dnnl_MatMulInteger_Add_Round_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Add_Round_PostOpTester(int opset_version = 11)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 5u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];
    NodeArg* c1 = graph_input_defs[4];

    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());
    auto& add1_out = graph.GetOrCreateNodeArg("add1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("add", "Add", "", {&cast_out, c1}, {&add1_out});
    graph.AddNode("round1", "Round", "", {&add1_out}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Add_Round) {
  Dnnl_MatMulInteger_Add_Round_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddInput<float>("c1", {2, 4},
                        {0.4f, 0.6f, 0.4f, 0.6f,
                         0.4f, 0.6f, 0.5f, 0.5f});
  test.AddOutput<float>("Y", {2, 4},
                        {-55.0f, 17.0f, 89.0f, -43.0f,
                          122.0f, 155.0f, 68.0f, -38.0f});

  test.Run();
}

class Dnnl_MatMulInteger_Cast_Sigmoid_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Cast_Sigmoid_PostOpTester(int opset_version = 13)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];

    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("sigmoid1", "Sigmoid", "", {&cast_out}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Sigmoid) {
  Dnnl_MatMulInteger_Cast_Sigmoid_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddOutput<float>("Y", {2, 4},
                        {1.2995814e-24f, 0.9999999f, 1.0f, 7.781132e-20f,
                         1.0f, 1.0f, 1.0f, 1.1548225e-17f});

  test.Run();
}

class Dnnl_MatMulInteger_Mul_Softplus_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Mul_Softplus_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 5u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];
    NodeArg* c1 = graph_input_defs[4];

    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());
    auto& mul1_out = graph.GetOrCreateNodeArg("add1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("mul1", "Mul", "", {&cast_out, c1}, {&mul1_out});
    graph.AddNode("softplus1", "Softplus", "", {&mul1_out}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Mul_Softplus) {
  Dnnl_MatMulInteger_Mul_Softplus_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddInput<float>("c1", {},
                       {0.01f});
  test.AddOutput<float>("Y", {2, 4},
                        {0.4554924814633376f, 0.7763437730407398f,
                         1.2340546691512106f, 0.4971544503321099f,
                         1.4786884144349526f, 1.7342345654720792f,
                         1.0898667349636622f, 0.5170403966954268f});

  test.Run();
}

class Dnnl_MatMulInteger_Cast_Abs_Sqrt_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Cast_Abs_Sqrt_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];

    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());
    auto& abs_out = graph.GetOrCreateNodeArg("abs_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("abs1", "Abs", "", {&cast_out}, {&abs_out});
    graph.AddNode("sqrt1", "Sqrt", "", {&abs_out}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Abs_Sqrt) {
  Dnnl_MatMulInteger_Cast_Abs_Sqrt_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddOutput<float>("Y", {2, 4},
                        {7.4161983f, 4.0f, 9.433981f, 6.6332498f,
                         11.045361f, 12.409674f, 8.246211f, 6.244998f});

  test.Run();
}

class Dnnl_MatMulInteger_Mul_Tanh_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Mul_Tanh_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 5u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];
    NodeArg* c1 = graph_input_defs[4];

    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());
    auto& mul1_out = graph.GetOrCreateNodeArg("add1_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("mul1", "Mul", "", {&cast_out, c1}, {&mul1_out});
    graph.AddNode("tanh1", "Tanh", "", {&mul1_out}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Mul_Tanh) {
  Dnnl_MatMulInteger_Mul_Tanh_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddInput<float>("c1", {}, {0.01f});
  test.AddOutput<float>("Y", {2, 4},
                        {-0.5005202111902353f, 0.1586485042974989f,
                         0.7113937318189626f, -0.41364444218713514f,
                         0.8396541756543753f, 0.9121203692077173f,
                         0.5915193954318164f, -0.3713602278765077f});

  test.Run();
}

class Dnnl_MatMulInteger_36_ops_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_36_ops_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 5u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];
    NodeArg* c1 = graph_input_defs[4];

    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());

    auto& m1_out = graph.GetOrCreateNodeArg("mul1_out", y->TypeAsProto());
    auto& a1_out = graph.GetOrCreateNodeArg("add1_out", y->TypeAsProto());
    auto& s1_out = graph.GetOrCreateNodeArg("sub1_out", y->TypeAsProto());
    auto& d1_out = graph.GetOrCreateNodeArg("div1_out", y->TypeAsProto());

    auto& m2_out = graph.GetOrCreateNodeArg("mul2_out", y->TypeAsProto());
    auto& a2_out = graph.GetOrCreateNodeArg("add2_out", y->TypeAsProto());
    auto& s2_out = graph.GetOrCreateNodeArg("sub2_out", y->TypeAsProto());
    auto& d2_out = graph.GetOrCreateNodeArg("div2_out", y->TypeAsProto());

    auto& m3_out = graph.GetOrCreateNodeArg("mul3_out", y->TypeAsProto());
    auto& a3_out = graph.GetOrCreateNodeArg("add3_out", y->TypeAsProto());
    auto& s3_out = graph.GetOrCreateNodeArg("sub3_out", y->TypeAsProto());
    auto& d3_out = graph.GetOrCreateNodeArg("div3_out", y->TypeAsProto());

    auto& m4_out = graph.GetOrCreateNodeArg("mul4_out", y->TypeAsProto());
    auto& a4_out = graph.GetOrCreateNodeArg("add4_out", y->TypeAsProto());
    auto& s4_out = graph.GetOrCreateNodeArg("sub4_out", y->TypeAsProto());
    auto& d4_out = graph.GetOrCreateNodeArg("div4_out", y->TypeAsProto());

    auto& m5_out = graph.GetOrCreateNodeArg("mul5_out", y->TypeAsProto());
    auto& a5_out = graph.GetOrCreateNodeArg("add5_out", y->TypeAsProto());
    auto& s5_out = graph.GetOrCreateNodeArg("sub5_out", y->TypeAsProto());
    auto& d5_out = graph.GetOrCreateNodeArg("div5_out", y->TypeAsProto());

    auto& m6_out = graph.GetOrCreateNodeArg("mul6_out", y->TypeAsProto());
    auto& a6_out = graph.GetOrCreateNodeArg("add6_out", y->TypeAsProto());
    auto& s6_out = graph.GetOrCreateNodeArg("sub6_out", y->TypeAsProto());
    auto& d6_out = graph.GetOrCreateNodeArg("div6_out", y->TypeAsProto());

    auto& m7_out = graph.GetOrCreateNodeArg("mul7_out", y->TypeAsProto());
    auto& a7_out = graph.GetOrCreateNodeArg("add7_out", y->TypeAsProto());
    auto& s7_out = graph.GetOrCreateNodeArg("sub7_out", y->TypeAsProto());
    auto& d7_out = graph.GetOrCreateNodeArg("div7_out", y->TypeAsProto());

    auto& m8_out = graph.GetOrCreateNodeArg("mul8_out", y->TypeAsProto());
    auto& a8_out = graph.GetOrCreateNodeArg("add8_out", y->TypeAsProto());
    auto& s8_out = graph.GetOrCreateNodeArg("sub8_out", y->TypeAsProto());
    auto& d8_out = graph.GetOrCreateNodeArg("div8_out", y->TypeAsProto());

    auto& m9_out = graph.GetOrCreateNodeArg("mul9_out", y->TypeAsProto());
    auto& a9_out = graph.GetOrCreateNodeArg("add9_out", y->TypeAsProto());
    auto& s9_out = graph.GetOrCreateNodeArg("sub9_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("mul1", "Mul", "", {&cast_out, c1}, {&m1_out});
    graph.AddNode("add1", "Add", "", {&m1_out, c1}, {&a1_out});
    graph.AddNode("sub1", "Sub", "", {&a1_out, c1}, {&s1_out});
    graph.AddNode("div1", "Div", "", {&s1_out, c1}, {&d1_out});
    graph.AddNode("mul2", "Mul", "", {&d1_out, c1}, {&m2_out});
    graph.AddNode("add2", "Add", "", {&m2_out, c1}, {&a2_out});
    graph.AddNode("sub2", "Sub", "", {&a2_out, c1}, {&s2_out});
    graph.AddNode("div2", "Div", "", {&s2_out, c1}, {&d2_out});
    graph.AddNode("mul3", "Mul", "", {&d2_out, c1}, {&m3_out});
    graph.AddNode("add3", "Add", "", {&m3_out, c1}, {&a3_out});
    graph.AddNode("sub3", "Sub", "", {&a3_out, c1}, {&s3_out});
    graph.AddNode("div3", "Div", "", {&s3_out, c1}, {&d3_out});
    graph.AddNode("mul4", "Mul", "", {&d3_out, c1}, {&m4_out});
    graph.AddNode("add4", "Add", "", {&m4_out, c1}, {&a4_out});
    graph.AddNode("sub4", "Sub", "", {&a4_out, c1}, {&s4_out});
    graph.AddNode("div4", "Div", "", {&s4_out, c1}, {&d4_out});
    graph.AddNode("mul5", "Mul", "", {&d4_out, c1}, {&m5_out});
    graph.AddNode("add5", "Add", "", {&m5_out, c1}, {&a5_out});
    graph.AddNode("sub5", "Sub", "", {&a5_out, c1}, {&s5_out});
    graph.AddNode("div5", "Div", "", {&s5_out, c1}, {&d5_out});
    graph.AddNode("mul6", "Mul", "", {&d5_out, c1}, {&m6_out});
    graph.AddNode("add6", "Add", "", {&m6_out, c1}, {&a6_out});
    graph.AddNode("sub6", "Sub", "", {&a6_out, c1}, {&s6_out});
    graph.AddNode("div6", "Div", "", {&s6_out, c1}, {&d6_out});
    graph.AddNode("mul7", "Mul", "", {&d6_out, c1}, {&m7_out});
    graph.AddNode("add7", "Add", "", {&m7_out, c1}, {&a7_out});
    graph.AddNode("sub7", "Sub", "", {&a7_out, c1}, {&s7_out});
    graph.AddNode("div7", "Div", "", {&s7_out, c1}, {&d7_out});
    graph.AddNode("mul8", "Mul", "", {&d7_out, c1}, {&m8_out});
    graph.AddNode("add8", "Add", "", {&m8_out, c1}, {&a8_out});
    graph.AddNode("sub8", "Sub", "", {&a8_out, c1}, {&s8_out});
    graph.AddNode("div8", "Div", "", {&s8_out, c1}, {&d8_out});
    graph.AddNode("mul9", "Mul", "", {&d8_out, c1}, {&m9_out});
    graph.AddNode("add9", "Add", "", {&m9_out, c1}, {&a9_out});
    graph.AddNode("sub9", "Sub", "", {&a9_out, c1}, {&s9_out});
    graph.AddNode("div9", "Div", "", {&s9_out, c1}, {y});
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_36_ops) {
  Dnnl_MatMulInteger_36_ops_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddInput<float>("c1", {}, {2.5f});
  test.AddOutput<float>("Y", {2, 4},
                        {-55.0f, 16.0f, 89.0f, -44.0f,
                         122.0f, 154.0f, 68.0f, -39.0f});

  test.Run();
}

class Dnnl_MatMulInteger_Cast_Elu_LeakyRelu_PostOpTester : public OpTester {
 public:
  explicit Dnnl_MatMulInteger_Cast_Elu_LeakyRelu_PostOpTester(int opset_version = 10)
      : OpTester("MatMulInteger", opset_version) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_input_defs.size(), 4u);
    ASSERT_EQ(graph_output_defs.size(), 1u);

    NodeArg* a = graph_input_defs[0];
    NodeArg* b = graph_input_defs[1];
    NodeArg* a_zp = graph_input_defs[2];
    NodeArg* b_zp = graph_input_defs[3];

    NodeArg* y = graph_output_defs[0];

    const onnx::TensorShapeProto* output_shape = y->Shape();
    onnx::TypeProto output_int32;
    output_int32.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
    for (int i = 0; i < output_shape->dim_size(); ++i) {
      auto dim = output_int32.mutable_tensor_type()->mutable_shape()->add_dim();
      *dim = output_shape->dim(i);
    }
    auto& matmul_out = graph.GetOrCreateNodeArg("matmul_out", &output_int32);
    auto& cast_out = graph.GetOrCreateNodeArg("cast_out", y->TypeAsProto());
    auto& elu_out = graph.GetOrCreateNodeArg("elu_out", y->TypeAsProto());

    graph.AddNode("matmul1", "MatMulInteger", "", {a, b, a_zp, b_zp}, {&matmul_out});
    auto& cast_node = graph.AddNode("cast1", "Cast", "", {&matmul_out}, {&cast_out});
    cast_node.AddAttribute("to", int64_t{ONNX_NAMESPACE::TensorProto_DataType_FLOAT});
    graph.AddNode("elu1", "Elu", "", {&cast_out}, {&elu_out}).AddAttribute("alpha", 1.5f);
    graph.AddNode("leakyrelu1", "LeakyRelu", "", {&elu_out}, {y}).AddAttribute("alpha", 0.1f);
  }
};

TEST(DnnlMatMulIntegerFusion, MatMulInteger_Cast_Elu_LeakyRelu) {
  Dnnl_MatMulInteger_Cast_Elu_LeakyRelu_PostOpTester test;
  test.AddInput<int8_t>("a", {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("b", {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zp", {}, {5});
  test.AddInput<int8_t>("b_zp", {}, {5});
  test.AddOutput<float>("Y", {2, 4},
                        {-0.15f, 16.0f, 89.0f, -0.15f,
                         122.0f, 154.0f, 68.0f, -0.15f});

  test.Run();
}
#endif  // USE_DNNL
}  // namespace test
}  // namespace onnxruntime
