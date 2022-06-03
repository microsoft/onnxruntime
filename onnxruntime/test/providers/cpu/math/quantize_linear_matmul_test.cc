// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/quantization/quantize_linear_matmul.h"
#include "core/mlas/inc/mlas.h"

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
namespace test {

class OpTesterRequant : public OpTester {
 public:
  explicit OpTesterRequant(
    const char* op,
    int opset_version = 7,
    const char* domain = onnxruntime::kOnnxDomain,
    bool verify_output = true) : OpTester(op, opset_version, domain, verify_output) {}

  void Run(ExpectResult expect_result = ExpectResult::kExpectSuccess, const std::string& expected_failure_string = "",
           const std::unordered_set<std::string>& excluded_provider_types = {},
           const RunOptions* run_options = nullptr,
           std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr,
           const Graph::ResolveOptions& resolve_options = {}) {
    SessionOptions session_options;
    SetUpDefaultSessionOptions(session_options);
    for (auto use_fixed_point : {"0", "1"}) {
      ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
        kOrtSessionOptionsConfigFixedPointRequantOnARM64,
        use_fixed_point));
      OpTester::Run(session_options,
                    expect_result,
                    expected_failure_string,
                    excluded_provider_types,
                    run_options,
                    execution_providers,
                    resolve_options);
    }
  }

  void Run(SessionOptions session_options,
           ExpectResult expect_result = ExpectResult::kExpectSuccess,
           const std::string& expected_failure_string = "",
           const std::unordered_set<std::string>& excluded_provider_types = {},
           const RunOptions* run_options = nullptr,
           std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr,
           const Graph::ResolveOptions& resolve_options = {},
           /*out*/ size_t* number_of_pre_packed_weights_counter = nullptr,
           /*out*/ size_t* number_of_shared_pre_packed_weights_counter = nullptr) {
    for (auto use_fixed_point : {"0", "1"}) {
      ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
        kOrtSessionOptionsConfigFixedPointRequantOnARM64,
        use_fixed_point));
      OpTester::Run(session_options,
                    expect_result,
                    expected_failure_string,
                    excluded_provider_types,
                    run_options,
                    execution_providers,
                    resolve_options,
                    number_of_pre_packed_weights_counter,
                    number_of_shared_pre_packed_weights_counter);
    }
  }
};

TEST(QuantizeLinearMatmulOpTest, QLinearMatMul3D_U8U8) {
  OpTesterRequant test("QLinearMatMul", 10);
  test.AddInput<uint8_t>("T1", {2, 2, 4},
                         {208, 236, 0, 238,
                          3, 214, 255, 29,

                          208, 236, 0, 238,
                          3, 214, 255, 29});

  test.AddInput<float>("a_scale", {}, {0.0066f});
  test.AddInput<uint8_t>("a_zero_point", {}, {113});

  test.AddInput<uint8_t>("T2", {2, 4, 3},
                         {152, 51, 244,
                          60, 26, 255,
                          0, 127, 246,
                          127, 254, 247,

                          152, 51, 244,
                          60, 26, 255,
                          0, 127, 246,
                          127, 254, 247});

  test.AddInput<float>("b_scale", {}, {0.00705f});
  test.AddInput<uint8_t>("b_zero_point", {}, {114});

  test.AddInput<float>("y_scale", {}, {0.0107f});
  test.AddInput<uint8_t>("y_zero_point", {}, {118});
  test.AddOutput<uint8_t>("T3", {2, 2, 3},
                          {168, 115, 255,
                           1, 66, 151,

                           168, 115, 255,
                           1, 66, 151});

  test.Run();
}

TEST(QuantizeLinearMatmulOpTest, QLinearMatMul3D_U8S8) {
  OpTesterRequant test("QLinearMatMul", 10);
  test.AddInput<uint8_t>("T1", {2, 2, 4},
                         {208, 126, 0, 238,
                          3, 214, 255, 29,

                          208, 236, 0, 238,
                          3, 214, 255, 29});

  test.AddInput<float>("a_scale", {}, {0.0066f});
  test.AddInput<uint8_t>("a_zero_point", {}, {113});

  test.AddInput<int8_t>("T2", {2, 4, 3},
                        {-43, 51, -34,
                         60, 26, -17,
                         0, 63, -55,
                         47, -29, -31,

                         -62, 51, -42,
                         60, 26, -22,
                         0, -8, -19,
                         37, -2, -47});

  test.AddInput<float>("b_scale", {}, {0.00802f});
  test.AddInput<int8_t>("b_zero_point", {}, {-2});

  test.AddInput<float>("y_scale", {}, {0.0123f});
  test.AddInput<uint8_t>("y_zero_point", {}, {118});
  test.AddOutput<uint8_t>("T3", {2, 2, 3},
                          {130, 95, 114,
                           148, 155, 105,

                           146, 157, 75,
                           160, 101, 134});

  test.Run();
}

TEST(QuantizeLinearMatmulOpTest, QLinearMatMul3D_S8S8) {
  OpTesterRequant test("QLinearMatMul", 10);
  test.AddInput<int8_t>("T1", {2, 2, 4},
                        {80, -2, -128, 110,
                         -125, 86, 127, -99,

                         80, 108, -128, 110,
                         -125, 86, 127, -99});

  test.AddInput<float>("a_scale", {}, {0.0066f});
  test.AddInput<int8_t>("a_zero_point", {}, {-15});

  test.AddInput<int8_t>("T2", {2, 4, 3},
                        {-43, 51, -34,
                         60, 26, -17,
                         0, 63, -55,
                         47, -29, -31,

                         -62, 51, -42,
                         60, 26, -22,
                         0, -8, -19,
                         37, -2, -47});

  test.AddInput<float>("b_scale", {}, {0.00802f});
  test.AddInput<int8_t>("b_zero_point", {}, {-2});

  test.AddInput<float>("y_scale", {}, {0.0123f});
  test.AddInput<int8_t>("y_zero_point", {}, {-10});
  test.AddOutput<int8_t>("T3", {2, 2, 3},
                         {2, -33, -14,
                          20, 27, -23,

                          18, 29, -53,
                          32, -27, 6});

  test.Run();
}

TEST(QuantizeLinearMatmulOpTest, QLinearMatMul2D_U8U8) {
  auto run_test = [](bool only_t1_not_initializer) {
    OpTesterRequant test("QLinearMatMul", 10);
    test.AddInput<uint8_t>("T1", {2, 4},
                           {208, 236, 0, 238,
                            3, 214, 255, 29});

    test.AddInput<float>("a_scale", {}, {0.0066f}, only_t1_not_initializer);
    test.AddInput<uint8_t>("a_zero_point", {}, {113}, only_t1_not_initializer);

    test.AddInput<uint8_t>("T2", {4, 3},
                           {152, 51, 244,
                            60, 26, 255,
                            0, 127, 246,
                            127, 254, 247},
                           only_t1_not_initializer);

    test.AddInput<float>("b_scale", {}, {0.00705f}, only_t1_not_initializer);
    test.AddInput<uint8_t>("b_zero_point", {}, {114}, only_t1_not_initializer);

    test.AddInput<float>("y_scale", {}, {0.0107f}, only_t1_not_initializer);
    test.AddInput<uint8_t>("y_zero_point", {}, {118}, only_t1_not_initializer);
    test.AddOutput<uint8_t>("T3", {2, 3},
                            {168, 115, 255,
                             1, 66, 151});

    test.Run();
  };

  run_test(false);

  // NNAPI will require all inputs except T1 to be initializers
  run_test(true);
}

TEST(QuantizeLinearMatmulOpTest, QLinearMatMul2D_U8S8) {
  auto run_test = [](bool only_t1_not_initializer) {
    OpTesterRequant test("QLinearMatMul", 10);
    test.AddInput<uint8_t>("T1", {2, 4},
                           {208, 126, 0, 238,
                            3, 214, 255, 29});

    test.AddInput<float>("a_scale", {}, {0.0066f}, only_t1_not_initializer);
    test.AddInput<uint8_t>("a_zero_point", {}, {113}, only_t1_not_initializer);

    test.AddInput<int8_t>("T2", {4, 3},
                          {-43, 51, -34,
                           60, 26, -17,
                           0, 63, -55,
                           47, -29, -31},
                          only_t1_not_initializer);

    test.AddInput<float>("b_scale", {}, {0.00802f}, only_t1_not_initializer);
    test.AddInput<int8_t>("b_zero_point", {}, {0}, only_t1_not_initializer);

    test.AddInput<float>("y_scale", {}, {0.0123f}, only_t1_not_initializer);
    test.AddInput<uint8_t>("y_zero_point", {}, {118}, only_t1_not_initializer);
    test.AddOutput<uint8_t>("T3", {2, 3},
                            {129, 94, 113,
                             147, 154, 104});

    test.Run();
  };

  run_test(false);

  // NNAPI will require all inputs except T1 to be initializers
  run_test(true);
}

TEST(QuantizeLinearMatmulOpTest, QLinearMatMul2D_S8S8) {
  auto run_test = [](bool only_t1_not_initializer) {
    OpTesterRequant test("QLinearMatMul", 10);
    test.AddInput<int8_t>("T1", {2, 4},
                          {80, -2, -128, 110,
                           -125, 86, 127, -99});

    test.AddInput<float>("a_scale", {}, {0.0066f}, only_t1_not_initializer);
    test.AddInput<int8_t>("a_zero_point", {}, {-15}, only_t1_not_initializer);

    test.AddInput<int8_t>("T2", {4, 3},
                          {-43, 51, -34,
                           60, 26, -17,
                           0, 63, -55,
                           47, -29, -31},
                          only_t1_not_initializer);

    test.AddInput<float>("b_scale", {}, {0.00802f}, only_t1_not_initializer);
    test.AddInput<int8_t>("b_zero_point", {}, {0}, only_t1_not_initializer);

    test.AddInput<float>("y_scale", {}, {0.0123f}, only_t1_not_initializer);
    test.AddInput<int8_t>("y_zero_point", {}, {-10}, only_t1_not_initializer);
    test.AddOutput<int8_t>("T3", {2, 3},
                           {1, -34, -15,
                            19, 26, -24});

    test.Run();
  };

  run_test(false);

  // NNAPI will require all inputs except T1 to be initializers
  run_test(true);
}

static void QLinearMatMul2DTest(bool only_t1_not_initializer) {
  // Test non-empty inputs
  OpTesterRequant test_non_empty("QLinearMatMul", 10);
  test_non_empty.AddInput<uint8_t>("T1", {2, 4}, {208, 236, 0, 238, 3, 214, 255, 29});
  test_non_empty.AddInput<float>("a_scale", {1}, {0.0066f}, only_t1_not_initializer);
  test_non_empty.AddInput<uint8_t>("a_zero_point", {1}, {113}, only_t1_not_initializer);
  test_non_empty.AddInput<uint8_t>("T2", {4, 3}, {152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247}, only_t1_not_initializer);
  test_non_empty.AddInput<float>("b_scale", {1}, {0.00705f}, only_t1_not_initializer);
  test_non_empty.AddInput<uint8_t>("b_zero_point", {1}, {114}, only_t1_not_initializer);
  test_non_empty.AddInput<float>("y_scale", {1}, {0.0107f}, only_t1_not_initializer);
  test_non_empty.AddInput<uint8_t>("y_zero_point", {1}, {118}, only_t1_not_initializer);
  test_non_empty.AddOutput<uint8_t>("T3", {2, 3}, {168, 115, 255, 1, 66, 151});
  test_non_empty.Run();

  // Test with an empty input
  OpTesterRequant test_empty("QLinearMatMul", 10);
  test_empty.AddInput<uint8_t>("T1", {0, 4}, {});
  test_empty.AddInput<float>("a_scale", {1}, {0.0066f}, only_t1_not_initializer);
  test_empty.AddInput<uint8_t>("a_zero_point", {1}, {113}, only_t1_not_initializer);
  test_empty.AddInput<uint8_t>("T2", {4, 3}, {152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247}, only_t1_not_initializer);
  test_empty.AddInput<float>("b_scale", {1}, {0.00705f}, only_t1_not_initializer);
  test_empty.AddInput<uint8_t>("b_zero_point", {1}, {114}, only_t1_not_initializer);
  test_empty.AddInput<float>("y_scale", {1}, {0.0107f}, only_t1_not_initializer);
  test_empty.AddInput<uint8_t>("y_zero_point", {1}, {118}, only_t1_not_initializer);
  test_empty.AddOutput<uint8_t>("T3", {0, 3}, {});

  // Skip NNAPI as it doesn't support empty output for now
  test_empty.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNnapiExecutionProvider});
}

TEST(QuantizeLinearMatmulOpTest, QLinearMatMul) {
  QLinearMatMul2DTest(false);
}

// NNAPI EP requires weight to be an initializer
TEST(QuantizeLinearMatmulOpTest, QLinearMatMulAllInputExceptT1AreInitializers) {
  QLinearMatMul2DTest(true);
}

TEST(QuantizeLinearMatmulOpTest, PerColumn_2D) {
  OpTesterRequant test("QLinearMatMul", 10);
  test.AddInput<uint8_t>("a",
                         {2, 4},
                         {125, 135, 133, 122,
                          132, 123, 136, 135});
  test.AddInput<float>("a_scale", {}, {0.1f});
  test.AddInput<uint8_t>("a_zero_point", {}, {133});
  test.AddInput<int8_t>("b",
                        {4, 4},
                        {0, -8, 2, 3,
                         -11, -13, -8, 1,
                         2, 4, 4, -10,
                         3, 2, -11, 2});
  test.AddInput<float>("b_scale", {4},
                       {0.1f, 0.2f, 0.3f, 0.4f});
  test.AddInput<int8_t>("b_zero_point",
                        {1, 4},
                        {1, -2, 2, -1});
  test.AddInput<float>("y_scale", {}, {0.2f});
  test.AddInput<uint8_t>("y_zero_point", {}, {130});

  test.AddOutput<uint8_t>("y",
                          {2, 4},
                          {128, 128, 148, 118,
                           136, 144, 142, 121});

  test.Run();
}

TEST(QuantizeLinearMatmulOpTest, PerColumn_2D_S8S8) {
  OpTesterRequant test("QLinearMatMul", 10);
  test.AddInput<int8_t>("a",
                        {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<float>("a_scale", {}, {0.1f});
  test.AddInput<int8_t>("a_zero_point", {}, {5});
  test.AddInput<int8_t>("b",
                        {4, 4},
                        {0, -8, 2, 3,
                         -11, -13, -8, 1,
                         2, 4, 4, -10,
                         3, 2, -11, 2});
  test.AddInput<float>("b_scale", {4},
                       {0.1f, 0.2f, 0.3f, 0.4f});
  test.AddInput<int8_t>("b_zero_point",
                        {1, 4},
                        {1, -2, 2, -1});
  test.AddInput<float>("y_scale", {}, {0.2f});
  test.AddInput<int8_t>("y_zero_point", {}, {2});

  test.AddOutput<int8_t>("y",
                         {2, 4},
                         {0, 0, 20, -10,
                          8, 16, 14, -7});

  test.Run();
}

TEST(QuantizeLinearMatmulOpTest, PerColumn_ND) {
  OpTesterRequant test("QLinearMatMul", 10);
  test.AddInput<uint8_t>("a",
                         {2, 2, 4},
                         {125, 135, 133, 122,
                          132, 123, 136, 135,

                          125, 135, 133, 122,
                          132, 123, 136, 135});
  test.AddInput<float>("a_scale", {}, {0.1f});
  test.AddInput<uint8_t>("a_zero_point", {}, {133});
  test.AddInput<int8_t>("b",
                        {2, 4, 4},
                        {0, -8, 2, 3,
                         -11, -13, -8, 1,
                         2, 4, 4, -10,
                         3, 2, -11, 2,

                         0, -8, 2, 3,
                         -11, -13, -8, 1,
                         2, 4, 4, -10,
                         3, 2, -11, 2});
  test.AddInput<float>("b_scale", {2, 1, 4},
                       {0.1f, 0.2f, 0.3f, 0.4f,
                        0.4f, 0.3f, 0.2f, 0.1f});
  test.AddInput<int8_t>("b_zero_point",
                        {2, 1, 4},
                        {1, -2, 2, -1,
                         2, -4, -1, 0});
  test.AddInput<float>("y_scale", {}, {0.2f});
  test.AddInput<uint8_t>("y_zero_point", {}, {130});

  test.AddOutput<uint8_t>("y",
                          {2, 2, 4},
                          {128, 128, 148, 118,
                           136, 144, 142, 121,

                           126, 122, 137, 128,
                           157, 150, 136, 128});

  test.Run();
}

TEST(QuantizeLinearMatmulOpTest, PerColumn_ND_S8S8) {
  OpTesterRequant test("QLinearMatMul", 10);
  test.AddInput<int8_t>("a",
                        {2, 2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7,

                         -3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<float>("a_scale", {}, {0.1f});
  test.AddInput<int8_t>("a_zero_point", {}, {5});
  test.AddInput<int8_t>("b",
                        {2, 4, 4},
                        {0, -8, 2, 3,
                         -11, -13, -8, 1,
                         2, 4, 4, -10,
                         3, 2, -11, 2,

                         0, -8, 2, 3,
                         -11, -13, -8, 1,
                         2, 4, 4, -10,
                         3, 2, -11, 2});
  test.AddInput<float>("b_scale", {2, 1, 4},
                       {0.1f, 0.2f, 0.3f, 0.4f,
                        0.4f, 0.3f, 0.2f, 0.1f});
  test.AddInput<int8_t>("b_zero_point",
                        {2, 1, 4},
                        {1, -2, 2, -1,
                         2, -4, -1, 0});
  test.AddInput<float>("y_scale", {}, {0.2f});
  test.AddInput<int8_t>("y_zero_point", {}, {2});

  test.AddOutput<int8_t>("y",
                         {2, 2, 4},
                         {0, 0, 20, -10,
                          8, 16, 14, -7,

                          -2, -6, 9, 0,
                          29, 22, 8, 0});

  test.Run();
}

/**
 * @brief Extend QLinearMatMul for verifying prepacking behavior
 */
struct PrePackTestOp {
  // TODO!! use template and macro to extract a common utility out of this
  //   for grey box kernel testing by extending kernel classes.
  static constexpr const char* OpName = "QLinearMatMulPrePack";
  static constexpr const char* OpDomain = "testing";

  static ONNX_NAMESPACE::OpSchema OpSchema() {
    // Mock our own schema, using the existing QLinearMatMul schema as template
    auto p_original = ONNX_NAMESPACE::OpSchemaRegistry::Schema("QLinearMatMul", 10, "");
    ONNX_NAMESPACE::OpSchema modified;

    modified.SetDoc("Return success, error, or throw based on the input.")
        .SetName(OpName)
        .SetDomain(OpDomain)
        .SinceVersion(10);

    const auto& inputs = p_original->inputs();
    for (int i = 0; i < static_cast<int>(inputs.size()); i++) {
      const auto& in = inputs[i];
      modified.Input(i, in.GetName(), in.GetDescription(), in.GetTypeStr(),
                     in.GetOption(), in.GetIsHomogeneous(), in.GetMinArity(), in.GetDifferentiationCategory());
    }

    const auto& outputs = p_original->outputs();
    for (int oi = 0; oi < static_cast<int>(outputs.size()); oi++) {
      const auto& out = outputs[oi];
      modified.Output(oi, out.GetName(), out.GetDescription(), out.GetTypeStr(),
                      out.GetOption(), out.GetIsHomogeneous(), out.GetMinArity(), out.GetDifferentiationCategory());
    }

    for (const auto& ty : p_original->typeConstraintParams()) {
      modified.TypeConstraint(ty.type_param_str, ty.allowed_type_strs, ty.description);
    }
    return modified;
  }

  class QLinearMatMulPrePackT : public QLinearMatMul {
   public:
    QLinearMatMulPrePackT(const OpKernelInfo& info) : QLinearMatMul(info) {
    }

    Status Compute(OpKernelContext* context) const override {
      if (!(bool(packed_b_) || (b_shape_ == context->Input<Tensor>(IN_B)->Shape()))) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Expected prepacking did not happen.");
      }
      return QLinearMatMul::Compute(context);
    }
  };

  static KernelDefBuilder KernelDef() {
    // TODO extract this out of existing OP's kernel def instead of copying code!
    KernelDefBuilder def;
    def.SetName(OpName)
        .SetDomain(OpDomain)
        .SinceVersion(10)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .Provider(onnxruntime::kCpuExecutionProvider);

    return def;
  }
};

#ifndef ENABLE_TRAINING
TEST(QuantizeLinearMatmulOpTest, QLinearMatMulPrePack) {
  auto registry = std::make_shared<CustomRegistry>();
  std::vector<ONNX_NAMESPACE::OpSchema> schemas{PrePackTestOp::OpSchema()};
  Status status;
  ASSERT_TRUE((status = registry->RegisterOpSet(schemas, PrePackTestOp::OpDomain, 10, 11)).IsOK()) << status;
  KernelCreateFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) { out = std::make_unique<typename PrePackTestOp::QLinearMatMulPrePackT>(info); return Status::OK(); };
  auto kernel_def = PrePackTestOp::KernelDef();
  ASSERT_TRUE((status = registry->RegisterCustomKernel(kernel_def, kernel_create_fn)).IsOK()) << status;

  OpTesterRequant test_non_empty(PrePackTestOp::OpName, 10, PrePackTestOp::OpDomain);
  test_non_empty.AddCustomOpRegistry(registry);

  test_non_empty.AddInput<uint8_t>("T1", {2, 4}, {208, 236, 0, 238, 3, 214, 255, 29});
  test_non_empty.AddInput<float>("a_scale", {1}, {0.0066f}, true);
  test_non_empty.AddInput<uint8_t>("a_zero_point", {1}, {113}, true);
  test_non_empty.AddInput<uint8_t>("T2", {4, 3}, {152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247}, true);
  test_non_empty.AddInput<float>("b_scale", {1}, {0.00705f}, true);
  test_non_empty.AddInput<uint8_t>("b_zero_point", {1}, {114}, true);
  test_non_empty.AddInput<float>("y_scale", {1}, {0.0107f}, true);
  test_non_empty.AddInput<uint8_t>("y_zero_point", {1}, {118}, true);
  test_non_empty.AddOutput<uint8_t>("T3", {2, 3}, {168, 115, 255, 1, 66, 151});
  test_non_empty.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
#endif

}  // namespace test
}  // namespace onnxruntime
