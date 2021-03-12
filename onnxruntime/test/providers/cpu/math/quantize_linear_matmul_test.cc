// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/quantize_linear_matmul.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(QuantizeLinearMatmulOpTest, QLinearMatMul3D_U8U8) {
  OpTester test("QLinearMatMul", 10);
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
  OpTester test("QLinearMatMul", 10);
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

TEST(QuantizeLinearMatmulOpTest, QLinearMatMul2D_U8U8) {
  auto run_test = [](bool only_t1_not_initializer) {
    OpTester test("QLinearMatMul", 10);
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
    OpTester test("QLinearMatMul", 10);
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

static void QLinearMatMul2DTest(bool only_t1_not_initializer) {
  // Test non-empty inputs
  OpTester test_non_empty("QLinearMatMul", 10);
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
  OpTester test_empty("QLinearMatMul", 10);
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

using namespace ONNX_NAMESPACE;

// Test kernel that extends QLearnMatMul for extra verification
struct PrePackTestOp {
  static constexpr const char* OpName = "QLinearMatMulPrePack";
  static constexpr const char* OpDomain = "testing";

  // TODO!! query QLinearMatMul schema instead of copy this much code!!
  static ONNX_NAMESPACE::OpSchema OpSchema() {
    // Get QLinearMatMul schema from global registry and hack
    auto p_original = ONNX_NAMESPACE::OpSchemaRegistry::Schema("QLinearMatMul", 10, "");
    ONNX_NAMESPACE::OpSchema modified;

    modified.SetDoc("Return success, error, or throw based on the input.")
        .SetName(OpName)
        .SetDomain(OpDomain)
        .SinceVersion(10);

    const auto& inputs = p_original->inputs();
    for (int i = 0; i < inputs.size(); i++) {
      const auto& in = inputs[i];
      modified.Input(i, in.GetName(), in.GetDescription(), in.GetTypeStr(),
          in.GetOption(), in.GetIsHomogeneous(), in.GetMinArity(), in.GetDifferentiationCategory());
    }

    const auto& outputs = p_original->outputs();
    for (int oi = 0; oi < outputs.size(); oi++) {
      const auto& out = outputs[oi];
      modified.Output(oi, out.GetName(), out.GetDescription(), out.GetTypeStr(),
          out.GetOption(), out.GetIsHomogeneous(), out.GetMinArity(), out.GetDifferentiationCategory());
    }
    
    for (const auto& ty : p_original->typeConstraintParams()) {
      modified.TypeConstraint(ty.type_param_str, ty.allowed_type_strs, ty.description);
    }
    return modified;

/*    ONNX_NAMESPACE::OpSchema schema;
    schema.SetDoc("Return success, error, or throw based on the input.")
        .SetName(OpName)
        .SetDomain(OpDomain)
        .SinceVersion(10)
        .Input(
            0,
            "a",
            "N-dimensional quantized matrix a",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            1,
            "a_scale",
            "scale of quantized input a",
            "tensor(float)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "a_zero_point",
            "zero point of quantized input a",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            3,
            "b",
            "N-dimensional quantized matrix b",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            4,
            "b_scale",
            "scale of quantized input b",
            "tensor(float)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            5,
            "b_zero_point",
            "zero point of quantized input b",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            6,
            "y_scale",
            "scale of quantized output y",
            "tensor(float)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            7,
            "y_zero_point",
            "zero point of quantized output y",
            "T3",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "y",
            "Quantized matrix multiply results from a * b",
            "T3",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T1",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain input a and its zero point data type to 8-bit integer tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain input b and its zero point data type to 8-bit integer tensor.")
        .TypeConstraint(
            "T3",
            {"tensor(int8)", "tensor(uint8)"},
            "Constrain output y and its zero point data type to 8-bit integer tensor.");

    return schema;*/
  }

  class QLinearMatMulPrePackT : public QLinearMatMul {
   public:
    QLinearMatMulPrePackT(const OpKernelInfo& info) : QLinearMatMul(info) {
    }

    Status Compute(OpKernelContext* context) const override {
      ORT_ENFORCE(bool(packed_b_), "QLinearMatMul input B should be pre-packed, but it is not!");
      return QLinearMatMul::Compute(context);
    }
  };

  static KernelDefBuilder KernelDef() {
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

TEST(QuantizeLinearMatmulOpTest, QLinearMatMulPrePack) {
  auto registry = std::make_shared<CustomRegistry>();
  std::vector<OpSchema> schemas{PrePackTestOp::OpSchema()};
  Status status;
  ASSERT_TRUE((status = registry->RegisterOpSet(schemas, PrePackTestOp::OpDomain, 10, 11)).IsOK()) << status;
  KernelCreateFn kernel_create_fn = [](const OpKernelInfo& info) { return new typename PrePackTestOp::QLinearMatMulPrePackT(info); };
  auto kernel_def = PrePackTestOp::KernelDef();
  ASSERT_TRUE((status = registry->RegisterCustomKernel(kernel_def, kernel_create_fn)).IsOK()) << status;

  OpTester test_non_empty(PrePackTestOp::OpName, 10, PrePackTestOp::OpDomain);
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
  test_non_empty.Run();
}


}  // namespace test
}  // namespace onnxruntime
