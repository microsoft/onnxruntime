// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>

#include "core/framework/customregistry.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

// These minimal, session-local schemas exercise the real kernels without changing the ONNX pin
// or adding unreleased schemas to production. Shape validation is deliberately left to the kernels.
std::shared_ptr<CustomRegistry> Matrix29TestSchemas() {
  using ONNX_NAMESPACE::OpSchema;
  auto registry = std::make_shared<CustomRegistry>();
  const std::vector<std::string> types{"tensor(float)", "tensor(double)", "tensor(float16)", "tensor(bfloat16)"};
  OpSchema gemm;
  gemm.SetName("Gemm")
      .SetDomain(kOnnxDomain)
      .SinceVersion(29)
      .Input(0, "A", "", "T")
      .Input(1, "B", "", "T")
      .Input(2, "C", "", "T", OpSchema::Optional)
      .Output(0, "Y", "", "T")
      .TypeConstraint("T", types, "")
      .Attr("transA", "", ONNX_NAMESPACE::AttributeProto::INT, int64_t{0})
      .Attr("transB", "", ONNX_NAMESPACE::AttributeProto::INT, int64_t{0})
      .Attr("alpha", "", ONNX_NAMESPACE::AttributeProto::FLOAT, 1.0f)
      .Attr("beta", "", ONNX_NAMESPACE::AttributeProto::FLOAT, 1.0f);
  OpSchema grouped;
  grouped.SetName("GroupedMatMul")
      .SetDomain(kOnnxDomain)
      .SinceVersion(29)
      .Input(0, "input", "", "T")
      .Input(1, "weights", "", "T")
      .Input(2, "group_indices", "", "Tind")
      .Input(3, "bias", "", "T", OpSchema::Optional)
      .Output(0, "output", "", "T")
      .TypeConstraint("T", types, "")
      .TypeConstraint("Tind", {"tensor(int32)", "tensor(int64)"}, "")
      .Attr("transB", "", ONNX_NAMESPACE::AttributeProto::INT, int64_t{0});
  std::vector<OpSchema> schemas{std::move(gemm), std::move(grouped)};
  ORT_THROW_IF_ERROR(registry->RegisterOpSet(schemas, kOnnxDomain, 29, 29));
  return registry;
}

class Matrix29Test : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    if (GetParam() && !DefaultCudaExecutionProvider()) GTEST_SKIP() << "CUDA EP is unavailable";
  }

  void Run(OpTester& tester, const std::string& error = {}) {
    tester.SetAllowUnreleasedOnnxOpset();
    tester.AddCustomOpRegistry(Matrix29TestSchemas());
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    if (GetParam()) {
      ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    }
    tester.Config(options)
        .ConfigEp(GetParam() ? DefaultCudaExecutionProvider() : DefaultCpuExecutionProvider())
        .Config(error.empty() ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure, error)
        .RunWithConfig();
  }

  template <typename T>
  void GemmCase(const std::vector<int64_t>& a_shape, int64_t trans_a, int64_t trans_b,
                const std::vector<int64_t>* c_shape, bool constant_b = false, float beta = 0.5f, float alpha = 1.5f) {
    const size_t rank = a_shape.size();
    const bool transpose = trans_a != 0 && rank > 1;
    const int64_t k = a_shape[transpose ? rank - 2 : rank - 1];
    const int64_t n = 5;
    std::vector<int64_t> y_shape = a_shape;
    if (transpose) std::swap(y_shape[rank - 1], y_shape[rank - 2]);
    y_shape.back() = n;
    auto values = [](int64_t count) {
      std::vector<T> data;
      for (int64_t i = 0; i < count; ++i) data.push_back(static_cast<T>(static_cast<float>(i % 7 - 3)));
      return data;
    };
    auto a = values(TensorShape(a_shape).Size());
    auto b = values(k * n);
    auto c = c_shape ? values(TensorShape(*c_shape).Size()) : std::vector<T>{};
    std::vector<T> expected;
    const int64_t m = rank == 1 ? 1 : y_shape[rank - 2];
    const int64_t rows = TensorShape(y_shape).Size() / n;
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t col = 0; col < n; ++col) {
        float sum = 0;
        for (int64_t j = 0; j < k; ++j) {
          const int64_t ai = transpose ? (row / m) * m * k + j * m + row % m : row * k + j;
          const int64_t bi = trans_b ? col * k + j : j * n + col;
          sum += static_cast<float>(a[ai]) * static_cast<float>(b[bi]);
        }
        float result = sum * alpha;
        if (c_shape && beta != 0) {
          int64_t ci = 0, stride = 1, flat = row * n + col;
          for (size_t d = y_shape.size(); d-- > 0;) {
            if (d + c_shape->size() >= y_shape.size()) {
              const auto dim = (*c_shape)[d + c_shape->size() - y_shape.size()];
              if (dim != 1) ci += flat % y_shape[d] * stride;
              stride *= dim;
            }
            flat /= y_shape[d];
          }
          result += beta * static_cast<float>(c[ci]);
        }
        expected.push_back(static_cast<T>(result));
      }
    }
    OpTester test("Gemm", 29);
    test.AddAttribute("transA", trans_a);
    test.AddAttribute("transB", trans_b);
    test.AddAttribute("alpha", alpha);
    test.AddAttribute("beta", beta);
    test.AddInput<T>("A", a_shape, a);
    test.AddInput<T>("B", trans_b ? std::vector<int64_t>{n, k} : std::vector<int64_t>{k, n}, b, constant_b);
    if (c_shape) test.AddInput<T>("C", *c_shape, c);
    test.AddOutput<T>("Y", y_shape, expected);
    Run(test);
  }
};

TEST_P(Matrix29Test, GemmVectorAndHigherRank) {
  const std::vector<int64_t> scalar{}, columns{5}, batch_bias{2, 1, 5}, partial_bias{1, 3, 1};
  for (int64_t trans_a : {int64_t{0}, int64_t{-1}, int64_t{2}}) {
    for (int64_t trans_b : {int64_t{0}, int64_t{2}}) {
      GemmCase<float>({4}, trans_a, trans_b, &columns, true);
      GemmCase<double>({4}, trans_a, trans_b, &scalar);
      GemmCase<float>({2, 3, 4}, trans_a, trans_b, &batch_bias, true);
      GemmCase<float>({2, 3, 4}, trans_a, trans_b, nullptr);
      GemmCase<float>({2, 3, 4}, trans_a, trans_b, &scalar);
    }
  }
  GemmCase<float>({2, 3, 4}, 0, 0, &partial_bias);
  const std::vector<int64_t> rank4_bias{2, 1, 4, 1};
  GemmCase<float>({2, 3, 4, 6}, 0, 0, &rank4_bias, true);
  const std::vector<int64_t> full_bias{2, 3, 5};
  GemmCase<float>({2, 3, 4}, 0, 0, &full_bias, true);
  GemmCase<float>({2, 4, 3}, 1, 1, &full_bias);
}

TEST_P(Matrix29Test, GemmHalf) {
  const std::vector<int64_t> bias{2, 1, 5};
  GemmCase<MLFloat16>({2, 3, 4}, 0, 0, &bias);
  GemmCase<MLFloat16>({2, 3, 4}, 1, 1, &bias);
  const std::vector<int64_t> columns{1, 1, 5};
  GemmCase<MLFloat16>({2, 3, 4}, 0, 0, &columns, false, 1.0f, 1.0f);
}

TEST_P(Matrix29Test, GemmEmptyAndZeroK) {
  const std::vector<int64_t> bias{5};
  GemmCase<float>({2, 0, 4}, 0, 0, &bias);
  GemmCase<float>({0, 3, 4}, 1, 0, &bias);
  GemmCase<float>({2, 3, 0}, 0, 0, &bias, true);
  GemmCase<float>({2, 0, 3}, 1, 1, &bias);
  GemmCase<float>({0}, 2, 0, &bias);
  GemmCase<float>({2, 3, 0}, 0, 0, nullptr);
}

TEST_P(Matrix29Test, GemmRejectsBiasEvenWhenBetaZero) {
  OpTester test("Gemm", 29);
  test.AddAttribute("beta", 0.0f);
  test.AddInput<float>("A", {2}, {1, 2});
  test.AddInput<float>("B", {2, 3}, {1, 2, 3, 4, 5, 6});
  test.AddInput<float>("C", {1, 3}, {1, 2, 3});
  test.AddOutput<float>("Y", {3}, {0, 0, 0});
  Run(test, "Remove excess leading dimensions");
}

TEST_P(Matrix29Test, GemmRejectsInvalidRanksAndContraction) {
  for (int scenario = 0; scenario < 3; ++scenario) {
    OpTester test("Gemm", 29);
    test.AddInput<float>("A", scenario == 0 ? std::vector<int64_t>{} : std::vector<int64_t>{1}, {1});
    test.AddInput<float>("B", scenario == 1 ? std::vector<int64_t>{1} : std::vector<int64_t>{2, 1},
                         scenario == 1 ? std::vector<float>{1} : std::vector<float>{1, 2});
    test.AddOutput<float>("Y", {1}, {0});
    Run(test, scenario == 2 ? "dimension mismatch" : "Reshape the inputs to the required ranks");
  }
}

TEST_P(Matrix29Test, GemmCudaBFloat16) {
  if (!GetParam() || !HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA with bfloat16 support";
  const std::vector<int64_t> bias{2, 1, 5};
  GemmCase<BFloat16>({2, 3, 4}, 0, 0, &bias);
  GemmCase<BFloat16>({2, 3, 4}, 1, 1, &bias);
}

TEST_P(Matrix29Test, GroupedSelectionsTransposeAndTypes) {
  for (int64_t trans_b : {int64_t{0}, int64_t{-1}, int64_t{2}}) {
    for (bool with_bias : {false, true}) {
      auto run = [&]<typename T, typename Index>(int64_t tokens) {
        OpTester test("GroupedMatMul", 29);
        test.AddAttribute("transB", trans_b);
        std::vector<T> weights;
        for (int i = 0; i < 18; ++i) weights.push_back(static_cast<T>(static_cast<float>(i)));
        const std::array<float, 6> input_pattern{1, 2, -1, 3, 0, 4};
        const std::array<Index, 6> index_pattern{2, 0, 2, 1, 2, 0};
        std::vector<T> input;
        std::vector<Index> indices;
        for (int64_t i = 0; i < tokens * 3; ++i) {
          input.push_back(static_cast<T>(input_pattern[i % 6]));
          indices.push_back(index_pattern[i % 6]);
        }
        const std::vector<T> bias{T(0.5f), T(1.0f), T(1.5f), T(-2.0f), T(3.0f), T(0.25f)};
        std::vector<T> expected;
        for (size_t i = 0; i < indices.size(); ++i) {
          for (int64_t col = 0; col < 2; ++col) {
            float value = with_bias ? static_cast<float>(bias[indices[i] * 2 + col]) : 0;
            for (int64_t k = 0; k < 3; ++k) {
              value += static_cast<float>(input[(i / 3) * 3 + k]) *
                       static_cast<float>(weights[indices[i] * 6 + (trans_b ? col * 3 + k : k * 2 + col)]);
            }
            expected.push_back(static_cast<T>(value));
          }
        }
        test.AddInput<T>("input", {tokens, 3}, input);
        test.AddInput<T>("weights", trans_b ? std::vector<int64_t>{3, 2, 3} : std::vector<int64_t>{3, 3, 2}, weights);
        test.AddInput<Index>("group_indices", {tokens, 3}, indices);
        if (with_bias) test.AddInput<T>("bias", {3, 2}, bias);
        test.AddOutput<T>("output", {tokens, 3, 2}, expected);
        Run(test);
      };
      for (int64_t tokens : {int64_t{2}, int64_t{33}}) {
        run.template operator()<float, int32_t>(tokens);
        run.template operator()<double, int64_t>(tokens);
        run.template operator()<MLFloat16, int64_t>(tokens);
        if (GetParam() && HasCudaEnvironment(800)) run.template operator()<BFloat16, int32_t>(tokens);
      }
    }
  }
}

TEST_P(Matrix29Test, GroupedZeroKAndEmpty) {
  for (bool trans_b : {false, true}) {
    for (bool with_bias : {false, true}) {
      OpTester test("GroupedMatMul", 29);
      test.AddAttribute("transB", static_cast<int64_t>(trans_b));
      test.AddInput<float>("input", {2, 0}, {});
      test.AddInput<float>("weights", trans_b ? std::vector<int64_t>{2, 3, 0} : std::vector<int64_t>{2, 0, 3}, {});
      test.AddInput<int64_t>("group_indices", {2, 1}, {1, 0});
      if (with_bias) test.AddInput<float>("bias", {2, 3}, {1, 2, 3, 4, 5, 6});
      test.AddOutput<float>("output", {2, 1, 3}, with_bias ? std::vector<float>{4, 5, 6, 1, 2, 3} : std::vector<float>(6, 0));
      Run(test);
    }
    for (const auto& dims : {std::vector<int64_t>{0, 2}, std::vector<int64_t>{2, 0}}) {
      OpTester test("GroupedMatMul", 29);
      test.AddAttribute("transB", static_cast<int64_t>(trans_b));
      test.AddInput<float>("input", {dims[0], 3}, std::vector<float>(dims[0] * 3));
      test.AddInput<float>("weights", trans_b ? std::vector<int64_t>{2, 4, 3} : std::vector<int64_t>{2, 3, 4},
                           std::vector<float>(24));
      test.AddInput<int32_t>("group_indices", dims, {});
      test.AddOutput<float>("output", {dims[0], dims[1], 4}, {});
      Run(test);
    }
  }
}

TEST_P(Matrix29Test, GroupedRejectsInvalidIndicesEvenWhenNZero) {
  for (int64_t index : {int64_t{-1}, int64_t{2}}) {
    OpTester test("GroupedMatMul", 29);
    test.AddInput<float>("input", {1, 3}, {1, 2, 3});
    test.AddInput<float>("weights", {2, 3, 0}, {});
    test.AddInput<int64_t>("group_indices", {1, 1}, {index});
    test.AddOutput<float>("output", {1, 1, 0}, {});
    Run(test, "Supply a valid nonnegative group index");
  }
}

TEST_P(Matrix29Test, GroupedZeroKDoesNotAllocatePerUnusedGroup) {
  OpTester test("GroupedMatMul", 29);
  test.AddInput<float>("input", {2, 0}, {});
  test.AddInput<float>("weights", {1000000000, 0, 3}, {});
  test.AddInput<int32_t>("group_indices", {2, 1}, {0, 999999999});
  test.AddOutput<float>("output", {2, 1, 3}, std::vector<float>(6, 0));
  Run(test);
}

TEST_P(Matrix29Test, GroupedRejectsInvalidShapes) {
  for (int scenario = 0; scenario < 3; ++scenario) {
    OpTester test("GroupedMatMul", 29);
    test.AddInput<float>("input", {1, 2}, {1, 2});
    test.AddInput<float>("weights", scenario == 0 ? std::vector<int64_t>{1, 1, 2} : std::vector<int64_t>{1, 2, 1},
                         {1, 2});
    test.AddInput<int32_t>("group_indices", scenario == 1 ? std::vector<int64_t>{2, 1} : std::vector<int64_t>{1, 1},
                           scenario == 1 ? std::vector<int32_t>{0, 0} : std::vector<int32_t>{0});
    if (scenario == 2) test.AddInput<float>("bias", {1}, {1});
    test.AddOutput<float>("output", {1, 1, 1}, {0});
    Run(test, scenario == 2 ? "Reshape the per-group bias" : "match the contraction dimension K");
  }
}

INSTANTIATE_TEST_SUITE_P(CpuAndCuda, Matrix29Test, testing::Bool());

}  // namespace
}  // namespace test
}  // namespace onnxruntime
