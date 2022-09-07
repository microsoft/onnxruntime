// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qordered_test_utils.h"

namespace onnxruntime {
namespace test {

// Only the CUDA EP supports ordered quantized ops for now
#ifdef USE_CUDA

static void RunQOrdered_MatMul_Test(
    std::vector<int64_t> const& shape_A,
    std::vector<int64_t> const& shape_B,
    std::vector<int64_t> const& shape_Y,
    OrderCublasLt weight_order,
    float scale_A, float scale_B, float scale_C, float scale_Y,
    bool add_bias = false, bool broadcast_c_batch = false) {
  // Needs Turing or higher architecture
  if (NeedSkipIfCudaArchLowerThan(750)) {
    return;
  }

  scale_A = MLFloat16(scale_A).ToFloat();
  scale_B = MLFloat16(scale_B).ToFloat();
  scale_C = MLFloat16(scale_C).ToFloat();
  scale_Y = MLFloat16(scale_Y).ToFloat();
  int64_t num_elements_Y = std::accumulate(shape_Y.begin(), shape_Y.end(), int64_t{1LL}, std::multiplies<int64_t>());
  std::vector<int8_t> vec_A = GenData<int8_t>(shape_A, 1.0f);
  std::vector<int8_t> vec_B = GenData<int8_t>(shape_B, 1.0f);
  std::vector<int8_t> vec_Y(num_elements_Y);

  int64_t cols_A = 0, rows_A = 0, batch_A = 0;
  BatchRowColFromShape(shape_A, batch_A, rows_A, cols_A);
  OrderedIndex indexerA(ORDER_ROW, rows_A, cols_A);

  int64_t cols_B = 0, rows_B = 0, batch_B = 0;
  BatchRowColFromShape(shape_B, batch_B, rows_B, cols_B);
  OrderedIndex indexerB(ORDER_ROW, cols_B, rows_B);  // B need Transpose

  int64_t cols_Y = 0, rows_Y = 0, batch_Y = 0;
  BatchRowColFromShape(shape_Y, batch_Y, rows_Y, cols_Y);
  OrderedIndex indexerY(ORDER_ROW, rows_Y, cols_Y);

  std::vector<int64_t> bias_shape = {cols_Y};
  std::vector<float> bias;
  if (add_bias) {  // make bias not too big
    float bias_scale = MLFloat16(scale_Y / 27.0f).ToFloat();
    bias = GenData<float>(bias_shape, bias_scale);
  }
  std::vector<int8_t> vec_C;
  std::vector<int64_t> shape_C = {broadcast_c_batch ? 1 : batch_Y, rows_Y, cols_Y};
  if (scale_C != 0.0f) {
    vec_C = GenData<int8_t>(shape_C, 1.0f);
  }

  // TODO: broadcasting higher dims
  float alpha = scale_A * scale_B / scale_Y;
  float beta = scale_C / scale_Y;
  int8_t const* A = vec_A.data();
  int8_t const* B = vec_B.data();
  int8_t* Y = vec_Y.data();
  int8_t* C = vec_C.data();
  for (int64_t b = 0; b < batch_Y; b++) {
    for (int64_t m = 0; m < rows_A; m++) {
      for (int64_t n = 0; n < cols_B; n++) {
        int32_t isum = 0;
        for (int64_t k = 0; k < cols_A; k++) {
          auto posA = indexerA(m, k);
          auto posB = indexerB(n, k);  // Transpose B
          isum += (A[posA] * B[posB]);
        }
        float sum = alpha * isum;
        if (add_bias) {
          sum += bias[n];
        }
        auto posY = indexerY(m, n);
        if (scale_C != 0.0f) {
          sum += beta * C[posY];
        }
        Y[posY] = static_cast<int8_t>(std::nearbyintf(std::min(127.0f, std::max(-128.0f, sum))));
      }
    }
    A += (batch_A <= 1 ? int64_t{0} : (rows_A * cols_A));
    B += (batch_B <= 1 ? int64_t{0} : (rows_B * cols_B));
    Y += (batch_Y <= 1 ? int64_t{0} : (rows_Y * cols_Y));
    C += (shape_C[0] == 1 ? int64_t{0} : (rows_Y * cols_Y));
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  OpTester test_qorder("QOrderedMatMul", 1, onnxruntime::kMSDomain);
  test_qorder.AddAttribute("order_B", static_cast<int64_t>(weight_order));
  test_qorder.AddAttribute("order_A", static_cast<int64_t>(ORDER_ROW));
  test_qorder.AddAttribute("order_Y", static_cast<int64_t>(ORDER_ROW));
  test_qorder.AddInput<int8_t>("A", shape_A, vec_A);
  test_qorder.AddInput<float>("scale_A", {}, {scale_A});
  test_qorder.AddInput<int8_t>("B", shape_B, vec_B);
  test_qorder.AddInput<float>("scale_B", {}, {scale_B});
  test_qorder.AddInput<float>("scale_Y", {}, {scale_Y});
  if (add_bias) {
    test_qorder.AddInput<float>("bias", bias_shape, bias);
  } else {
    test_qorder.AddOptionalInputEdge<float>();
  }

  if (scale_C != 0.0f) {
    test_qorder.AddInput<int8_t>("C", shape_C, vec_C);
    test_qorder.AddInput<float>("scale_C", {}, {scale_C});
  }
  test_qorder.AddOutput<int8_t>("Y", shape_Y, vec_Y, false, 0.0f, 0.0f /* abs error */);
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QOrderedTest, MatMul_COL_16x64x32) {
  std::vector<int64_t> shape_A = {16, 32};
  std::vector<int64_t> shape_B = {32, 64};
  std::vector<int64_t> shape_Y = {16, 64};
  RunQOrdered_MatMul_Test(shape_A, shape_B, shape_Y, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_bias_COL_16x64x32) {
  std::vector<int64_t> shape_A = {16, 32};
  std::vector<int64_t> shape_B = {32, 64};
  std::vector<int64_t> shape_Y = {16, 64};
  RunQOrdered_MatMul_Test(shape_A, shape_B, shape_Y, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          true /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_COL_16x64x32) {
  std::vector<int64_t> shape_A = {16, 32};
  std::vector<int64_t> shape_B = {32, 64};
  std::vector<int64_t> shape_Y = {16, 64};
  RunQOrdered_MatMul_Test(shape_A, shape_B, shape_Y, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 4.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_bias_addC_COL_16x64x32) {
  std::vector<int64_t> shape_A = {16, 32};
  std::vector<int64_t> shape_B = {32, 64};
  std::vector<int64_t> shape_Y = {16, 64};
  RunQOrdered_MatMul_Test(shape_A, shape_B, shape_Y, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 4.0f /*scaleC*/, 2.0f,
                          true /* add bias */, true /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_COL_16x64x32_b3_1) {
  std::vector<int64_t> shape_A = {3, 16, 32};
  std::vector<int64_t> shape_B = {1, 32, 64};
  std::vector<int64_t> shape_Y = {3, 16, 64};
  RunQOrdered_MatMul_Test(shape_A, shape_B, shape_Y, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_bias_COL_16x64x32_b2_1) {
  std::vector<int64_t> shape_A = {2, 16, 32};
  std::vector<int64_t> shape_B = {1, 32, 64};
  std::vector<int64_t> shape_Y = {2, 16, 64};
  RunQOrdered_MatMul_Test(shape_A, shape_B, shape_Y, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          true /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_COL_16x64x32_b2_1) {
  std::vector<int64_t> shape_A = {2, 16, 32};
  std::vector<int64_t> shape_B = {1, 32, 64};
  std::vector<int64_t> shape_Y = {2, 16, 64};
  RunQOrdered_MatMul_Test(shape_A, shape_B, shape_Y, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 4.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_broadcastC_COL_16x64x32_b2_1) {
  std::vector<int64_t> shape_A = {2, 16, 32};
  std::vector<int64_t> shape_B = {1, 32, 64};
  std::vector<int64_t> shape_Y = {2, 16, 64};
  RunQOrdered_MatMul_Test(shape_A, shape_B, shape_Y, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, true /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_bias_COL_16x64x32_b2_1) {
  std::vector<int64_t> shape_A = {2, 16, 32};
  std::vector<int64_t> shape_B = {1, 32, 64};
  std::vector<int64_t> shape_Y = {2, 16, 64};
  RunQOrdered_MatMul_Test(shape_A, shape_B, shape_Y, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          true /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_bias_addC_broadcastC_COL_16x64x32_b2_1) {
  std::vector<int64_t> shape_A = {2, 16, 32};
  std::vector<int64_t> shape_B = {1, 32, 64};
  std::vector<int64_t> shape_Y = {2, 16, 64};
  RunQOrdered_MatMul_Test(shape_A, shape_B, shape_Y, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          true /* add bias */, true /* broadcast batch c */);
}

#endif

}  // namespace test
}  // namespace onnxruntime
