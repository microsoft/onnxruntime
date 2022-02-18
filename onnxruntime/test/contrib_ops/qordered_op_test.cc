// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #if defined(USE_CUDA) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "contrib_ops/cpu/bert/longformer_attention_base.h"

#include <numeric>
#include <functional>
#include <iostream>
#include <math.h>

namespace onnxruntime {
namespace test {

enum OrderCublasLt {
  ORDER_COL = 0,
  ORDER_ROW = 1,
  ORDER_COL32 = 2,
  ORDER_COL4_4R2_8C = 3,
  ORDER_COL32_2R_4R4 = 4
};

// generate random data without precision loss if quantized.
template <typename T>
static std::vector<T> GenData(std::vector<int64_t> const& shape, float scale) {
  int64_t n = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

  scale = std::is_same<T, int8_t>::value ? 1.0f : scale;  // using scale = 1.0f to generate int8_t data,
  std::vector<T> r(n);
  RandomValueGenerator random{};
  std::vector<int> tmp = random.Uniform<int32_t>(shape, -128, 127);
  for (int64_t i = 0; i < n; i++) {
    r[i] = static_cast<T>(tmp[i] * scale);
  }
  return r;
}

class OrderedIndex {
  OrderCublasLt order_;
  int64_t rows_;
  int64_t cols_;

 public:
  OrderedIndex(OrderCublasLt order, int64_t rows, int64_t cols)
      : order_(order), rows_(rows), cols_(cols) {}

  int64_t operator()(int64_t r, int64_t c);
};

int64_t OrderedIndex::operator()(int64_t r, int64_t c) {
  switch (order_) {
    case ORDER_ROW:
      return r * cols_ + c;
    case ORDER_COL:
      return c * rows_ + r;
    case ORDER_COL32: {
      int64_t tile_id = c / 32;
      int64_t tile_stride = 32 * rows_;
      return tile_id * tile_stride + r * 32 + (c % 32);
    }
    case ORDER_COL4_4R2_8C: {
      int64_t tiles_c = c / 32;
      int64_t tiles_r = r / 8;
      int64_t tile_idx = tiles_c * (rows_ / 8) + tiles_r;
      int64_t offset = tile_idx * (32 * 8);
      offset += (r & 0x1) * (32 * 4);
      int64_t in_4x4x8_tile_c = c % 32;
      int64_t in_4x4x8_tile_r = (r % 8) / 2;
      int64_t in_4x4x8_idx = (in_4x4x8_tile_c / 4) * (4 * 4) + in_4x4x8_tile_r * 4 + (in_4x4x8_tile_c % 4);
      offset += in_4x4x8_idx;
      return offset;
    }
    case ORDER_COL32_2R_4R4: {
      // TODO:
    }
    default:
      return 0;
  }
}

void BatchRowColFromShape(std::vector<int64_t> const& shape, int64_t& batch, int64_t& rows, int64_t& cols) {
  cols = shape.back();
  rows = (shape.size() > 1 ? shape[shape.size() - 2] : 1LL);
  batch = (shape.size() <= 2)
              ? 1LL
              : std::accumulate(shape.data(), shape.data() + (shape.size() - 2), 1LL, std::multiplies<int64_t>());
}

template <typename TSrc, typename TDst>
static std::vector<TDst> ReorderAndTransform(const std::vector<int64_t>& shape, const std::vector<TSrc>& src,
                                             OrderCublasLt order_from, OrderCublasLt order_to,
                                             std::function<TDst(TSrc)> transform) {
  int64_t cols = 0, rows = 0, batch = 0;
  BatchRowColFromShape(shape, batch, rows, cols);

  OrderedIndex src_indexer(order_from, rows, cols);
  OrderedIndex dst_indexer(order_to, rows, cols);

  std::vector<TDst> dst(batch * cols * rows);
  const TSrc* bsrc = src.data();
  TDst* bdst = dst.data();
  for (int64_t b = 0, batch_stride = rows * cols; b < batch; b++) {
    for (int64_t r = 0; r < rows; r++) {
      for (int64_t c = 0; c < cols; c++) {
        int64_t src_idx = src_indexer(r, c);
        int64_t dst_idx = dst_indexer(r, c);
        if (src_idx >= batch_stride || dst_idx >= batch_stride) {
          throw std::runtime_error("Out of bound index calculated, error found in OrderedIndexer");
        }
        bdst[dst_idx] = transform(bsrc[src_idx]);
      }
    }
    bsrc += batch_stride;
    bdst += batch_stride;
  }
  return dst;
}

template <typename T>
static std::vector<int8_t> QuantizeTransform(std::vector<int64_t> const& shape, float scale,
                                             const std::vector<T>& src, OrderCublasLt order) {
  return ReorderAndTransform<T, int8_t>(shape, src, ORDER_ROW, order,
                                        [scale](T source_value) -> int8_t {
                                          float v = (float)source_value / scale;
                                          v = std::max(-128.0f, v);
                                          v = std::min(127.0f, v);
                                          return static_cast<int8_t>(std::nearbyintf(v));
                                        });
}

template <typename T>
static std::vector<T> DequantizeTransform(std::vector<int64_t> const& shape, float scale,
                                          const std::vector<int8_t>& src, OrderCublasLt order) {
  return ReorderAndTransform<int8_t, T>(shape, src, order, ORDER_ROW,
                                        [scale](int8_t source_value) -> T { return T(scale * float(source_value)); });
}

template <typename T>
static std::vector<T> Reorder(std::vector<int64_t> const& shape, const std::vector<T>& src,
                              OrderCublasLt order_from, OrderCublasLt order_to) {
  return ReorderAndTransform<T, T>(shape, src, order_from, order_to, [](T v) -> T { return v; });
}

template <typename T>
static void RunQOrdered_Quantize_Test(
    std::vector<T> const& fvec,
    std::vector<int64_t> const& shape,
    OrderCublasLt order_q,
    float scale) {
  auto qvec = QuantizeTransform(shape, scale, fvec, order_q);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  OpTester test_qorder("QuantizeWithOrder", 1, onnxruntime::kMSDomain);
  test_qorder.AddAttribute("order_input", (int64_t)ORDER_ROW);
  test_qorder.AddAttribute("order_output", (int64_t)order_q);
  test_qorder.AddInput<T>("input", shape, fvec);
  test_qorder.AddInput<float>("scale_input", {}, {scale});
  test_qorder.AddOutput("output", shape, qvec, false, 0.0f, 0.0f);
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QOrderedTest, FP32_Quantize_COL) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<float> fvec = GenData<float>(shape, scale * 0.7685f);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL, scale);
}

TEST(QOrderedTest, FP32_Quantize_ROW) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<float> fvec = GenData<float>(shape, scale * 0.7685f);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_ROW, scale);
}

TEST(QOrderedTest, FP32_Quantize_COL32) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<float> fvec = GenData<float>(shape, scale * 0.7685f);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL32, scale);
}

TEST(QOrderedTest, FP32_Quantize_COL4_4R2_8C) {
  std::vector<int64_t> shape = {3, 8 * 3, 32 * 2};
  float scale(2.0f);
  std::vector<float> fvec = GenData<float>(shape, scale * 0.7685f);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL4_4R2_8C, scale);
}

TEST(QOrderedTest, FP16_Quantize_COL) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<MLFloat16> fvec = GenData<MLFloat16>(shape, scale * 0.7685f);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL, scale);
}
TEST(QOrderedTest, FP16_Quantize_ROW) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<MLFloat16> fvec = GenData<MLFloat16>(shape, scale * 0.7685f);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_ROW, scale);
}

TEST(QOrderedTest, FP16_Quantize_COL32) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<MLFloat16> fvec = GenData<MLFloat16>(shape, scale * 0.7685f);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL32, scale);
}

TEST(QOrderedTest, FP16_Quantize_COL4_4R2_8C) {
  std::vector<int64_t> shape = {3, 8 * 3, 32 * 2};
  float scale = 2.0f;
  std::vector<MLFloat16> fvec = GenData<MLFloat16>(shape, scale * 0.7685f);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL4_4R2_8C, scale);
}

template <typename T>
static void RunQOrdered_Dequantize_Test(
    std::vector<int8_t> const& qvec,
    std::vector<int64_t> const& shape,
    OrderCublasLt order_q,
    float scale) {
  auto fvec = DequantizeTransform<T>(shape, scale, qvec, order_q);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  OpTester test_qorder("DequantizeWithOrder", 1, onnxruntime::kMSDomain);
  test_qorder.AddAttribute("to", (int64_t)(std::is_same<T, float>::value ? onnx::TensorProto_DataType_FLOAT : onnx::TensorProto_DataType_FLOAT16));
  test_qorder.AddAttribute("order_input", (int64_t)order_q);
  test_qorder.AddAttribute("order_output", (int64_t)ORDER_ROW);
  test_qorder.AddInput<int8_t>("input", shape, qvec);
  test_qorder.AddInput<float>("scale_input", {}, {scale});
  test_qorder.AddOutput("output", shape, fvec);
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Dequantize only work for ORDER_COL32 and ORDER_ROW input
TEST(QOrderedTest, FP32_Dequantize_COL32) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<int8_t> qvec = GenData<int8_t>(shape, 1.0f);
  RunQOrdered_Dequantize_Test<float>(qvec, shape, ORDER_COL32, scale);
}

TEST(QOrderedTest, FP32_Dequantize_ROW) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<int8_t> qvec = GenData<int8_t>(shape, 1.0f);
  RunQOrdered_Dequantize_Test<float>(qvec, shape, ORDER_ROW, scale);
}

TEST(QOrderedTest, FP16_Dequantize_COL32) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<int8_t> qvec = GenData<int8_t>(shape, 1.0f);
  RunQOrdered_Dequantize_Test<MLFloat16>(qvec, shape, ORDER_COL32, scale);
}

TEST(QOrderedTest, FP16_Dequantize_ROW) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<int8_t> qvec = GenData<int8_t>(shape, 1.0f);
  RunQOrdered_Dequantize_Test<MLFloat16>(qvec, shape, ORDER_ROW, scale);
}

static void RunQOrdered_MatMul_Test(
    std::vector<int64_t> const& shapeA,
    std::vector<int64_t> const& shapeB,
    std::vector<int64_t> const& shapeY,
    OrderCublasLt order_weight,
    float scaleA, float scaleB, float scaleC, float scaleY,
    bool add_bias = false, bool broadcast_c_batch = false) {
  scaleA = MLFloat16(scaleA).ToFloat();
  scaleB = MLFloat16(scaleB).ToFloat();
  scaleC = MLFloat16(scaleC).ToFloat();
  scaleY = MLFloat16(scaleY).ToFloat();
  int64_t nY = std::accumulate(shapeY.begin(), shapeY.end(), int64_t{1LL}, std::multiplies<int64_t>());
  std::vector<int8_t> vecA = GenData<int8_t>(shapeA, 1.0f);
  std::vector<int8_t> vecB = GenData<int8_t>(shapeB, 1.0f);
  std::vector<int8_t> vecY(nY);

  OrderCublasLt real_order_A = (order_weight == ORDER_COL ? ORDER_ROW : ORDER_COL32);
  OrderCublasLt real_order_w = (order_weight == ORDER_COL ? ORDER_ROW : order_weight);

  int64_t colsA = 0, rowsA = 0, batchA = 0;
  BatchRowColFromShape(shapeA, batchA, rowsA, colsA);
  OrderedIndex indexerA(real_order_A, rowsA, colsA);

  int64_t colsB = 0, rowsB = 0, batchB = 0;
  BatchRowColFromShape(shapeB, batchB, rowsB, colsB);
  OrderedIndex indexerB(real_order_w, colsB, rowsB);  // B need Transpose

  int64_t colsY = 0, rowsY = 0, batchY = 0;
  BatchRowColFromShape(shapeY, batchY, rowsY, colsY);
  OrderedIndex indexerY(real_order_A, rowsY, colsY);

  std::vector<int64_t> shapeBias = {colsY};
  std::vector<float> vecBias;
  if (add_bias) { // make bias not too big
    float scaleBias = MLFloat16(scaleY / 27.0f).ToFloat();
    vecBias = GenData<float>(shapeBias, scaleBias);
  }
  std::vector<int8_t> vecC;
  std::vector<int64_t> shapeC = {broadcast_c_batch ? 1 : batchY, rowsY, colsY};
  if (scaleC != 0.0f) {
    vecC = GenData<int8_t>(shapeC, 1.0f);
  }

  // TODO: brocasting higher dims
  float alpha = scaleA * scaleB / scaleY;
  float beta = scaleC / scaleY;
  int8_t const* A = vecA.data();
  int8_t const* B = vecB.data();
  int8_t* Y = vecY.data();
  int8_t* C = vecC.data();
  for (int64_t b = 0; b < batchY; b++) {
    for (int64_t m = 0; m < rowsA; m++) {
      for (int64_t n = 0; n < colsB; n++) {
        int32_t isum = 0;
        for (int64_t k = 0; k < colsA; k++) {
          auto posA = indexerA(m, k);
          auto posB = indexerB(n, k);  // Transpose B
          isum += (A[posA] * B[posB]);
        }
        float sum = alpha * isum;
        if (add_bias) {
          sum += vecBias[n];
        }
        auto posY = indexerY(m, n);
        if (scaleC != 0.0f) {
          sum += beta * C[posY];
        }
        Y[posY] = static_cast<int8_t>((int)std::nearbyintf(std::min(127.0f, std::max(-128.0f, sum))));
      }
    }
    A += (batchA <= 1 ? int64_t{0} : (rowsA * colsA));
    B += (batchB <= 1 ? int64_t{0} : (rowsB * colsB));
    Y += (batchY <= 1 ? int64_t{0} : (rowsY * colsY));
    C += (shapeC[0] == 1 ? int64_t{0} : (rowsY * colsY));
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  OpTester test_qorder("QOrderedMatMul", 1, onnxruntime::kMSDomain);
  test_qorder.AddAttribute("order_B", (int64_t)order_weight);
  test_qorder.AddAttribute("order_A", (int64_t)real_order_A);
  test_qorder.AddAttribute("order_Y", (int64_t)real_order_A);
  test_qorder.AddInput<int8_t>("A", shapeA, vecA);
  test_qorder.AddInput<float>("scale_A", {}, {scaleA});
  test_qorder.AddInput<int8_t>("B", shapeB, vecB);
  test_qorder.AddInput<float>("scale_B", {}, {scaleB});
  test_qorder.AddInput<float>("scale_Y", {}, {scaleY});
  if (add_bias) {
    test_qorder.AddInput<float>("bias", shapeBias, vecBias);
  } else {
    test_qorder.AddOptionalInputEdge<float>();
  }

  if (scaleC != 0.0f) {
    test_qorder.AddInput<int8_t>("C", shapeC, vecC);
    test_qorder.AddInput<float>("scale_C", {}, {scaleC});
  }
  test_qorder.AddOutput<int8_t>("Y", shapeY, vecY, false, 0.0f, 0.0f /* abs error */);
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QOrderedTest, MatMul_COL4_4R2_8C_16x32x32) {
  std::vector<int64_t> shapeA = {16, 32};
  std::vector<int64_t> shapeB = {32, 32};
  std::vector<int64_t> shapeY = {16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_COL4_4R2_8C_16x32x32) {
  std::vector<int64_t> shapeA = {16, 32};
  std::vector<int64_t> shapeB = {32, 32};
  std::vector<int64_t> shapeY = {16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_COL4_4R2_8C_16x32x32_b2_1) {
  std::vector<int64_t> shapeA = {2, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 32};
  std::vector<int64_t> shapeY = {2, 16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_COL4_4R2_8C_16x32x32_b2_1) {
  std::vector<int64_t> shapeA = {2, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 32};
  std::vector<int64_t> shapeY = {2, 16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_broadcastC_COL4_4R2_8C_16x32x32_b2_1) {
  std::vector<int64_t> shapeA = {2, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 32};
  std::vector<int64_t> shapeY = {2, 16, 32};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL4_4R2_8C,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, true /* broadcast batch c */);
}

///////////////////////////////////////////////////////////////////////////////////////////
// QOrderMatMul with weight order using ORDER_COL
///////////////////////////////////////////////////////////////////////////////////////////
TEST(QOrderedTest, MatMul_COL_16x64x32) {
  std::vector<int64_t> shapeA = {16, 32};
  std::vector<int64_t> shapeB = {32, 64};
  std::vector<int64_t> shapeY = {16, 64};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_bias_COL_16x64x32) {
  std::vector<int64_t> shapeA = {16, 32};
  std::vector<int64_t> shapeB = {32, 64};
  std::vector<int64_t> shapeY = {16, 64};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          true /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_COL_16x64x32) {
  std::vector<int64_t> shapeA = {16, 32};
  std::vector<int64_t> shapeB = {32, 64};
  std::vector<int64_t> shapeY = {16, 64};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 4.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_bias_addC_COL_16x64x32) {
  std::vector<int64_t> shapeA = {16, 32};
  std::vector<int64_t> shapeB = {32, 64};
  std::vector<int64_t> shapeY = {16, 64};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 4.0f /*scaleC*/, 2.0f,
                          true /* add bias */, true /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_COL_16x64x32_b3_1) {
  std::vector<int64_t> shapeA = {3, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 64};
  std::vector<int64_t> shapeY = {3, 16, 64};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_bias_COL_16x64x32_b2_1) {
  std::vector<int64_t> shapeA = {2, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 64};
  std::vector<int64_t> shapeY = {2, 16, 64};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          true /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_COL_16x64x32_b2_1) {
  std::vector<int64_t> shapeA = {2, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 64};
  std::vector<int64_t> shapeY = {2, 16, 64};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 4.0f /*scaleC*/, 2.0f,
                          false /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_broadcastC_COL_16x64x32_b2_1) {
  std::vector<int64_t> shapeA = {2, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 64};
  std::vector<int64_t> shapeY = {2, 16, 64};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          false /* add bias */, true /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_addC_bias_COL_16x64x32_b2_1) {
  std::vector<int64_t> shapeA = {2, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 64};
  std::vector<int64_t> shapeY = {2, 16, 64};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          true /* add bias */, false /* broadcast batch c */);
}

TEST(QOrderedTest, MatMul_bias_addC_broadcastC_COL_16x64x32_b2_1) {
  std::vector<int64_t> shapeA = {2, 16, 32};
  std::vector<int64_t> shapeB = {1, 32, 64};
  std::vector<int64_t> shapeY = {2, 16, 64};
  RunQOrdered_MatMul_Test(shapeA, shapeB, shapeY, ORDER_COL,
                          1.0f / 32.0f, 1.0f / 32.0f, 0.0f /*scaleC*/, 2.0f,
                          true /* add bias */, true /* broadcast batch c */);
}

template <typename T> // MLFloat16 or float
static void
RunQOrdered_LayerNorm_WithData(std::vector<int64_t> const& shape, int axis, float epsilon, OrderCublasLt order,
                               const std::vector<int8_t>& vecX, float scale_x,
                               const std::vector<T>& vecGamma, const std::vector<T>* vecBeta,
                               float scale_y, const std::vector<int8_t>& vecY) {
  std::vector<int64_t> bias_shape = {shape.back()};
  OpTester test_qorder("QOrderedLayerNormalization", 1, onnxruntime::kMSDomain);
  test_qorder.AddAttribute("axis", (int64_t)axis);
  test_qorder.AddAttribute("epsilon", epsilon);
  test_qorder.AddAttribute("order_X", (int64_t)order);
  test_qorder.AddAttribute("order_Y", (int64_t)order);
  test_qorder.AddInput<int8_t>("X", shape, vecX);
  test_qorder.AddInput<float>("scale_X", {}, {scale_x});
  test_qorder.AddInput<T>("scale", bias_shape, vecGamma);
  if (vecBeta) {
    test_qorder.AddInput<T>("B", bias_shape, *vecBeta);
  } else {
    test_qorder.AddOptionalInputEdge<T>();
  }
  test_qorder.AddInput<float>("scale_Y", {}, {scale_y});
  test_qorder.AddOutput<int8_t>("Y", shape, vecY, false, 0.0f, 0.0f /* abs error */);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

static void
RunQOrdered_LayerNorm_Test(std::vector<int64_t> const& shape, int axis, bool has_bias, OrderCublasLt order) {
  int64_t N = std::accumulate(shape.begin(), shape.end(), int64_t{1LL}, std::multiplies<int64_t>());
  int64_t cols = 0, rows = 0, batch = 0;
  BatchRowColFromShape(shape, batch, rows, cols);

  std::vector<int64_t> bias_shape = {cols};
  std::vector<int8_t> vecX = GenData<int8_t>(shape, 1.0f);
  RandomValueGenerator random{};
  std::vector<MLFloat16> vecGamma = ToFloat16(random.Uniform<float>(bias_shape, -1.0f, 1.0f));
  std::vector<MLFloat16> vecBeta(cols);
  if (has_bias) {
    vecBeta = ToFloat16(random.Uniform<float>(bias_shape, -1.0f / 16.0f, -1.0f / 16.0f));
  }
  std::vector<int8_t> vecY(N);
  float scale_x = 2.0f;
  float scale_y = 2.0f / 128.0f;
  float epsilon = 0.00001f;

  const int8_t* bsrc = vecX.data();
  int8_t* bdst = vecY.data();
  for (int b = 0; b < batch; b++) {
    for (int r = 0; r < rows; r++) {
      // Var(X)=E[X*X]− E[X]* E[X]
      int64_t sum_x = std::accumulate(bsrc, bsrc + cols, 0LL, std::plus<int64_t>());
      int64_t sum_x2 = std::accumulate(bsrc, bsrc + cols, 0LL, [](int64_t s, int64_t v) { return s + v * v; });
      float u_x_scaled = scale_x * static_cast<float>(sum_x) / cols; // no precision lost in static_cast<float>(sum_x)
      float var_x_scaled = static_cast<float>(((double)sum_x2 - (double)sum_x * sum_x / cols) / cols) * scale_x * scale_x;
      float var_episilon = var_x_scaled + epsilon;
      float rsqrt_var = 1.0f / ::sqrtf(var_episilon);
      for (int c = 0; c < cols; c++) {
        float v = (scale_x * static_cast<float>(bsrc[c])) - u_x_scaled;
        v = (v * rsqrt_var * vecGamma[c].ToFloat()) + (has_bias ? vecBeta[c].ToFloat() : 0.0f);
        v = v / scale_y;
        v = std::max(-128.0f, std::min(v, 127.0f));
        bdst[c] = static_cast<int8_t>((int)std::nearbyintf(v));
      }
      bsrc += cols;
      bdst += cols;
    }
  }
  if (order == ORDER_COL32) {
    vecX = Reorder(shape, vecX, ORDER_ROW, order);
    vecY = Reorder(shape, vecY, ORDER_ROW, order);
  }

  RunQOrdered_LayerNorm_WithData(shape, axis, epsilon, order, vecX, scale_x, vecGamma, has_bias ? &vecBeta : nullptr, scale_y, vecY);
}

TEST(QOrderedTest, LayerNorm_Data_1x32) {
  float scale_x = 1.0;
  std::vector<int8_t> vecX = {
      -103, -24, -11, 125, 103, -117, 44, -109, 27, 122, 113,
      0, -7, 26, 110, -34, 28, -61, -107, -35, -54, 69,
      -3, -58, -119, 61, -19, -115, 35, 76, 16, 54};
  std::vector<float> vecGamma32 = {
      0.9536133f, 0.9160156f, 1.3310547f, 1.0068359f, 0.8095703f,
      1.0126953f, 0.8876953f, 1.4316406f, 1.0947266f, 1.0498047f,
      0.7373047f, 1.0615234f, 1.015625f, 1.0751953f, 1.0068359f,
      1.0908203f, 1.2011719f, 1.1962891f, 0.91796875f, 1.0947266f,
      1.3183594f, 1.0185547f, 1.0791016f, 1.0273438f, 0.8364258f,
      0.94873047f, 1.0292969f, 1.09375f, 1.0371094f, 1.1240234f,
      1.4384766f, 1.0068359f};
  std::vector<float> vecBeta32 = {
      0.01411438f, 0.18273926f, -0.12414551f, 0.09887695f, -0.1114502f,
      -0.08227539f, -0.04598999f, -0.11322021f, 0.12731934f, -0.06591797f,
      -0.00662994f, 0.04962158f, -0.04281616f, 0.07476807f, 0.23010254f,
      0.1036377f, 0.10852051f, 0.10919189f, -0.02905273f, -0.0512085f,
      -0.1194458f, 0.02661133f, 0.05789185f, -0.05239868f, 0.17907715f,
      -0.01765442f, -0.12255859f, -0.09729004f, 0.06591797f, 0.02258301f,
      -0.01844788f, -0.11999512f};
  std::vector<int8_t> vecY = {
      -84, -8, -22, 114, 64, -108, 30, -128, 33, 105, 71,
      2, -10, 28, 109, -26, 35, -57, -87, -37, -70, 61,
      0, -56, -75, 48, -26, -115, 35, 74, 17, 38};
  float scale_y = 1.0 / 64.0f;
  auto vecGamma = ToFloat16(vecGamma32);
  auto vecBeta = ToFloat16(vecBeta32);

  RunQOrdered_LayerNorm_WithData({1, 1, 32}, -1, 0.00001f, ORDER_ROW, vecX, scale_x, vecGamma, &vecBeta, scale_y, vecY);
  RunQOrdered_LayerNorm_WithData({1, 1, 32}, -1, 0.00001f, ORDER_COL32, vecX, scale_x, vecGamma, &vecBeta, scale_y, vecY);
  RunQOrdered_LayerNorm_WithData({1, 1, 32}, -1, 0.00001f, ORDER_ROW, vecX, scale_x, vecGamma32, &vecBeta32, scale_y, vecY);
  RunQOrdered_LayerNorm_WithData({1, 1, 32}, -1, 0.00001f, ORDER_COL32, vecX, scale_x, vecGamma32, &vecBeta32, scale_y, vecY);
}

TEST(QOrderedTest, LayerNorm_OrderRow_3x7x600) {
  RunQOrdered_LayerNorm_Test({3, 9, 80}, -1, true, ORDER_ROW);
}

TEST(QOrderedTest, LayerNorm_OrderRow_3x11x96) {
  RunQOrdered_LayerNorm_Test({3, 11, 96}, -1, true, ORDER_ROW);
}

TEST(QOrderedTest, LayerNorm_OrderRow_NoBias_3x11x96) {
  RunQOrdered_LayerNorm_Test({3, 11, 96}, -1, false, ORDER_ROW);
}

TEST(QOrderedTest, LayerNorm_OrderCol32_3x11x96) {
  RunQOrdered_LayerNorm_Test({3, 11, 96}, -1, true, ORDER_COL32);
}

TEST(QOrderedTest, LayerNorm_OrderCol32_NoBias_3x11x96) {
  RunQOrdered_LayerNorm_Test({3, 11, 96}, -1, false, ORDER_COL32);
}

static void
RunQOrdered_Gelu_Test(std::vector<int64_t> const& shape, float scale_X, float scale_Y, OrderCublasLt order) {
  static const float sqrt_of_2 = std::sqrt(2.0f);

  int64_t N = std::accumulate(shape.begin(), shape.end(), int64_t{1LL}, std::multiplies<int64_t>());
  std::vector<int8_t> vecX = GenData<int8_t>(shape, 1.0f);
  std::vector<int8_t> vecY(N);
  for (int64_t i = 0; i < N; i++) {
    float x = scale_X * (float)vecX[i];
    float r = (x * (0.5f * (1.0f + std::erff(x / sqrt_of_2)))) / scale_Y;
    int8_t q = static_cast<int8_t>((int)std::nearbyintf(std::min(127.0f, std::max(-128.0f, r)))); 
    vecY[i] = q;
  }

  OpTester test_qorder("QOrderedGelu", 1, onnxruntime::kMSDomain);
  test_qorder.AddAttribute("order_X", (int64_t)order);
  test_qorder.AddAttribute("order_Y", (int64_t)order);
  test_qorder.AddInput<int8_t>("X", shape, vecX);
  test_qorder.AddInput<float>("scale_X", {}, {scale_X});
  test_qorder.AddInput<float>("scale_Y", {}, {scale_Y});
  test_qorder.AddOutput<int8_t>("Y", shape, vecY, false, 0.0f, 0.0f /* abs error */);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QOrderedTest, Gelu_3x11x12) {
  RunQOrdered_Gelu_Test({3, 11, 12}, 1.0f / 32.0f, 1.0f/128.0f, ORDER_COL32);
  RunQOrdered_Gelu_Test({3, 11, 12}, 1.0f / 32.0f, 1.0f/128.0f, ORDER_ROW);
}

}  // namespace test
}  // namespace onnxruntime

// #endif
