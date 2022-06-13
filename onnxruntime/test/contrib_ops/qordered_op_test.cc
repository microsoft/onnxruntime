#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4305)
#endif

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
static std::vector<T> GenData(std::vector<int64_t> const& shape, float scale,
                              int32_t min = -128, int32_t max = 127) {
  int64_t n = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

  scale = std::is_same<T, int8_t>::value ? 1.0f : scale;  // using scale = 1.0f to generate int8_t data,
  std::vector<T> r(n);
  RandomValueGenerator random{};
  std::vector<int> tmp = random.Uniform<int32_t>(shape, min, max);
  for (int64_t i = 0; i < n; i++) {
    r[i] = static_cast<T>(tmp[i] * scale);
  }
  return r;
}

template <>
static std::vector<float> GenData(std::vector<int64_t> const& shape, float scale,
                                  int32_t min, int32_t max) {
  int64_t n = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

  ORT_IGNORE_RETURN_VALUE(scale);
  std::vector<float> r(n);
  RandomValueGenerator random{};
  return random.Uniform<float>(shape, static_cast<float>(min), static_cast<float>(max));
}

template <>
static std::vector<MLFloat16> GenData(std::vector<int64_t> const& shape, float scale,
                                      int32_t min, int32_t max) {
  int64_t n = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

  ORT_IGNORE_RETURN_VALUE(scale);
  std::vector<MLFloat16> r(n);
  RandomValueGenerator random{};
  auto float_data = random.Uniform<float>(shape, static_cast<float>(min), static_cast<float>(max));

  std::vector<MLFloat16> fp16_data;
  fp16_data.reserve(float_data.size());

  for (auto e : float_data) {
    fp16_data.emplace_back(e);
  }

  return fp16_data;
}

template <typename T>
static std::vector<T> GenZerosData(std::vector<int64_t> const& shape) {
  int64_t n = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

  std::vector<T> r(n);
  for (int64_t i = 0; i < n; i++) {
    r[i] = static_cast<T>(0.f);
  }
  return r;
}

template <typename T>
static std::vector<T> GenOnesData(std::vector<int64_t> const& shape) {
  int64_t n = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());

  std::vector<T> r(n);
  for (int64_t i = 0; i < n; i++) {
    r[i] = static_cast<T>(1.f);
  }
  return r;
}

static std::vector<MLFloat16> CastFloatToFp16(const std::vector<float>& float_data) {
  std::vector<MLFloat16> fp16_vector;
  fp16_vector.reserve(float_data.size());

  for (auto e : float_data) {
    fp16_vector.emplace_back(e);
  }

  return fp16_vector;
}

static std::vector<float> CastFp16ToFloat(const std::vector<MLFloat16>& fp16_vector) {
  std::vector<float> float_data;
  float_data.reserve(fp16_vector.size());

  for (const auto& e : fp16_vector) {
    float_data.emplace_back(e.ToFloat());
  }

  return float_data;
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
                                          return static_cast<int8_t>(std::round(v));
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
    T scale) {
  auto qvec = QuantizeTransform(shape, scale, fvec, order_q);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  OpTester test_qorder("QuantizeWithOrder", 1, onnxruntime::kMSDomain);
  test_qorder.AddAttribute("order_input", (int64_t)ORDER_ROW);
  test_qorder.AddAttribute("order_output", (int64_t)order_q);
  test_qorder.AddInput<T>("input", shape, fvec);
  test_qorder.AddInput<T>("scale_input", {}, {scale});
  test_qorder.AddOutput("output", shape, qvec, false, 0.0f, 1.0f);
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QOrderedTest, FP32_Quantize_COL) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<float> fvec = GenData<float>(shape, scale);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL, scale);
}

TEST(QOrderedTest, FP32_Quantize_ROW) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<float> fvec = GenData<float>(shape, scale);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_ROW, scale);
}

TEST(QOrderedTest, FP32_Quantize_COL32) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<float> fvec = GenData<float>(shape, scale);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL32, scale);
}

TEST(QOrderedTest, FP32_Quantize_COL4_4R2_8C) {
  std::vector<int64_t> shape = {3, 8 * 3, 32 * 2};
  float scale(2.0f);
  std::vector<float> fvec = GenData<float>(shape, scale);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL4_4R2_8C, scale);
}

TEST(QOrderedTest, FP16_Quantize_COL) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  MLFloat16 scale = MLFloat16(2.0f);
  std::vector<MLFloat16> fvec = GenData<MLFloat16>(shape, scale);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL, scale);
}
TEST(QOrderedTest, FP16_Quantize_ROW) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  MLFloat16 scale = MLFloat16(2.0f);
  std::vector<MLFloat16> fvec = GenData<MLFloat16>(shape, scale);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_ROW, scale);
}

TEST(QOrderedTest, FP16_Quantize_COL32) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  MLFloat16 scale = MLFloat16(2.0f);
  std::vector<MLFloat16> fvec = GenData<MLFloat16>(shape, scale);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL32, scale);
}

TEST(QOrderedTest, FP16_Quantize_COL4_4R2_8C) {
  std::vector<int64_t> shape = {3, 8 * 3, 32 * 2};
  MLFloat16 scale = MLFloat16(2.0f);
  std::vector<MLFloat16> fvec = GenData<MLFloat16>(shape, scale);
  RunQOrdered_Quantize_Test(fvec, shape, ORDER_COL4_4R2_8C, scale);
}

template <typename T>
static void RunQOrdered_Dequantize_Test(
    std::vector<int8_t> const& qvec,
    std::vector<int64_t> const& shape,
    OrderCublasLt order_q,
    T scale) {
  auto fvec = DequantizeTransform<T>(shape, scale, qvec, order_q);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  OpTester test_qorder("DequantizeWithOrder", 1, onnxruntime::kMSDomain);
  test_qorder.AddAttribute("order_input", (int64_t)order_q);
  test_qorder.AddAttribute("order_output", (int64_t)ORDER_ROW);
  test_qorder.AddInput<int8_t>("input", shape, qvec);
  test_qorder.AddInput<T>("scale_input", {}, {scale});
  test_qorder.AddOutput("output", shape, fvec);
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Dequantize only work for ORDER_COL32 and ORDER_ROW input
TEST(QOrderedTest, FP32_Dequantize_COL32) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<int8_t> qvec = GenData<int8_t>(shape, 1.0f);
  RunQOrdered_Dequantize_Test(qvec, shape, ORDER_COL32, scale);
}

TEST(QOrderedTest, FP32_Dequantize_ROW) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  float scale = 2.0f;
  std::vector<int8_t> qvec = GenData<int8_t>(shape, 1.0f);
  RunQOrdered_Dequantize_Test(qvec, shape, ORDER_ROW, scale);
}

TEST(QOrderedTest, FP16_Dequantize_COL32) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  MLFloat16 scale(2.0f);
  std::vector<int8_t> qvec = GenData<int8_t>(shape, 1.0f);
  RunQOrdered_Dequantize_Test(qvec, shape, ORDER_COL32, scale);
}

TEST(QOrderedTest, FP16_Dequantize_ROW) {
  std::vector<int64_t> shape = {3, 5, 32 * 2};
  MLFloat16 scale(2.0f);
  std::vector<int8_t> qvec = GenData<int8_t>(shape, 1.0f);
  RunQOrdered_Dequantize_Test(qvec, shape, ORDER_ROW, scale);
}

static void RunQOrdered_MatMul_Test(
    std::vector<int64_t> const& shapeA,
    std::vector<int64_t> const& shapeB,
    std::vector<int64_t> const& shapeY,
    OrderCublasLt order_weight,
    float scaleA, float scaleB, float scaleC, float scaleY,
    bool add_bias = false, bool broadcast_c_batch = false) {
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
  if (add_bias) {
    vecBias = GenData<float>(shapeBias, scaleY);
  }
  std::vector<int8_t> vecC;
  std::vector<int64_t> shapeC = {broadcast_c_batch ? 1 : batchY, rowsY, colsY};
  if (scaleC != 0.0f) {
    vecC = GenData<int8_t>(shapeC, 1.0f);
  }

  // TODO: brocasting higher dims
  float alpha = scaleA * scaleB / scaleY;
  int8_t const* A = vecA.data();
  int8_t const* B = vecB.data();
  int8_t* Y = vecY.data();
  int8_t* C = vecC.data();
  for (int64_t b = 0; b < batchY; b++) {
    for (int64_t m = 0; m < rowsA; m++) {
      for (int64_t n = 0; n < colsB; n++) {
        float sum = 0.0f;
        for (int64_t k = 0; k < colsA; k++) {
          auto posA = indexerA(m, k);
          auto posB = indexerB(n, k);  // Transpose B
          sum += A[posA] * B[posB];
        }
        sum *= alpha;
        if (add_bias) {
          sum += vecBias[n];
        }
        auto posY = indexerY(m, n);
        if (scaleC != 0.0f) {
          sum += scaleC * C[posY];
        }
        Y[posY] = static_cast<int8_t>((int)std::round(std::min(127.0f, std::max(-128.0f, sum))));
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
  }
  if (scaleC != 0.0f) {
    test_qorder.AddInput<int8_t>("C", shapeC, vecC);
    test_qorder.AddInput<float>("scale_C", {}, {scaleC});
  }
  test_qorder.AddOutput<int8_t>("Y", shapeY, vecY, false, 0.0f, 1.0f /* abs error */);
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

static void RunQOrderedAddBiasResidual_LayerNorm_WithData(std::vector<int64_t> const& shape, int axis, float epsilon, OrderCublasLt order,
                                                          const std::vector<int8_t>& vecX, float scale_x,
                                                          const std::vector<int8_t>* vecR, const float* scale_r,
                                                          const std::vector<MLFloat16>* vecBias,
                                                          const std::vector<MLFloat16>& vecGamma, const std::vector<MLFloat16>* vecBeta,
                                                          float scale_y, const std::vector<int8_t>& vecY) {
  std::vector<int64_t> bias_shape = {shape.back()};
  OpTester test_qorder("QOrderedAddBiasResidualLayerNorm", 1, onnxruntime::kMSDomain);
  test_qorder.AddAttribute("axis", (int64_t)axis);
  test_qorder.AddAttribute("epsilon", epsilon);
  test_qorder.AddAttribute("order_X", (int64_t)order);
  test_qorder.AddAttribute("order_R", (int64_t)order);
  test_qorder.AddAttribute("order_Y", (int64_t)order);

  test_qorder.AddInput<int8_t>("X", shape, vecX);
  test_qorder.AddInput<float>("scale_X", {}, {scale_x});

  if (vecR) {
    test_qorder.AddInput<int8_t>("R", shape, *vecR);
    test_qorder.AddInput<float>("scale_R", {}, {*scale_r});
  } else {
    test_qorder.AddOptionalInputEdge<int8_t>();
    test_qorder.AddOptionalInputEdge<float>();
  }

  if (vecBias) {
    test_qorder.AddInput<MLFloat16>("B", bias_shape, *vecBias);
  } else {
    test_qorder.AddOptionalInputEdge<MLFloat16>();
  }

  test_qorder.AddInput<float>("scale_Y", {}, {scale_y});

  test_qorder.AddInput<MLFloat16>("gamma", bias_shape, vecGamma);

  if (vecBeta) {
    test_qorder.AddInput<MLFloat16>("beta", bias_shape, *vecBeta);
  } else {
    test_qorder.AddOptionalInputEdge<MLFloat16>();
  }

  test_qorder.AddOutput<int8_t>("Y", shape, vecY, false, 0.0f, 1.0f /* abs error */);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

static void RunQOrdered_LayerNorm_Test(std::vector<int64_t> const& shape, int axis,
                                       bool has_beta, bool has_residual_and_bias, OrderCublasLt order) {
  int64_t N = std::accumulate(shape.begin(), shape.end(), int64_t{1LL}, std::multiplies<int64_t>());
  int64_t cols = 0, rows = 0, batch = 0;
  BatchRowColFromShape(shape, batch, rows, cols);

  std::vector<int64_t> beta_shape = {cols};
  std::vector<int8_t> vecX = GenData<int8_t>(shape, 1.0f);

  // Residual's shape is same as X's
  std::vector<int8_t> vecR = GenData<int8_t>(shape, 1.0f);
  std::vector<MLFloat16> vecBias(cols);

  RandomValueGenerator random{};

  std::vector<MLFloat16> vecGamma = ToFloat16(random.Uniform<float>(beta_shape, 3.0f, 3.0f));
  std::vector<MLFloat16> vecBeta(cols);
  if (has_beta) {
    vecBeta = ToFloat16(random.Uniform<float>(beta_shape, -1.0f / 16.0f, -1.0f / 16.0f));
  }
  if (has_residual_and_bias) {
    // shape is same as beta_shape
    vecBias = ToFloat16(random.Uniform<float>(beta_shape, -1.0f / 16.0f, -1.0f / 16.0f));
  }

  std::vector<int8_t> vecY(N);
  float scale_x = 1.0f;
  float scale_r = 2.0f;
  float scale_y = 2.0f / 128.0f;
  float epsilon = 0.00001;

  int8_t* bdst = vecY.data();

  if (!has_residual_and_bias) {
    const int8_t* bsrc = vecX.data();

    for (int b = 0; b < batch; b++) {
      for (int r = 0; r < rows; r++) {
        int64_t sum = std::accumulate(bsrc, bsrc + cols, 0LL, std::plus<int64_t>());
        double u = double(sum) / cols;
        double diff_square_sum = std::accumulate(bsrc, bsrc + cols, 0.0,
                                                 [u, scale_x](double sum, int8_t v) { double d= (double)v - u; return sum + d * d; });
        double u_scaled = u * scale_x;
        double dss_episilon = diff_square_sum * (scale_x * scale_x) / (cols - 1) + epsilon;
        double sqrt_var = std::sqrt(dss_episilon);
        for (int c = 0; c < cols; c++) {
          double v = ((double)bsrc[c] * scale_x - u_scaled) / sqrt_var * vecGamma[c].ToFloat() + vecBeta[c].ToFloat();
          v = v / scale_y;
          v = std::max(-128.0, std::min(v, 127.0));
          bdst[c] = static_cast<int8_t>((int)std::nearbyint(v));
        }
        bsrc += cols;
        bdst += cols;
      }
    }
  } else {
    std::vector<double> intermediate(vecX.size());
    double* bsrc = intermediate.data();

    for (int b = 0; b < batch; b++) {
      for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
          size_t index = b * rows * cols + r * cols + c;
          double temp = ((double)vecX[index] * scale_x) + ((double)vecR[index] * scale_r) + ((double)vecBias[c].ToFloat());
          bsrc[c] = temp;
        }

        double sum = std::accumulate(bsrc, bsrc + cols, 0.0, std::plus<double>());

        double u = double(sum) / cols;
        double diff_square_sum = std::accumulate(bsrc, bsrc + cols, 0.0,
                                                 [u, scale_x](double sum, double v) { double d= v - u; return sum + d * d; });
        double dss_episilon = diff_square_sum / (cols - 1) + epsilon;
        double sqrt_var = std::sqrt(dss_episilon);

        for (int c = 0; c < cols; c++) {
          double v = ((double)bsrc[c] - u) / sqrt_var * vecGamma[c].ToFloat() + vecBeta[c].ToFloat();

          v = v / scale_y;
          v = std::max(-128.0, std::min(v, 127.0));
          bdst[c] = static_cast<int8_t>((int)std::nearbyint(v));
          //bdst[c] = static_cast<int8_t>(u);
        }
        bsrc += cols;
        bdst += cols;
      }
    }
  }

  if (order == ORDER_COL32) {
    vecX = Reorder(shape, vecX, ORDER_ROW, order);
    vecY = Reorder(shape, vecY, ORDER_ROW, order);

    if (has_residual_and_bias) {
      vecR = Reorder(shape, vecR, ORDER_ROW, order);
    }
  }

  RunQOrderedAddBiasResidual_LayerNorm_WithData(shape, axis, epsilon, order, vecX, scale_x,
                                                has_residual_and_bias ? &vecR : nullptr,
                                                &scale_r,  // pass in irrespective of whether bias and residual exist (the call site will choose to use or ignore it)
                                                has_residual_and_bias ? &vecBias : nullptr,
                                                vecGamma,
                                                has_beta ? &vecBeta : nullptr,
                                                scale_y, vecY);
}

TEST(QOrderedTest, LayerNorm_OrderCol32_Beta_BiasAndResidual_4x12x256) {
  RunQOrdered_LayerNorm_Test({4, 12, 256}, -1, true, true, ORDER_COL32);
}

TEST(QOrderedTest, LayerNorm_OrderCol32_Beta_BiasAndResidual_3x11x512) {
  RunQOrdered_LayerNorm_Test({3, 11, 512}, -1, true, true, ORDER_COL32);
}

TEST(QOrderedTest, LayerNorm_OrderCol32_Beta_BiasAndResidual_2x22x768) {
  RunQOrdered_LayerNorm_Test({2, 22, 768}, -1, true, true, ORDER_COL32);
}

TEST(QOrderedTest, LayerNorm_OrderCol32_Beta_BiasAndResidual_3x34x1024) {
  RunQOrdered_LayerNorm_Test({3, 34, 1024}, -1, true, true, ORDER_COL32);
}

TEST(QOrderedTest, LayerNorm_Data_1x32) {
  float scale_x = 1.0;
  std::vector<int8_t> vecX = {
      -103, -24, -11, 125, 103, -117, 44, -109, 27, 122, 113,
      0, -7, 26, 110, -34, 28, -61, -107, -35, -54, 69,
      -3, -58, -119, 61, -19, -115, 35, 76, 16, 54};
  std::vector<float> vecGamma32 = {
      0.9536133, 0.9160156, 1.3310547, 1.0068359, 0.8095703,
      1.0126953, 0.8876953, 1.4316406, 1.0947266, 1.0498047,
      0.7373047, 1.0615234, 1.015625, 1.0751953, 1.0068359,
      1.0908203, 1.2011719, 1.1962891, 0.91796875, 1.0947266,
      1.3183594, 1.0185547, 1.0791016, 1.0273438, 0.8364258,
      0.94873047, 1.0292969, 1.09375, 1.0371094, 1.1240234,
      1.4384766, 1.0068359};
  std::vector<float> vecBeta32 = {
      0.01411438, 0.18273926, -0.12414551, 0.09887695, -0.1114502,
      -0.08227539, -0.04598999, -0.11322021, 0.12731934, -0.06591797,
      -0.00662994, 0.04962158, -0.04281616, 0.07476807, 0.23010254,
      0.1036377, 0.10852051, 0.10919189, -0.02905273, -0.0512085,
      -0.1194458, 0.02661133, 0.05789185, -0.05239868, 0.17907715,
      -0.01765442, -0.12255859, -0.09729004, 0.06591797, 0.02258301,
      -0.01844788, -0.11999512};
  std::vector<int8_t> vecY = {
      -84, -8, -21, 113, 63, -108, 29, -128, 32, 104, 70,
      2, -9, 27, 109, -26, 34, -56, -87, -37, -70, 61,
      0, -55, -74, 47, -25, -115, 34, 73, 17, 38};
  float scale_y = 1.0 / 64.0f;
  auto vecGamma = ToFloat16(vecGamma32);
  auto vecBeta = ToFloat16(vecBeta32);

  RunQOrderedAddBiasResidual_LayerNorm_WithData({1, 1, 32}, -1, 0.00001f, ORDER_ROW, vecX, scale_x,
                                                nullptr, nullptr, nullptr,
                                                vecGamma, &vecBeta, scale_y, vecY);
  RunQOrderedAddBiasResidual_LayerNorm_WithData({1, 1, 32}, -1, 0.00001f, ORDER_COL32, vecX, scale_x,
                                                nullptr, nullptr, nullptr,
                                                vecGamma, &vecBeta, scale_y, vecY);
}
TEST(QOrderedTest, LayerNorm_OrderRow_3x7x600) {
  RunQOrdered_LayerNorm_Test({3, 9, 80}, -1, true, false, ORDER_ROW);
}

TEST(QOrderedTest, LayerNorm_OrderRow_3x11x96) {
  RunQOrdered_LayerNorm_Test({3, 11, 96}, -1, true, false, ORDER_ROW);
}

TEST(QOrderedTest, LayerNorm_OrderRow_NoBias_3x11x96) {
  RunQOrdered_LayerNorm_Test({3, 11, 96}, -1, false, false, ORDER_ROW);
}

TEST(QOrderedTest, LayerNorm_OrderCol32_3x11x96) {
  RunQOrdered_LayerNorm_Test({3, 11, 96}, -1, true, false, ORDER_COL32);
}

TEST(QOrderedTest, LayerNorm_OrderCol32_NoBeta_3x11x96) {
  RunQOrdered_LayerNorm_Test({3, 11, 96}, -1, false, false, ORDER_COL32);
}

TEST(QOrderedTest, LayerNorm_OrderCol32_Beta_3x11x96) {
  RunQOrdered_LayerNorm_Test({3, 11, 96}, -1, true, false, ORDER_COL32);
}

static void RunQOrdered_Gelu_Test(std::vector<int64_t> const& shape, float scale_X, float scale_Y, OrderCublasLt order) {
  float sqrt_of_2 = std::sqrt(2.0f);

  int64_t N = std::accumulate(shape.begin(), shape.end(), int64_t{1LL}, std::multiplies<int64_t>());
  std::vector<int8_t> vecX = GenData<int8_t>(shape, 1.0f);
  std::vector<int8_t> vecY(N);
  for (int64_t i = 0; i < N; i++) {
    float x = scale_X * (float)vecX[i];
    float r = (x * (0.5f * (1.0f + std::erff(x / sqrt_of_2)))) / scale_Y;
    int8_t q = static_cast<int8_t>((int)std::nearbyintf(std::min(127.0f, std::max(-128.0f, r))));
    vecY[i] = q;
  }

  // Ordering doesn't really matter for a unary activation op
  // This is just us being too pedantic.
  if (order == ORDER_COL32) {
    vecX = Reorder(shape, vecX, ORDER_ROW, order);
    vecY = Reorder(shape, vecY, ORDER_ROW, order);
  }

  OpTester test_qorder("QOrderedGelu", 1, onnxruntime::kMSDomain);
  test_qorder.AddAttribute("order_X", (int64_t)order);
  test_qorder.AddAttribute("order_Y", (int64_t)order);
  test_qorder.AddInput<int8_t>("X", shape, vecX);
  test_qorder.AddInput<float>("scale_X", {}, {scale_X});
  test_qorder.AddInput<float>("scale_Y", {}, {scale_Y});
  test_qorder.AddOutput<int8_t>("Y", shape, vecY, false, 0.0f, 1.0f /* abs error */);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QOrderedTest, Gelu_3x11x32) {
  RunQOrdered_Gelu_Test({3, 11, 32}, 1.0f / 32.0f, 1.0f / 128.0f, ORDER_COL32);
  RunQOrdered_Gelu_Test({3, 11, 32}, 1.0f / 32.0f, 1.0f / 128.0f, ORDER_ROW);
}

static void RunQOrdered_BiasGelu_Test(std::vector<int64_t> const& shape, float input_scale,
                                      std::vector<int64_t> const& bias_shape, float bias_scale,
                                      float output_scale, OrderCublasLt order) {
  float sqrt_of_2 = std::sqrt(2.0f);

  int64_t N = std::accumulate(shape.begin(), shape.end(), int64_t{1LL}, std::multiplies<int64_t>());
  std::vector<int8_t> vecX = GenData<int8_t>(shape, 1.0f);
  std::vector<int8_t> vecBias = GenData<int8_t>(bias_shape, 1.0f);
  std::vector<int8_t> vecY(N);
  int64_t cols = bias_shape[0];
  if (order != ORDER_COL32) {
    throw std::runtime_error("Only COL32 order supported currently");
  }

  if ((shape.size() < 1) || (cols != shape[shape.size() - 1]) || (cols % 32 != 0)) {
    throw std::runtime_error("Shape requirements unmet");
  }
  for (int64_t i = 0; i < N; i++) {
    auto bias_id = i % cols;
    float x = input_scale * (float)vecX[i] + bias_scale * (float)vecBias[bias_id];
    float r = (x * (0.5f * (1.0f + std::erff(x / sqrt_of_2)))) / output_scale;
    int8_t q = static_cast<int8_t>((int)std::nearbyintf(std::min(127.0f, std::max(-128.0f, r))));
    vecY[i] = q;
  }

  vecX = Reorder(shape, vecX, ORDER_ROW, order);
  vecY = Reorder(shape, vecY, ORDER_ROW, order);
  vecBias = Reorder(bias_shape, vecBias, ORDER_ROW, order);

  OpTester test_qorder("QOrderedBiasGelu", 1, onnxruntime::kMSDomain);
  test_qorder.AddAttribute("order_A", (int64_t)order);
  test_qorder.AddAttribute("order_B", (int64_t)order);
  test_qorder.AddAttribute("order_Y", (int64_t)order);
  test_qorder.AddInput<int8_t>("A", shape, vecX);
  test_qorder.AddInput<float>("scale_A", {}, {input_scale});
  test_qorder.AddInput<int8_t>("B", bias_shape, vecBias);
  test_qorder.AddInput<float>("scale_B", {}, {bias_scale});
  test_qorder.AddInput<float>("scale_Y", {}, {output_scale});
  test_qorder.AddOutput<int8_t>("Y", shape, vecY, false, 0.0f, 1.0f /* abs error */);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QOrderedTest, BiasGelu_3x11x64) {
  RunQOrdered_BiasGelu_Test(
      {3, 11, 64}, 1.0f / 32.0f, {64}, 1.0f / 32.0f, 1.0f / 128.0f, ORDER_COL32);
}

TEST(QOrderedTest, LongformerAttention_1) {
  OpTester test_qorder("QOrderedLongformerAttention", 1, onnxruntime::kMSDomain);

  test_qorder.AddAttribute("num_heads", (int64_t)2);
  test_qorder.AddAttribute("window", (int64_t)2);
  test_qorder.AddAttribute("order_input", (int64_t)1);
  test_qorder.AddAttribute("order_output", (int64_t)1);
  test_qorder.AddAttribute("order_weight", (int64_t)0);
  test_qorder.AddAttribute("order_global_weight", (int64_t)0);

  float scale = 2.f / 256;
  float qkv_gemm_scale = 0.02f;

  int64_t batch = 2;
  int64_t sequence = 8;
  int64_t hidden = 30;

  int64_t size = batch * sequence * hidden;

  std::vector<int8_t> input = GenOnesData<int8_t>({batch, sequence, hidden});
  std::vector<int8_t> output = GenZerosData<int8_t>({batch, sequence, hidden});
  std::vector<int8_t> weight = GenOnesData<int8_t>({hidden, 3 * hidden});
  std::vector<float> bias = GenZerosData<float>({3 * hidden});
  std::vector<MLFloat16> mask = GenZerosData<MLFloat16>({batch, sequence});
  std::vector<int32_t> global = GenZerosData<int32_t>({batch, sequence});

  // TODO: Re-order weight

  test_qorder.AddInput<int8_t>("input", {batch, sequence, hidden}, input);
  test_qorder.AddInput<float>("scale_input", {1}, {scale});

  test_qorder.AddInput<int8_t>("weight", {hidden, hidden * 3}, weight);
  test_qorder.AddInput<float>("scale_weight", {1}, {scale});

  test_qorder.AddInput<float>("bias", {hidden * 3}, bias);
  test_qorder.AddInput<float>("scale_bias", {1}, {scale});

  test_qorder.AddInput<float>("scale_qkv_gemm", {1}, {qkv_gemm_scale});

  test_qorder.AddInput<MLFloat16>("mask", {batch, sequence}, mask);

  test_qorder.AddInput<int8_t>("global_weight", {hidden, hidden * 3}, weight);
  test_qorder.AddInput<float>("scale_global_weight", {1}, {scale});

  test_qorder.AddInput<float>("global_bias", {hidden * 3}, bias);
  test_qorder.AddInput<float>("scale_global_gemm", {1}, {scale});

  test_qorder.AddInput<int32_t>("global", {batch, sequence}, global);
  test_qorder.AddInput<float>("scale_output", {1}, {scale});

  test_qorder.AddOutput<int8_t>("Y", {batch, sequence, hidden},
                                output, false, 0.0f, 1.0f /* abs error */);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

static void QuantizeAndFindScale(const std::vector<MLFloat16>& data, std::vector<int8_t>& quantized_data, float& scale) {
  float max = std::abs(data[0].ToFloat());

  for (const auto& e : data) {
    auto val = std::abs(e.ToFloat());

    if (val > max) {
      max = e;
    }
  }

  scale = (max * 2.f) / 255.f;

  quantized_data.reserve(data.size());

  for (const auto& e : data) {
    float val = e.ToFloat() / scale;
    val = std::max(-128.0f, val);
    val = std::min(127.0f, val);
    quantized_data.push_back(static_cast<int8_t>(std::round(val)));
  }
}

static void QKVGemmScale(MLFloat16* mat_A, MLFloat16* mat_B, int64_t batch,
                         int64_t sequence, int64_t hidden, float& scale) {
  // Do the MatMul and along the way compute the max value needed to get the scale
  float max = 0;

  for (int64_t b = 0; b < batch; ++b) {
    int64_t base_offset = b * sequence * hidden;

    for (int64_t i = 0; i < sequence; ++i) {
      for (int64_t j = 0; j < 3 * hidden; ++j) {
        float sum = 0;

        for (int64_t k = 0; k < hidden; ++k) {
          sum += (mat_A[base_offset + i * hidden + k].ToFloat() * mat_B[k * 3 * hidden + j].ToFloat());
        }

        auto abs_sum = std::abs(sum);
        if (abs_sum > max) {
          max = abs_sum;
        }
      }
    }
  }

  scale = (max * 2.f) / 255.f;
}

static std::vector<MLFloat16> QDQ(const std::vector<MLFloat16>& input, float scale) {
  std::vector<MLFloat16> output;
  output.reserve(input.size());

  for (auto& e : input) {
    // Q

    float val = e.ToFloat() / scale;
    val = std::max(-128.0f, val);
    val = std::min(127.0f, val);
    int8_t quant_val = static_cast<int8_t>(std::round(val));

    // DQ
    float dequant_val = quant_val * scale;
    output.emplace_back(dequant_val);
  }

  return output;
}
TEST(QOrderedTest, LongformerAttention_2) {
  OpTester test_qorder("QOrderedLongformerAttention", 1, onnxruntime::kMSDomain);

  test_qorder.AddAttribute("num_heads", (int64_t)2);
  test_qorder.AddAttribute("window", (int64_t)2);
  test_qorder.AddAttribute("order_input", (int64_t)1);
  test_qorder.AddAttribute("order_output", (int64_t)1);
  test_qorder.AddAttribute("order_weight", (int64_t)0);
  test_qorder.AddAttribute("order_global_weight", (int64_t)0);

  int64_t batch = 1;
  int64_t sequence = 4;
  int64_t hidden = 32;

  int64_t size = batch * sequence * hidden;

  // Input
  std::vector<MLFloat16> input_data = GenData<MLFloat16>({batch, sequence, hidden}, 1.f, -1, 1);
  std::vector<int8_t> input;
  float input_scale;

  QuantizeAndFindScale(input_data, input, input_scale);

  test_qorder.AddInput<int8_t>("input", {batch, sequence, hidden}, input);

  test_qorder.AddInput<float>("scale_input", {1}, {input_scale});

  // Weight
  std::vector<MLFloat16> weight_data = GenData<MLFloat16>({hidden, hidden * 3}, 1.f, -1, 1);

  std::vector<int8_t> weight;
  float weight_scale;

  QuantizeAndFindScale(weight_data, weight, weight_scale);

  auto weight_reordered = Reorder({hidden, hidden * 3}, weight, ORDER_ROW, ORDER_COL);

  test_qorder.AddInput<int8_t>("weight", {hidden, hidden * 3}, weight_reordered);

  test_qorder.AddInput<float>("scale_weight", {1}, {weight_scale});

  // Bias
  std::vector<float> bias;
  bias.resize(3 * hidden, 0.f);
  test_qorder.AddInput<float>("bias", {hidden * 3}, bias);
  test_qorder.AddInput<float>("scale_bias", {1}, {1.f});  // Not used anyway

  // QKV Gemm scale
  float qkv_gemm_scale = 0.1f;
  QKVGemmScale(input_data.data(), weight_data.data(), batch, sequence, hidden, qkv_gemm_scale);
  test_qorder.AddInput<float>("scale_qkv_gemm", {1}, {qkv_gemm_scale});

  // Mask
  std::vector<MLFloat16> mask = GenZerosData<MLFloat16>({batch, sequence});
  test_qorder.AddInput<MLFloat16>("mask", {batch, sequence}, mask);

  // Global Weight
  std::vector<MLFloat16> global_weight_data = GenData<MLFloat16>({hidden, hidden * 3}, 1.f, -1, 1);

  std::vector<int8_t> global_weight;
  float global_weight_scale;

  QuantizeAndFindScale(global_weight_data, global_weight, global_weight_scale);

  auto global_weight_data_reordered = Reorder({hidden, hidden * 3}, global_weight, ORDER_ROW, ORDER_COL);

  test_qorder.AddInput<int8_t>("global_weight", {hidden, hidden * 3}, global_weight_data_reordered);

  test_qorder.AddInput<float>("scale_global_weight", {1}, {global_weight_scale});

  // Global bias
  std::vector<float> global_bias;
  global_bias.resize(3 * hidden, 0.f);
  test_qorder.AddInput<float>("global_bias", {hidden * 3}, global_bias);

  // Scale global gemm
  float global_qkv_gemm_scale = 0.1f;
  QKVGemmScale(input_data.data(), global_weight_data.data(), batch, sequence, hidden, global_qkv_gemm_scale);
  test_qorder.AddInput<float>("scale_global_gemm", {1}, {0.1});

  // Global
  std::vector<int> global = {1, 1, 0, 0};
  test_qorder.AddInput<int32_t>("global", {batch, sequence}, global);

  // Non-quantized model
  // inputs=['input', 'weight', 'bias', 'mask_float32', 'global_weight', 'global_bias', 'global'],

  OpTester test_nonq("LongformerAttention", 1, onnxruntime::kMSDomain);
  test_nonq.AddAttribute("num_heads", (int64_t)2);
  test_nonq.AddAttribute("window", (int64_t)2);
  test_nonq.AddInput<MLFloat16>("input", {batch, sequence, hidden}, QDQ(input_data, input_scale));
  test_nonq.AddInput<MLFloat16>("weight", {hidden, 3 * hidden}, QDQ(weight_data, weight_scale));
  test_nonq.AddInput<MLFloat16>("bias", {3 * hidden}, CastFloatToFp16(bias));

  test_nonq.AddInput<MLFloat16>("mask", {batch, sequence}, mask);
  test_nonq.AddInput<MLFloat16>("global_weight", {hidden, 3 * hidden}, QDQ(global_weight_data, global_weight_scale));
  test_nonq.AddInput<MLFloat16>("global_bias", {3 * hidden}, CastFloatToFp16(global_bias));
  test_nonq.AddInput<int32_t>("global", {batch, sequence}, global);
  std::vector<MLFloat16> dummy_output(size, MLFloat16(0.f));
  test_nonq.AddOutput<MLFloat16>("output", {batch, sequence, hidden},
                                 dummy_output, false, 0.0f, 0.0f /* abs error */);

  std::vector<OrtValue> non_quantized_fetches;
  test_nonq.Run(non_quantized_fetches, DefaultCudaExecutionProvider());

  const MLFloat16* raw_out_data = non_quantized_fetches[0].GetMutable<Tensor>()->Data<MLFloat16>();

  // Output of the quantized model
  std::vector<MLFloat16> output_data;
  for (int64_t i = 0; i < size; ++i) {
    output_data.push_back(raw_out_data[i]);
  }

  std::vector<int8_t> output;
  float output_scale;

  QuantizeAndFindScale(output_data, output, output_scale);

  test_qorder.AddInput<float>("scale_output", {1}, {output_scale});

  test_qorder.AddOutput<int8_t>("output", {batch, sequence, hidden},
                                output, false, 0.0f, 1.0f /* abs error */);

  // Run the quantized model

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  std::vector<OrtValue> quantized_fetches;

  test_qorder.Run(quantized_fetches, DefaultCudaExecutionProvider());

  const int8_t* raw_quantized_data = quantized_fetches[0].GetMutable<Tensor>()->Data<int8_t>();

  std::unordered_map<int, int> diff_count;
  int total_diff = 0;

  for (int64_t i = 0; i < size; ++i) {
    auto diff = output[i] - raw_quantized_data[i];

    if (diff != 0) {
      ++total_diff;
    }

    if (diff_count.find(diff) != diff_count.end()) {
      ++diff_count[diff];
    } else {
      diff_count[diff] = 1;
    }
  }

  float a = 1.f;
  ORT_IGNORE_RETURN_VALUE(a);
}

}  // namespace test
}  // namespace onnxruntime

//#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif
