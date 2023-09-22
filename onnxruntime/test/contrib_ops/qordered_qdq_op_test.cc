// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/contrib_ops/qordered_test_utils.h"

#if defined(USE_CUDA)

#include <cuda.h>

#if defined(USE_CUDA)

namespace onnxruntime {
namespace test {

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
  int cuda_runtime_version = 0;
  // Need 11.4 or higher cuda runtime
  if ((cudaRuntimeGetVersion(&cuda_runtime_version) != cudaSuccess) || (cuda_runtime_version < 11040)) {
    return;
  }

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
  int cuda_runtime_version = 0;
  // Need 11.4 or higher cuda runtime
  if ((cudaRuntimeGetVersion(&cuda_runtime_version) != cudaSuccess) || (cuda_runtime_version < 11040)) {
    return;
  }

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

}  // namespace test
}  // namespace onnxruntime

#endif  // CUDA_VERSION

#endif  // USE_CUDA
