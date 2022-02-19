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
#include <cublasLt.h>

namespace onnxruntime {
namespace test {

template <typename T>
static std::vector<T> GenData(std::vector<int64_t> const & shape, float scale) {
  int64_t n = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  std::vector<T> r(n);
  for (int64_t i = 0; i < n; i++) {
    r[i] = static_cast<T>(((i % 256) - 128) * scale);
  }
  return r;
}

class OrderedIndex {
  cublasLtOrder_t order_;
  int64_t rows_;
  int64_t cols_;
public:
  OrderedIndex(cublasLtOrder_t order, int64_t rows, int64_t cols) : order_(order), rows_(rows), cols_(cols) { }
  int64_t operator()(int64_t r, int64_t c);
};

int64_t OrderedIndex::operator()(int64_t r, int64_t c) {
  switch (order_) {
    case CUBLASLT_ORDER_ROW:
      return r * cols_ + c;
    case CUBLASLT_ORDER_COL:
      return c * rows_ + r;
    case CUBLASLT_ORDER_COL32:
      {
        int64_t tile_id = c / 32;
        int64_t tile_stride = 32 * rows_;
        return tile_id * tile_stride + r * 32 + (c % 32);
      }
    case CUBLASLT_ORDER_COL4_4R2_8C:
      {
        int64_t tiles_c = c / 32;
        int64_t tiles_r = r / 8;
        int64_t tile_idx = tiles_c * (rows_ / 8) + tiles_r;
        int64_t offset = tile_idx * (32 * 8);
        offset += (r & 0x1) * (32 * 4);
        int64_t in_4x4x8_tile_c = c % 32;
        int64_t in_4x4x8_tile_r = (r % 8) / 2;
        int64_t in_4x4x8_idx = (in_4x4x8_tile_c / 4) * (4*4) + in_4x4x8_tile_r * 4 + (in_4x4x8_tile_c % 4);
        offset += in_4x4x8_idx;
        return offset;
      }
    case CUBLASLT_ORDER_COL32_2R_4R4:
    {
      // TODO:
    }
    default:
      return 0;
  }
}

template <typename TSrc>
static std::vector<int8_t> QuantizeTransform(std::vector<int64_t> const& shape, float scale, const std::vector<TSrc>& src, cublasLtOrder_t order) {
  int64_t cols = shape.back();
  int64_t rows = (shape.size() > 1 ? shape[shape.size() - 2] : 1LL);
  int64_t batch = (shape.size() <= 2 ? 1LL : std::accumulate(shape.data(), shape.data() + (shape.size() - 2), 1LL, std::multiplies<int64_t>()));
  
  OrderedIndex src_indexer(CUBLASLT_ORDER_ROW, rows, cols);
  OrderedIndex dst_indexer(order, rows, cols);

  std::vector<int8_t> dst(batch * cols * rows, 0);
  const TSrc* bsrc = src.data();
  int8_t* bdst = dst.data();
  for (int64_t b = 0, batch_stride = rows * cols; b < batch; b++) {
    for (int64_t r = 0; r < rows; r++) {
      for (int64_t c = 0; c < cols; c++) {
        int64_t src_idx = src_indexer(r, c);
        int64_t dst_idx = dst_indexer(r, c);
        if (src_idx >= batch_stride || dst_idx >= batch_stride || bdst[dst_idx] != 0) {
          std::cout << "out of bound index calculated, error found in OrderedIndexer" << std::endl;
        }
        float v = (float)bsrc[src_idx] * scale;
        v = std::max(TSrc(-128.0f), v);
        v = std::min(TSrc(127.0f), v);
        bdst[dst_idx] = static_cast<int8_t>(std::round(v));
      }
    }
    bsrc += batch_stride;
    bdst += batch_stride;
  }
  return dst;
}

template <typename T>
static void RunQOrdered_Quantize_Test(
    std::vector<T> const& fvec,
    std::vector<int64_t> const& shape,
    cublasLtOrder_t order_q,
    T scale) {
  auto qvec = QuantizeTransform(shape, scale, fvec, order_q);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  OpTester test_q("QuantizeWithOrder", 1, onnxruntime::kMSDomain);
  test_q.AddAttribute("order_input", (int64_t)CUBLASLT_ORDER_ROW);
  test_q.AddAttribute("order_output", (int64_t)order_q);
  test_q.AddInput<T>("input", shape, fvec);
  test_q.AddInput<T>("scale_input", {}, {scale});
  test_q.AddOutput("output", shape, qvec);
  test_q.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// template <typename T>
// static void RunQOrdered_Dequantize_Test(
//     std::vector<int8_t> const& qvec,
//     cublasLtOrder_t order_q,
//     std::vector<int64_t> const& shape,
//     std::vector<T> const& fvec,
//     cublasLtOrder_t order_f,
//     T scale) {
//   std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
//   execution_providers.push_back(DefaultCudaExecutionProvider());

//   OpTester test_dq("DequantizeWithOrder", 1, onnxruntime::kMSDomain);
//   test_dq.AddAttribute("order_input", (int64_t)order_q);
//   test_dq.AddAttribute("order_output", (int64_t)order_f);
//   test_dq.template AddInput("input", shape, qvec);
//   test_dq.AddInput<T>("scale_input", {}, {scale});
//   test_dq.AddOutput("output", shape, fvec);
//   test_dq.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
// }


TEST(QOrderedTest, FP32_Quantize_COL32) {
  std::vector<int64_t> shape = {1, 5, 32 * 2};
  float scale = 1.0f;
  std::vector<float> fvec = GenData<float>(shape, scale);
  RunQOrdered_Quantize_Test(fvec, shape, CUBLASLT_ORDER_COL32, scale);
}

TEST(QOrderedTest, FP32_Quantize_COL4_4R2_8C) {
  std::vector<int64_t> shape = {1, 8 * 3, 32 * 2};
  float scale(1.0f);
  std::vector<float> fvec = GenData<float>(shape, scale);
  RunQOrdered_Quantize_Test(fvec, shape, CUBLASLT_ORDER_COL4_4R2_8C, scale);
}


}  // namespace test
}  // namespace onnxruntime

// #endif
