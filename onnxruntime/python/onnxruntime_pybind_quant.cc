// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "core/mlas/inc/mlas_q4.h"
#include "contrib_ops/cpu/quantization/dequantize_blockwise_bnb4.h"
#include "core/util/thread_utils.h"

#include <stdexcept>
#include <memory>
#include <sstream>

namespace pybind11 {
namespace detail {
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 23;
template <>
struct npy_format_descriptor<onnxruntime::MLFloat16> {
  static constexpr auto name = _("float16");
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
  static std::string format() {
    // following: https://docs.python.org/3/library/struct.html#format-characters
    return "e";
  }
};
}  // namespace detail
}  // namespace pybind11

namespace onnxruntime {
namespace python {

namespace py = pybind11;
using namespace onnxruntime;

template <typename T, int qbits>
void QuantizeMatMulNBitsBlockwise(
    py::array_t<uint8_t> dst,          // shape: [ N, block_per_K, block_blob_size ]
    py::array_t<T> src,                // shape: [K, N]
    py::array_t<T> scale,              // shape: [N, block_per_K]
    py::array_t<uint8_t> zero_points,  // shape: [N, block_per_K] if bits > 4 else [N, (block_per_K + 1) / 2]
    int32_t block_size,
    int32_t N,
    int32_t K,
    bool is_symmetric) {
  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                          concurrency::ThreadPoolType::INTRA_OP);

  py::buffer_info dst_buf = dst.request();
  py::buffer_info src_buf = src.request();
  py::buffer_info scale_buf = scale.request();
  py::buffer_info zp_buf = zero_points.request();

  MlasQuantizeBlockwise<T, qbits>(
      reinterpret_cast<uint8_t*>(dst_buf.ptr),
      reinterpret_cast<T*>(scale_buf.ptr),
      is_symmetric ? nullptr : reinterpret_cast<uint8_t*>(zp_buf.ptr),
      reinterpret_cast<const T*>(src_buf.ptr),
      block_size,
      true,
      K,
      N,
      N,
      tp.get());
}

template <typename T, int qbits>
bool QuantizeQDQMatMulNBitsBlockwise(
    py::array_t<uint8_t> dst,
    py::array_t<T> src,
    py::array_t<T> scale,
    py::array_t<uint8_t> zero_points,
    int32_t quant_block_size,
    int32_t N,
    int32_t K,
    bool is_symmetric) {
  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                          concurrency::ThreadPoolType::INTRA_OP);

  py::buffer_info dst_buf = dst.request();
  py::buffer_info src_buf = src.request();
  py::buffer_info scale_buf = scale.request();
  py::buffer_info zp_buf = zero_points.request();

  return MlasQDQQuantizeBlockwise<T, qbits>(
      reinterpret_cast<const T*>(src_buf.ptr),
      reinterpret_cast<T*>(scale_buf.ptr),
      is_symmetric ? nullptr : reinterpret_cast<uint8_t*>(zp_buf.ptr),
      reinterpret_cast<uint8_t*>(dst_buf.ptr),
      true,
      K,
      N,
      quant_block_size,
      tp.get());
}

template <typename T>
bool QuantizeQDQMatMul4BitsBlockwise(
    py::array_t<uint8_t> dst,          // shape: [K, N / 2]
    py::array_t<T> src,                // shape: [K, N]
    py::array_t<T> scale,              // shape: [block_per_K, N]
    py::array_t<uint8_t> zero_points,  // shape: [block_per_K, N / 2]
    int32_t quant_block_size,
    int32_t N,
    int32_t K,
    bool is_symmetric) {
  return QuantizeQDQMatMulNBitsBlockwise<T, 4>(dst, src, scale, zero_points, quant_block_size, N, K, is_symmetric);
}

template <typename T>
void QuantizeMatMulBnb4Blockwise(
    py::array_t<uint8_t> dst,
    py::array_t<T> src,
    py::array_t<T> absmax,
    int32_t block_size,
    int32_t quant_type,
    int32_t N,
    int32_t K) {
  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                          concurrency::ThreadPoolType::INTRA_OP);

  py::buffer_info dst_buf = dst.request();
  py::buffer_info src_buf = src.request();
  py::buffer_info absmax_buf = absmax.request();

  contrib::QuantizeBlockwiseBnb4<T>(
      static_cast<uint8_t*>(dst_buf.ptr),
      static_cast<const T*>(src_buf.ptr),
      static_cast<T*>(absmax_buf.ptr),
      block_size,
      quant_type,
      N,
      K,
      tp.get());
}

// Pack FP4 (MXFP4) weights for MoE GEMM kernels.
//
// Input:  q_weights in [N, K/2] layout (FP4 packed 2 per byte along K dimension, row-major)
// Output: Packed weights in [K, N/2] layout (FP4 packed 2 per byte along N dimension, column-major)
//
// Unlike INT4 which requires architecture-specific row permutation and interleaving,
// FP4 (SM90+ TMA path) only needs a simple transpose at the nibble level.
// This function is CPU-only and does not require CUDA to be present.
py::array_t<uint8_t> PackFP4WeightsForMoE(
    py::array_t<uint8_t> q_weights,
    int32_t N,
    int32_t K) {
  py::buffer_info in_buf = q_weights.request();
  const uint8_t* src = static_cast<const uint8_t*>(in_buf.ptr);

  if (N % 2 != 0 || K % 2 != 0) {
    throw std::invalid_argument("N and K must be even for FP4 packing");
  }
  if (N <= 0 || K <= 0) {
    throw std::invalid_argument("N and K must be positive");
  }
  if (in_buf.ndim != 2 || in_buf.shape[0] != N || in_buf.shape[1] != K / 2) {
    throw std::invalid_argument("q_weights must have shape (N, K / 2)");
  }

  int K_half = K / 2;
  int N_half = N / 2;
  size_t out_size = static_cast<size_t>(K) * static_cast<size_t>(N_half);
  py::array_t<uint8_t> output(static_cast<pybind11::ssize_t>(out_size));
  py::buffer_info out_buf = output.request();
  uint8_t* dst = static_cast<uint8_t*>(out_buf.ptr);
  std::memset(dst, 0, out_size);

  // Transpose FP4 nibbles from [N, K] (packed as [N, K/2] bytes) to
  // [K, N] (packed as [K, N/2] bytes).
  // Source packing: byte at (n, k/2) has value(n, k&~1) in low nibble, value(n, k|1) in high nibble
  // Dest packing:   byte at (k, n/2) has value(k, n&~1) in low nibble, value(k, n|1) in high nibble
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      // Read nibble at logical position (n, k) from src [N, K/2]
      int src_byte = n * K_half + k / 2;
      uint8_t nibble = (k % 2 == 0) ? (src[src_byte] & 0x0F) : (src[src_byte] >> 4);

      // Write nibble at logical position (k, n) to dst [K, N/2]
      int dst_byte = k * N_half + n / 2;
      if (n % 2 == 0) {
        dst[dst_byte] |= nibble;  // low nibble
      } else {
        dst[dst_byte] |= (nibble << 4);  // high nibble
      }
    }
  }

  return output;
}

void CreateQuantPybindModule(py::module& m) {
  m.def("quantize_matmul_2bits", &QuantizeMatMulNBitsBlockwise<float, 2>);
  m.def("quantize_matmul_2bits", &QuantizeMatMulNBitsBlockwise<MLFloat16, 2>);
  m.def("quantize_matmul_4bits", &QuantizeMatMulNBitsBlockwise<float, 4>);
  m.def("quantize_matmul_4bits", &QuantizeMatMulNBitsBlockwise<MLFloat16, 4>);
  m.def("quantize_matmul_8bits", &QuantizeMatMulNBitsBlockwise<float, 8>);
  m.def("quantize_matmul_8bits", &QuantizeMatMulNBitsBlockwise<MLFloat16, 8>);
  m.def("quantize_matmul_bnb4", &QuantizeMatMulBnb4Blockwise<float>);
  m.def("quantize_matmul_bnb4", &QuantizeMatMulBnb4Blockwise<MLFloat16>);
  m.def("quantize_qdq_matmul_2bits", &QuantizeQDQMatMulNBitsBlockwise<float, 2>);
  m.def("quantize_qdq_matmul_2bits", &QuantizeQDQMatMulNBitsBlockwise<MLFloat16, 2>);
  m.def("quantize_qdq_matmul_4bits", &QuantizeQDQMatMul4BitsBlockwise<float>);
  m.def("quantize_qdq_matmul_4bits", &QuantizeQDQMatMul4BitsBlockwise<MLFloat16>);
  m.def("pack_fp4_weights_for_cuda_moe_gemm", &PackFP4WeightsForMoE,
        "Pack FP4 (MXFP4) weights for CUDA MoE GEMM: transpose [N,K/2] to column-major [K,N/2]",
        py::arg("q_weights"), py::arg("N"), py::arg("K"));
}

}  // namespace python
}  // namespace onnxruntime
