// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "contrib_ops/cpu/quantization/dequantize_blockwise.h"
#include "contrib_ops/cpu/quantization/dequantize_blockwise_bnb4.h"
#include "core/util/thread_utils.h"

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

template <typename T>
void QuantizeMatMul4BitsBlockwise(
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

  contrib::QuantizeBlockwise<T>(
      static_cast<uint8_t*>(dst_buf.ptr),
      static_cast<const T*>(src_buf.ptr),
      static_cast<T*>(scale_buf.ptr),
      is_symmetric ? nullptr : static_cast<uint8_t*>(zp_buf.ptr),
      block_size,
      4,
      N,
      K,
      tp.get());
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

void CreateQuantPybindModule(py::module& m) {
  m.def("quantize_matmul_4bits", &QuantizeMatMul4BitsBlockwise<float>);
  m.def("quantize_matmul_4bits", &QuantizeMatMul4BitsBlockwise<MLFloat16>);
  m.def("quantize_matmul_bnb4", &QuantizeMatMulBnb4Blockwise<float>);
  m.def("quantize_matmul_bnb4", &QuantizeMatMulBnb4Blockwise<MLFloat16>);
}

}  // namespace python
}  // namespace onnxruntime
