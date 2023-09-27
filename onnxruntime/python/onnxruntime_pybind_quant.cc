// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <numpy/npy_common.h>

#include "contrib_ops/cpu/quantization/dequantize_blockwise.h"
#include "core/util/thread_utils.h"

namespace onnxruntime {
namespace python {

namespace py = pybind11;
using namespace onnxruntime;

template <typename T>
class ToOrtType {
 public:
  typedef T MappedType;
};

template <>
class ToOrtType<npy_half> {
 public:
  typedef MLFloat16 MappedType;
};

template <typename T>
void QuantizeMatMul4BitsBlockwise(
    py::array_t<uint8_t> dst,          // shape: [ N, block_per_K, block_blob_size ]
    py::array_t<T> src,                // shape: [K, N]
    py::array_t<T> scale,              // shape: [N, block_per_K]
    py::array_t<uint8_t> zero_points,  // shape: [N, block_per_K]
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

  typedef typename ToOrtType<T>::MappedType OrtType;
  contrib::QuantizeBlockwise<OrtType>(
      static_cast<uint8_t*>(dst_buf.ptr),
      static_cast<const OrtType*>(src_buf.ptr),
      static_cast<OrtType*>(scale_buf.ptr),
      is_symmetric ? nullptr : static_cast<uint8_t*>(zp_buf.ptr),
      block_size,
      4,
      N,
      K,
      tp.get());
}

void CreateQuantPybindModule(py::module& m) {
  m.def("quantize_matmul_4bits_float", &QuantizeMatMul4BitsBlockwise<float>);
  m.def("quantize_matmul_4bits_fp16", &QuantizeMatMul4BitsBlockwise<npy_half>);
}

}  // namespace python
}  // namespace onnxruntime
