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

template<typename T>
void QuantizeMatMul4BitsBlockwise(
    py::array_t<uint8_t> dst,    // shape: [ N, block_per_K, block_blob_size ]
    py::array_t<T> src,          // shape: [K, N]
    py::array_t<T> scale,        // shape: [N, block_per_K]
    py::array_t<uint8_t> zero_points,  // shape: [N, block_per_K]
    int32_t block_size,
    int32_t N,
    int32_t K,
    bool is_symmetric) {
  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                          concurrency::ThreadPoolType::INTRA_OP);

  if constexpr (std::is_same_v<T, npy_half>) {
    contrib::QuantizeBlockwise<MLFloat16>(
        dst.mutable_data(),
        reinterpret_cast<const MLFloat16*>(src.data()),
        reinterpret_cast<MLFloat16*>(scale.mutable_data()),
        is_symmetric ? nullptr : zero_points.mutable_data(),
        block_size,
        4,
        N,
        K,
        tp.get());
  } else {
    contrib::QuantizeBlockwise<T>(
        dst.mutable_data(),
        src.data(),
        scale.mutable_data(),
        is_symmetric ? nullptr : zero_points.mutable_data(),
        block_size,
        4,
        N,
        K,
        tp.get());
  }
}

void CreateQuantPybindModule(py::module& m) {
  m.def("quantize_matmul_4bits_float", &QuantizeMatMul4BitsBlockwise<float>);
  m.def("quantize_matmul_4bits_fp16", &QuantizeMatMul4BitsBlockwise<npy_half>);
}

}  // namespace python
}  // namespace onnxruntime
