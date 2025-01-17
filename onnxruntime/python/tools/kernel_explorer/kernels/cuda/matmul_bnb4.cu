// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file serve as a simple example for adding a tunable op to onnxruntime.

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <pybind11/pybind11.h>

#include <string>

#include "core/providers/cuda/tunable/cuda_tunable.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"
#include "python/tools/kernel_explorer/kernels/vector_add_kernel.cuh"
#include "contrib_ops/cuda/quantization/dequantize_blockwise_bnb4.cuh"
#include "contrib_ops/cuda/quantization/matmul_bnb4.cuh"

namespace py = pybind11;

namespace onnxruntime {

// Extend the OpParams so that all specializations have the same parameter passing interface
template <typename T>
struct MatrixFloatBnb4Params : cuda::tunable::OpParams {
  std::string Signature() const override { return std::to_string(n_); }

  int quant_type_;
  T* output_;
  const T* a_;
  const uint8_t* b_;
  const T* absmax_;
  T* quant_map_buffer_;
  int m_;
  int n_;
  int k_;
};

template <typename T>
class MatrixFloatBnb4 : public IKernelExplorer {
 public:
  MatrixFloatBnb4(DeviceArray& output,
                  DeviceArray& a,
                  DeviceArray& b,
                  DeviceArray& absmax,
                  DeviceArray& quant_map_buffer,
                  int quant_type, int m, int n, int k) {
    params_.tuning_ctx = TuningContext();
    params_.stream = Stream();
    params_.output_ = static_cast<T*>(output.ptr());
    params_.a_ = static_cast<T*>(a.ptr());
    params_.b_ = static_cast<uint8_t*>(b.ptr());
    params_.absmax_ = static_cast<T*>(absmax.ptr());
    params_.quant_map_buffer_ = static_cast<T*>(quant_map_buffer.ptr());
    params_.quant_type_ = quant_type;
    params_.m_ = m;
    params_.n_ = n;
    params_.k_ = k;
  }

  void Run() override {
    ORT_THROW_IF_ERROR(contrib::cuda::SetBnbQuantMap(
        params_.quant_type_,
        params_.quant_map_buffer_,
        params_.StreamHandle()));
    contrib::cuda::TryMatMulBnb4(
        params_.quant_map_buffer_,
        params_.output_,
        params_.a_,
        params_.b_,
        params_.absmax_,
        params_.m_,
        params_.n_,
        params_.k_,
        64,
        params_.StreamHandle());
  }

 private:
  // A VectorAddOp<T> is a callable that can process const VectorAddParams<T>*
  using ParamsT = MatrixFloatBnb4Params<T>;
  ParamsT params_{};
};

#define REGISTER_OP(name, type)                                                                                  \
  py::class_<name<type>>(m, #name "_" #type)                                                                     \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, int, int, int, int>()) \
      .def("SetRepeats", &name<type>::SetRepeats)                                                                \
      .def("Profile", &name<type>::Profile)                                                                      \
      .def("Run", &name<type>::Run);

KE_REGISTER(m) {
  REGISTER_OP(MatrixFloatBnb4, half);
  REGISTER_OP(MatrixFloatBnb4, float);
}

}  // namespace onnxruntime
