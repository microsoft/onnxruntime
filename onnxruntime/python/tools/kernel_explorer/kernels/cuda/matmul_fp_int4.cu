// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file serve as a simple example for adding a tunable op to onnxruntime.

#if USE_CUDA
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#elif USE_ROCM
#include <hip/hip_fp16.h>
#endif
#include <pybind11/pybind11.h>

#include <string>

#if USE_CUDA
#include "core/providers/cuda/tunable/cuda_tunable.h"
#elif USE_ROCM
#include "core/providers/rocm/tunable/rocm_tunable.h"
#endif
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"
#include "python/tools/kernel_explorer/kernels/vector_add_kernel.cuh"
#include "contrib_ops/cuda/quantization/matmul_with_quant_weight.cuh"

namespace py = pybind11;

namespace onnxruntime {

// Extend the OpParams so that all specializations have the same parameter passing interface
template <typename T>
struct MatrixFloatInt4Params :
#if USE_CUDA
    cuda::tunable::OpParams
#elif USE_ROCM
    rocm::tunable::OpParams
#endif
{
  std::string Signature() const override { return std::to_string(n_); }

  T* output_;
  const T* a_;
  const uint8_t* b_;
  const T* scales_;
  const uint8_t* zero_points_;
  int m_;
  int n_;
  int k_;
};

template <typename T>
class MatrixFloatInt4 : public IKernelExplorer {
 public:
  MatrixFloatInt4(DeviceArray& output, DeviceArray& a, DeviceArray& b, DeviceArray& scales, int m, int n, int k) {
    params_.tuning_ctx = TuningContext();
    params_.stream = Stream();
    params_.output_ = static_cast<T*>(output.ptr());
    params_.a_ = static_cast<T*>(a.ptr());
    params_.b_ = static_cast<uint8_t*>(b.ptr());
    params_.scales_ = static_cast<T*>(scales.ptr());
    params_.zero_points_ = nullptr;
    params_.m_ = m;
    params_.n_ = n;
    params_.k_ = k;
  }

  void Run() override {
    contrib::cuda::TryMatMul4BitsWeight<T>(
        params_.output_,
        params_.a_,
        params_.b_,
        params_.scales_,
        params_.zero_points_,
        params_.m_,
        params_.n_,
        params_.k_,
        32,
        params_.StreamHandle());
  }

 private:
  // A VectorAddOp<T> is a callable that can process const VectorAddParams<T>*
  using ParamsT = MatrixFloatInt4Params<T>;
  ParamsT params_{};
};

#define REGISTER_OP(name, type)                                                               \
  py::class_<name<type>>(m, #name "_" #type)                                                  \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, int, int, int>()) \
      .def("SetRepeats", &name<type>::SetRepeats)                                             \
      .def("Profile", &name<type>::Profile)                                                   \
      .def("Run", &name<type>::Run);

KE_REGISTER(m) {
  REGISTER_OP(MatrixFloatInt4, half);
  REGISTER_OP(MatrixFloatInt4, float);
}

}  // namespace onnxruntime
