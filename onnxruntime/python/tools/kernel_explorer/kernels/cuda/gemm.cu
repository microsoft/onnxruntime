// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file serve as a simple example for adding a tunable op to onnxruntime.

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <pybind11/pybind11.h>

#include <string>

#include "core/providers/cuda/tunable/cuda_tunable.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/cuda_stream_handle.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"
#include "python/tools/kernel_explorer/kernels/vector_add_kernel.cuh"
#include "contrib_ops/cuda/quantization/matmul_nbits.cuh"

namespace py = pybind11;

namespace onnxruntime {

// Extend the OpParams so that all specializations have the same parameter passing interface
template <typename T>
struct GemmBenchmarkParams : cuda::tunable::OpParams {
  std::string Signature() const override { return std::to_string(n_); }

  T* output_;
  const T* a_;
  const T* b_;
  int m_;
  int n_;
  int k_;
  cublasHandle_t cublas_handle;
};

template <typename T>
class GemmBenchmark : public IKernelExplorer {
 public:
  GemmBenchmark(DeviceArray& output, DeviceArray& a, DeviceArray& b, int m, int n, int k) {
    params_.tuning_ctx = TuningContext();
    params_.stream = Stream();
    params_.output_ = static_cast<T*>(output.ptr());
    params_.a_ = static_cast<T*>(a.ptr());
    params_.b_ = static_cast<T*>(b.ptr());
    params_.m_ = m;
    params_.n_ = n;
    params_.k_ = k;

    CUBLAS_CALL_THROW(cublasCreate(&(params_.cublas_handle)));
    CUDA_CALL_THROW(cudaGetDeviceProperties(&device_prop_, 0));
  }

  void Run() override {
    typedef typename ToCudaType<T>::MappedType CudaT;
    CudaT one = ToCudaType<T>::FromFloat(1.0f);
    CudaT zero = ToCudaType<T>::FromFloat(0.0f);

    // TF32 is enable by default. To disable TF32, set environment variable NVIDIA_TF32_OVERRIDE = 0
    constexpr bool use_tf32 = true;
    CUBLAS_CALL_THROW(cublasGemmHelper(
        params_.cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        params_.n_, params_.m_, params_.k_,
        &one,
        reinterpret_cast<const CudaT*>(params_.b_),
        params_.n_,
        reinterpret_cast<const CudaT*>(params_.a_),
        params_.k_,
        &zero,
        params_.output_,
        params_.n_,
        device_prop_,
        use_tf32));
  }

 private:
  // A VectorAddOp<T> is a callable that can process const VectorAddParams<T>*
  using ParamsT = GemmBenchmarkParams<T>;
  ParamsT params_{};
  cudaDeviceProp device_prop_;
};

#define REGISTER_OP(name, type)                                                 \
  py::class_<name<type>>(m, #name "_" #type)                                    \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int, int, int>()) \
      .def("SetRepeats", &name<type>::SetRepeats)                               \
      .def("Profile", &name<type>::Profile)                                     \
      .def("Run", &name<type>::Run);

KE_REGISTER(m) {
  REGISTER_OP(GemmBenchmark, half);
  REGISTER_OP(GemmBenchmark, float);
}

}  // namespace onnxruntime
