// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file serve as a simple example for adding a tunable op to onnxruntime.

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <pybind11/pybind11.h>

#include <string>

#include "core/providers/cuda/tunable/cuda_tunable.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "contrib_ops/cuda/quantization/dequantize_blockwise.cuh"

namespace py = pybind11;

namespace onnxruntime {

// Extend the OpParams so that all specializations have the same parameter passing interface
template <typename T>
struct DequantizeInt4Params : cuda::tunable::OpParams {
  std::string Signature() const override { return std::to_string(n_); }

  T* output_;
  const uint8_t* quant_;
  const T* scales_;
  const uint8_t* zero_points_;
  int n_;
  int k_;
};

template <typename T>
class DequantizeInt4 : public IKernelExplorer {
 public:
  DequantizeInt4(DeviceArray& output, DeviceArray& quant, DeviceArray& scales, int n, int k) {
    params_.tuning_ctx = TuningContext();
    params_.stream = Stream();
    params_.output_ = static_cast<T*>(output.ptr());
    params_.quant_ = static_cast<uint8_t*>(quant.ptr());
    params_.scales_ = static_cast<T*>(scales.ptr());
    params_.zero_points_ = nullptr;
    params_.n_ = n;
    params_.k_ = k;
  }

  void Run() override {
    ORT_THROW_IF_ERROR(contrib::cuda::DequantizeBlockwise4b(
        params_.output_,
        params_.quant_,
        params_.scales_,
        params_.zero_points_,
        32,
        true,
        params_.k_,
        params_.n_,
        params_.StreamHandle()));
  }

 private:
  // A VectorAddOp<T> is a callable that can process const VectorAddParams<T>*
  using ParamsT = DequantizeInt4Params<T>;
  ParamsT params_{};
};

#define REGISTER_OP(name, type)                                            \
  py::class_<name<type>>(m, #name "_" #type)                               \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int, int>()) \
      .def("SetRepeats", &name<type>::SetRepeats)                          \
      .def("Profile", &name<type>::Profile)                                \
      .def("Run", &name<type>::Run);

KE_REGISTER(m) {
  REGISTER_OP(DequantizeInt4, half);
  REGISTER_OP(DequantizeInt4, float);
}

}  // namespace onnxruntime
