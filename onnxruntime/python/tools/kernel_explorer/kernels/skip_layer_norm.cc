// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/skip_layer_norm.h"

#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>

#include "contrib_ops/rocm/bert/skip_layer_norm_op.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

namespace py = pybind11;

namespace onnxruntime {

template <typename T, int ThreadsPerBlock, int VecSize>
class SkipLayerNormSmall : public IKernelExplorer {
 public:
  SkipLayerNormSmall(DeviceArray& output, DeviceArray& input, DeviceArray& skip,
                     DeviceArray& gamma, DeviceArray& beta, DeviceArray& bias,
                     float epsilon, int hidden_size, int element_count)
      : params_(this->Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(input.ptr()),
                static_cast<T*>(skip.ptr()), static_cast<T*>(gamma.ptr()), static_cast<T*>(beta.ptr()),
                static_cast<T*>(bias.ptr()), epsilon, hidden_size, element_count) {}

  void Run() override {
    ORT_THROW_IF_ERROR((contrib::rocm::SkipLayerNormSmallOp<T, ThreadsPerBlock, VecSize>(&params_)));
  }

  bool IsSupported() {
    Status status = contrib::rocm::SkipLayerNormSmallOp<T, ThreadsPerBlock, VecSize>(&params_);
    return status.IsOK();
  }

 private:
  using ParamsT = contrib::rocm::SkipLayerNormParams<T>;
  ParamsT params_{};
};

#define REGISTER_OP(name, type, threads_per_block, vec_size)                                                   \
  py::class_<name<type, threads_per_block, vec_size>>(m, #name "_" #type "_" #threads_per_block "_" #vec_size) \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&,                                    \
                    DeviceArray&, DeviceArray&,                                                                \
                    float, int, int>())                                                                        \
      .def("SetRepeats", &name<type, threads_per_block, vec_size>::SetRepeats)                                 \
      .def("Profile", &name<type, threads_per_block, vec_size>::Profile)                                       \
      .def("Run", &name<type, threads_per_block, vec_size>::Run)                                               \
      .def("IsSupported", &name<type, threads_per_block, vec_size>::IsSupported);

#define REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, threads_per_block) \
  REGISTER_OP(name, type, threads_per_block, 1)                     \
  REGISTER_OP(name, type, threads_per_block, 2)                     \
  REGISTER_OP(name, type, threads_per_block, 4)                     \
  REGISTER_OP(name, type, threads_per_block, 8)                     \
  REGISTER_OP(name, type, threads_per_block, 16)

#define REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(name, type) \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 64)                         \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 128)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 192)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 256)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 320)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 384)

void InitSkipLayerNorm(py::module m) {
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormSmall, half);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormSmall, float);
}

}  // namespace onnxruntime
