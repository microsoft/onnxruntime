// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <hip/hip_fp16.h>
#include "contrib_ops/rocm/bert/skip_layer_norm_impl_kernel.h"
#include "contrib_ops/rocm/bert/skip_layer_norm_tunable_op.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

namespace py = pybind11;

namespace onnxruntime {

template <typename T, int ThreadsPerBlock, int VecSize>
class SkipLayerNorm : public IKernelExplorer {
 public:
  SkipLayerNorm(DeviceArray& output, DeviceArray& input, DeviceArray& skip,
                DeviceArray& gamma, DeviceArray& beta, DeviceArray& bias,
                float epsilon, int hidden_size, int element_count)
      : params_(this->Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(input.ptr()),
                static_cast<T*>(skip.ptr()), static_cast<T*>(gamma.ptr()), static_cast<T*>(beta.ptr()),
                static_cast<T*>(bias.ptr()), epsilon, hidden_size, element_count) {}

  void Run() override {
    ORT_THROW_IF_ERROR((contrib::rocm::SkipLayerNormOp<T, ThreadsPerBlock, VecSize>(&params_)));
  }

 private:
  using ParamsT = contrib::rocm::SkipLayerNormParams<T>;
  ParamsT params_{};
};

template <typename T>
class SkipLayerNormTunable : public IKernelExplorer {
 public:
  SkipLayerNormTunable(DeviceArray& output, DeviceArray& input, DeviceArray& skip,
                       DeviceArray& gamma, DeviceArray& beta, DeviceArray& bias,
                       float epsilon, int hidden_size, int element_count)
      : params_(this->Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(input.ptr()),
                static_cast<T*>(skip.ptr()), static_cast<T*>(gamma.ptr()), static_cast<T*>(beta.ptr()),
                static_cast<T*>(bias.ptr()), epsilon, hidden_size, element_count) {
    op_.EnableTuning();
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

 private:
  using ParamsT = contrib::rocm::SkipLayerNormParams<T>;
  ParamsT params_{};
  contrib::rocm::SkipLayerNormTunableOp<T> op_{};
};

#define REGISTER_OP(name, type, threads_per_block, vec_size)                                                   \
  py::class_<name<type, threads_per_block, vec_size>>(m, #name "_" #type "_" #threads_per_block "_" #vec_size) \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&,                                    \
                    DeviceArray&, DeviceArray&,                                                                \
                    float, int, int>())                                                                        \
      .def("SetRepeats", &name<type, threads_per_block, vec_size>::SetRepeats)                                 \
      .def("Profile", &name<type, threads_per_block, vec_size>::Profile)                                       \
      .def("Run", &name<type, threads_per_block, vec_size>::Run);

#define REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, threads_per_block) \
  REGISTER_OP(name, type, threads_per_block, 1)                     \
  REGISTER_OP(name, type, threads_per_block, 2)                     \
  REGISTER_OP(name, type, threads_per_block, 4)                     \
  REGISTER_OP(name, type, threads_per_block, 8)                     \
  REGISTER_OP(name, type, threads_per_block, 16)

#define REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(name, type) \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 64)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 128)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 192)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 256)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 320)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 384)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 448)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 512)

#define REGISTER_TUNABLE_OP(type)                                              \
  py::class_<SkipLayerNormTunable<type>>(m, "SkipLayerNorm_" #type "_Tunable") \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&,    \
                    DeviceArray&, DeviceArray&,                                \
                    float, int, int>())                                        \
      .def("SetRepeats", &SkipLayerNormTunable<type>::SetRepeats)              \
      .def("Profile", &SkipLayerNormTunable<type>::Profile)                    \
      .def("Run", &SkipLayerNormTunable<type>::Run);

void InitSkipLayerNorm(py::module m) {
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(SkipLayerNorm, half);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(SkipLayerNorm, float);

  REGISTER_TUNABLE_OP(half);
  REGISTER_TUNABLE_OP(float);
}

}  // namespace onnxruntime
