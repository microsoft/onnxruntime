// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <hip/hip_fp16.h>
#include "contrib_ops/rocm/bert/fast_gelu_impl_kernel.h"
#include "contrib_ops/rocm/bert/fast_gelu_tunable_op.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

namespace py = pybind11;

namespace onnxruntime {

template <typename T, int ThreadsPerBlock, int VecSize>
class FastGelu : public IKernelExplorer {
 public:
  FastGelu(DeviceArray& input, DeviceArray& bias, DeviceArray& output, int input_length, int bias_length)
      : params_(TuningContext(), Stream(), static_cast<T*>(input.ptr()), static_cast<T*>(bias.ptr()),
                static_cast<T*>(output.ptr()), input_length, bias_length) {}

  bool IsSupported() {
    Status status = op_.IsSupported(&params_);
    return status.IsOK();
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

 private:
  using ParamsT = contrib::rocm::FastGeluParams<T>;
  ParamsT params_{};
  contrib::rocm::FastGeluOp<T, ThreadsPerBlock, VecSize> op_{};
};

template <typename T>
class FastGeluStaticSelection : public IKernelExplorer {
 public:
  FastGeluStaticSelection(DeviceArray& input, DeviceArray& bias, DeviceArray& output, int input_length, int bias_length)
      : params_(TuningContext(), Stream(), static_cast<T*>(input.ptr()), static_cast<T*>(bias.ptr()),
                static_cast<T*>(output.ptr()), input_length, bias_length) {}

  bool IsSupported() {
    return true;
  }

  void Run() override {
    ORT_THROW_IF_ERROR((contrib::rocm::FastGeluStaticSelection<T>(&params_)));
  }

 private:
  using ParamsT = contrib::rocm::FastGeluParams<T>;
  ParamsT params_{};
};

template <typename T>
class FastGeluTunable : public IKernelExplorer {
 public:
  FastGeluTunable(DeviceArray& input, DeviceArray& bias, DeviceArray& output, int input_length, int bias_length)
      : params_(TuningContext(), Stream(), static_cast<T*>(input.ptr()), static_cast<T*>(bias.ptr()),
                static_cast<T*>(output.ptr()), input_length, bias_length) {
    params_.TuningContext()->EnableTunableOpAndTuning();
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

  bool IsSupported() {
    return true;
  }

 private:
  using ParamsT = contrib::rocm::FastGeluParams<T>;
  ParamsT params_{};
  contrib::rocm::FastGeluTunableOp<T> op_{};
};

#define REGISTER_OP_COMMON(name, type)                                     \
  KE_REGISTER_OP_COMMON(m, name, TEMPLATED_TYPENAME(type))                 \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int, int>()) \
      .def("IsSupported", &type::IsSupported);

#define REGISTER_OP(dtype, threads_per_block, vec_size)                       \
  REGISTER_OP_COMMON("FastGelu_" #dtype "_" #threads_per_block "_" #vec_size, \
                     TEMPLATED_TYPENAME(FastGelu<dtype, threads_per_block, vec_size>))

#define REGISTER_OP_FOR_ALL_VEC_SIZE(dtype, threads_per_block) \
  REGISTER_OP(dtype, threads_per_block, 1)                     \
  REGISTER_OP(dtype, threads_per_block, 2)                     \
  REGISTER_OP(dtype, threads_per_block, 4)                     \
  REGISTER_OP(dtype, threads_per_block, 8)                     \
  REGISTER_OP(dtype, threads_per_block, 16)

#define REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(dtype) \
  REGISTER_OP_FOR_ALL_VEC_SIZE(dtype, 64)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(dtype, 128)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(dtype, 192)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(dtype, 256)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(dtype, 320)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(dtype, 384)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(dtype, 448)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(dtype, 512)

#define REGISTER_OP_TYPED(tpl, dtype) REGISTER_OP_COMMON(#tpl "_" #dtype, tpl<dtype>)

KE_REGISTER(m) {
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(half);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(float);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(double);

  REGISTER_OP_TYPED(FastGeluTunable, half);
  REGISTER_OP_TYPED(FastGeluTunable, float);
  REGISTER_OP_TYPED(FastGeluTunable, double);

  REGISTER_OP_TYPED(FastGeluStaticSelection, half);
  REGISTER_OP_TYPED(FastGeluStaticSelection, float);
  REGISTER_OP_TYPED(FastGeluStaticSelection, double);
}

}  // namespace onnxruntime
