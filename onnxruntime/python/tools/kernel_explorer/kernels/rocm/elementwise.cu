// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <hip/hip_fp16.h>
#include "contrib_ops/rocm/bert/elementwise.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

namespace py = pybind11;
using namespace onnxruntime::contrib::rocm;

namespace onnxruntime {

template <typename Fn, typename T, int ThreadsPerBlock, int VecSize>
class Elementwise : public IKernelExplorer {
 public:
  Elementwise(DeviceArray& input, DeviceArray& bias, DeviceArray& output, int input_length, int bias_length)
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
  using ParamsT = internal::ElementwiseParams<T>;
  ParamsT params_{};
  internal::ElementwiseOp<Fn, T, ThreadsPerBlock, VecSize> op_{};
};

template <typename Fn, typename T>
class ElementwiseStaticSelection : public IKernelExplorer {
 public:
  ElementwiseStaticSelection(DeviceArray& input, DeviceArray& bias, DeviceArray& output, int input_length, int bias_length)
      : params_(TuningContext(), Stream(), static_cast<T*>(input.ptr()), static_cast<T*>(bias.ptr()),
                static_cast<T*>(output.ptr()), input_length, bias_length) {}

  bool IsSupported() {
    return true;
  }

  void Run() override {
    ORT_THROW_IF_ERROR((internal::ElementwiseStaticSelection<Fn, T>(&params_)));
  }

 private:
  using ParamsT = internal::ElementwiseParams<T>;
  ParamsT params_{};
};

template <typename Fn, typename T>
class ElementwiseTunable : public IKernelExplorer {
 public:
  ElementwiseTunable(DeviceArray& input, DeviceArray& bias, DeviceArray& output, int input_length, int bias_length)
      : params_(TuningContext(), Stream(), static_cast<T*>(input.ptr()), static_cast<T*>(bias.ptr()),
                static_cast<T*>(output.ptr()), input_length, bias_length) {
    params_.TuningContext()->EnableTunableOpAndTuning();
  }

  void Run() override {
    WithMaxTuningDurationMs max_duration(TuningContext(), 250);
    ORT_THROW_IF_ERROR(op_(&params_));
  }

  bool IsSupported() {
    return true;
  }

 private:
  using ParamsT = internal::ElementwiseParams<T>;
  ParamsT params_{};
  internal::ElementwiseTunableOp<Fn, T> op_{};
};

#define REGISTER_OP(registered_name, tpl, functor_name, dtype, threads_per_block, vec_size)           \
  py::class_<tpl<functor::functor_name, dtype, threads_per_block, vec_size>>(                         \
      m, #registered_name "_" #dtype "_" #threads_per_block "_" #vec_size)                            \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int, int>())                            \
      .def("SetRepeats", &tpl<functor::functor_name, dtype, threads_per_block, vec_size>::SetRepeats) \
      .def("Profile", &tpl<functor::functor_name, dtype, threads_per_block, vec_size>::Profile)       \
      .def("Run", &tpl<functor::functor_name, dtype, threads_per_block, vec_size>::Run)               \
      .def("IsSupported", &tpl<functor::functor_name, dtype, threads_per_block, vec_size>::IsSupported);

#define REGISTER_OP_FOR_ALL_VEC_SIZE(registered_name, tpl, functor_name, dtype, threads_per_block) \
  REGISTER_OP(functor_name, tpl, functor_name, dtype, threads_per_block, 1)                        \
  REGISTER_OP(functor_name, tpl, functor_name, dtype, threads_per_block, 2)                        \
  REGISTER_OP(functor_name, tpl, functor_name, dtype, threads_per_block, 4)                        \
  REGISTER_OP(functor_name, tpl, functor_name, dtype, threads_per_block, 8)                        \
  REGISTER_OP(functor_name, tpl, functor_name, dtype, threads_per_block, 16)

#define REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(registered_name, tpl, functor_name, dtype) \
  REGISTER_OP_FOR_ALL_VEC_SIZE(registered_name, tpl, functor_name, dtype, 64)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(registered_name, tpl, functor_name, dtype, 128)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(registered_name, tpl, functor_name, dtype, 192)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(registered_name, tpl, functor_name, dtype, 256)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(registered_name, tpl, functor_name, dtype, 320)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(registered_name, tpl, functor_name, dtype, 384)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(registered_name, tpl, functor_name, dtype, 448)           \
  REGISTER_OP_FOR_ALL_VEC_SIZE(registered_name, tpl, functor_name, dtype, 512)

#define REGISTER_OP_TYPED(registered_name, tpl, functor_name, dtype)           \
  py::class_<tpl<functor::functor_name, dtype>>(m, registered_name "_" #dtype) \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int, int>())     \
      .def("SetRepeats", &tpl<functor::functor_name, dtype>::SetRepeats)       \
      .def("Profile", &tpl<functor::functor_name, dtype>::Profile)             \
      .def("Run", &tpl<functor::functor_name, dtype>::Run)                     \
      .def("IsSupported", &tpl<functor::functor_name, dtype>::IsSupported);

KE_REGISTER(m) {
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK("FastGeLU", Elementwise, FastGeLU, half);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK("FastGeLU", Elementwise, FastGeLU, float);

  REGISTER_OP_TYPED("FastGeLUTunable", ElementwiseTunable, FastGeLU, half);
  REGISTER_OP_TYPED("FastGeLUTunable", ElementwiseTunable, FastGeLU, float);

  REGISTER_OP_TYPED("FastGeLUStaticSelection", ElementwiseStaticSelection, FastGeLU, half);
  REGISTER_OP_TYPED("FastGeLUStaticSelection", ElementwiseStaticSelection, FastGeLU, float);
}

KE_REGISTER(m) {
REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK("GeLU", Elementwise, GeLU, half);
REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK("GeLU", Elementwise, GeLU, float);

REGISTER_OP_TYPED("GeLUTunable", ElementwiseTunable, GeLU, half);
REGISTER_OP_TYPED("GeLUTunable", ElementwiseTunable, GeLU, float);

REGISTER_OP_TYPED("GeLUStaticSelection", ElementwiseStaticSelection, GeLU, half);
REGISTER_OP_TYPED("GeLUStaticSelection", ElementwiseStaticSelection, GeLU, float);
}

KE_REGISTER(m) {
REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK("ReLU", Elementwise, ReLU, half);
REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK("ReLU", Elementwise, ReLU, float);

REGISTER_OP_TYPED("ReLUTunable", ElementwiseTunable, ReLU, half);
REGISTER_OP_TYPED("ReLUTunable", ElementwiseTunable, ReLU, float);

REGISTER_OP_TYPED("ReLUStaticSelection", ElementwiseStaticSelection, ReLU, half);
REGISTER_OP_TYPED("ReLUStaticSelection", ElementwiseStaticSelection, ReLU, float);
}

}  // namespace onnxruntime
