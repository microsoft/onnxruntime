// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/rocm/softmax.h"

#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>

#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"
#include "core/providers/rocm/math/softmax_tunable_op.cuh"
#include "core/providers/rocm/shared_inc/accumulation_type.h"

namespace py = pybind11;

namespace onnxruntime {

template <typename T, int VecSize>
class SoftmaxBlockwise : public IKernelExplorer {
 public:
  SoftmaxBlockwise(DeviceArray& output, DeviceArray& input, int softmax_elements,
                   int input_stride, int output_stride, int batch_count, bool is_log_softmax)
      : params_(this->Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(input.ptr()),
                softmax_elements, input_stride, output_stride, batch_count, is_log_softmax) {}

  void Run() override {
    ORT_THROW_IF_ERROR((rocm::SoftmaxBlockwiseOp<T, T, rocm::AccumulationType_t<T>, VecSize>(&params_)));
  }

  bool IsSupported() {
    Status status = rocm::SoftmaxBlockwiseOp<T, T, rocm::AccumulationType_t<T>, VecSize>(&params_);
    return status.IsOK();
  }

 private:
  using ParamsT = rocm::SoftmaxParams<T, T>;
  ParamsT params_{};
};

template <typename T>
class SoftmaxBlockwiseStaticSelection : public IKernelExplorer {
 public:
  SoftmaxBlockwiseStaticSelection(DeviceArray& output, DeviceArray& input, int softmax_elements,
                                  int input_stride, int output_stride, int batch_count, bool is_log_softmax)
      : params_(this->Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(input.ptr()),
                softmax_elements, input_stride, output_stride, batch_count, is_log_softmax) {}

  bool IsSupported() {
    return true;
  }

  void Run() override {
    ORT_THROW_IF_ERROR((rocm::SoftmaxBlockwiseStaticSelection<T, T, rocm::AccumulationType_t<T>>(&params_)));
  }

 private:
  using ParamsT = rocm::SoftmaxParams<T, T>;
  ParamsT params_{};
};

template <typename T>
class SoftmaxTunable : public IKernelExplorer {
 public:
  SoftmaxTunable(DeviceArray& output, DeviceArray& input, int softmax_elements,
                 int input_stride, int output_stride, int batch_count, bool is_log_softmax)
      : params_(this->Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(input.ptr()),
                softmax_elements, input_stride, output_stride, batch_count, is_log_softmax) {
    op_.EnableTuning();
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

  bool IsSupported() {
    return true;
  }

 private:
  using ParamsT = rocm::SoftmaxParams<T, T>;
  ParamsT params_{};
  rocm::SoftmaxTunableOp<T, T, rocm::AccumulationType_t<T>> op_{};
};

#define REGISTER_OP(name, type, vec_size)                                    \
  py::class_<name<type, vec_size>>(m, #name "_" #type "_" #vec_size)         \
      .def(py::init<DeviceArray&, DeviceArray&, int, int, int, int, bool>()) \
      .def("SetRepeats", &name<type, vec_size>::SetRepeats)                  \
      .def("Profile", &name<type, vec_size>::Profile)                        \
      .def("Run", &name<type, vec_size>::Run)                                \
      .def("IsSupported", &name<type, vec_size>::IsSupported);

#define REGISTER_OP_FOR_ALL_VEC_SIZE(name, type) \
  REGISTER_OP(name, type, 1)                     \
  REGISTER_OP(name, type, 2)                     \
  REGISTER_OP(name, type, 4)                     \
  REGISTER_OP(name, type, 8)                     \
  REGISTER_OP(name, type, 16)

#define REGISTER_OP_TYPED(name, type)                                        \
  py::class_<name<type>>(m, #name "_" #type)                                 \
      .def(py::init<DeviceArray&, DeviceArray&, int, int, int, int, bool>()) \
      .def("SetRepeats", &name<type>::SetRepeats)                            \
      .def("Profile", &name<type>::Profile)                                  \
      .def("Run", &name<type>::Run)                                          \
      .def("IsSupported", &name<type>::IsSupported);

void InitSoftmax(py::module m) {
  REGISTER_OP_FOR_ALL_VEC_SIZE(SoftmaxBlockwise, half);
  REGISTER_OP_FOR_ALL_VEC_SIZE(SoftmaxBlockwise, float);

  REGISTER_OP_TYPED(SoftmaxBlockwiseStaticSelection, half);
  REGISTER_OP_TYPED(SoftmaxBlockwiseStaticSelection, float);

  REGISTER_OP_TYPED(SoftmaxTunable, half);
  REGISTER_OP_TYPED(SoftmaxTunable, float);
}

}  // namespace onnxruntime
