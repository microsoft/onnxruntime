// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>
#include "python/tools/kernel_explorer/kernels/vector_add.h"
#include "python/tools/kernel_explorer/kernels/vector_add_kernel.h"
#include "python/tools/kernel_explorer/kernels/vector_add_tunable_op.h"

namespace py = pybind11;

namespace onnxruntime {

template <typename T, int ThreadsPerBlock, int VecSize>
class VectorAdd: public Operator {
 public:
  VectorAdd(DeviceArray& x, DeviceArray& y, DeviceArray& z, int n) :
    x_(reinterpret_cast<T*>(x.ptr())),
    y_(reinterpret_cast<T*>(y.ptr())),
    z_(reinterpret_cast<T*>(z.ptr())),
    n_(n) {}

  void Run() {
    LaunchVectorAdd<T, ThreadsPerBlock, VecSize>(stream_, x_, y_, z_, n_);
  }

 private:
  T* x_;
  T* y_;
  T* z_;
  int n_;
};

template <typename T>
class VectorAddTunable: public Operator {
 public:
  VectorAddTunable(DeviceArray& x, DeviceArray& y, DeviceArray& z, int n) :
    x_(reinterpret_cast<T*>(x.ptr())),
    y_(reinterpret_cast<T*>(y.ptr())),
    z_(reinterpret_cast<T*>(z.ptr())),
    n_(n) {
    op_.EnableTuning();
  }

  void Run() {
    VectorAddParams<T> op_params(stream_, x_, y_, z_, n_);
    op_.Run(&op_params);
  }

 private:
  T* x_;
  T* y_;
  T* z_;
  int n_;
  VectorAddTunableOp<T> op_;
};

#define REGISTER_OP(name, type, threads_per_block, vec_size)                                              \
  py::class_<name<type, threads_per_block, vec_size>>(m, #name"_"#type"_"#threads_per_block"_"#vec_size)  \
    .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int>())                                       \
    .def("SetRepeats", &name<type, threads_per_block, vec_size>::SetRepeats)                              \
    .def("Profile", &name<type, threads_per_block, vec_size>::Profile)                                    \
    .def("Run", &name<type, threads_per_block, vec_size>::Run);

#define REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, threads_per_block)  \
  REGISTER_OP(name, type, threads_per_block, 1)                      \
  REGISTER_OP(name, type, threads_per_block, 2)                      \
  REGISTER_OP(name, type, threads_per_block, 4)                      \
  REGISTER_OP(name, type, threads_per_block, 8)

#define REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(name, type)  \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 64)             \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 128)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 192)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 256)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 320)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 384)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 448)            \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 512)

void InitVectorAdd(py::module m) {
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(VectorAdd, half);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(VectorAdd, float);
  py::class_<VectorAddTunable<half>>(m, "VectorAdd_half_Tunable")
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int>())
      .def("SetRepeats", &VectorAddTunable<half>::SetRepeats)
      .def("Profile", &VectorAddTunable<half>::Profile)
      .def("Run", &VectorAddTunable<half>::Run);
}

}  // namespace onnxruntime
