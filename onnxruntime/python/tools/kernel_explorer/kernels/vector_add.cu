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

namespace py = pybind11;

namespace onnxruntime {

// #####################################################################################################################
//  In practice, VectorAddParam, VectorAddOp and VectorAddTunableOp should be tightly integrated to onnxruntime.
//  We place them here purely for demo purpose.
// #####################################################################################################################

// Extend the OpParams so that all specializations have the same parameter passing interface
template <typename T>
struct VectorAddParams :
#if USE_CUDA
    cuda::tunable::OpParams
#elif USE_ROCM
    rocm::tunable::OpParams
#endif
{
  std::string Signature() const override { return std::to_string(n); }

  T* x;
  T* y;
  T* z;
  int n;
};

// Wrap the kernel function, so that we have a unified launch interface. If the kernel has state, the state can also
// be managed at this level via a functor
template <typename T, int TPB, int Vec>
Status VectorAddOp(const VectorAddParams<T>* params) {
  return LaunchVectorAdd<T, TPB, Vec>(
      params->StreamHandle(),
      params->x,
      params->y,
      params->z,
      params->n);
}

#define ADD_OP(threads_per_block)                         \
  this->RegisterOp(VectorAddOp<T, threads_per_block, 1>); \
  this->RegisterOp(VectorAddOp<T, threads_per_block, 2>); \
  this->RegisterOp(VectorAddOp<T, threads_per_block, 4>); \
  this->RegisterOp(VectorAddOp<T, threads_per_block, 8>);

// A Tunable VectorAddOp is a collection of non-tunable VectorAddOps implementations that have variable performance
// characteristics. Those implementations may be put into a C++ container for tuner to select.
template <typename T>
class VectorAddTunableOp :
#if USE_CUDA
    public cuda::tunable::TunableOp<VectorAddParams<T>>
#elif USE_ROCM
    public rocm::tunable::TunableOp<VectorAddParams<T>>
#endif
{
 public:
  VectorAddTunableOp() {
    ADD_OP(64);
    ADD_OP(128);
    ADD_OP(192);
    ADD_OP(256);
    ADD_OP(320);
    ADD_OP(384);
    ADD_OP(448);
    ADD_OP(512);
  }
};

#undef ADD_OP

// #####################################################################################################################
//  Following code just wraps our kernel implementation and expose them as python interface. This is the code that
//  should be in the kernel_explorer directory.
// #####################################################################################################################

template <typename T, int TPB, int Vec>
class VectorAdd : public IKernelExplorer {
 public:
  VectorAdd(DeviceArray& x, DeviceArray& y, DeviceArray& z, int n) {
    params_.tuning_ctx = TuningContext();
    params_.stream = Stream();
    params_.x = static_cast<T*>(x.ptr());
    params_.y = static_cast<T*>(y.ptr());
    params_.z = static_cast<T*>(z.ptr());
    params_.n = n;
  }

  void Run() override {
    ORT_THROW_IF_ERROR((VectorAddOp<T, TPB, Vec>(&params_)));
  }

 private:
  // A VectorAddOp<T> is a callable that can process const VectorAddParams<T>*
  using ParamsT = VectorAddParams<T>;
  ParamsT params_{};
};

template <typename T>
class VectorAddTunable : public IKernelExplorer {
 public:
  VectorAddTunable(DeviceArray& x, DeviceArray& y, DeviceArray& z, int n) {
    params_.tuning_ctx = TuningContext();
    params_.stream = Stream();
    params_.x = static_cast<T*>(x.ptr());
    params_.y = static_cast<T*>(y.ptr());
    params_.z = static_cast<T*>(z.ptr());
    params_.n = n;

    params_.TuningContext()->EnableTunableOpAndTuning();
  }

  void Run() override {
    ORT_THROW_IF_ERROR(impl_(&params_));
  }

 private:
  using ParamsT = VectorAddParams<T>;
  ParamsT params_;

  // tunable is stateful, store it as an instance
  VectorAddTunableOp<T> impl_;
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

#define REGISTER_TUNABLE_OP(type)                                      \
  py::class_<VectorAddTunable<type>>(m, "VectorAdd_" #type "_Tunable") \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int>())  \
      .def("SetRepeats", &VectorAddTunable<type>::SetRepeats)          \
      .def("Profile", &VectorAddTunable<type>::Profile)                \
      .def("Run", &VectorAddTunable<type>::Run);

KE_REGISTER(m) {
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(VectorAdd, half);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK(VectorAdd, float);

  REGISTER_TUNABLE_OP(half);
  REGISTER_TUNABLE_OP(float)
}

}  // namespace onnxruntime
