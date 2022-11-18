// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file serve as a simple example for adding a tunable op to onnxruntime.
#include "python/tools/kernel_explorer/kernels/codegen/vector_add_triton.h"
#include "python/tools/kernel_explorer/kernels/codegen/vector_add_triton_kernel.h"

#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>

#include <string>

#include "core/providers/rocm/tunable/tunable.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

namespace py = pybind11;

namespace onnxruntime {

//#####################################################################################################################
// In practice, TritonVectorAddParam, TritonVectorAddOp and TritonVectorAddTunableOp should be tightly integrated to onnxruntime.
// We place them here purely for demo purpose.
//#####################################################################################################################

// Extend the OpParams so that all specializations have the same parameter passing interface
template <typename T>
struct TritonVectorAddParams : rocm::tunable::OpParams {
  std::string Signature() const override { return std::to_string(n); }

  T* x;
  T* y;
  T* z;
  int n;
};

// Wrap the kernel function, so that we have a unified launch interface. If the kernel has state, the state can also
// be managed at this level via a functor
template <typename T, int BS>
Status TritonVectorAddOp(const TritonVectorAddParams<T>* params) {
  return LaunchTritonVectorAdd<T, BS>(
      params->stream,
      params->x,
      params->y,
      params->z,
      params->n);
}

#define ADD_OP(block_size)                                \
  this->ops_.emplace_back(TritonVectorAddOp<T, block_size>);

// A Tunable TritonVectorAddOp is a collection of non-tunable TritonVectorAddOps implementations that have variable performance
// characteristics. Those implementations may be put into a C++ container for tuner to select.
template <typename T>
class TritonVectorAddTunableOp : public rocm::tunable::TunableOp<TritonVectorAddParams<T>> {
 public:
  TritonVectorAddTunableOp() {
    ADD_OP(1024);
  }

 private:
  // This Op is always tunable, you generally don't need to implement it.
  virtual bool Condition(const TritonVectorAddParams<T>* /*params*/) {
    return true;
  }
};

#undef ADD_OP

//#####################################################################################################################
// Following code just wraps our kernel implementation and expose them as python interface. This is the code that
// should be in the kernel_explorer directory.
//#####################################################################################################################

template <typename T, int BS>
class TritonVectorAdd : public IKernelExplorer {
 public:
  TritonVectorAdd(DeviceArray& x, DeviceArray& y, DeviceArray& z, int n) {
    params_.stream = Stream();
    params_.x = static_cast<T*>(x.ptr());
    params_.y = static_cast<T*>(y.ptr());
    params_.z = static_cast<T*>(z.ptr());
    params_.n = n;
  }

  void Run() override {
    ORT_THROW_IF_ERROR((TritonVectorAddOp<T, BS>(&params_)));
  }

 private:
 // A TritonVectorAddOp<T> is a callable that can process const TritonVectorAddParams<T>*
  using ParamsT = TritonVectorAddParams<T>;
  ParamsT params_{};
};

template <typename T>
class TritonVectorAddTunable : public IKernelExplorer {
 public:
  TritonVectorAddTunable(DeviceArray& x, DeviceArray& y, DeviceArray& z, int n) {
    params_.stream = Stream();
    params_.x = static_cast<T*>(x.ptr());
    params_.y = static_cast<T*>(y.ptr());
    params_.z = static_cast<T*>(z.ptr());
    params_.n = n;

    impl_.EnableTuning();
  }

  void Run() override {
    ORT_THROW_IF_ERROR(impl_(&params_));
  }

 private:
  using ParamsT = TritonVectorAddParams<T>;
  ParamsT params_;

  // tunable is stateful, store it as an instance
  TritonVectorAddTunableOp<T> impl_;
};


#define REGISTER_OP(name, type, block_size)                                             \
  py::class_<name<type, block_size>>(m, #name"_"#type"_"#block_size)  \
    .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int>())                                       \
    .def("SetRepeats", &name<type, block_size>::SetRepeats)                              \
    .def("Profile", &name<type, block_size>::Profile)                                    \
    .def("Run", &name<type, block_size>::Run);

#define REGISTER_OP_FOR_ALL_BLOCK_SIZE(name, type, block_size)  \
  REGISTER_OP(name, type, block_size)



#define REGISTER_TUNABLE_OP(type)                                      \
  py::class_<TritonVectorAddTunable<type>>(m, "TritonVectorAdd_" #type "_Tunable") \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, int>())  \
      .def("SetRepeats", &TritonVectorAddTunable<type>::SetRepeats)          \
      .def("Profile", &TritonVectorAddTunable<type>::Profile)                \
      .def("Run", &TritonVectorAddTunable<type>::Run);

void InitTritonVectorAdd(py::module m) {
  initTritonKernels<float,1024>(); 
  REGISTER_OP_FOR_ALL_BLOCK_SIZE(TritonVectorAdd, float, 1024);
  REGISTER_TUNABLE_OP(float)
}

}  // namespace onnxruntime
