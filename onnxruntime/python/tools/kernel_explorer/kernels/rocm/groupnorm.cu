// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdio.h>
#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>

#include "contrib_ops/rocm/diffusion/group_norm_common.h"
#include "contrib_ops/rocm/diffusion/group_norm_tunable_op.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

namespace py = pybind11;

namespace onnxruntime {

template <typename T, int ThreadsPerBlock, int VecSize>
class GroupNormNHWC : public IKernelExplorer {
 public:
  GroupNormNHWC(DeviceArray& output, DeviceArray& workspace, DeviceArray& input, DeviceArray& gamma, DeviceArray& beta,
                int batch_size, int height, int width, int num_channels, int num_groups, float epsilon, bool use_swish)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<float*>(workspace.ptr()),
                static_cast<T*>(input.ptr()), static_cast<float*>(gamma.ptr()), static_cast<float*>(beta.ptr()),
                batch_size, height, width, num_channels, num_groups, epsilon, use_swish) {}

  bool IsSupported() {
    Status status = op_.IsSupported(&params_);
    return status.IsOK();
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

 private:
  using ParamsT = contrib::rocm::GroupNormNHWCParams<T>;
  ParamsT params_{};
  contrib::rocm::GroupNormNHWCOp<T, ThreadsPerBlock, VecSize> op_{};
};

template <typename T>
class GroupNormNHWCStaticSelection : public IKernelExplorer {
 public:
  GroupNormNHWCStaticSelection(DeviceArray& output, DeviceArray& workspace, DeviceArray& input, DeviceArray& gamma, DeviceArray& beta,
                               int batch_size, int height, int width, int num_channels, int num_groups, float epsilon, bool use_swish)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<float*>(workspace.ptr()),
                static_cast<T*>(input.ptr()), static_cast<float*>(gamma.ptr()), static_cast<float*>(beta.ptr()),
                batch_size, height, width, num_channels, num_groups, epsilon, use_swish) {}

  void Run() override {
    ORT_THROW_IF_ERROR((contrib::rocm::GroupNormNHWCStaticSelection<T>(&params_)));
  }

  bool IsSupported() {
    Status status = contrib::rocm::GroupNormNHWCStaticSelection<T>(&params_);
    return status.IsOK();
  }

 private:
  using ParamsT = contrib::rocm::GroupNormNHWCParams<T>;
  ParamsT params_{};
};

template <typename T>
class GroupNormNHWCTunable : public IKernelExplorer {
 public:
  GroupNormNHWCTunable(DeviceArray& output, DeviceArray& workspace, DeviceArray& input, DeviceArray& gamma, DeviceArray& beta,
                       int batch_size, int height, int width, int num_channels, int num_groups, float epsilon, bool use_swish)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<float*>(workspace.ptr()),
                static_cast<T*>(input.ptr()), static_cast<float*>(gamma.ptr()), static_cast<float*>(beta.ptr()),
                batch_size, height, width, num_channels, num_groups, epsilon, use_swish) {
    params_.TuningContext()->EnableTunableOp();
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

  bool IsSupported() {
    return true;
  }

 private:
  using ParamsT = contrib::rocm::GroupNormNHWCParams<T>;
  ParamsT params_{};
  contrib::rocm::GroupNormNHWCTunableOp<T> op_{};
};

#define REGISTER_OP(name, type, threads_per_block, vec_size)                                                   \
  py::class_<name<type, threads_per_block, vec_size>>(m, #name "_" #type "_" #threads_per_block "_" #vec_size) \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&,                      \
                    int, int, int, int, int, float, bool>())                                                   \
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
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 384)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 448)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 512)

#define REGISTER_OP_TYPED(name, type)                                                     \
  py::class_<name<type>>(m, #name "_" #type)                                              \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, \
                    int, int, int, int, int, float, bool>())                              \
      .def("SetRepeats", &name<type>::SetRepeats)                                         \
      .def("Profile", &name<type>::Profile)                                               \
      .def("Run", &name<type>::Run)                                                       \
      .def("IsSupported", &name<type>::IsSupported);

KE_REGISTER(m) {
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(GroupNormNHWC, half);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(GroupNormNHWC, float);

  REGISTER_OP_TYPED(GroupNormNHWCTunable, half);
  REGISTER_OP_TYPED(GroupNormNHWCTunable, float);

  REGISTER_OP_TYPED(GroupNormNHWCStaticSelection, half);
  REGISTER_OP_TYPED(GroupNormNHWCStaticSelection, float);
}

}  // namespace onnxruntime
