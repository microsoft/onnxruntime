// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/tools/kernel_explorer/kernels/rocm/skip_layer_norm.h"

#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>

#include "contrib_ops/rocm/bert/skip_layer_norm_tunable_op.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

namespace py = pybind11;

namespace onnxruntime {

template <typename T, int ThreadsPerBlock, int VecSize, bool Simplified>
class SkipLayerNormSmall : public IKernelExplorer {
 public:
  SkipLayerNormSmall(DeviceArray& output, DeviceArray& skip_input_bias_add_output, DeviceArray& input, DeviceArray& skip,
                     DeviceArray& gamma, DeviceArray& beta, DeviceArray& bias,
                     float epsilon, int hidden_size, int element_count)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(skip_input_bias_add_output.ptr()), 
                static_cast<T*>(input.ptr()), static_cast<T*>(skip.ptr()), static_cast<T*>(gamma.ptr()), 
                static_cast<T*>(beta.ptr()), static_cast<T*>(bias.ptr()), epsilon, hidden_size, element_count) {}

  void Run() override {
    ORT_THROW_IF_ERROR((contrib::rocm::SkipLayerNormSmallOp<T, ThreadsPerBlock, VecSize, Simplified>(&params_)));
  }

  bool IsSupported() {
    Status status = contrib::rocm::SkipLayerNormSmallOp<T, ThreadsPerBlock, VecSize, Simplified>(&params_);
    return status.IsOK();
  }

 private:
  using ParamsT = contrib::rocm::SkipLayerNormParams<T>;
  ParamsT params_{};
};

template <typename T, int ThreadsPerBlock, int VecSize, bool Simplified>
class SkipLayerNormRegular : public IKernelExplorer {
 public:
  SkipLayerNormRegular(DeviceArray& output, DeviceArray& skip_input_bias_add_output, DeviceArray& input, DeviceArray& skip,
                       DeviceArray& gamma, DeviceArray& beta, DeviceArray& bias,
                       float epsilon, int hidden_size, int element_count)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(skip_input_bias_add_output.ptr()), 
                static_cast<T*>(input.ptr()), static_cast<T*>(skip.ptr()), static_cast<T*>(gamma.ptr()), 
                static_cast<T*>(beta.ptr()), static_cast<T*>(bias.ptr()), epsilon, hidden_size, element_count) {}

  void Run() override {
    ORT_THROW_IF_ERROR((contrib::rocm::SkipLayerNormRegularOp<T, ThreadsPerBlock, VecSize, Simplified>(&params_)));
  }

  bool IsSupported() {
    Status status = contrib::rocm::SkipLayerNormRegularOp<T, ThreadsPerBlock, VecSize, Simplified>(&params_);
    return status.IsOK();
  }

 private:
  using ParamsT = contrib::rocm::SkipLayerNormParams<T>;
  ParamsT params_{};
};

template <typename T, bool Simplified>
class SkipLayerNormStaticSelection : public IKernelExplorer {
 public:
  SkipLayerNormStaticSelection(DeviceArray& output, DeviceArray& skip_input_bias_add_output, DeviceArray& input, 
                               DeviceArray& skip, DeviceArray& gamma, DeviceArray& beta, DeviceArray& bias,
                               float epsilon, int hidden_size, int element_count)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(skip_input_bias_add_output.ptr()), 
                static_cast<T*>(input.ptr()), static_cast<T*>(skip.ptr()), static_cast<T*>(gamma.ptr()), 
                static_cast<T*>(beta.ptr()), static_cast<T*>(bias.ptr()), epsilon, hidden_size, element_count) {}

  void Run() override {
    ORT_THROW_IF_ERROR((contrib::rocm::SkipLayerNormStaticSelection<T, Simplified>(&params_)));
  }

  bool IsSupported() {
    Status status = contrib::rocm::SkipLayerNormStaticSelection<T, Simplified>(&params_);
    return status.IsOK();
  }

 private:
  using ParamsT = contrib::rocm::SkipLayerNormParams<T>;
  ParamsT params_{};
};

template <typename T, bool Simplified>
class SkipLayerNormTunable : public IKernelExplorer {
 public:
  SkipLayerNormTunable(DeviceArray& output, DeviceArray& skip_input_bias_add_output, DeviceArray& input, DeviceArray& skip,
                       DeviceArray& gamma, DeviceArray& beta, DeviceArray& bias,
                       float epsilon, int hidden_size, int element_count)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(skip_input_bias_add_output.ptr()), 
                static_cast<T*>(input.ptr()), static_cast<T*>(skip.ptr()), static_cast<T*>(gamma.ptr()), 
                static_cast<T*>(beta.ptr()), static_cast<T*>(bias.ptr()), epsilon, hidden_size, element_count) {

    params_.TuningContext()->EnableTunableOp();
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

  bool IsSupported() {
    return true;
  }

 private:
  using ParamsT = contrib::rocm::SkipLayerNormParams<T>;
  ParamsT params_{};
  contrib::rocm::SkipLayerNormTunableOp<T, Simplified> op_{};
};

#define REGISTER_OP(name, type, threads_per_block, vec_size, Simplified)                                                                    \
  py::class_<name<type, threads_per_block, vec_size, Simplified>>(m, #name "_" #type "_" #threads_per_block "_" #vec_size "_" #Simplified)  \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&,                                                                 \
                    DeviceArray&, DeviceArray&, DeviceArray&,                                                                               \
                    float, int, int>())                                                                                                     \
      .def("SetRepeats", &name<type, threads_per_block, vec_size, Simplified>::SetRepeats)                                                  \
      .def("Profile", &name<type, threads_per_block, vec_size, Simplified>::Profile)                                                        \
      .def("Run", &name<type, threads_per_block, vec_size, Simplified>::Run)                                                                \
      .def("IsSupported", &name<type, threads_per_block, vec_size, Simplified>::IsSupported);

#define REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, threads_per_block, Simplified) \
  REGISTER_OP(name, type, threads_per_block, 1, Simplified)                     \
  REGISTER_OP(name, type, threads_per_block, 2, Simplified)                     \
  REGISTER_OP(name, type, threads_per_block, 4, Simplified)                     \
  REGISTER_OP(name, type, threads_per_block, 8, Simplified)                     \
  REGISTER_OP(name, type, threads_per_block, 16, Simplified)

#define REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(name, type, Simplified) \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 64, Simplified)                         \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 128, Simplified)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 192, Simplified)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 256, Simplified)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 320, Simplified)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 384, Simplified)

#define REGISTER_OP_TYPED(name, type, Simplified)                           \
  py::class_<name<type, Simplified>>(m, #name "_" #type "_" #Simplified)    \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, \
                    DeviceArray&, DeviceArray&, DeviceArray&,               \
                    float, int, int>())                                     \
      .def("SetRepeats", &name<type, Simplified>::SetRepeats)               \
      .def("Profile", &name<type, Simplified>::Profile)                     \
      .def("Run", &name<type, Simplified>::Run)                             \
      .def("IsSupported", &name<type, Simplified>::IsSupported);

void InitSkipLayerNorm(py::module m) {
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormSmall, half, true);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormSmall, float, true);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormSmall, half, false);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormSmall, float, false);

  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormRegular, half, true);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormRegular, float, true);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormRegular, half, false);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormRegular, float, false);

  REGISTER_OP_TYPED(SkipLayerNormTunable, half, true);
  REGISTER_OP_TYPED(SkipLayerNormTunable, float, true);
  REGISTER_OP_TYPED(SkipLayerNormTunable, half, false);
  REGISTER_OP_TYPED(SkipLayerNormTunable, float, false);

  REGISTER_OP_TYPED(SkipLayerNormStaticSelection, half, true);
  REGISTER_OP_TYPED(SkipLayerNormStaticSelection, float, true);
  REGISTER_OP_TYPED(SkipLayerNormStaticSelection, half, false);
  REGISTER_OP_TYPED(SkipLayerNormStaticSelection, float, false);
}

}  // namespace onnxruntime
