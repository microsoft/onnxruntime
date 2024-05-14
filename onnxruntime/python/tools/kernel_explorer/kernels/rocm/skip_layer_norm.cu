// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
    ORT_THROW_IF_ERROR((contrib::rocm::SkipLayerNormSmallOp<T, float, T, ThreadsPerBlock, VecSize, Simplified>(&params_)));
  }

  bool IsSupported() {
    Status status = contrib::rocm::SkipLayerNormSmallOp<T, float, T, ThreadsPerBlock, VecSize, Simplified>(&params_);
    return status.IsOK();
  }

 private:
  using ParamsT = contrib::rocm::SkipLayerNormParams<T, T>;
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
    ORT_THROW_IF_ERROR((contrib::rocm::SkipLayerNormRegularOp<T, float, T, ThreadsPerBlock, VecSize, Simplified>(&params_)));
  }

  bool IsSupported() {
    Status status = contrib::rocm::SkipLayerNormRegularOp<T, float, T, ThreadsPerBlock, VecSize, Simplified>(&params_);
    return status.IsOK();
  }

 private:
  using ParamsT = contrib::rocm::SkipLayerNormParams<T, T>;
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
    ORT_THROW_IF_ERROR((contrib::rocm::SkipLayerNormStaticSelection<T, float, T, Simplified>(&params_)));
  }

  bool IsSupported() {
    Status status = contrib::rocm::SkipLayerNormStaticSelection<T, float, T, Simplified>(&params_);
    return status.IsOK();
  }

 private:
  using ParamsT = contrib::rocm::SkipLayerNormParams<T, T>;
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
    params_.TuningContext()->EnableTunableOpAndTuning();
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

  bool IsSupported() {
    return true;
  }

 private:
  using ParamsT = contrib::rocm::SkipLayerNormParams<T, T>;
  ParamsT params_{};
  contrib::rocm::SkipLayerNormTunableOp<T, float, T, Simplified> op_{};
};

#define REGISTER_OP_COMMON(name, type, ...)                                 \
  py::class_<type<__VA_ARGS__>>(m, name)                                    \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, \
                    DeviceArray&, DeviceArray&, DeviceArray&,               \
                    float, int, int>())                                     \
      .def("SetRepeats", &type<__VA_ARGS__>::SetRepeats)                    \
      .def("Profile", &type<__VA_ARGS__>::Profile)                          \
      .def("Run", &type<__VA_ARGS__>::Run)                                  \
      .def("IsSupported", &type<__VA_ARGS__>::IsSupported);

#define REGISTER_OP(name, type, threads_per_block, vec_size)                                                                           \
  REGISTER_OP_COMMON("Simplified" #name "_" #type "_" #threads_per_block "_" #vec_size, name, type, threads_per_block, vec_size, true) \
  REGISTER_OP_COMMON(#name "_" #type "_" #threads_per_block "_" #vec_size, name, type, threads_per_block, vec_size, false)

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
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 512)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 576)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 640)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 704)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 768)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 832)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 896)                        \
  REGISTER_OP_FOR_ALL_VEC_SIZE(name, type, 1024)

#define REGISTER_COMMON(name, type, ...)                                    \
  py::class_<type<__VA_ARGS__>>(m, name)                                    \
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, \
                    DeviceArray&, DeviceArray&, DeviceArray&,               \
                    float, int, int>())                                     \
      .def("SetRepeats", &type<__VA_ARGS__>::SetRepeats)                    \
      .def("Profile", &type<__VA_ARGS__>::Profile)                          \
      .def("Run", &type<__VA_ARGS__>::Run)                                  \
      .def("IsSupported", &type<__VA_ARGS__>::IsSupported);

#define REGISTER_OP_TYPED(name, type)                             \
  REGISTER_COMMON("Simplified" #name "_" #type, name, type, true) \
  REGISTER_COMMON(#name "_" #type, name, type, false)

KE_REGISTER(m) {
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormSmall, half);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormSmall, float);

  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormRegular, half);
  REGISTER_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormRegular, float);

  REGISTER_OP_TYPED(SkipLayerNormTunable, half);
  REGISTER_OP_TYPED(SkipLayerNormTunable, float);

  REGISTER_OP_TYPED(SkipLayerNormStaticSelection, half);
  REGISTER_OP_TYPED(SkipLayerNormStaticSelection, float);
}

}  // namespace onnxruntime
