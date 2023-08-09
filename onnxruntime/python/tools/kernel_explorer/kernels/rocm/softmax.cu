// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

#include "core/providers/rocm/math/softmax_ck.cuh"
#include "core/providers/rocm/math/softmax_tunable_op.cuh"
#include "core/providers/rocm/shared_inc/accumulation_type.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"
#include "core/providers/rocm/math/softmax_triton.cuh"

namespace py = pybind11;

namespace onnxruntime {

template <typename T, int VecSize>
class SoftmaxBlockwise : public IKernelExplorer {
 public:
  SoftmaxBlockwise(DeviceArray& output, DeviceArray& input, int softmax_elements,
                   int input_stride, int output_stride, int batch_count, bool is_log_softmax)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(input.ptr()),
                softmax_elements, input_stride, output_stride, batch_count, is_log_softmax) {
    type_string_ = "SoftmaxBlockwise_" + std::to_string(VecSize);
  }

  void Run() override {
    ORT_THROW_IF_ERROR((rocm::SoftmaxBlockwiseOp<T, T, rocm::AccumulationType_t<T>, VecSize>(&params_)));
  }

  std::vector<std::string> ListOps() const {
    return {type_string_};
  }

  bool SelectOp(const std::string& name) {
    Status status = rocm::SoftmaxBlockwiseOp<T, T, rocm::AccumulationType_t<T>, VecSize>(&params_);
    return status.IsOK() && name == type_string_;
  }

 private:
  using ParamsT = rocm::SoftmaxParams<T, T>;
  ParamsT params_{};
  std::string type_string_{};
};

template <typename T>
class SoftmaxWarpwiseStaticSelection : public IKernelExplorer {
 public:
  SoftmaxWarpwiseStaticSelection(DeviceArray& output, DeviceArray& input, int softmax_elements,
                                 int input_stride, int output_stride, int batch_count, bool is_log_softmax)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(input.ptr()),
                softmax_elements, input_stride, output_stride, batch_count, is_log_softmax) {}

  void Run() override {
    ORT_THROW_IF_ERROR((rocm::SoftmaxWarpwiseStaticSelection<T, T, rocm::AccumulationType_t<T>>(&params_)));
  }

  std::vector<std::string> ListOps() const {
    return {"SoftmaxWarpwiseStaticSelection"};
  }

  bool SelectOp(const std::string& name) {
    auto status = rocm::SoftmaxWarpwiseStaticSelection<T, T, rocm::AccumulationType_t<T>>(&params_);
    return status.IsOK() && name == "SoftmaxWarpwiseStaticSelection";
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
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(input.ptr()),
                softmax_elements, input_stride, output_stride, batch_count, is_log_softmax) {}

  void Run() override {
    ORT_THROW_IF_ERROR((rocm::SoftmaxBlockwiseStaticSelection<T, T, rocm::AccumulationType_t<T>>(&params_)));
  }

  std::vector<std::string> ListOps() const {
    return {"SoftmaxBlockwiseStaticSelection"};
  }

  bool SelectOp(const std::string& name) {
    return name == "SoftmaxBlockwiseStaticSelection";
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
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(input.ptr()),
                softmax_elements, input_stride, output_stride, batch_count, is_log_softmax) {
    params_.TuningContext()->EnableTunableOpAndTuning();
  }

  void Run() override {
    ORT_THROW_IF_ERROR(op_(&params_));
  }

  std::vector<std::string> ListOps() const {
    return {"SoftmaxTunable"};
  }

  bool SelectOp(const std::string& name) {
    return name == "SoftmaxTunable";
  }

 private:
  using ParamsT = rocm::SoftmaxParams<T, T>;
  ParamsT params_{};
  rocm::SoftmaxTunableOp<T, T, rocm::AccumulationType_t<T>> op_{};
};

#ifdef USE_COMPOSABLE_KERNEL
template <typename T>
class CKSoftmax : public IKernelExplorer {
 public:
  CKSoftmax(DeviceArray& output, DeviceArray& input, int softmax_elements,
            int input_stride, int output_stride, int batch_count, bool is_log_softmax)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(input.ptr()),
                softmax_elements, input_stride, output_stride, batch_count, is_log_softmax) {
    for (auto&& [type_string, op] : rocm::GetCKSoftmaxTypeStringAndOps<T, T, rocm::AccumulationType_t<T>>()) {
      type_strings_.emplace_back(std::move(type_string));
      ops_.emplace_back(std::move(op));
    }
  }

  void Run() override {
    ORT_THROW_IF_ERROR(ops_[selected_op_](&params_));
  }

  std::vector<std::string> ListOps() const {
    return type_strings_;
  }

  bool SelectOp(const std::string& name) {
    for (size_t i = 0; i < ops_.size(); i++) {
      if (type_strings_[i] == name) {
        selected_op_ = i;
        Status status = ops_[i](&params_);
        return status.IsOK();
      }
    }

    ORT_THROW("Cannot find implementation ", name);
  }

 private:
  using ParamsT = rocm::SoftmaxParams<T, T>;
  using OpT = rocm::tunable::Op<ParamsT>;
  ParamsT params_{};
  std::vector<OpT> ops_;
  std::vector<std::string> type_strings_;
  size_t selected_op_{};
};
#endif  // USE_COMPOSABLE_KERNEL

#ifdef USE_TRITON_KERNEL
template <typename T>
class SoftmaxTriton : public IKernelExplorer {
 public:
  SoftmaxTriton(DeviceArray& output, DeviceArray& input, int softmax_elements,
                int input_stride, int output_stride, int batch_count, bool is_log_softmax)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(input.ptr()),
                softmax_elements, input_stride, output_stride, batch_count, is_log_softmax) {
    for (auto&& [name, op] : rocm::GetSoftmaxTritonOps<T, T>()) {
      name_strings_.emplace_back(name);
      ops_.emplace_back(std::move(op));
    }
  }

  void Run() override {
    ORT_THROW_IF_ERROR(ops_[selected_op_](&params_));
  }

  std::vector<std::string> ListOps() const {
    return name_strings_;
  }

  bool SelectOp(const std::string& name) {
    for (size_t i = 0; i < ops_.size(); i++) {
      if (name_strings_[i] == name) {
        selected_op_ = i;
        Status status = ops_[i](&params_);
        return status.IsOK();
      }
    }

    ORT_THROW("Cannot find implementation ", name);
  }

 private:
  using ParamsT = rocm::SoftmaxParams<T, T>;
  using OpT = rocm::tunable::Op<ParamsT>;
  ParamsT params_{};
  std::vector<OpT> ops_;
  std::vector<std::string> name_strings_;
  size_t selected_op_{};
};

#endif  // USE_TRITON_KERNEL

#define REGISTER_OP(name, type, vec_size)                                    \
  py::class_<name<type, vec_size>>(m, #name "_" #type "_" #vec_size)         \
      .def(py::init<DeviceArray&, DeviceArray&, int, int, int, int, bool>()) \
      .def("SetRepeats", &name<type, vec_size>::SetRepeats)                  \
      .def("Profile", &name<type, vec_size>::Profile)                        \
      .def("Run", &name<type, vec_size>::Run)                                \
      .def("ListOps", &name<type, vec_size>::ListOps)                        \
      .def("SelectOp", &name<type, vec_size>::SelectOp);

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
      .def("ListOps", &name<type>::ListOps)                                  \
      .def("SelectOp", &name<type>::SelectOp);

KE_REGISTER(m) {
  REGISTER_OP_FOR_ALL_VEC_SIZE(SoftmaxBlockwise, half);
  REGISTER_OP_FOR_ALL_VEC_SIZE(SoftmaxBlockwise, float);

  REGISTER_OP_TYPED(SoftmaxWarpwiseStaticSelection, half);
  REGISTER_OP_TYPED(SoftmaxWarpwiseStaticSelection, float);

  REGISTER_OP_TYPED(SoftmaxBlockwiseStaticSelection, half);
  REGISTER_OP_TYPED(SoftmaxBlockwiseStaticSelection, float);

  REGISTER_OP_TYPED(SoftmaxTunable, half);
  REGISTER_OP_TYPED(SoftmaxTunable, float);
}

#ifdef USE_COMPOSABLE_KERNEL
KE_REGISTER(m) {
  REGISTER_OP_TYPED(CKSoftmax, half);
  REGISTER_OP_TYPED(CKSoftmax, float);
}
#endif  // USE_COMPOSABLE_KERNEL

#ifdef USE_TRITON_KERNEL
KE_REGISTER(m) {
  REGISTER_OP_TYPED(SoftmaxTriton, half);
  REGISTER_OP_TYPED(SoftmaxTriton, float);
}
#endif  // USE_TRITON_KERNEL

}  // namespace onnxruntime
