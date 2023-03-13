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

namespace py = pybind11;

namespace onnxruntime {

template <typename T, int VecSize>
class SoftmaxBlockwise : public ISelectableKernelExplorer {
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

  std::vector<std::string> ListOps() const override {
    return {type_string_};
  }

  bool SelectOp(const std::string& name) override {
    Status status = rocm::SoftmaxBlockwiseOp<T, T, rocm::AccumulationType_t<T>, VecSize>(&params_);
    return status.IsOK() && name == type_string_;
  }

 private:
  using ParamsT = rocm::SoftmaxParams<T, T>;
  ParamsT params_{};
  std::string type_string_{};
};

template <typename T>
class SoftmaxWarpwiseStaticSelection : public ISelectableKernelExplorer {
 public:
  SoftmaxWarpwiseStaticSelection(DeviceArray& output, DeviceArray& input, int softmax_elements,
                                 int input_stride, int output_stride, int batch_count, bool is_log_softmax)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(input.ptr()),
                softmax_elements, input_stride, output_stride, batch_count, is_log_softmax) {}

  void Run() override {
    ORT_THROW_IF_ERROR((rocm::SoftmaxWarpwiseStaticSelection<T, T, rocm::AccumulationType_t<T>>(&params_)));
  }

  std::vector<std::string> ListOps() const override {
    return {"SoftmaxWarpwiseStaticSelection"};
  }

  bool SelectOp(const std::string& name) override {
    auto status = rocm::SoftmaxWarpwiseStaticSelection<T, T, rocm::AccumulationType_t<T>>(&params_);
    return status.IsOK() && name == "SoftmaxWarpwiseStaticSelection";
  }

 private:
  using ParamsT = rocm::SoftmaxParams<T, T>;
  ParamsT params_{};
};

template <typename T>
class SoftmaxBlockwiseStaticSelection : public ISelectableKernelExplorer {
 public:
  SoftmaxBlockwiseStaticSelection(DeviceArray& output, DeviceArray& input, int softmax_elements,
                                  int input_stride, int output_stride, int batch_count, bool is_log_softmax)
      : params_(TuningContext(), Stream(), static_cast<T*>(output.ptr()), static_cast<T*>(input.ptr()),
                softmax_elements, input_stride, output_stride, batch_count, is_log_softmax) {}

  void Run() override {
    ORT_THROW_IF_ERROR((rocm::SoftmaxBlockwiseStaticSelection<T, T, rocm::AccumulationType_t<T>>(&params_)));
  }

  std::vector<std::string> ListOps() const override {
    return {"SoftmaxBlockwiseStaticSelection"};
  }

  bool SelectOp(const std::string& name) override {
    return name == "SoftmaxBlockwiseStaticSelection";
  }

 private:
  using ParamsT = rocm::SoftmaxParams<T, T>;
  ParamsT params_{};
};

template <typename T>
class SoftmaxTunable : public ISelectableKernelExplorer {
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

  std::vector<std::string> ListOps() const override {
    return {"SoftmaxTunable"};
  }

  bool SelectOp(const std::string& name) override {
    return name == "SoftmaxTunable";
  }

 private:
  using ParamsT = rocm::SoftmaxParams<T, T>;
  ParamsT params_{};
  rocm::SoftmaxTunableOp<T, T, rocm::AccumulationType_t<T>> op_{};
};

#ifdef USE_COMPOSABLE_KERNEL
template <typename T>
class CKSoftmax : public ISelectableKernelExplorer {
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

  std::vector<std::string> ListOps() const override {
    return type_strings_;
  }

  bool SelectOp(const std::string& name) override {
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

#define REGISTER_OP(tpl, dtype, vec_size)                                                                      \
  KE_REGISTER_SELECTABLE_OP_COMMON(m, #tpl "_" #dtype "_" #vec_size, TEMPLATED_TYPENAME(tpl<dtype, vec_size>)) \
      .def(py::init<DeviceArray&, DeviceArray&, int, int, int, int, bool>());

#define REGISTER_OP_FOR_ALL_VEC_SIZE(name, type) \
  REGISTER_OP(name, type, 1)                     \
  REGISTER_OP(name, type, 2)                     \
  REGISTER_OP(name, type, 4)                     \
  REGISTER_OP(name, type, 8)                     \
  REGISTER_OP(name, type, 16)

#define REGISTER_OP_TYPED(tpl, dtype)                                                  \
  KE_REGISTER_SELECTABLE_OP_COMMON(m, #tpl "_" #dtype, TEMPLATED_TYPENAME(tpl<dtype>)) \
      .def(py::init<DeviceArray&, DeviceArray&, int, int, int, int, bool>())

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

}  // namespace onnxruntime
