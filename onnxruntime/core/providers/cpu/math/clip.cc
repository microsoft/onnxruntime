// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/clip.h"

#include "core/framework/data_types_internal.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/providers/op_kernel_type_control_utils.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Clip,
    6,
    10,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Clip_6<float>);

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Clip, 11, Input, 0,
    float);
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES(
    kCpuExecutionProvider, kOnnxDomain, Clip, 12, Input, 0,
    float, double, int8_t, uint8_t, int64_t, uint64_t);
}  // namespace op_kernel_type_control

using Clip11Types = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Clip, 11, Input, 0);
using EnabledClip11Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Clip, 11, Input, 0);
using Clip12Types = ORT_OP_KERNEL_ARG_DEFAULT_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Clip, 12, Input, 0);
using EnabledClip12Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Clip, 12, Input, 0);

using AllEnabledClipTypes =
    utils::TypeSetUnion<
        EnabledClip11Types,
        EnabledClip12Types>;

#define REG_KERNEL_VERSIONED_NONTEMPL(                                                 \
    OP_TYPE, START_VER, END_VER, KERNEL_CLASS, DEFAULT_TYPE_LIST, ENABLED_TYPE_LIST)   \
  ONNX_CPU_OPERATOR_VERSIONED_KERNEL(                                                  \
      OP_TYPE,                                                                         \
      START_VER,                                                                       \
      END_VER,                                                                         \
      KernelDefBuilder()                                                               \
          .MayInplace(0, 0)                                                            \
          .TypeConstraint("T",                                                         \
                          BuildKernelDefConstraintsFromTypeList<DEFAULT_TYPE_LIST>(),  \
                          BuildKernelDefConstraintsFromTypeList<ENABLED_TYPE_LIST>()), \
      KERNEL_CLASS);

#define REG_KERNEL_NONTEMPL(                                                           \
    OP_TYPE, VERSION, KERNEL_CLASS, DEFAULT_TYPE_LIST, ENABLED_TYPE_LIST)              \
  ONNX_CPU_OPERATOR_KERNEL(                                                            \
      OP_TYPE,                                                                         \
      VERSION,                                                                         \
      KernelDefBuilder()                                                               \
          .MayInplace(0, 0)                                                            \
          .TypeConstraint("T",                                                         \
                          BuildKernelDefConstraintsFromTypeList<DEFAULT_TYPE_LIST>(),  \
                          BuildKernelDefConstraintsFromTypeList<ENABLED_TYPE_LIST>()), \
      KERNEL_CLASS);

REG_KERNEL_VERSIONED_NONTEMPL(Clip, 11, 11, Clip, Clip11Types, EnabledClip11Types);
REG_KERNEL_VERSIONED_NONTEMPL(Clip, 12, 12, Clip, Clip12Types, EnabledClip12Types);
REG_KERNEL_NONTEMPL(Clip, 13, Clip, Clip12Types, EnabledClip12Types);

#undef REG_KERNEL_VERSIONED_NONTEMPL
#undef REG_KERNEL_NONTEMPL

template <typename T>
Status Clip_6<T>::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  Tensor* Y = ctx->Output(0, X->Shape());
  EigenVectorMap<T>(Y->template MutableData<T>(), Y->Shape().Size()) =
      ConstEigenVectorMap<T>(X->template Data<T>(), X->Shape().Size())
          .cwiseMax(this->min_)
          .cwiseMin(this->max_);
  return Status::OK();
}

template <typename T>
struct Clip::ComputeImpl {
  void operator()(const Tensor* X, const Tensor* min, const Tensor* max, Tensor* Y) const {
    auto min_val = std::numeric_limits<T>::lowest();
    auto max_val = std::numeric_limits<T>::max();
    if (min) {
      ORT_ENFORCE(min->Shape().IsScalar(), "min should be a scalar.");
      min_val = *(min->template Data<T>());
    }
    if (max) {
      ORT_ENFORCE(max->Shape().IsScalar(), "max should be a scalar.");
      max_val = *(max->template Data<T>());
    }

    EigenVectorMap<T>(Y->template MutableData<T>(), Y->Shape().Size()) =
        ConstEigenVectorMap<T>(X->template Data<T>(), X->Shape().Size())
            .cwiseMax(min_val)
            .cwiseMin(max_val);
  }
};

Status Clip::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  const auto* min = ctx->Input<Tensor>(1);
  const auto* max = ctx->Input<Tensor>(2);
  Tensor* Y = ctx->Output(0, X->Shape());

  utils::MLTypeCallDispatcherFromTypeList<AllEnabledClipTypes> t_disp(X->GetElementType());

  t_disp.Invoke<ComputeImpl>(X, min, max, Y);

  return Status::OK();
}

}  // namespace onnxruntime
