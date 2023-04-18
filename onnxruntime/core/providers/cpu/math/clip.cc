// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/clip.h"

#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/providers/op_kernel_type_control.h"
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
    float, double, int8_t, uint8_t, int32_t, uint32_t, int64_t, uint64_t);
}  // namespace op_kernel_type_control

using EnabledClip11Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Clip, 11, Input, 0);
using EnabledClip12Types = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(
    kCpuExecutionProvider, kOnnxDomain, Clip, 12, Input, 0);

using AllEnabledClipTypes =
    utils::TypeSetUnion<
        EnabledClip11Types,
        EnabledClip12Types>;

#define REG_KERNEL_VERSIONED_NONTEMPL(                                                 \
    OP_TYPE, START_VER, END_VER, KERNEL_CLASS, ENABLED_TYPE_LIST)                      \
  ONNX_CPU_OPERATOR_VERSIONED_KERNEL(                                                  \
      OP_TYPE,                                                                         \
      START_VER,                                                                       \
      END_VER,                                                                         \
      KernelDefBuilder()                                                               \
          .MayInplace(0, 0)                                                            \
          .TypeConstraint("T",                                                         \
                          BuildKernelDefConstraintsFromTypeList<ENABLED_TYPE_LIST>()), \
      KERNEL_CLASS);

#define REG_KERNEL_NONTEMPL(                                                           \
    OP_TYPE, VERSION, KERNEL_CLASS, ENABLED_TYPE_LIST)                                 \
  ONNX_CPU_OPERATOR_KERNEL(                                                            \
      OP_TYPE,                                                                         \
      VERSION,                                                                         \
      KernelDefBuilder()                                                               \
          .MayInplace(0, 0)                                                            \
          .TypeConstraint("T",                                                         \
                          BuildKernelDefConstraintsFromTypeList<ENABLED_TYPE_LIST>()), \
      KERNEL_CLASS);

REG_KERNEL_VERSIONED_NONTEMPL(Clip, 11, 11, Clip, EnabledClip11Types);
REG_KERNEL_VERSIONED_NONTEMPL(Clip, 12, 12, Clip, EnabledClip12Types);
REG_KERNEL_NONTEMPL(Clip, 13, Clip, EnabledClip12Types);

#undef REG_KERNEL_VERSIONED_NONTEMPL
#undef REG_KERNEL_NONTEMPL

template <typename T>
Status Clip_6<T>::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  Tensor* Y = ctx->Output(0, X->Shape());

  // We split input/output data into chunks. Each chunk has N elements
  // (except, maybe, the last chunk), and we use a thread pool to process
  // the chunks in parallel. N = 16384 was selected based on performance
  // results on input tensors of 10^i elements for i in [1 .. 6].
  static constexpr int64_t length_per_task = 16384;

  const int64_t elem_count = Y->Shape().Size();
  const int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;

  concurrency::ThreadPool::TryBatchParallelFor(
      ctx->GetOperatorThreadPool(),
      static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        const int64_t start = task_idx * length_per_task;
        const size_t count = narrow<size_t>(std::min(length_per_task, elem_count - start));

        EigenVectorMap<T>(Y->MutableData<T>() + start, count) =
            ConstEigenVectorMap<T>(X->Data<T>() + start, count)
                .cwiseMax(this->min_)
                .cwiseMin(this->max_);
      },
      0);

  return Status::OK();
}

template <typename T>
struct Clip::ComputeImpl {
  void operator()(const Tensor* X, const Tensor* min, const Tensor* max, Tensor* Y,
                  concurrency::ThreadPool* tp) const {
    auto min_val = std::numeric_limits<T>::lowest();
    auto max_val = std::numeric_limits<T>::max();
    if (min) {
      ORT_ENFORCE(min->Shape().IsScalar(), "min should be a scalar.");
      min_val = *(min->Data<T>());
    }
    if (max) {
      ORT_ENFORCE(max->Shape().IsScalar(), "max should be a scalar.");
      max_val = *(max->Data<T>());
    }

    // We split input/output data into chunks. Each chunk has N elements
    // (except, maybe, the last chunk), and we use a thread pool to process
    // the chunks in parallel. N = 16384 was selected based on performance
    // results on input tensors of 10^i elements for i in [1 .. 6].
    static constexpr int64_t length_per_task = 16384;

    const int64_t elem_count = Y->Shape().Size();
    const int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;

    concurrency::ThreadPool::TryBatchParallelFor(
        tp, static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          const int64_t start = task_idx * length_per_task;
          const size_t count = narrow<size_t>(std::min(length_per_task, elem_count - start));

          EigenVectorMap<T>(Y->MutableData<T>() + start, count) =
              ConstEigenVectorMap<T>(X->Data<T>() + start, count)
                  .cwiseMax(min_val)
                  .cwiseMin(max_val);
        },
        0);
  }
};

Status Clip::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  const auto* min = ctx->Input<Tensor>(1);
  const auto* max = ctx->Input<Tensor>(2);
  Tensor* Y = ctx->Output(0, X->Shape());

  utils::MLTypeCallDispatcherFromTypeList<AllEnabledClipTypes> t_disp(X->GetElementType());

  t_disp.Invoke<ComputeImpl>(X, min, max, Y, ctx->GetOperatorThreadPool());

  return Status::OK();
}

}  // namespace onnxruntime
