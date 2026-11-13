// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace {

template <typename T>
using AccumulationType = std::conditional_t<std::is_same_v<T, double>, double, float>;

template <typename T>
using EigenType = std::conditional_t<std::is_same_v<T, MLFloat16>, Eigen::half,
                                     std::conditional_t<std::is_same_v<T, BFloat16>, Eigen::bfloat16, T>>;

template <typename Operation, typename T>
struct StableUnary {
  void operator()(const Tensor& input, Tensor& output, concurrency::ThreadPool* tp) const {
    using E = EigenType<T>;
    using A = AccumulationType<T>;
    const auto* x = reinterpret_cast<const E*>(input.Data<T>());
    auto* y = reinterpret_cast<E*>(output.MutableData<T>());
    concurrency::ThreadPool::TryParallelFor(
        tp, onnxruntime::narrow<std::ptrdiff_t>(input.Shape().Size()), {sizeof(T), sizeof(T), 30.0},
        [x, y](std::ptrdiff_t first, std::ptrdiff_t last) {
          ConstEigenVectorArrayMap<E> xm(x + first, last - first);
          EigenVectorArrayMap<E> ym(y + first, last - first);
          const auto wide = xm.template cast<A>();
          // Explicitly select zeros to retain their sign on vectorized paths.
          if constexpr (Operation::value) {
            ym = (wide == A{0}).select(wide, wide.expm1()).template cast<E>();
          } else {
            ym = (wide == A{0}).select(wide, wide.log1p()).template cast<E>();
          }
        });
  }
};

template <typename T>
T LogAddExp(T a, T b) {
  if (std::isnan(a) || std::isnan(b)) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  const T maximum = std::max(a, b);
  if (std::isinf(maximum)) {
    return maximum;
  }
  return maximum + std::log1p(std::exp(std::min(a, b) - maximum));
}

template <typename T>
struct CumulativeLogSumExp {
  void operator()(const Tensor& input, Tensor& output, int64_t axis, concurrency::ThreadPool* tp) const {
    using A = AccumulationType<T>;
    const int64_t width = input.Shape()[onnxruntime::narrow<size_t>(axis)];
    const int64_t inner = input.Shape().SizeFromDimension(onnxruntime::narrow<size_t>(axis + 1));
    const int64_t lanes = input.Shape().Size() / width;
    const T* x = input.Data<T>();
    T* y = output.MutableData<T>();
    if (width == 1) {
      std::copy_n(x, lanes, y);
      return;
    }
    concurrency::ThreadPool::TryParallelFor(
        tp, onnxruntime::narrow<std::ptrdiff_t>(lanes), {static_cast<double>(width * sizeof(T)), static_cast<double>(width * sizeof(T)), static_cast<double>(width) * 40},
        [=](std::ptrdiff_t first, std::ptrdiff_t last) {
          for (int64_t lane = first; lane < last; ++lane) {
            int64_t offset = (lane / inner) * width * inner + lane % inner;
            A prefix = -std::numeric_limits<A>::infinity();
            for (int64_t j = 0; j < width; ++j, offset += inner) {
              prefix = LogAddExp(prefix, static_cast<A>(x[offset]));
              y[offset] = static_cast<T>(prefix);
            }
          }
        });
  }
};

using FloatDispatcher = utils::MLTypeCallDispatcher<float, double, MLFloat16, BFloat16>;

}  // namespace

template <bool IsExpm1>
class StableUnaryKernel final : public OpKernel {
 public:
  explicit StableUnaryKernel(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto& input = *ctx->Input<Tensor>(0);
    auto& output = *ctx->Output(0, input.Shape());
    if (input.Shape().Size() != 0) {
      FloatDispatcher(input.GetElementType()).InvokeWithLeadingTemplateArgs<StableUnary, TypeList<std::bool_constant<IsExpm1>>>(input, output, ctx->GetOperatorThreadPool());
    }
    return Status::OK();
  }
};

class LogCumSumExp final : public OpKernel {
 public:
  explicit LogCumSumExp(const OpKernelInfo& info)
      : OpKernel(info), axis_(info.GetAttrOrDefault<int64_t>("axis", 0)) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto& input = *ctx->Input<Tensor>(0);
    const int64_t rank = static_cast<int64_t>(input.Shape().NumDimensions());
    ORT_RETURN_IF_NOT(rank > 0, "LogCumSumExp requires rank >= 1; reshape scalar X to a one-element tensor.");
    ORT_RETURN_IF_NOT(axis_ >= -rank && axis_ < rank, "LogCumSumExp axis ", axis_,
                      " is outside [-", rank, ", ", rank - 1, "]; choose an axis within X's rank.");
    const int64_t axis = HandleNegativeAxis(axis_, rank);
    auto& output = *ctx->Output(0, input.Shape());
    if (input.Shape().Size() != 0) {
      FloatDispatcher(input.GetElementType()).Invoke<CumulativeLogSumExp>(input, output, axis, ctx->GetOperatorThreadPool());
    }
    return Status::OK();
  }

 private:
  int64_t axis_;
};

ONNX_CPU_OPERATOR_KERNEL(
    Expm1, 29,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", BuildKernelDefConstraints<float, double, MLFloat16, BFloat16>()),
    StableUnaryKernel<true>);

ONNX_CPU_OPERATOR_KERNEL(
    Log1p, 29,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", BuildKernelDefConstraints<float, double, MLFloat16, BFloat16>()),
    StableUnaryKernel<false>);

ONNX_CPU_OPERATOR_KERNEL(
    LogCumSumExp, 29,
    KernelDefBuilder().TypeConstraint("T", BuildKernelDefConstraints<float, double, MLFloat16, BFloat16>()),
    LogCumSumExp);

}  // namespace onnxruntime
