// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/providers/mean_variance_normalization_helper.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

template <typename T>
class MeanVarianceNormalization29 final : public OpKernel {
 public:
  explicit MeanVarianceNormalization29(const OpKernelInfo& info)
      : OpKernel(info), epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-9f)) {
    const auto axes = info.GetAttrsOrDefault<int64_t>("axes", {0, 2, 3});
    axes_.assign(axes.begin(), axes.end());
  }

  Status Compute(OpKernelContext* context) const override {
    const auto& input = context->RequiredInput<Tensor>(0);
    MeanVarianceNormalizationPlan plan;
    ORT_RETURN_IF_ERROR(PrepareMeanVarianceNormalization(input.Shape(), axes_, plan));
    auto& output = context->RequiredOutput(0, input.Shape());
    if (input.Shape().Size() == 0) return Status::OK();

    Tensor transposed_input, transposed_output;
    if (plan.transpose) {
      AllocatorPtr allocator;
      ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
      transposed_input = Tensor(input.DataType(), plan.shape, allocator);
      transposed_output = Tensor(input.DataType(), plan.shape, allocator);
      ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(plan.permutation, input, transposed_input));
    }
    const T* x = plan.transpose ? transposed_input.Data<T>() : input.Data<T>();
    T* y = plan.transpose ? transposed_output.MutableData<T>() : output.MutableData<T>();
    using AccT = std::conditional_t<std::is_same_v<T, double>, double, float>;
    const auto columns = narrow<Eigen::Index>(plan.columns);
    concurrency::ThreadPool::TryParallelFor(
        context->GetOperatorThreadPool(), narrow<std::ptrdiff_t>(plan.rows),
        static_cast<double>(plan.columns) * 5,
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          // Keep the centered values in the compute type, not fp16/bf16.
          Eigen::Array<AccT, Eigen::Dynamic, 1> centered(columns);
          for (auto row = first; row < last; ++row) {
            Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> values(x + row * plan.columns, columns);
            centered = values.template cast<AccT>();
            const AccT mean = centered.mean();
            centered -= mean;
            const AccT denominator = std::sqrt(centered.square().mean()) + static_cast<AccT>(epsilon_);
            Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>(y + row * plan.columns, columns) =
                (centered / denominator).template cast<T>();
          }
        });
    if (plan.transpose) {
      ORT_RETURN_IF_ERROR(TransposeBase::DoTranspose(plan.inverse_permutation, transposed_output, output));
    }
    return Status::OK();
  }

 private:
  InlinedVector<int64_t> axes_;
  const float epsilon_;
};

#define REGISTER_MVN29(T)                                                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(MeanVarianceNormalization, kOnnxDomain, 29, T,                            \
                                kCpuExecutionProvider,                                                    \
                                KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                                MeanVarianceNormalization29<T>);

REGISTER_MVN29(float)
REGISTER_MVN29(double)
REGISTER_MVN29(MLFloat16)
REGISTER_MVN29(BFloat16)
#undef REGISTER_MVN29

}  // namespace onnxruntime
