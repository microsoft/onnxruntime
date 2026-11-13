// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include <type_traits>

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cuda/tensor/mean_variance_normalization_impl.h"
#include "core/providers/mean_variance_normalization_helper.h"

namespace onnxruntime::cuda {

template <typename T>
class MeanVarianceNormalization final : public CudaKernel {
 public:
  explicit MeanVarianceNormalization(const OpKernelInfo& info)
      : CudaKernel(info), epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-9f)) {
    const auto axes = info.GetAttrsOrDefault<int64_t>("axes", {0, 2, 3});
    axes_.assign(axes.begin(), axes.end());
  }

  Status ComputeInternal(OpKernelContext* context) const override {
    const auto* input = context->Input<Tensor>(0);
    MeanVarianceNormalizationPlan plan;
    ORT_RETURN_IF_ERROR(PrepareMeanVarianceNormalization(input->Shape(), axes_, plan));
    auto* output = context->Output(0, input->Shape());
    if (input->Shape().Size() == 0) return Status::OK();

    // Reduction over outer and inner dimensions (including NCHW's default
    // per-channel normalization) can read the original tensor without transposes.
    int64_t inner = 0;
    bool transpose = plan.transpose;
    if (transpose && plan.unreduced_rank > 0) {
      bool contiguous_rows = true;
      for (size_t axis = 1; axis < plan.unreduced_rank; ++axis) {
        contiguous_rows &= plan.permutation[axis] == plan.permutation[0] + axis;
      }
      if (contiguous_rows) {
        inner = input->Shape().SizeFromDimension(plan.permutation[plan.unreduced_rank - 1] + 1);
        transpose = false;
      }
    }
    std::unique_ptr<Tensor> transposed_input, transposed_output;
    if (transpose) {
      AllocatorPtr allocator;
      ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
      transposed_input = Tensor::Create(input->DataType(), plan.shape, allocator);
      transposed_output = Tensor::Create(input->DataType(), plan.shape, allocator);
      ORT_RETURN_IF_ERROR(Transpose::DoTranspose(GetDeviceProp(), Stream(context), GetCublasHandle(context),
                                                 plan.permutation, *input, *transposed_input));
    }
    using CudaT = typename ToCudaType<T>::MappedType;
    using AccT = std::conditional_t<std::is_same_v<T, double>, double, float>;
    const int64_t chunks = MeanVarianceNormalizationChunks(plan.rows, plan.columns);
    const size_t workspace_size = chunks > 1
                                      ? static_cast<size_t>(SafeInt<size_t>(plan.rows) * (chunks + 2))
                                      : size_t{0};
    auto workspace = GetScratchBuffer<AccT>(workspace_size, GetComputeStream(context));
    MeanVarianceNormalizationImpl(
        Stream(context),
        reinterpret_cast<const CudaT*>(transpose ? transposed_input->Data<T>() : input->Data<T>()),
        reinterpret_cast<CudaT*>(transpose ? transposed_output->MutableData<T>() : output->MutableData<T>()),
        plan.rows, plan.columns, inner, epsilon_, workspace.get());
    if (transpose) {
      ORT_RETURN_IF_ERROR(Transpose::DoTranspose(GetDeviceProp(), Stream(context), GetCublasHandle(context),
                                                 plan.inverse_permutation, *transposed_output, *output));
    }
    return Status::OK();
  }

 private:
  InlinedVector<int64_t> axes_;
  const float epsilon_;
};

#define REGISTER_MVN29(T)                                                                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(MeanVarianceNormalization, kOnnxDomain, 29, T,                                       \
                                kCudaExecutionProvider,                                                              \
                                (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                                MeanVarianceNormalization<T>);

REGISTER_MVN29(float)
REGISTER_MVN29(double)
REGISTER_MVN29(MLFloat16)
REGISTER_MVN29(BFloat16)
#undef REGISTER_MVN29

}  // namespace onnxruntime::cuda
