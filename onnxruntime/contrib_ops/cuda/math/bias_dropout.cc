// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/bias_dropout.h"

#include "core/providers/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename T>
struct GetRatioDataImpl {
  void operator()(const Tensor* ratio, float& ratio_data) const {
    ratio_data = static_cast<float>(*(ratio->Data<T>()));
    ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f, "ratio_data is outside range [0, 1)");
  }
};

template <typename T>
struct BiasDropoutComputeImpl {
  Status operator()(const cudaDeviceProp& prop, cudaStream_t stream, const int64_t N, const int64_t mask_element_count,
                    const fast_divmod fdm_dim, const float ratio_data, PhiloxGenerator& generator, const Tensor& X,
                    const Tensor& bias, const Tensor* residual, Tensor& Y, void* mask_data, bool has_same_shape_bias,
                    bool use_bitmask) const {
    typedef typename ToCudaType<T>::MappedType CudaT;

    const CudaT* X_data = reinterpret_cast<const CudaT*>(X.Data<T>());
    const CudaT* bias_data = reinterpret_cast<const CudaT*>(bias.Data<T>());

    const CudaT* residual_data = nullptr;
    if (residual) {
      if (residual->Shape() != X.Shape()) {
        return Status(common::ONNXRUNTIME, common::FAIL, "Residual input shape does not match X input shape.");
      }
      residual_data = reinterpret_cast<const CudaT*>(residual->Data<T>());
    }

    CudaT* Y_data = reinterpret_cast<CudaT*>(Y.MutableData<T>());
    BiasDropoutKernelImpl<CudaT>(prop, stream, N, mask_element_count, fdm_dim, ratio_data, generator, X_data, bias_data,
                                 residual_data, Y_data, mask_data, has_same_shape_bias, use_bitmask);
    return Status::OK();
  }
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(BiasDropout, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T1", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
                            .InputMemoryType(OrtMemTypeCPUInput, 3)
                            .InputMemoryType(OrtMemTypeCPUInput, 4),
                        BiasDropout<false>);

ONNX_OPERATOR_KERNEL_EX(BitmaskBiasDropout, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T1", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
                            .TypeConstraint("T3", DataTypeImpl::GetTensorType<BitmaskElementType>())
                            .InputMemoryType(OrtMemTypeCPUInput, 3)
                            .InputMemoryType(OrtMemTypeCPUInput, 4),
                        BiasDropout<true>);

template <bool UseBitmask>
Status BiasDropout<UseBitmask>::ComputeInternal(OpKernelContext* context) const {
  // Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(X, "X Input is not available.");

  const TensorShape& x_shape = X->Shape();
  const int64_t N = x_shape.Size();

  // Get bias_data
  const Tensor* bias = context->Input<Tensor>(1);
  if (!bias) return Status(common::ONNXRUNTIME, common::FAIL, "Bias input of BiasDropout is not available.");
  const TensorShape& bias_shape = bias->Shape();
  const int64_t dim = bias_shape.GetDims().back();
  bool has_same_shape_bias = (bias_shape == x_shape);
  if (!has_same_shape_bias) {
    if (bias_shape.NumDimensions() != 1) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Bias input is not a 1D tensor.");
    }

    if (dim != x_shape.GetDims().back()) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Bias' dimension doesn't match input's last dimension.");
    }
  }

  // Get residual_data
  const Tensor* residual = context->Input<Tensor>(2);

  // Get Y_data
  auto Y = context->Output(0, x_shape);

  // Get mask_data
  Tensor* mask = nullptr;
  int64_t mask_element_count = N;
  if (UseBitmask) {
    mask_element_count = (N + kNumBitsPerBitmaskElement - 1) / kNumBitsPerBitmaskElement;
    mask = context->Output(1, {mask_element_count});
  } else {
    mask = context->Output(1, x_shape);
  }

  // Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(3);
  if (ratio) {
    utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(ratio->GetElementType());
    t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  }

  // Check for inference mode.
  const Tensor* training_mode = context->Input<Tensor>(4);
  bool is_training_mode = training_mode && *(training_mode->Data<bool>());
  if (!is_training_mode) {
    ratio_data = 0.0f;
  }

  IAllocatorUniquePtr<void> temp_mask_buffer{};  // buffer to use if mask is not provided
  void* const mask_data = [this, mask_element_count, mask, &temp_mask_buffer]() {
    if (mask) return mask->MutableDataRaw();
    temp_mask_buffer =
        GetScratchBuffer<void>(mask_element_count * (UseBitmask ? sizeof(BitmaskElementType) : sizeof(bool)));
    return temp_mask_buffer.get();
  }();

  const fast_divmod fdm_dim(gsl::narrow_cast<int>(dim));
  PhiloxGenerator& generator = generator_ ? *generator_ : PhiloxGenerator::Default();

  utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(X->GetElementType());
  return t_disp.InvokeRet<Status, BiasDropoutComputeImpl>(GetDeviceProp(), Stream(), N, mask_element_count, fdm_dim,
                                                          ratio_data, generator, *X, *bias, residual, *Y, mask_data,
                                                          has_same_shape_bias, UseBitmask);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
