// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/bias_dropout.h"
#include "core/providers/cuda/nn/dropout.h"

#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {


ONNX_OPERATOR_KERNEL_EX(
    BiasDropout,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", ALL_IEEE_FLOAT_TENSOR_TYPES)
        .TypeConstraint("T1", ALL_IEEE_FLOAT_TENSOR_TYPES)
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
        .InputMemoryType<OrtMemTypeCPUInput>(3)
        .InputMemoryType<OrtMemTypeCPUInput>(4),
    BiasDropout);

template <typename T>
struct BiasDropoutComputeImpl {
  Status operator()(const cudaDeviceProp& prop,
                    cudaStream_t stream,
                    const int64_t N,
                    const fast_divmod fdm_dim,
                    const float ratio_data,
                    PhiloxGenerator& generator,
                    const Tensor& X,
                    const Tensor& bias,
                    const Tensor* residual,
                    Tensor& Y,
                    bool* mask_data) const {
    typedef typename ToCudaType<T>::MappedType CudaT;

    const CudaT* X_data = reinterpret_cast<const CudaT*>(X.template Data<T>());
    const CudaT* bias_data = reinterpret_cast<const CudaT*>(bias.template Data<T>());

    const CudaT* residual_data = nullptr;
    if (residual) {
      if (residual->Shape() != X.Shape()) {
        return Status(common::ONNXRUNTIME, common::FAIL, "Residual input shape does not match X input shape.");
      }
      residual_data = reinterpret_cast<const CudaT*>(residual->template Data<T>());
    }

    CudaT* Y_data = reinterpret_cast<CudaT*>(Y.template MutableData<T>());

    BiasDropoutKernelImpl<CudaT>(prop, stream, N, fdm_dim, ratio_data, generator, X_data, bias_data, residual_data, Y_data, mask_data);

    return Status::OK();
  }
};

Status BiasDropout::ComputeInternal(OpKernelContext* context) const {
  //Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(X, "X Input is not available.");

  const TensorShape& x_shape = X->Shape();
  const int64_t N = x_shape.Size();

  //Get bias_data
  const Tensor* bias = context->Input<Tensor>(1);
  if (bias == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "Bias input of BiasDropout is not available.");
  const TensorShape& bias_shape = bias->Shape();
  if (bias_shape.NumDimensions() != 1) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Bias input is not a 1D tensor.");
  }
  const int64_t dim = bias_shape[0];
  if (dim != x_shape.GetDims().back()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Bias' dimension doesn't match input's last dimension.");
  }

  //Get residual_data
  const Tensor* residual = context->Input<Tensor>(2);

  //Get Y_data
  auto Y = context->Output(0, x_shape);

  //Get mask_data
  auto mask = context->Output(1, x_shape);

  //Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(3);
  if (ratio) {
    utils::MLTypeCallDispatcher<ALL_IEEE_FLOAT_DATA_TYPES> t_disp(ratio->GetElementType());
    t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  }

  //Check for inference mode.
  const Tensor* training_mode = context->Input<Tensor>(4);
  bool is_training_mode = (training_mode != nullptr) && *(training_mode->Data<bool>());
  if (!is_training_mode) {
    ratio_data = 0.0f;
  }

  IAllocatorUniquePtr<bool> temp_mask_buffer{};  // buffer to use if mask is not provided
  bool* const mask_data = [this, N, mask, &temp_mask_buffer]() {
    if (mask) return mask->MutableData<bool>();
    temp_mask_buffer = GetScratchBuffer<bool>(N);
    return temp_mask_buffer.get();
  }();

  const fast_divmod fdm_dim(gsl::narrow_cast<int>(dim));
  PhiloxGenerator& generator = generator_ ? *generator_ : PhiloxGenerator::Default();

  utils::MLTypeCallDispatcher<ALL_IEEE_FLOAT_DATA_TYPES> t_disp(X->GetElementType());
  return t_disp.InvokeRet<Status, BiasDropoutComputeImpl>(
      GetDeviceProp(), Stream(), N, fdm_dim, ratio_data, generator, *X, *bias, residual, *Y, mask_data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime