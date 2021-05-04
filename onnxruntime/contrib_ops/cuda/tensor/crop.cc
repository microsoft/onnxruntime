// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "crop.h"
#include "crop_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Crop,                                                       \
      kOnnxDomain,                                                \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Crop<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Status Crop<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(ValidateInput(X));

  const auto& dims = X->Shape().GetDims();
  const int64_t N = dims[0];
  const int64_t C = dims[1];
  const int64_t H = dims[2];
  const int64_t W = dims[3];

  // find the cropped region, and copy it to the destination matrix
  int64_t leftBorder = border_[0];
  int64_t topBorder = border_[1];
  int64_t rightBorder = border_[2];
  int64_t bottomBorder = border_[3];

  int64_t bottomLimit = H - bottomBorder;
  int64_t rightLimit = W - rightBorder;

  // scale = (height, width)
  if (!scale_.empty()) {
    bottomLimit = topBorder + scale_[0];
    rightLimit = leftBorder + scale_[1];
  }

  Tensor* Y = context->Output(0, TensorShape({N, C, bottomLimit - topBorder, rightLimit - leftBorder}));

  typedef typename ToCudaType<T>::MappedType CudaT;
  fast_divmod fdm_YW(gsl::narrow_cast<int>(rightLimit - leftBorder));
  fast_divmod fdm_YHW(gsl::narrow_cast<int>((bottomLimit - topBorder) * (rightLimit - leftBorder)));

  CropImpl<CudaT>(
      Stream(),
      reinterpret_cast<const CudaT*>(X->template Data<T>()),
      gsl::narrow_cast<int>(leftBorder),
      gsl::narrow_cast<int>(topBorder),
      gsl::narrow_cast<int>(W),
      gsl::narrow_cast<int>(W * H),
      fdm_YW,
      fdm_YHW,
      reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
      Y->Shape().Size());

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
