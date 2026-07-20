// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "roialign.h"
#include "roialign_impl.h"

namespace onnxruntime {
namespace cuda {

#define ADD_VERSIONED_TYPED_ROIALIGN_OP_10(T)                            \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                               \
      RoiAlign,                                                          \
      kOnnxDomain,                                                       \
      10,                                                                \
      15,                                                                \
      T,                                                                 \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()), \
      RoiAlign<T>);

#define ADD_VERSIONED_TYPED_ROIALIGN_OP_16(T)                            \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                               \
      RoiAlign,                                                          \
      kOnnxDomain,                                                       \
      16,                                                                \
      21,                                                                \
      T,                                                                 \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()), \
      RoiAlign<T>);

#define ADD_TYPED_ROIALIGN_OP_22(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      RoiAlign,                                                          \
      kOnnxDomain,                                                       \
      22,                                                                \
      T,                                                                 \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()), \
      RoiAlign<T>);

template <typename T>
Status RoiAlign<T>::ComputeInternal(OpKernelContext* context) const {
  // X
  const auto* X_ptr = context->Input<Tensor>(0);
  // rois
  const auto* rois_ptr = context->Input<Tensor>(1);
  // batch indices
  const auto* batch_indices_ptr = context->Input<Tensor>(2);

  const auto& x_dims = X_ptr->Shape();
  const auto& rois_dims = rois_ptr->Shape();
  const auto& batch_indices_dims = batch_indices_ptr->Shape();

  auto num_rois = batch_indices_dims[0];
  auto num_roi_cols = rois_dims[1];

  auto status = CheckROIAlignValidInput(X_ptr, rois_ptr, batch_indices_ptr);
  if (status != Status::OK()) {
    return status;
  }

  Tensor& Y = *context->Output(0, {num_rois, x_dims[1], this->output_height_, this->output_width_});
  int64_t output_size = Y.Shape().Size();

  if (output_size > 0) {
    RoiAlignImpl(
        Stream(context),
        output_size,  // num threads
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(X_ptr->Data<T>()),
        ToCudaType<T>::FromFloat(this->spatial_scale_),
        x_dims[1],  // num channels
        x_dims[2],  // height
        x_dims[3],  // width
        this->output_height_,
        this->output_width_,
        this->sampling_ratio_,
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(rois_ptr->Data<T>()),
        num_roi_cols,
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(Y.MutableData<T>()),
        this->mode_ == RoiAlignMode::avg,
        this->half_pixel_,
        batch_indices_ptr->Data<int64_t>(),
        x_dims[0]);  // batch_size
  }

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T)          \
  ADD_VERSIONED_TYPED_ROIALIGN_OP_10(T) \
  ADD_VERSIONED_TYPED_ROIALIGN_OP_16(T) \
  ADD_TYPED_ROIALIGN_OP_22(T)           \
  template Status RoiAlign<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
// MLFloat16 is available for RoiAlign op from version 16 (not version 10):
ADD_VERSIONED_TYPED_ROIALIGN_OP_16(MLFloat16)
ADD_TYPED_ROIALIGN_OP_22(MLFloat16)
template Status RoiAlign<MLFloat16>::ComputeInternal(OpKernelContext* ctx) const;

// BFloat16 is available for RoiAlign op from version 22:
ADD_TYPED_ROIALIGN_OP_22(BFloat16)
template Status RoiAlign<BFloat16>::ComputeInternal(OpKernelContext* ctx) const;

}  // namespace cuda
};  // namespace onnxruntime
