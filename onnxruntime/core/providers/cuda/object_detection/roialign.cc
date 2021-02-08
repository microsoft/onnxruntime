// Copyright (c) Microsoft Corporation. All rights reserved. 
// Licensed under the MIT License. 

#include "roialign.h"
#include "roialign_impl.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      RoiAlign,                                                          \
      kOnnxDomain,                                                       \
      10,                                                                \
      T,                                                                 \
      kCudaExecutionProvider,                                            \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
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

  auto& Y = *context->Output(0, {num_rois, x_dims[1], this->output_height_, this->output_width_});
  int64_t output_size = Y.Shape().Size();

  if (output_size > 0) {
    RoiAlignImpl(
        Stream(),
        output_size,  // num threads
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(X_ptr->template Data<T>()),
        ToCudaType<T>::FromFloat(this->spatial_scale_),
        x_dims[1],  // num channels
        x_dims[2],  // height
        x_dims[3],  // width
        this->output_height_,
        this->output_width_,
        this->sampling_ratio_,
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(rois_ptr->template Data<T>()),
        num_roi_cols,
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(Y.template MutableData<T>()),
        this->mode_ == RoiAlignMode::avg,
        batch_indices_ptr->template Data<int64_t>()
    );
  }

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status RoiAlign<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
//SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace cuda
};  // namespace onnxruntime
