// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/col2im.h"

#include "core/framework/element_type_lists.h"
#include "core/framework/TensorSeq.h"
#include "core/providers/common.h"
#include "core/framework/copy.h"
#include "core/common/safeint.h"
#include "core/providers/op_kernel_type_control.h"

namespace onnxruntime {

#define REGISTER_KERNEL_TYPED(T)                                                            \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                                           \
      Col2Im,                                                                               \
      1,                                                                                    \
      T,                                                                                    \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()), \
      Col2Im<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
Status Col2Im<T>::Compute(OpKernelContext* context) const {
  const auto* col_input = context->Input<Tensor>(0);
  const auto* image_shape = context->Input<Tensor>(1);
  const auto* kernel_shape = context->Input<Tensor>(2);

  TensorShape col_shape = col_input->Shape();
  const auto num_image_channels = image_shape->Shape()[1];
  const auto batch_size = col_shape[0];

  const int64_t image_size = image_shape->Shape().Size();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  const int64_t col_buffer_size = col_input->Shape().Size();
  auto col_data = alloc->Alloc(SafeInt<size_t>(sizeof(T)) * col_buffer_size);

  BufferUniquePtr col_buffer(col_data, BufferDeleter(std::move(alloc)));
  T* col_buffer_data = static_cast<T*>(col_buffer.get());

  TensorShapeVector Y_dims;
  Y_dims.insert(Y_dims.begin(), {batch_size, num_image_channels});
  TensorShape Yshape(Y_dims);
  Tensor* Y = context->Output(0, Yshape);
  T* Ydata = Y->template MutableData<T>();

  // template <typename T, class Provider, int order>
  // void Col2imNd(
  //     const T* data_col,
  //     const int64_t* img_shape,
  //     const int64_t* output_shape,
  //     int64_t channels_col,
  //     int64_t img_size,
  //     const int64_t* kernel_shape,
  //     const int64_t* stride,
  //     const int64_t* dilation,
  //     const int64_t* pad,
  //     ptrdiff_t N,
  //     T* data_img,
  //     Provider* provider);

  math::Col2imNd<T, CPUMathUtil, StorageOrder::NCHW>(
    col_buffer_data,
    image_shape->Shape().GetDims().data(),
    col_shape.GetDims().data(),
    num_image_channels,
    image_size,
    kernel_shape->Shape().GetDims().data(),
    col2im_attrs_.strides.data(),
    col2im_attrs_.dilations.data(),
    col2im_attrs_.pads.data(),
    static_cast<int>(kernel_shape->Shape().Size()),
    Ydata,
    &CPUMathUtil::Instance());

  return Status::OK();
}

}  // namespace onnxruntime
