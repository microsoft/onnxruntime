// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/dropout_grad.h"
#include "core/providers/cuda/nn/dropout.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_GRADIENT_KERNEL(OpName)                                 \
  ONNX_OPERATOR_KERNEL_EX(                                               \
      OpName,                                                            \
      kMSDomain,                                                         \
      1,                                                                 \
      kCudaExecutionProvider,                                            \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T", ALL_IEEE_FLOAT_TENSOR_TYPES)              \
          .TypeConstraint("T1", ALL_IEEE_FLOAT_TENSOR_TYPES)             \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())     \
          .InputMemoryType<OrtMemTypeCPUInput>(2),                       \
      DropoutGrad);

REGISTER_GRADIENT_KERNEL(DropoutGrad)

template <typename T>
struct DropoutGradComputeImpl {
  void operator()(cudaStream_t stream,
                  const int64_t N,
                  const Tensor& dY,
                  const bool* mask_data,
                  const float ratio_data,
                  Tensor& dX) const {
    typedef typename ToCudaType<T>::MappedType CudaT;

    const CudaT* dY_data = reinterpret_cast<const CudaT*>(dY.template Data<T>());
    CudaT* dX_data = reinterpret_cast<CudaT*>(dX.template MutableData<T>());
    DropoutGradientKernelImpl<CudaT>(stream, N, dY_data, mask_data, ratio_data, dX_data);
  }
};

Status DropoutGrad::ComputeInternal(OpKernelContext* context) const {
  auto dY = context->Input<Tensor>(0);
  const TensorShape& shape = dY->Shape();
  const int64_t N = shape.Size();

  auto mask = context->Input<Tensor>(1);
  ORT_ENFORCE(mask->Shape().Size() == N);
  const bool* mask_data = mask->template Data<bool>();

  //Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(2);
  if (ratio) {
    utils::MLTypeCallDispatcher<ALL_IEEE_FLOAT_DATA_TYPES> t_disp(ratio->GetElementType());
    t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  }

  auto dX = context->Output(0, shape);

  utils::MLTypeCallDispatcher<ALL_IEEE_FLOAT_DATA_TYPES> t_disp(dY->GetElementType());
  t_disp.Invoke<DropoutGradComputeImpl>(Stream(), N, *dY, mask_data, ratio_data, *dX);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
