// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/bitmask_dropout_grad.h"

#include "orttraining/training_ops/cuda/nn/bitmask_dropout_grad_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {

namespace {

template <typename T>
struct GetRatioDataImpl {
  void operator()(const Tensor* ratio, float& ratio_data) const {
    ratio_data = static_cast<float>(*(ratio->template Data<T>()));
    ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f, "ratio_data is outside range [0, 1)");
  }
};

}  // namespace

namespace cuda {

ONNX_OPERATOR_KERNEL_EX(BitmaskDropoutGrad, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T1", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
                            .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint32_t>())
                            .InputMemoryType(OrtMemTypeCPUInput, 2)
                            .InputMemoryType(OrtMemTypeCPUInput, 3),
                        BitmaskDropoutGrad);

template <typename T>
struct BitmaskDropoutGradComputeImpl {
  void operator()(
      const cudaDeviceProp& prop,
      cudaStream_t stream,
      const int64_t N,
      const Tensor& dY,
      const uint32_t* mask_data,
      const float ratio_data,
      Tensor& dX) const {
    typedef typename ToCudaType<T>::MappedType CudaT;

    const CudaT* dY_data = reinterpret_cast<const CudaT*>(dY.template Data<T>());
    CudaT* dX_data = reinterpret_cast<CudaT*>(dX.template MutableData<T>());
    BitmaskDropoutGradientKernelImpl<CudaT>(prop, stream, N, dY_data, mask_data, ratio_data, dX_data);
  }
};

Status BitmaskDropoutGrad::ComputeInternal(OpKernelContext* context) const {
  auto dY = context->Input<Tensor>(0);
  const TensorShape& shape = dY->Shape();
  const int64_t N = shape.Size();

  auto mask = context->Input<Tensor>(1);
  ORT_ENFORCE(mask->Shape().Size() == ((N + 31) / 32));
  const uint32_t* mask_data = mask->template Data<uint32_t>();

  //Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(2);
  if (ratio) {
    utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(ratio->GetElementType());
    t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  }

  //Check for inference mode.
  const Tensor* training_mode = context->Input<Tensor>(3);
  bool is_training_mode = (training_mode != nullptr) && *(training_mode->Data<bool>());
  if (!is_training_mode) {
    ratio_data = 0.0f;
  }

  auto dX = context->Output(0, shape);

  utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(dY->GetElementType());
  t_disp.Invoke<BitmaskDropoutGradComputeImpl>(GetDeviceProp(), Stream(), N, *dY, mask_data, ratio_data, *dX);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime