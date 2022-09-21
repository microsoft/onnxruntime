// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/dropout_grad.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "orttraining/training_ops/cuda/nn/dropout_grad_impl.h"

namespace onnxruntime {
namespace cuda {

namespace {

template <typename T>
struct GetRatioDataImpl {
  void operator()(const Tensor* ratio, float& ratio_data) const {
    ratio_data = static_cast<float>(*(ratio->template Data<T>()));
    ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f, "ratio_data is outside range [0, 1)");
  }
};

template <typename T>
struct DropoutGradComputeImpl {
  void operator()(cudaStream_t stream, const int64_t N, const Tensor& dY, const void* mask_data, const float ratio_data,
                  Tensor& dX, bool use_bitmask) const {
    typedef typename ToCudaType<T>::MappedType CudaT;

    const CudaT* dY_data = reinterpret_cast<const CudaT*>(dY.template Data<T>());
    CudaT* dX_data = reinterpret_cast<CudaT*>(dX.template MutableData<T>());
    DropoutGradientKernelImpl<CudaT>(stream, N, dY_data, mask_data, ratio_data, dX_data, use_bitmask);
  }
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(DropoutGrad, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T1", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
                            .InputMemoryType(OrtMemTypeCPUInput, 2),
                        DropoutGrad<false>);

ONNX_OPERATOR_KERNEL_EX(BitmaskDropoutGrad, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T1", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
                            .TypeConstraint("T3", DataTypeImpl::GetTensorType<BitmaskElementType>())
                            .InputMemoryType(OrtMemTypeCPUInput, 2)
                            .InputMemoryType(OrtMemTypeCPUInput, 3),
                        DropoutGrad<true>);

template <bool UseBitmask>
Status DropoutGrad<UseBitmask>::ComputeInternal(OpKernelContext* context) const {
  auto dY = context->Input<Tensor>(0);
  const TensorShape& shape = dY->Shape();
  const int64_t N = shape.Size();

  auto mask = context->Input<Tensor>(1);
  if (UseBitmask) {
    ORT_ENFORCE(mask->Shape().Size() == (N + kNumBitsPerBitmaskElement - 1) / kNumBitsPerBitmaskElement);
  } else {
    ORT_ENFORCE(mask->Shape().Size() == N);
  }

  const void* mask_data = mask->DataRaw();

  // Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(2);
  if (ratio) {
    utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(ratio->GetElementType());
    t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  }

  auto dX = context->Output(0, shape);

  utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(dY->GetElementType());
  t_disp.Invoke<DropoutGradComputeImpl>(Stream(), N, *dY, mask_data, ratio_data, *dX, UseBitmask);
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
