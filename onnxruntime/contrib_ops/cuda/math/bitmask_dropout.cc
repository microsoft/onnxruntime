// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/bitmask_dropout.h"

#include "contrib_ops/cuda/math/bitmask_dropout_impl.h"
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

namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    BitmaskDropout,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
        .TypeConstraint("T1", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint32_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2),
    BitmaskDropout);

template <typename T>
struct BitmaskDropoutComputeImpl {
  Status operator()(const cudaDeviceProp& prop,
                    cudaStream_t stream,
                    const int64_t N,
                    const float ratio_data,
                    PhiloxGenerator& generator,
                    const Tensor& X,
                    Tensor& Y,
                    uint32_t* mask_data) const {
    typedef typename ToCudaType<T>::MappedType CudaT;

    const CudaT* X_data = reinterpret_cast<const CudaT*>(X.template Data<T>());
    CudaT* Y_data = reinterpret_cast<CudaT*>(Y.template MutableData<T>());

    BitmaskDropoutKernelImpl<CudaT>(prop, stream, N, ratio_data, generator, X_data, Y_data, mask_data);

    return Status::OK();
  }
};

Status BitmaskDropout::ComputeInternal(OpKernelContext* context) const {
  //Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(X, "X Input is not available.");

  const TensorShape& x_shape = X->Shape();
  const int64_t N = x_shape.Size();

  //Get Y_data
  Tensor* Y = context->Output(0, x_shape);

  //Get mask_data
  int64_t input_elements = x_shape.Size();
  int64_t mask_elements = (input_elements + 31) / 32;
  Tensor* mask = context->Output(1, {mask_elements});

  //Get the ratio_data
  float ratio_data = default_ratio_;
  const Tensor* ratio = context->Input<Tensor>(1);
  if (ratio) {
    utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(ratio->GetElementType());
    t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  }

  //Check for inference mode.
  const Tensor* training_mode = context->Input<Tensor>(2);
  bool is_training_mode = (training_mode != nullptr) && *(training_mode->Data<bool>());
  if (!is_training_mode) {
    ratio_data = 0.0f;
  }

  IAllocatorUniquePtr<uint32_t> temp_mask_buffer{};  // buffer to use if mask is not provided
  uint32_t* const mask_data = [this, N, mask, &temp_mask_buffer]() {
    if (mask) return mask->MutableData<uint32_t>();
    temp_mask_buffer = GetScratchBuffer<uint32_t>(NumBitmaskElements(N));
    return temp_mask_buffer.get();
  }();

  PhiloxGenerator& generator = generator_ ? *generator_ : PhiloxGenerator::Default();

  utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(X->GetElementType());
  return t_disp.InvokeRet<Status, BitmaskDropoutComputeImpl>(GetDeviceProp(), Stream(), N, ratio_data, generator, *X, *Y, mask_data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
