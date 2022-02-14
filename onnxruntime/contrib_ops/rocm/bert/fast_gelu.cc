// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/miopen_common.h"
#include "core/providers/rocm/math/unary_elementwise_ops_impl.h"
#include "fast_gelu.h"
#include "fast_gelu_impl.h"
#include "contrib_ops/cpu/bert/bias_gelu_helper.h"
#include "transformer_common.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FastGelu,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FastGelu<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
FastGelu<T>::FastGelu(const OpKernelInfo& op_kernel_info) : RocmKernel(op_kernel_info) {
  const TransformerOptions* options = TransformerOptions::GetInstance();
  use_half2_ = !options->DisableHalf2();
}

template <typename T>
Status FastGelu<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(bias_gelu_helper::CheckInputs(context));

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* bias = context->Input<Tensor>(1);
  Tensor* output = context->Output(0, input->Shape());

  int64_t input_length = input->Shape().Size();
  if (input_length == 0) {
    return Status::OK();
  }
  int64_t bias_length = (nullptr == bias) ? 0 : bias->Shape().Size();
  typedef typename ToHipType<T>::MappedType HipT;

  // HIP math for BFP16 is not supported, so convert input to fp32 then call kernel
  IAllocatorUniquePtr<float> temp_X;
  IAllocatorUniquePtr<float> temp_B;
  ORT_IF_CONSTEXPR (std::is_same<T, BFloat16>::value) {
    temp_X = GetScratchBuffer<float>(input_length);
    Impl_Cast<HipT, float>(Stream(), reinterpret_cast<const HipT*>(input->template Data<T>()), temp_X.get(), input_length);
  }

  if (temp_X) {
    int64_t output_length = output->Shape().Size();
    auto temp_output = GetScratchBuffer<float>(output_length);
    if (nullptr != bias) {
        temp_B = GetScratchBuffer<float>(bias_length);
        Impl_Cast<HipT, float>(Stream(), reinterpret_cast<const HipT*>(bias->template Data<T>()), temp_B.get(), bias_length);
    }

    if (!LaunchFastGeluKernel<float>(GetDeviceProp(),
                                    Stream(),
                                    static_cast<int>(input_length),
                                    static_cast<int>(bias_length),
                                    temp_X.get(),
                                    (nullptr != bias) ? temp_B.get() : nullptr,
                                    temp_output.get(),
                                    false)) {
      HIP_CALL(hipGetLastError());
      return Status(common::ONNXRUNTIME, common::FAIL);
    }

    Impl_Cast<float, HipT>(Stream(), temp_output.get(), reinterpret_cast<HipT*>(output->template MutableData<T>()), output_length); 
  } else {
    if (!LaunchFastGeluKernel<HipT>(GetDeviceProp(),
                                    Stream(),
                                    static_cast<int>(input_length),
                                    static_cast<int>(bias_length),
                                    reinterpret_cast<const HipT*>(input->template Data<T>()),
                                    (nullptr != bias) ? reinterpret_cast<const HipT*>(bias->template Data<T>()) : nullptr,
                                    reinterpret_cast<HipT*>(output->template MutableData<T>()),
                                    use_half2_)) {
      HIP_CALL(hipGetLastError());
      return Status(common::ONNXRUNTIME, common::FAIL);
    }
  }

  return Status::OK();
}

}  //namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
