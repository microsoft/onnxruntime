// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/s2s_split_quickgelu_fusion.h"
#include "contrib_ops/cuda/math/s2s_split_quickgelu_fusion_impl.h"

using namespace onnxruntime::common;
namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    S2SModelSplitQuickGelu, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<float, MLFloat16, BFloat16>()),
    S2SModelSplitQuickGelu);

template <typename T>
void S2SModelSplitQuickGelu::KernelLaunchDispatcher<T>::operator()(cudaStream_t stream, int num_outputs,
                                                                   const Tensor& input, Tensor& output) const {
  using CudaT = typename ToCudaType<T>::MappedType;
  LaunchS2SModelSplitQuickGeluKernel<CudaT>(stream, num_outputs, reinterpret_cast<const CudaT*>(input.template Data<T>()),
                                            reinterpret_cast<CudaT*>(output.template MutableData<T>()));
}

Status S2SModelSplitQuickGelu::ComputeInternal(OpKernelContext* context) const {
  const auto* input = context->Input<Tensor>(0);
  ORT_ENFORCE(input);
  const auto& input_shape = input->Shape();
  auto output_shape = input_shape;
  output_shape[1] /= 2;
  auto* output = context->Output(0, output_shape);
  ORT_ENFORCE(output);

  utils::MLTypeCallDispatcher<float, MLFloat16, BFloat16> dispatcher{input->GetElementType()};
  dispatcher.Invoke<KernelLaunchDispatcher>(Stream(context), 2, *input, *output);
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
