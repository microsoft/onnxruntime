// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/s2s_split_quickgelu_fusion.h"

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/math/s2s_split_quickgelu_fusion_impl.h"

using namespace onnxruntime;
using namespace onnxruntime::cuda;
using namespace onnxruntime::contrib::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    S2SModelSplitQuickGelu, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>()),
    S2SModelSplitQuickGelu);

template <typename T>
void S2SModelSplitQuickGelu::KernelLaunchDispatcher<T>::operator()(cudaStream_t stream, int64_t input_size, int64_t axis,
                                                                   int64_t alpha, const Tensor& X, const Tensor& S,
                                                                   Tensor& Y) const {
  using CudaT = typename ToCudaType<T>::MappedType;
  LaunchS2SModelSplitQuickGeluKernel<CudaT>(stream, input_size, axis, alpha, reinterpret_cast<const CudaT*>(X.template Data<T>()),
                                            reinterpret_cast<const CudaT*>(S.template Data<T>()),
                                            reinterpret_cast<CudaT*>(Y.template MutableData<T>()));
}

Status S2SModelSplitQuickGelu::ComputeInternal(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X);
  const auto* S = context->Input<Tensor>(1);
  ORT_ENFORCE(S);

  const auto& input_shape = X->Shape();
  const auto& split_shape = S->Shape();
  ORT_ENFORCE(input_shape.NumDimensions() >= 1 && split_shape.NumDimensions() == 1,
              "S must be 1-dimensional.");

  // TODO: output_shape should be different
  auto* Y = context->Output(0, input_shape);
  ORT_ENFORCE(Y);

  const auto input_size = input_shape.Size();
  utils::MLTypeCallDispatcher<MLFloat16, float, double, BFloat16> dispatcher{X->GetElementType()};
  // TODO: Get axis value
  int64_t axis = info.GetAttrOrDefault<float>("axis", -1);

  // TODO: Get alpha value
  int64_t alpha_ = info.GetAttrOrDefault<float>("alpha", 1.0);

  dispatcher.Invoke<KernelLaunchDispatcher>(Stream(context), input_size, axis, alpha, *X, *S, *Y);

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
