// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/bias_softmax.h"

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/math/bias_softmax_impl.h"

using namespace onnxruntime;
using namespace onnxruntime::cuda;
using namespace onnxruntime::contrib::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename T>
struct DispatchBiasSoftmaxImpl {
  Status operator()(cudaStream_t stream, cudnnHandle_t cudnn_handle, Tensor* Y, const Tensor* X, const Tensor* B,
                    int element_count, int batch_count, bool is_inner_broadcast, int bias_broadcast_size,
                    size_t element_size) {
    typedef typename ToCudaType<T>::MappedType CudaT;
    CudaT* output_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());
    const CudaT* input_data = reinterpret_cast<const CudaT*>(X->template Data<T>());
    const CudaT* bias_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
    return BiasSoftmaxImpl<CudaT>(stream, cudnn_handle, output_data, input_data, bias_data, element_count, batch_count,
                                  is_inner_broadcast, bias_broadcast_size, element_size);
  }
};

}  // namespace

// MIOpen doesn't support double so ROCm kernel doesn't have double support for now.
#ifdef USE_ROCM
#define BIAS_SOFTMAX_KERNEL_DATATYPES \
  { DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<MLFloat16>() }
#define BIAS_SOFTMAX_DISPATCH_TYPES float, MLFloat16
#else
#define BIAS_SOFTMAX_KERNEL_DATATYPES DataTypeImpl::AllIEEEFloatTensorTypes()
#define BIAS_SOFTMAX_DISPATCH_TYPES float, MLFloat16, double
#endif

ONNX_OPERATOR_KERNEL_EX(BiasSoftmax, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create()).TypeConstraint("T", BIAS_SOFTMAX_KERNEL_DATATYPES), BiasSoftmax);

Status BiasSoftmax::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  const TensorShape& X_shape = X->Shape();
  const TensorShape& B_shape = B->Shape();
  Tensor* Y = ctx->Output(0, X_shape);

  const int axis = static_cast<int>(HandleNegativeAxis(axis_, X_shape.NumDimensions()));
  const int batch_count = static_cast<int>(X_shape.SizeToDimension(axis));
  const int element_count = static_cast<int>(X_shape.SizeFromDimension(axis));
  int bias_broadcast_size = static_cast<int>(B_shape.Size() / element_count);
  if (is_inner_broadcast_) bias_broadcast_size = batch_count / bias_broadcast_size;
  utils::MLTypeCallDispatcher<BIAS_SOFTMAX_DISPATCH_TYPES> t_disp(X->GetElementType());
  return t_disp.InvokeRet<Status, DispatchBiasSoftmaxImpl>(Stream(), CudnnHandle(), Y, X, B, element_count, batch_count,
                                                           is_inner_broadcast_, bias_broadcast_size,
                                                           X->DataType()->Size());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
