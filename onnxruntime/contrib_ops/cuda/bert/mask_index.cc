// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/framework/tensorprotoutils.h"
#include "onnx/defs/tensor_proto_util.h"
#include "mask_index.h"
#include "mask_index_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      MaskIndex,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      MaskIndex<T>);

REGISTER_KERNEL_TYPED(int32_t)
REGISTER_KERNEL_TYPED(int64_t)
REGISTER_KERNEL_TYPED(float)

using namespace ONNX_NAMESPACE;

template <typename T>
MaskIndex<T>::MaskIndex(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
}

template <typename T>
Status MaskIndex<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* mask = context->Input<Tensor>(0);

  const auto input_dims = mask->Shape().GetDims();
  if (input_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input is expected to have 2 dimensions, got ", input_dims.size());
  }

  std::vector<int64_t> mask_index_dims;
  mask_index_dims.push_back(2 * input_dims[0]);
  TensorShape mask_index_shape(mask_index_dims);
  Tensor* mask_index = context->Output(0, mask_index_shape);

  int batch_size = static_cast<int>(input_dims[0]);
  int sequence_length = static_cast<int>(input_dims[1]);

  cudaStream_t stream = nullptr;  // use default stream
  if (!LaunchMaskIndexKernel<T>(
          stream,
          mask->template Data<T>(),
          mask_index->template MutableData<int32_t>(),
          batch_size,
          sequence_length)) {
    // Get last error to reset it to cudaSuccess.
    CUDA_CALL(cudaGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  return Status::OK();
}

}  //namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
