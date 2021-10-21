// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/slice.h"
#include "core/providers/cuda/tensor/slice_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_TYPED_DYNAMICSLICE(TIND)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      DynamicSlice,                                                     \
      kOnnxDomain,                                                      \
      1,                                                                \
      TIND,                                                             \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                       \
          .InputMemoryType(OrtMemTypeCPUInput, 3)                       \
          .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()) \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()), \
      cuda::Slice<true>);

REGISTER_TYPED_DYNAMICSLICE(int32_t)
REGISTER_TYPED_DYNAMICSLICE(int64_t)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
