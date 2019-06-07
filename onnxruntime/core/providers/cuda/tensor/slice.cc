// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "slice.h"
#include "slice_impl.h"

namespace onnxruntime {
namespace cuda {
#define REGISTER_VERSIONED_TYPED_SLICE(TIND)                                              \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                \
      Slice,                                                                              \
      kOnnxDomain,                                                                        \
      1, 9,                                                                               \
      TIND,                                                                               \
      kCudaExecutionProvider,                                                             \
      KernelDefBuilder().InputMemoryType<OrtMemTypeCPUInput>(1).                          \
                         InputMemoryType<OrtMemTypeCPUInput>(2).                          \
                         InputMemoryType<OrtMemTypeCPUInput>(3).                          \
                         TypeConstraint("T",    DataTypeImpl::AllFixedSizeTensorTypes()). \
                         TypeConstraint("Tind", DataTypeImpl::GetTensorType<TIND>()),     \
      Slice<TIND,false>);

REGISTER_VERSIONED_TYPED_SLICE(int32_t) 
REGISTER_VERSIONED_TYPED_SLICE(int64_t) 
      
}  // namespace cuda
}  // namespace onnxruntime
