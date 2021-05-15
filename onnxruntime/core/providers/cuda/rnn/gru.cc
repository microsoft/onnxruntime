// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gru.h"
#include "rnn_impl.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/math/gemm_helper.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_VERSIONED_TYPED(T)                                      \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                      \
      GRU,                                                                      \
      kOnnxDomain,                                                              \
      7,                                                                        \
      13,                                                                       \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder()                                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>())         \
          .InputMemoryType<OrtMemTypeCPUInput>(RNN_Input_Index::sequence_lens), \
      GRU<T>);

#define REGISTER_KERNEL_TYPED(T)                                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      GRU,                                                                      \
      kOnnxDomain,                                                              \
      14,                                                                       \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder()                                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>())         \
          .InputMemoryType<OrtMemTypeCPUInput>(RNN_Input_Index::sequence_lens), \
      GRU<T>);

REGISTER_KERNEL_VERSIONED_TYPED(float);
REGISTER_KERNEL_VERSIONED_TYPED(double);
REGISTER_KERNEL_VERSIONED_TYPED(MLFloat16);

REGISTER_KERNEL_TYPED(float);
REGISTER_KERNEL_TYPED(double);
REGISTER_KERNEL_TYPED(MLFloat16);

}  // namespace cuda
}  // namespace onnxruntime
