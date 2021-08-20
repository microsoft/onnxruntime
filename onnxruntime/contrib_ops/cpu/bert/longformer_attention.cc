// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "contrib_ops/cpu/bert/longformer_attention.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    LongformerAttention,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    LongformerAttention<float>);

template <typename T>
LongformerAttention<T>::LongformerAttention(const OpKernelInfo& info) : OpKernel(info), LongformerAttentionBase(info) {
}

template <typename T>
Status LongformerAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask = context->Input<Tensor>(3);
  const Tensor* global_weights = context->Input<Tensor>(4);
  const Tensor* global_bias = context->Input<Tensor>(5);
  const Tensor* global_attention = context->Input<Tensor>(6);
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(), weights->Shape(), bias->Shape(), mask->Shape(),
                                  global_weights->Shape(), global_bias->Shape(), global_attention->Shape()));

  // Input and output shapes:
  //   Input 0 - input       : (batch_size, sequence_length, hidden_size)
  //   Output 0 - output     : (batch_size, sequence_length, hidden_size)
  const auto& shape = input->Shape();
  int batch_size = static_cast<int>(shape[0]);
  int sequence_length = static_cast<int>(shape[1]);
  int hidden_size = static_cast<int>(shape[2]);
  int head_size = hidden_size / num_heads_;

  Tensor* output = context->Output(0, shape);
  T* output_data = output->template MutableData<T>();

  //dummy output
  memset(output_data, 0, sizeof(T) * batch_size * sequence_length * hidden_size);

  ORT_UNUSED_PARAMETER(head_size);
  return Status::OK();
}


}  // namespace contrib
}  // namespace onnxruntime
