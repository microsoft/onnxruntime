// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/sequence/sequence_ops.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/cpu/tensor/utils.h"

using namespace onnxruntime::common;

namespace onnxruntime {

// SequenceLength
ONNX_CPU_OPERATOR_KERNEL(
    SequenceLength,
    11,
    KernelDefBuilder()
        .TypeConstraint("S", DataTypeImpl::AllSequenceTensorTypes())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
    SequenceLength);

Status SequenceLength::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<VectorTensor>(0);
  ORT_ENFORCE(X != nullptr, "Got nullptr for sequence input.");

  auto* Y = context->Output(0, {});
  auto* Y_data = Y->template MutableData<int64_t>();
  *Y_data = static_cast<int64_t>(X->size());

  return Status::OK();
}

// SequenceAt
ONNX_CPU_OPERATOR_KERNEL(
    SequenceAt,
    11,
    KernelDefBuilder()
        .TypeConstraint("S", DataTypeImpl::AllSequenceTensorTypes())
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("I", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int32_t>(),
                                 DataTypeImpl::GetTensorType<int64_t>()}),
    SequenceAt);

static int64_t GetSeqIdx(const Tensor& idx_tensor) {
  int64_t seq_idx = INT_MAX;
  auto idx_tensor_dtype = utils::GetTensorProtoType(idx_tensor);
  switch (idx_tensor_dtype) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      const auto* idx_data = idx_tensor.Data<int32_t>();
      seq_idx = static_cast<int64_t>(*idx_data);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      const auto* idx_data = idx_tensor.Data<int64_t>();
      seq_idx = *idx_data;
      break;
    }
    default:
      ORT_THROW("Unsupported data type: ", idx_tensor_dtype);
  }
  return seq_idx;
}

bool ValidateSeqIdx(int64_t input_seq_idx, int64_t seq_size) {
  bool retval = false;
  if (input_seq_idx < 0) {
    retval = input_seq_idx <= -1 && input_seq_idx >= -seq_size;
  } else {
    retval = input_seq_idx < seq_size;
  }
  return retval;
}

template <typename T>
static void CopyTensor(const Tensor& indexed_tensor, Tensor& output_tensor) {
  const auto* input_data = indexed_tensor.template Data<T>();
  auto* output_data = output_tensor.template MutableData<T>();
  memcpy(output_data, input_data, indexed_tensor.SizeInBytes());
}

Status SequenceAt::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<VectorTensor>(0);
  ORT_ENFORCE(X != nullptr, "Got nullptr for sequence input.");

  const auto* I = context->Input<Tensor>(1);
  ORT_ENFORCE(I != nullptr, "Got nullptr input for index tensor");
  int64_t input_seq_idx = GetSeqIdx(*I);
  if (!ValidateSeqIdx(input_seq_idx, static_cast<int64_t>(X->size()))) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid sequence index (", input_seq_idx, ") specified for sequence of size (", X->size(), ")");
  }

  if (input_seq_idx < 0) {
    input_seq_idx = static_cast<int64_t>(X->size()) + input_seq_idx;
  }
  const Tensor& indexed_tensor = (*X)[input_seq_idx];
  auto* Y = context->Output(0, indexed_tensor.Shape().GetDims());
  CopyCpuTensor(&indexed_tensor, Y);

  return Status::OK();
}

}  // namespace onnxruntime
