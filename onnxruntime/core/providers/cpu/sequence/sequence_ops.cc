// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/sequence/sequence_ops.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/framework/TensorSeq.h"

using namespace onnxruntime::common;

namespace onnxruntime {

// TODO: The current implementation of sequence ops relies on tensor copies. Ideally we should try to avoid
// these copies. This has been postponed due to lack of time.

// SequenceLength
ONNX_CPU_OPERATOR_KERNEL(
    SequenceLength,
    11,
    KernelDefBuilder()
        .TypeConstraint("S", DataTypeImpl::AllSequenceTensorTypes())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
    SequenceLength);

Status SequenceLength::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<TensorSeq>(0);
  ORT_ENFORCE(X != nullptr, "Got nullptr for sequence input.");

  auto* Y = context->Output(0, {});
  ORT_ENFORCE(Y != nullptr, "SequenceLength: Got nullptr for output tensor");
  auto* Y_data = Y->template MutableData<int64_t>();
  *Y_data = static_cast<int64_t>(X->tensors.size());

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

Status SequenceAt::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<TensorSeq>(0);
  ORT_ENFORCE(X != nullptr, "Got nullptr for sequence input.");

  const auto* I = context->Input<Tensor>(1);
  ORT_ENFORCE(I != nullptr, "Got nullptr input for index tensor");
  int64_t input_seq_idx = GetSeqIdx(*I);
  if (!ValidateSeqIdx(input_seq_idx, static_cast<int64_t>(X->tensors.size()))) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid sequence index (", input_seq_idx, ") specified for sequence of size (", X->tensors.size(), ")");
  }

  if (input_seq_idx < 0) {
    input_seq_idx = static_cast<int64_t>(X->tensors.size()) + input_seq_idx;
  }
  const Tensor& indexed_tensor = X->tensors[input_seq_idx];
  auto* Y = context->Output(0, indexed_tensor.Shape().GetDims());
  ORT_ENFORCE(Y != nullptr, "SequenceAt: Got nullptr for output tensor");
  CopyCpuTensor(&indexed_tensor, Y);

  return Status::OK();
}

// SequenceEmpty
ONNX_CPU_OPERATOR_KERNEL(
    SequenceEmpty,
    11,
    KernelDefBuilder()
        .TypeConstraint("S", DataTypeImpl::AllSequenceTensorTypes()),
    SequenceEmpty);

SequenceEmpty::SequenceEmpty(const OpKernelInfo& info) : OpKernel(info) {
  if (!info.GetAttr("dtype", &dtype_).IsOK()) {
    dtype_ = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  }
}

Status SequenceEmpty::Compute(OpKernelContext* context) const {
  auto* Y = context->Output<TensorSeq>(0);
  ORT_ENFORCE(Y != nullptr, "SequenceEmpty: Got nullptr for output sequence");
  MLDataType seq_dtype{};
  switch (dtype_) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      seq_dtype = DataTypeImpl::GetType<float>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      seq_dtype = DataTypeImpl::GetType<bool>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      seq_dtype = DataTypeImpl::GetType<int>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      seq_dtype = DataTypeImpl::GetType<double>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      seq_dtype = DataTypeImpl::GetType<std::string>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      seq_dtype = DataTypeImpl::GetType<int8_t>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      seq_dtype = DataTypeImpl::GetType<uint8_t>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      seq_dtype = DataTypeImpl::GetType<uint16_t>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      seq_dtype = DataTypeImpl::GetType<int16_t>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      seq_dtype = DataTypeImpl::GetType<int64_t>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      seq_dtype = DataTypeImpl::GetType<uint32_t>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      seq_dtype = DataTypeImpl::GetType<uint64_t>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      seq_dtype = DataTypeImpl::GetType<MLFloat16>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      seq_dtype = DataTypeImpl::GetType<BFloat16>();
      break;
    default:
      ORT_THROW("Unsupported 'dtype' value: ", dtype_);
  }

  Y->dtype = seq_dtype;
  return Status::OK();
}

// SequenceInsert
ONNX_CPU_OPERATOR_KERNEL(
    SequenceInsert,
    11,
    KernelDefBuilder()
        .TypeConstraint("S", DataTypeImpl::AllSequenceTensorTypes())
        .TypeConstraint("I", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int32_t>(),
                                 DataTypeImpl::GetTensorType<int64_t>()}),
    SequenceInsert);

Status CreateAndCopyCpuTensor(const Tensor& in_tensor, OpKernelContext* context, Tensor& out_tensor) {
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  out_tensor = Tensor(in_tensor.DataType(), onnxruntime::TensorShape(in_tensor.Shape()), alloc);
  CopyCpuTensor(&in_tensor, &out_tensor);
  return Status::OK();
}

Status SequenceInsert::Compute(OpKernelContext* context) const {
  const auto* S = context->Input<TensorSeq>(0);
  ORT_ENFORCE(S != nullptr, "Got nullptr for sequence input.");

  const auto* X = context->Input<Tensor>(1);
  ORT_ENFORCE(X != nullptr, "Got nullptr for input tensor.");

  // Data type of the input tensor MUST be same as that of the input sequence
  if (S->dtype != X->DataType()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Data type of the input tensor MUST be same as that of the input sequence. Sequence data type (",
                           DataTypeImpl::ToString(S->dtype), "), input tensor data type (", DataTypeImpl::ToString(X->DataType()), ")");
  }

  const auto* I = context->Input<Tensor>(2);
  int64_t num_tensors_input_seq = static_cast<int64_t>(S->tensors.size());
  int64_t input_seq_idx = num_tensors_input_seq + 1;  // default is append
  if (I) {                                            // position is optional
    input_seq_idx = GetSeqIdx(*I);
    if (!ValidateSeqIdx(input_seq_idx, static_cast<int64_t>(num_tensors_input_seq))) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Invalid sequence index (", input_seq_idx, ") specified for sequence of size (", num_tensors_input_seq, ")");
    }

    if (input_seq_idx < 0) {
      input_seq_idx = static_cast<int64_t>(num_tensors_input_seq) + input_seq_idx;
    }
  }

  auto* Y = context->Output<TensorSeq>(0);
  ORT_ENFORCE(Y != nullptr, "SequenceInsert: Got nullptr for output sequence");
  Y->dtype = S->dtype;
  Y->tensors.resize(num_tensors_input_seq + 1);
  for (int i = 0, yidx = 0; i < num_tensors_input_seq; ++i, ++yidx) {
    if (i == input_seq_idx) {
      CreateAndCopyCpuTensor(*X, context, Y->tensors[yidx]);
      ++yidx;
      CreateAndCopyCpuTensor(S->tensors[i], context, Y->tensors[yidx]);
    } else {
      CreateAndCopyCpuTensor(S->tensors[i], context, Y->tensors[yidx]);
    }
  }
  if (input_seq_idx == num_tensors_input_seq + 1) {
    CreateAndCopyCpuTensor(*X, context, Y->tensors[Y->tensors.size() - 1]);
  }

  return Status::OK();
}

// SequenceErase
ONNX_CPU_OPERATOR_KERNEL(
    SequenceErase,
    11,
    KernelDefBuilder()
        .TypeConstraint("S", DataTypeImpl::AllSequenceTensorTypes())
        .TypeConstraint("I", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int32_t>(),
                                 DataTypeImpl::GetTensorType<int64_t>()}),
    SequenceErase);

Status SequenceErase::Compute(OpKernelContext* context) const {
  const auto* S = context->Input<TensorSeq>(0);
  ORT_ENFORCE(S != nullptr, "Got nullptr for sequence input.");

  const auto* I = context->Input<Tensor>(1);
  int64_t num_tensors_input_seq = static_cast<int64_t>(S->tensors.size());
  int64_t input_seq_idx = num_tensors_input_seq - 1;  // default is erase last one
  if (I) {                                            // position is optional
    input_seq_idx = GetSeqIdx(*I);
    if (!ValidateSeqIdx(input_seq_idx, static_cast<int64_t>(num_tensors_input_seq))) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Invalid sequence index (", input_seq_idx, ") specified for sequence of size (", num_tensors_input_seq, ")");
    }

    if (input_seq_idx < 0) {
      input_seq_idx = static_cast<int64_t>(num_tensors_input_seq) + input_seq_idx;
    }
  }

  auto* Y = context->Output<TensorSeq>(0);
  ORT_ENFORCE(Y != nullptr, "SequenceErase: Got nullptr for output sequence");
  Y->dtype = S->dtype;
  Y->tensors.resize(num_tensors_input_seq - 1);
  for (int i = 0, yidx = 0; i < num_tensors_input_seq; ++i, ++yidx) {
    if (i == input_seq_idx) {
      --yidx;
    } else {
      CreateAndCopyCpuTensor(S->tensors[i], context, Y->tensors[yidx]);
    }
  }

  return Status::OK();
}

// SequenceConstruct
ONNX_CPU_OPERATOR_KERNEL(
    SequenceConstruct,
    11,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("S", DataTypeImpl::AllSequenceTensorTypes()),
    SequenceConstruct);

Status SequenceConstruct::Compute(OpKernelContext* context) const {
  auto num_inputs = Node().InputArgCount().front();
  ORT_ENFORCE(num_inputs >= 1, "Must have 1 or more inputs");

  auto* Y = context->Output<TensorSeq>(0);
  ORT_ENFORCE(Y != nullptr, "SequenceConstruct: Got nullptr for output sequence");

  MLDataType first_dtype = context->Input<Tensor>(0)->DataType();
  // Before copying check if all tensors are of the same type.
  for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
    const auto* X = context->Input<Tensor>(input_idx);
    if (input_idx > 0 && X->DataType() != first_dtype) {
      ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                      "Violation of the requirment that all input tensors must have the same data type.");
    }
  }

  // now copy the tensors to the output sequence
  Y->dtype = first_dtype;
  Y->tensors.resize(num_inputs);
  for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
    const auto* X = context->Input<Tensor>(input_idx);
    CreateAndCopyCpuTensor(*X, context, Y->tensors[input_idx]);
  }

  return Status::OK();
}
}  // namespace onnxruntime
