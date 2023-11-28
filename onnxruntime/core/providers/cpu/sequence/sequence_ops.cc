// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/sequence/sequence_ops.h"

#include "core/common/narrow.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

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

  auto* Y = context->Output(0, {});
  auto* Y_data = Y->MutableData<int64_t>();
  *Y_data = static_cast<int64_t>(X->Size());

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
  auto idx_tensor_dtype = idx_tensor.GetElementType();
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

constexpr bool ValidateSeqIdx(int64_t input_seq_idx, int64_t seq_size) {
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

  const auto* I = context->Input<Tensor>(1);

  int64_t input_seq_idx = GetSeqIdx(*I);
  if (!ValidateSeqIdx(input_seq_idx, static_cast<int64_t>(X->Size()))) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Invalid sequence index (", input_seq_idx, ") specified for sequence of size (", X->Size(), ")");
  }

  if (input_seq_idx < 0) {
    input_seq_idx = static_cast<int64_t>(X->Size()) + input_seq_idx;
  }
  const Tensor& indexed_tensor = X->Get(onnxruntime::narrow<size_t>(input_seq_idx));
  auto* Y = context->Output(0, indexed_tensor.Shape().GetDims());

  // Using DataTransferManager here allows other non-CPU EPs to use this implementation of the sequence ops
  ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(indexed_tensor, *Y));

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
#if !defined(DISABLE_FLOAT8_TYPES)
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:
      seq_dtype = DataTypeImpl::GetType<Float8E4M3FN>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ:
      seq_dtype = DataTypeImpl::GetType<Float8E4M3FNUZ>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2:
      seq_dtype = DataTypeImpl::GetType<Float8E5M2>();
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ:
      seq_dtype = DataTypeImpl::GetType<Float8E5M2FNUZ>();
      break;
#endif
    default:
      ORT_THROW("Unsupported 'dtype' value: ", dtype_);
  }

  Y->SetType(seq_dtype);
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

// Using DataTransferManager here allows other non-CPU EPs to use this implementation of the sequence ops
static Tensor CloneTensor(const Tensor& in_tensor, OpKernelContext* context, const DataTransferManager& dtm) {
  AllocatorPtr alloc;
  ORT_THROW_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  Tensor tmp(in_tensor.DataType(), onnxruntime::TensorShape(in_tensor.Shape()), alloc);
  ORT_THROW_IF_ERROR(dtm.CopyTensor(in_tensor, tmp));
  return tmp;
}

Status SequenceInsert::Compute(OpKernelContext* context) const {
  const auto* S = context->Input<TensorSeq>(0);
  const auto* X = context->Input<Tensor>(1);

  // Data type of the input tensor MUST be same as that of the input sequence
  if (!S->IsSameDataType(*X)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Data type of the input tensor MUST be same as that of the input sequence. Sequence data type (",
                           DataTypeImpl::ToString(S->DataType()), "), input tensor data type (", DataTypeImpl::ToString(X->DataType()), ")");
  }

  const auto* I = context->Input<Tensor>(2);
  int64_t num_tensors_input_seq = static_cast<int64_t>(S->Size());
  int64_t input_seq_idx = num_tensors_input_seq;  // default is append
  if (I) {                                        // position is optional
    input_seq_idx = GetSeqIdx(*I);
    if (!ValidateSeqIdx(input_seq_idx, num_tensors_input_seq) && input_seq_idx != num_tensors_input_seq) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Invalid sequence index (", input_seq_idx, ") specified for sequence of size (", num_tensors_input_seq, ")");
    }

    if (input_seq_idx < 0) {
      input_seq_idx = static_cast<int64_t>(num_tensors_input_seq) + input_seq_idx;
    }
  }

  auto* Y = context->Output<TensorSeq>(0);
  Y->SetType(S->DataType());
  Y->Reserve(SafeInt<size_t>(num_tensors_input_seq) + 1);

  for (int i = 0; i < num_tensors_input_seq; ++i) {
    if (i == input_seq_idx) {
      // Using DataTransferManager here allows other non-CPU EPs to use this implementation of the sequence ops
      Y->Add(CloneTensor(*X, context, Info().GetDataTransferManager()));
      Y->Add(S->GetAt(i));
    } else {
      Y->Add(S->GetAt(i));
    }
  }
  if (input_seq_idx == num_tensors_input_seq) {
    // Using DataTransferManager here allows other non-CPU EPs to use this implementation of the sequence ops
    Y->Add(CloneTensor(*X, context, Info().GetDataTransferManager()));
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
  const auto* I = context->Input<Tensor>(1);

  int64_t num_tensors_input_seq = static_cast<int64_t>(S->Size());
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
  Y->SetType(S->DataType());
  Y->Reserve(SafeInt<size_t>(num_tensors_input_seq) - 1);

  for (int i = 0; i < num_tensors_input_seq; ++i) {
    if (i == input_seq_idx) {
      continue;
    }
    Y->Add(S->GetAt(i));
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

  MLDataType first_dtype = context->Input<Tensor>(0)->DataType();
  // Before copying check if all tensors are of the same type.
  for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
    const auto* X = context->Input<Tensor>(input_idx);
    if (input_idx > 0 && X->DataType() != first_dtype) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Violation of the requirment that all input tensors must have the same data type.");
    }
  }

  // now copy the tensors to the output sequence
  Y->SetType(first_dtype);
  Y->Reserve(SafeInt<size_t>(num_inputs));
  for (int input_idx = 0; input_idx < num_inputs; ++input_idx) {
    const auto* X = context->Input<Tensor>(input_idx);
    // Using DataTransferManager here allows other non-CPU EPs to use this implementation of the sequence ops
    Y->Add(CloneTensor(*X, context, Info().GetDataTransferManager()));
  }
  return Status::OK();
}

// SplitToSequence

ONNX_CPU_OPERATOR_KERNEL(
    SplitToSequence,
    11,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraints<float, MLFloat16, double, int32_t, int64_t, std::string>())
        .TypeConstraint("S", DataTypeImpl::AllSequenceTensorTypes())
        .TypeConstraint("I", BuildKernelDefConstraints<int32_t, int64_t>()),
    SplitToSequence);

SplitToSequence::SplitToSequence(const OpKernelInfo& info) : OpKernel(info) {
  axis_ = info.GetAttrOrDefault<int64_t>("axis", 0);
  keepdims_ = info.GetAttrOrDefault<int64_t>("keepdims", 1);
}

Status SplitToSequence::Compute(OpKernelContext* context) const {
  const Tensor& input = *context->Input<Tensor>(0);
  const Tensor* p_split_input = context->Input<Tensor>(1);

  return ComputeImpl(*context, input, p_split_input);
}

Status SplitToSequence::PrepareForCompute(const TensorShape& input_shape, int64_t split_scalar, bool is_split_input_scalar,
                                          int64_t& num_outputs, int64_t& axis, int& before_dims,
                                          int& after_dims_including_split_axis, int& after_dims_excluding_split,
                                          bool& is_uneven_split, int& num_remaining_splits,
                                          InlinedVector<int64_t>& split_sizes) const {
  auto input_dims = input_shape.GetDims();
  const auto num_dimensions = gsl::narrow_cast<int64_t>(input_shape.NumDimensions());
  axis = HandleNegativeAxis(axis_, num_dimensions);  // handle negative and enforce axis is valid
  const int64_t split_dim_size = input_dims[onnxruntime::narrow<size_t>(axis)];

  before_dims = narrow<int>(input_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis)));
  after_dims_including_split_axis = narrow<int>(input_shape.SizeFromDimension(onnxruntime::narrow<size_t>(axis)));
  after_dims_excluding_split = (axis + 1 == num_dimensions)
                                   ? 1  // we multiply by this value so must be 1 not 0
                                   : narrow<int>(input_shape.SizeFromDimension(SafeInt<size_t>(axis) + 1));

  if (is_split_input_scalar) {
    auto num_even_splits = split_dim_size / split_scalar;
    num_remaining_splits = gsl::narrow_cast<int>(split_dim_size % split_scalar);
    num_outputs = num_even_splits;
    if (num_remaining_splits != 0) {
      is_uneven_split = true;
      num_outputs += 1;
    }
    split_sizes.resize(onnxruntime::narrow<size_t>(num_outputs));
    std::fill(split_sizes.begin(), split_sizes.begin() + onnxruntime::narrow<size_t>(num_even_splits), split_scalar);
    std::fill(split_sizes.begin() + onnxruntime::narrow<size_t>(num_even_splits), split_sizes.end(), num_remaining_splits);
  } else {
    if (split_sizes.empty()) {
      // populate split_sizes with the same size for each output
      num_outputs = split_dim_size;
      // https://github.com/onnx/onnx/issues/2396
      split_sizes = InlinedVector<int64_t>(static_cast<size_t>(num_outputs), DEFAULT_LENGTH_EACH_OUTPUT_);
    } else {
      auto split_size_sum = std::accumulate(split_sizes.cbegin(), split_sizes.cend(), 0LL);
      if (split_size_sum != split_dim_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "split_size_sum (", split_size_sum, ") != split_dim_size (", split_dim_size, ")");
      }
      num_outputs = split_sizes.size();
    }
  }

  return Status::OK();
}

template <typename T>
inline void copy_data(const T* src, T* dst, size_t count) {
  memcpy(dst, src, count * sizeof(T));
}

template <>
inline void copy_data<std::string>(const std::string* src, std::string* dst, size_t count) {
  const std::string* end = src + count;
  std::copy(src, end, dst);
}

static int64_t GetScalarSplitInput(const Tensor& tensor) {
  int64_t retval = INT_MAX;
  if (tensor.IsDataType<int32_t>()) {
    retval = *(tensor.Data<int32_t>());
  } else if (tensor.IsDataType<int64_t>()) {
    retval = *(tensor.Data<int64_t>());
  } else {
    ORT_THROW("Invalid data type for split tensor ", DataTypeImpl::ToString(tensor.DataType()));
  }
  return retval;
}

static void GetSplitSizesInput(const Tensor& tensor, InlinedVector<int64_t>& split_sizes) {
  auto num_elems = tensor.Shape().Size();
  split_sizes.reserve(onnxruntime::narrow<size_t>(num_elems));
  if (tensor.IsDataType<int32_t>()) {
    const int32_t* data_ptr = tensor.Data<int32_t>();
    std::copy(data_ptr, data_ptr + num_elems, std::back_inserter(split_sizes));
  } else if (tensor.IsDataType<int64_t>()) {
    const int64_t* data_ptr = tensor.Data<int64_t>();
    std::copy(data_ptr, data_ptr + num_elems, std::back_inserter(split_sizes));
  } else {
    ORT_THROW("Invalid data type for split tensor ", DataTypeImpl::ToString(tensor.DataType()));
  }
}

Status SplitToSequence::ComputeImpl(OpKernelContext& context, const Tensor& input,
                                    const Tensor* p_split_input) const {
  auto& input_shape = input.Shape();
  int64_t num_outputs = 0;
  int64_t axis = axis_;
  int before_dims = 0;
  int after_dims_including_split_axis = 0;
  int after_dims_excluding_split = 0;
  int64_t split_scalar = INT_MAX;
  bool is_split_input_scalar = false;
  bool is_uneven_split = false;
  int num_remaining_splits = 0;
  InlinedVector<int64_t> split_sizes;
  const bool is_string_type = input.IsDataTypeString();
  const size_t element_size = (is_string_type) ? 0U : input.DataType()->Size();

  // figure out split_scalar or split_sizes
  if (p_split_input) {
    if (p_split_input->Shape().NumDimensions() == 0) {  // scalar
      split_scalar = GetScalarSplitInput(*p_split_input);
      ORT_ENFORCE(split_scalar > 0, "Split should be > 0");
      is_split_input_scalar = true;
    } else {
      GetSplitSizesInput(*p_split_input, split_sizes);
      ORT_ENFORCE(std::all_of(split_sizes.cbegin(), split_sizes.cend(), [](int64_t value) { return value >= 0; }),
                  "Invalid value in 'split' input. All values must be >= 0");
    }
  }

  // Keep the split dimension or not. Default 1, which means we keep split dimension.
  // If input 'split' is specified, this attribute is ignored.
  bool use_keep_dims = split_sizes.empty();

  ORT_RETURN_IF_ERROR(PrepareForCompute(input_shape,
                                        split_scalar,
                                        is_split_input_scalar,
                                        num_outputs,
                                        axis,
                                        before_dims,
                                        after_dims_including_split_axis,
                                        after_dims_excluding_split,
                                        is_uneven_split,
                                        num_remaining_splits,
                                        split_sizes));
  auto tseq = context.Output<TensorSeq>(0);
  tseq->SetType(input.DataType());
  tseq->Reserve(static_cast<size_t>(num_outputs));

  // copy dimensions so we can update the selected axis in place
  auto output_dimensions = input_shape.AsShapeVector();
  SafeInt<size_t> input_offset = 0;
  const void* input_data = input.DataRaw();
  for (int i = 0; i < num_outputs; ++i) {
    // update size of dimension for axis we're splitting on while considering uneven split
    int split_size;
    if (is_uneven_split && i == num_outputs - 1) {  // only for the last output that has a size different from the rest
      split_size = num_remaining_splits;
    } else {
      split_size = narrow<int>(split_sizes[i]);
    }
    output_dimensions[onnxruntime::narrow<size_t>(axis)] = split_size;

    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(context.GetTempSpaceAllocator(&alloc));
    Tensor output_tensor(input.DataType(), onnxruntime::TensorShape(output_dimensions), alloc);
    void* output_data = output_tensor.MutableDataRaw();

    const auto M = before_dims;
    const auto* A = static_cast<const char*>(input_data) + static_cast<size_t>(input_offset * element_size);
    const auto lda = after_dims_including_split_axis;
    auto* B = output_data;

    const auto N = split_size * after_dims_excluding_split;
    const auto ldb = N;

    if (is_string_type) {
      const auto* src = reinterpret_cast<const std::string*>(A);
      auto* dst = reinterpret_cast<std::string*>(B);
      if (lda == N) {
        copy_data<std::string>(src, dst, static_cast<size_t>(M * N));
      } else {
        size_t lda_offset = 0;
        size_t ldb_offset = 0;
        for (size_t idx = 0; idx < static_cast<size_t>(M); ++idx,
                    lda_offset += lda, ldb_offset += ldb) {
          copy_data<std::string>(src + lda_offset, dst + ldb_offset, static_cast<size_t>(N));
        }
      }
    } else {
      if (lda == N) {
        // if the data is contiguous, we can just copy the data
        const size_t bytes_to_copy = static_cast<size_t>(N) * static_cast<size_t>(M) * element_size;
        memcpy(B, A, bytes_to_copy);
      } else {
        // otherwise we need to copy each row
        const size_t row_bytes = SafeInt<size_t>(N) * element_size;
        const auto lda_bytes_inc = SafeInt<size_t>(lda) * element_size;
        const auto ldb_bytes_inc = SafeInt<size_t>(ldb) * element_size;
        SafeInt<size_t> lda_bytes_offset = 0;
        SafeInt<size_t> ldb_bytes_offset = 0;
        for (size_t idx = 0; idx < static_cast<size_t>(M); ++idx,
                    lda_bytes_offset += lda_bytes_inc, ldb_bytes_offset += ldb_bytes_inc) {
          memcpy(reinterpret_cast<char*>(B) + static_cast<size_t>(ldb_bytes_offset),
                 reinterpret_cast<const char*>(A) + static_cast<size_t>(lda_bytes_offset), row_bytes);
        }
      }
    }

    input_offset += SafeInt<size_t>(split_size) * after_dims_excluding_split;  // offset by the N data we used in this iteration

    // if keep_dims = 0, reshape the tensor by dropping the dimension corresponding to 'axis'
    if (use_keep_dims && keepdims_ == 0) {
      TensorShapeVector new_dims;
      new_dims.reserve(output_dimensions.size() - 1);
      for (int64_t idx = 0, end = static_cast<int64_t>(output_dimensions.size()); idx < end; ++idx) {
        if (idx != axis) {
          new_dims.push_back(output_dimensions[onnxruntime::narrow<size_t>(idx)]);
        }
      }
      output_tensor.Reshape(new_dims);
    }

    // finally move the resulting tensor to the output sequence
    tseq->Add(std::move(output_tensor));
  }

  return Status::OK();
}
}  // namespace onnxruntime
