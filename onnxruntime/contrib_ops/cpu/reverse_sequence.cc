#include "reverse_sequence.h"
#include "onnx/defs/schema.h"

#include <cmath>
#include <gsl/gsl_algorithm>

namespace onnxruntime {
namespace contrib {

// reverse sequence which has shape [max_seq_len, batch_size, input_size]
template <typename T, typename TIndex>
static Status ReverseSequenceSequenceBatch(
    gsl::span<const T> inputs,
    gsl::span<T> inputs_reverse,
    gsl::span<const TIndex> sequence_lengths,
    const int64_t max_seq_len,
    const int64_t batch_size,
    const int64_t input_size) {

  for (int64_t batch = 0; batch < batch_size; batch++) {
    int64_t seq_len = static_cast<int64_t>(sequence_lengths[batch]);

    ORT_ENFORCE(seq_len <= max_seq_len && seq_len >= 0LL, 
                "Seq_lengths[", batch, "] = ", seq_len, " is out of range [0,", max_seq_len, "]");

    for (int64_t seq = 0; seq < seq_len; seq++) {
      gsl::span<const T> src = inputs.subspan(seq * batch_size * input_size + batch * input_size, input_size);
      gsl::span<T> dest = inputs_reverse.subspan((seq_len - seq - 1) * batch_size * input_size + batch * input_size, input_size);
      gsl::copy(src, dest);
    }

    for (int64_t seq = seq_len; seq < max_seq_len; seq++) {
      gsl::span<const T> src = inputs.subspan(seq * batch_size * input_size + batch * input_size, input_size);
      gsl::span<T> dest = inputs_reverse.subspan(seq * batch_size * input_size + batch * input_size, input_size);
      gsl::copy(src, dest);
    }
  }
  return Status::OK();  
}

// reverse sequence which has shape [batch_size, max_seq_len, input_size]
template <typename T, typename TIndex>
static Status ReverseSequenceBatchSequence(
    gsl::span<const T> inputs,
    gsl::span<T> inputs_reverse,
    gsl::span<const TIndex> sequence_lengths,
    const int64_t max_seq_len,
    const int64_t batch_size,
    const int64_t input_size) {

  for (int64_t batch = 0; batch < batch_size; batch++) {
    int64_t seq_len = sequence_lengths[batch];

    ORT_ENFORCE(seq_len <= max_seq_len && seq_len >= 0LL, 
                "Seq_lengths[", batch, "] = ", seq_len, " is out of range [0,", max_seq_len, "]");

    int64_t batch_start = batch * max_seq_len * input_size;
    for (int64_t seq = 0; seq < seq_len; seq++) {
      gsl::span<const T> src = inputs.subspan(batch_start + seq * input_size, input_size);
      gsl::span<T> dest = inputs_reverse.subspan(batch_start + (seq_len - seq - 1) * input_size, input_size);
      gsl::copy(src, dest);
    }

    if (seq_len < max_seq_len) {
      gsl::span<const T> src = inputs.subspan(batch_start + seq_len * input_size, (max_seq_len - seq_len) * input_size);
      gsl::span<T> dest = inputs_reverse.subspan(batch_start + seq_len * input_size, (max_seq_len - seq_len) * input_size);
      gsl::copy(src, dest);
    }
  }
  return Status::OK();
}

template <typename T, typename TIndex>
static Status ComputeReverseSequence(OpKernelContext* ctx, int64_t p_seq_axis, int64_t p_batch_axis) {
  if (! ((p_batch_axis == 0 && p_seq_axis == 1) || (p_batch_axis == 1 && p_seq_axis == 0))) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,  
            "Current only support (batch_axis, seq_axis): (1,0) or (0,1), yet got:(",
            p_batch_axis, ", ", p_seq_axis);
  }

  auto& input = *ctx->Input<Tensor>(0);
  auto& seq_lengths = *ctx->Input<Tensor>(1);
  TensorShape shape = input.Shape();
  if (shape.NumDimensions() <= (size_t)std::max(p_seq_axis, p_batch_axis)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
            "Input shape dims_size:", shape.NumDimensions(), 
            " less than seq_axis or batch_axis", std::max(p_seq_axis, p_batch_axis));
  }

  TensorShape seq_len_shape = seq_lengths.Shape();
  if (seq_len_shape.NumDimensions() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, 
            "seq_lengths must be 1-D tensor, not ", seq_len_shape.NumDimensions()); 
  }

  if (seq_len_shape[0] != shape[p_batch_axis]) {
    return  ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, 
            "seq_lengths dim size not equal to batch size of input:",
            seq_len_shape.NumDimensions(), " vs ", shape[p_batch_axis]); 
  }

  auto* y = ctx->Output(0, shape);

  gsl::span<const T> input_span = input.DataAsSpan<T>();
  gsl::span<const TIndex> seq_lengths_span = seq_lengths.DataAsSpan<TIndex>();
  gsl::span<T> result_span = y->MutableDataAsSpan<T>();

  int64_t max_seq_len = shape[p_seq_axis];
  int64_t batch_size = shape[p_batch_axis];
  int64_t input_size = shape.SizeFromDimension(2);
  return (p_batch_axis == 0) ?
    ReverseSequenceBatchSequence<T, TIndex>(input_span, result_span, seq_lengths_span, max_seq_len, batch_size, input_size) :
    ReverseSequenceSequenceBatch<T, TIndex>(input_span, result_span, seq_lengths_span, max_seq_len, batch_size, input_size);
}

Status ReverseSequence::Compute(OpKernelContext* ctx) const {
  auto& input_tensor = *ctx->Input<Tensor>(0);
  auto data_type = input_tensor.DataType();
  auto& seq_lengths_tensor = *ctx->Input<Tensor>(1);
  auto index_type = seq_lengths_tensor.DataType();
  if (index_type != DataTypeImpl::GetType<int32_t>() && index_type != DataTypeImpl::GetType<int64_t>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported index type:", index_type);
  }
  
  if (data_type == DataTypeImpl::GetType<float>()) {
      return (index_type == DataTypeImpl::GetType<int32_t>()) ?
                ComputeReverseSequence<float, int32_t>(ctx, seq_axis_, batch_axis_) :
                ComputeReverseSequence<float, int64_t>(ctx, seq_axis_, batch_axis_);     
  }
  else if (data_type == DataTypeImpl::GetType<int32_t>()) {
      return (index_type == DataTypeImpl::GetType<int32_t>()) ?
                ComputeReverseSequence<int32_t, int32_t>(ctx, seq_axis_, batch_axis_) :
                ComputeReverseSequence<int32_t, int64_t>(ctx, seq_axis_, batch_axis_);     
  }
  else if (data_type == DataTypeImpl::GetType<int16_t>()) {
      return (index_type == DataTypeImpl::GetType<int32_t>()) ?
                ComputeReverseSequence<int16_t, int32_t>(ctx, seq_axis_, batch_axis_) :
                ComputeReverseSequence<int16_t, int64_t>(ctx, seq_axis_, batch_axis_);     
  }
  else if (data_type == DataTypeImpl::GetType<int64_t>()) {
      return (index_type == DataTypeImpl::GetType<int32_t>()) ?
                ComputeReverseSequence<int64_t, int32_t>(ctx, seq_axis_, batch_axis_) :
                ComputeReverseSequence<int64_t, int64_t>(ctx, seq_axis_, batch_axis_);     
  }
  else if (data_type == DataTypeImpl::GetType<double>()) {
      return (index_type == DataTypeImpl::GetType<int32_t>()) ?
                ComputeReverseSequence<double, int32_t>(ctx, seq_axis_, batch_axis_) :
                ComputeReverseSequence<double, int64_t>(ctx, seq_axis_, batch_axis_);     
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,  
            "Unsupportted tensor data type:", data_type);
}


/* Range operator */
ONNX_OPERATOR_KERNEL_EX(
    ReverseSequence,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
      .TypeConstraint("T", {
        DataTypeImpl::GetTensorType<float>(), 
        DataTypeImpl::GetTensorType<double>(),
        DataTypeImpl::GetTensorType<int16_t>(), 
        DataTypeImpl::GetTensorType<int32_t>(), 
        DataTypeImpl::GetTensorType<int64_t>()})
      .TypeConstraint("TIndex", {
        DataTypeImpl::GetTensorType<int32_t>(), 
        DataTypeImpl::GetTensorType<int64_t>()}),
    ReverseSequence);


}  // namespace contrib
}  // namespace onnxruntime
