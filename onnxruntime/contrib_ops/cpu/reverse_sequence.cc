// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse_sequence.h"
#include "onnx/defs/schema.h"

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "gsl/gsl_algorithm"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "core/framework/utils.h"
#include "core/framework/tensor.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(ReverseSequence,
                        kMSDomain,
                        1,
                        kCpuExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
                            .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
                        ReverseSequenceOp);

template <typename T>
static Status ReverseSequence(OpKernelContext& context, bool time_major);

Status ReverseSequenceOp::Compute(OpKernelContext* context) const {
  Status status = Status::OK();
  const auto data_type = context->Input<Tensor>(0)->DataType();
  DispatchOnTensorTypeWithReturn(data_type, status, ReverseSequence, *context, time_major_);
  return status;
}

static int64_t TimeMajorInputOffset(const int64_t max_seq_len,
                                    const int64_t batch_size,
                                    const int64_t input_size,
                                    const int64_t batch_num,
                                    const int64_t seq_num) {
  return seq_num * batch_size * input_size + batch_num * input_size;
}

static int64_t BatchMajorInputOffset(const int64_t max_seq_len,
                                     const int64_t batch_size,
                                     const int64_t input_size,
                                     const int64_t batch_num,
                                     const int64_t seq_num) {
  return batch_num * max_seq_len * input_size + seq_num * input_size;
}

static int64_t TimeMajorOutputOffset(const int64_t max_seq_len,
                                     const int64_t batch_size,
                                     const int64_t input_size,
                                     const int64_t batch_num,
                                     const int64_t seq_num,
                                     const int64_t seq_len) {
  return (seq_len - seq_num - 1) * batch_size * input_size + batch_num * input_size;
}

static int64_t BatchMajorOutputOffset(const int64_t max_seq_len,
                                      const int64_t batch_size,
                                      const int64_t input_size,
                                      const int64_t batch_num,
                                      const int64_t seq_num,
                                      const int64_t seq_len) {
  return batch_num * max_seq_len * input_size + (seq_len - seq_num - 1) * input_size;
}

template <typename T>
void ReverseSequenceImpl(gsl::span<const T> inputs,
                         gsl::span<T> inputs_reverse,
                         gsl::span<const int> sequence_lengths,
                         const int64_t max_seq_len,
                         const int64_t batch_size,
                         const int64_t input_size,
                         bool time_major) {
  auto input_offset = time_major ? TimeMajorInputOffset : BatchMajorInputOffset;

  auto reversed_output_offset = time_major ? TimeMajorOutputOffset : BatchMajorOutputOffset;

  for (int i = 0; i < batch_size; i++) {
    int seq_len = sequence_lengths[i];

    if (seq_len == 0)
      continue;

#ifdef USE_OPENMP
// Parallel execute the loop.
#pragma omp parallel for
#endif
    for (int j = 0; j < seq_len; j++) {
      gsl::span<const T> src = inputs.subspan(input_offset(max_seq_len, batch_size, input_size, i, j), input_size);
      gsl::span<T> dest = inputs_reverse.subspan(
          reversed_output_offset(max_seq_len, batch_size, input_size, i, j, seq_len), input_size);

      // Use gsl::copy instead of std::copy() to allow compiler to optimize the code
      gsl::copy(src, dest);
    }

#ifdef USE_OPENMP
// Parallel execute the loop.
#pragma omp parallel for
#endif
    for (int j = seq_len; j < max_seq_len; j++) {
      const auto offset = input_offset(max_seq_len, batch_size, input_size, i, j);
      gsl::span<const T> src = inputs.subspan(offset, input_size);
      gsl::span<T> dest = inputs_reverse.subspan(offset, input_size);

      // Use gsl::copy instead of std::copy() to allow compiler to optimize the code
      gsl::copy(src, dest);
    }
  }
}

template <typename T>
static Status ReverseSequence(OpKernelContext& context, bool time_major) {
  const auto& X = *context.Input<Tensor>(0);

  const auto& dims = X.Shape();
  const auto batch_size = time_major ? dims[1] : dims[0];
  const auto max_seq_len = time_major ? dims[0] : dims[1];
  const auto input_size = dims.SizeFromDimension(2);

  const auto& seq_lengths = *context.Input<Tensor>(1);
  const auto& seq_len_shape = seq_lengths.Shape();

  if (seq_len_shape.NumDimensions() != 1 || seq_len_shape[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "sequence_lens shape must be {batch_size}. Got:",
                           seq_len_shape, ". batch_size=", batch_size);
  }

  auto& Y = *context.Output(0, dims);

  ReverseSequenceImpl(X.DataAsSpan<T>(), Y.MutableDataAsSpan<T>(), seq_lengths.DataAsSpan<int>(),
                      max_seq_len, batch_size, input_size, time_major);

  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
