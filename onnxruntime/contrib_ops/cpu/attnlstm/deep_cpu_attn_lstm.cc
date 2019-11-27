// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "deep_cpu_attn_lstm.h"
#include "activation_info.h"
#include "bahdanau_attention.h"
#include "uni_dir_attn_lstm.h"

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/platform/threadpool.h"
#include "core/framework/allocator.h"
#include "core/framework/op_kernel_context_internal.h"

namespace onnxruntime {
namespace contrib {

using ::onnxruntime::contrib::rnn::detail::UniDirectionalAttnLstm;
using ::onnxruntime::rnn::detail::Allocate;

extern template class BahdanauAttention<float>;

/* AttnLSTM operator */
ONNX_OPERATOR_KERNEL_EX(
    AttnLSTM,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    DeepCpuAttnLstmOp);

Status
DeepCpuAttnLstmOp::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]

  Status status;
  // auto& logger = context->Logger();

  switch (X.GetElementType()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      status = ComputeImpl<float>(*context);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      /* Need to update all the helpers to support double...
    status = ComputeImpl<double>(*context); */
      ORT_NOT_IMPLEMENTED("LSTM operator does not support double yet");
      break;
    default:
      ORT_THROW("Invalid data type for LSTM operator of ", X.DataType());
      break;
  }

  return status;
}

// #define DUMP_MATRIXES to provide lots of diagnostic output
#if defined(DUMP_MATRIXES)
#define DumpMatrix(...) ::onnxruntime::rnn::detail::DumpMatrixImpl(__VA_ARGS__)
#else
#define DumpMatrix(...) ((void)0)
#endif

template <typename T>
static gsl::span<const T> FirstHalfSpan(const gsl::span<const T>& dspan) {
  auto sz = dspan.size() / 2;
  return dspan.subspan(0, sz);
}

template <typename T>
static gsl::span<const T> SecondHalfSpan(const gsl::span<const T>& dspan) {
  auto sz = dspan.size() / 2;
  return dspan.subspan(sz);
}

template <typename T>
Status DeepCpuAttnLstmOp::ComputeImpl(OpKernelContext& context) const {
  auto& logger = context.Logger();

  // original lstm processing
  const Tensor& X = *context.Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size], input will concat with attention of previous state
  const Tensor& W = *context.Input<Tensor>(1);  // weights. [num_directions, 4*hidden_size, input_size + attention_size],
  const Tensor& R = *context.Input<Tensor>(2);  // recurrence weights. [num_directions, 4*hidden_size, hidden_size]

  // optional
  const Tensor* B = context.Input<Tensor>(3);              // bias. [num_directions, 8*hidden_size]
  const Tensor* sequence_lens = context.Input<Tensor>(4);  // [batch_size]
  const Tensor* initial_h = context.Input<Tensor>(5);      // initial hidden. [num_directions, batch_size, hidden_size]
  const Tensor* initial_c = context.Input<Tensor>(6);      // initial cell. [num_directions, batch_size, hidden_size]
  const Tensor* P = context.Input<Tensor>(7);              // peephole weights. [num_directions, 3*hidden_size]

  auto& X_shape = X.Shape();

  int seq_length = gsl::narrow<int>(X_shape[0]);
  int batch_size = gsl::narrow<int>(X_shape[1]);
  int input_size = gsl::narrow<int>(X_shape[2]);

  // Processing attention wrapper
  const int first_attn_input = 8;
  const Tensor& am_query_layer_weights = *context.Input<Tensor>(first_attn_input + 0);   // [num_directions, query_depth(hidden_size of lstm), am_attn_size]
  const Tensor& am_memory_layer_weights = *context.Input<Tensor>(first_attn_input + 1);  // [num_directions, memory_depth, am_attn_size]
  const Tensor& am_v_weights = *context.Input<Tensor>(first_attn_input + 2);             // [num_directions, am_attn_size]
  const Tensor& attn_memory = *context.Input<Tensor>(first_attn_input + 3);              // [batch_size, max_memory_step, memory_depth_]
  const Tensor* attn_memory_seq_lens = context.Input<Tensor>(first_attn_input + 4);      // [batch_size], int value
  const Tensor* attn_layer_weights = context.Input<Tensor>(first_attn_input + 5);        // [num_directions, memory_depth+cell_hidden_size, aw_attn_size]

  Status status = ValidateInputs(
      X, W, R, B, sequence_lens, initial_h, initial_c, P, batch_size,
      am_query_layer_weights, am_memory_layer_weights, am_v_weights, attn_memory, attn_memory_seq_lens, attn_layer_weights);
  ORT_RETURN_IF_ERROR(status);

  const int max_memory_step = gsl::narrow<int>(attn_memory.Shape()[1]);
  const int memory_depth = gsl::narrow<int>(am_memory_layer_weights.Shape()[1]);
  const int am_attn_size = gsl::narrow<int>(am_memory_layer_weights.Shape()[2]);
  const int query_depth = gsl::narrow<int>(am_query_layer_weights.Shape()[1]);  // it is equal to hidden_size
  const bool has_attention_layer = attn_layer_weights != nullptr;
  const int attn_layer_depth = has_attention_layer ? gsl::narrow<int>(attn_layer_weights->Shape()[2]) : 0;
  const int attention_size = has_attention_layer ? attn_layer_depth : memory_depth;

  const gsl::span<const T> attn_layer_weights_span = (has_attention_layer) ? attn_layer_weights->DataAsSpan<T>() : gsl::span<const T>();
  const gsl::span<const int> memory_seq_lens_span = (attn_memory_seq_lens != nullptr) ? attn_memory_seq_lens->DataAsSpan<int>() : gsl::span<const int>();

  // LSTM outputs are optional but must be in the same order
  std::vector<int64_t> Y_dims{seq_length, num_directions_, batch_size, hidden_size_};
  Tensor* Y = context.Output(/*index*/ 0, Y_dims);

  std::vector<int64_t> Y_h_dims{num_directions_, batch_size, hidden_size_};
  Tensor* Y_h = context.Output(/*index*/ 1, Y_h_dims);

  std::vector<int64_t> Y_c_dims{num_directions_, batch_size, hidden_size_};
  Tensor* Y_c = context.Output(/*index*/ 2, Y_c_dims);

  AllocatorPtr alloc;
  status = context.GetTempSpaceAllocator(&alloc);
  ORT_RETURN_IF_ERROR(status);

  gsl::span<const T> input_weights = W.DataAsSpan<T>();
  gsl::span<const T> recurrent_weights = R.DataAsSpan<T>();
  gsl::span<const T> bias = B != nullptr ? B->DataAsSpan<T>() : gsl::span<const T>();
  gsl::span<const T> peephole_weights = P != nullptr ? P->DataAsSpan<T>() : gsl::span<const T>();

  // spans for first direction
  const size_t input_weights_size_per_direction = 4 * hidden_size_ * (input_size + attention_size);
  const size_t hidden_weights_size_per_direction = 4 * hidden_size_ * hidden_size_;
  const size_t bias_size_per_direction = 8 * hidden_size_;
  const size_t peephole_weights_size_per_direction = 3 * hidden_size_;

  gsl::span<const T> input_weights_1 = input_weights.subspan(0, input_weights_size_per_direction);
  gsl::span<const T> recurrent_weights_1 = recurrent_weights.subspan(0, hidden_weights_size_per_direction);
  gsl::span<const T> bias_1 = bias.empty() ? bias : bias.subspan(0, bias_size_per_direction);
  gsl::span<const T> peephole_weights_1 =
      peephole_weights.empty() ? peephole_weights
                               : peephole_weights.subspan(0, peephole_weights_size_per_direction);

  gsl::span<const T> input = X.DataAsSpan<T>();
  gsl::span<const int> sequence_lens_span = sequence_lens != nullptr ? sequence_lens->DataAsSpan<int>()
                                                                     : gsl::span<const int>();

  const size_t initial_hidden_size_per_direction = batch_size * hidden_size_;
  gsl::span<const T> initial_hidden = initial_h != nullptr ? initial_h->DataAsSpan<T>() : gsl::span<const T>();
  gsl::span<const T> initial_hidden_1 =
      initial_hidden.empty() ? initial_hidden
                             : initial_hidden.subspan(0, initial_hidden_size_per_direction);

  const size_t initial_cell_size_per_direction = batch_size * hidden_size_;
  gsl::span<const T> initial_cell = initial_c != nullptr ? initial_c->DataAsSpan<T>() : gsl::span<const T>();
  gsl::span<const T> initial_cell_1 =
      initial_cell.empty() ? initial_cell
                           : initial_cell.subspan(0, initial_cell_size_per_direction);

  // output shape is [seq_length, num_directions, batch_size, hidden_size]
  // so it's not a case of all the output for one direction being first.
  // due to that we can only easily check that the end of the output for each direction is valid.
  const size_t output_size = Y != nullptr ? static_cast<size_t>(Y->Shape().Size()) : 0;
  const size_t per_direction_offset = batch_size * hidden_size_;
  gsl::span<T> output = Y != nullptr ? Y->MutableDataAsSpan<T>() : gsl::span<T>();
  gsl::span<T> output_1 =
      output.empty() ? output
                     : output.subspan(0, output_size - (num_directions_ - 1) * per_direction_offset);

  // UniDirectionalAttnLstm needs somewhere to write output, so even if we aren't returning Y_h and Y_c
  // we provide an appropriately sized buffer for that purpose.
  const size_t hidden_output_size_per_direction = batch_size * hidden_size_;
  IAllocatorUniquePtr<T> local_hidden_output;
  gsl::span<T> hidden_output =
      Y_h ? Y_h->MutableDataAsSpan<T>()
          : Allocate(alloc, hidden_output_size_per_direction * num_directions_, local_hidden_output);

  gsl::span<T> hidden_output_1 = hidden_output.subspan(0, hidden_output_size_per_direction);

  const size_t last_cell_size_per_direction = batch_size * hidden_size_;
  IAllocatorUniquePtr<T> local_last_cell;
  gsl::span<T> last_cell =
      Y_c ? Y_c->MutableDataAsSpan<T>()
          : Allocate(alloc, last_cell_size_per_direction * num_directions_, local_last_cell);

  gsl::span<T> last_cell_1 = last_cell.subspan(0, last_cell_size_per_direction);

  if (!output.empty() && !sequence_lens_span.empty()) {
    // clear tailing outputs
    int32_t max_seq_this_batch = *std::max_element(sequence_lens_span.cbegin(), sequence_lens_span.cend());
    if (max_seq_this_batch >= 0 && max_seq_this_batch < seq_length) {
      auto start = max_seq_this_batch * hidden_output_size_per_direction * num_directions_;
      std::fill(output.begin() + start, output.end(), T{});
    }
  }

  concurrency::ThreadPool* thread_pool = context.GetOperatorThreadPool();

  if (direction_ == Direction::kBidirectional) {
    // spans for second direction
    gsl::span<const T> input_weights_2 = input_weights.subspan(input_weights_size_per_direction,
                                                               input_weights_size_per_direction);
    gsl::span<const T> hidden_weights_2 = recurrent_weights.subspan(hidden_weights_size_per_direction,
                                                                    hidden_weights_size_per_direction);
    gsl::span<const T> bias_2 = bias.empty() ? bias : bias.subspan(bias_size_per_direction, bias_size_per_direction);
    gsl::span<const T> peephole_weights_2 =
        peephole_weights.empty() ? peephole_weights
                                 : peephole_weights.subspan(peephole_weights_size_per_direction,
                                                            peephole_weights_size_per_direction);

    gsl::span<const T> initial_hidden_2 =
        initial_hidden.empty() ? initial_hidden
                               : initial_hidden.subspan(initial_hidden_size_per_direction,
                                                        initial_hidden_size_per_direction);
    gsl::span<const T> initial_cell_2 =
        initial_cell.empty() ? initial_cell
                             : initial_cell.subspan(initial_cell_size_per_direction,
                                                    initial_cell_size_per_direction);
    gsl::span<T> output_2 =
        output.empty() ? output : output.subspan(per_direction_offset, output_size - per_direction_offset);

    gsl::span<T> hidden_output_2 = hidden_output.subspan(hidden_output_size_per_direction,
                                                         hidden_output_size_per_direction);
    gsl::span<T> last_cell_2 = last_cell.subspan(last_cell_size_per_direction,
                                                 last_cell_size_per_direction);

    BahdanauAttention<T> fam(
        alloc,
        logger,
        batch_size,
        max_memory_step,
        memory_depth,
        query_depth,
        am_attn_size,
        false, thread_pool);

    fam.SetWeights(
        FirstHalfSpan(am_v_weights.DataAsSpan<T>()),
        FirstHalfSpan(am_query_layer_weights.DataAsSpan<T>()),
        FirstHalfSpan(am_memory_layer_weights.DataAsSpan<T>()));
    fam.PrepareMemory(attn_memory.DataAsSpan<T>(), memory_seq_lens_span);

    AttentionWrapper<T> faw(
        alloc,
        logger,
        batch_size,
        memory_depth,
        attn_layer_depth,
        hidden_size_,
        has_attention_layer,
        fam, thread_pool);
    faw.SetWeights(FirstHalfSpan(attn_layer_weights_span));

    UniDirectionalAttnLstm<T> fw(
        alloc, logger,
        seq_length, batch_size, input_size,
        hidden_size_, Direction::kForward, input_forget_, faw,
        bias_1, peephole_weights_1, initial_hidden_1, initial_cell_1,
        activation_funcs_.Entries()[0],
        activation_funcs_.Entries()[1],
        activation_funcs_.Entries()[2],
        clip_, thread_pool);

    BahdanauAttention<T> bam(
        alloc,
        logger,
        batch_size,
        max_memory_step,
        memory_depth,
        query_depth,
        am_attn_size,
        false, thread_pool);
    bam.SetWeights(
        SecondHalfSpan(am_v_weights.DataAsSpan<T>()),
        SecondHalfSpan(am_query_layer_weights.DataAsSpan<T>()),
        SecondHalfSpan(am_memory_layer_weights.DataAsSpan<T>()));
    bam.PrepareMemory(attn_memory.DataAsSpan<T>(), memory_seq_lens_span);

    AttentionWrapper<T> baw(
        alloc,
        logger,
        batch_size,
        memory_depth,
        attn_layer_depth,
        hidden_size_,
        has_attention_layer,
        bam, thread_pool);
    baw.SetWeights(SecondHalfSpan(attn_layer_weights_span));

    UniDirectionalAttnLstm<T> bw(
        alloc, logger,
        seq_length, batch_size, input_size,
        hidden_size_, Direction::kReverse, input_forget_, baw,
        bias_2, peephole_weights_2, initial_hidden_2, initial_cell_2,
        activation_funcs_.Entries()[3],
        activation_funcs_.Entries()[4],
        activation_funcs_.Entries()[5],
        clip_, thread_pool);

    fw.Compute(input, sequence_lens_span, num_directions_, input_weights_1, recurrent_weights_1, output_1, hidden_output_1, last_cell_1);
    bw.Compute(input, sequence_lens_span, num_directions_, input_weights_2, hidden_weights_2, output_2, hidden_output_2, last_cell_2);

  } else {
    BahdanauAttention<T> fam(
        alloc,
        logger,
        batch_size,
        max_memory_step,
        memory_depth,
        query_depth,
        am_attn_size,
        false, thread_pool);

    fam.SetWeights(
        am_v_weights.DataAsSpan<T>(),
        am_query_layer_weights.DataAsSpan<T>(),
        am_memory_layer_weights.DataAsSpan<T>());
    fam.PrepareMemory(attn_memory.DataAsSpan<T>(), memory_seq_lens_span);

    AttentionWrapper<T> faw(
        alloc,
        logger,
        batch_size,
        memory_depth,
        attn_layer_depth,
        hidden_size_,
        has_attention_layer,
        fam, thread_pool);

    faw.SetWeights(attn_layer_weights_span);

    UniDirectionalAttnLstm<T> fw(
        alloc, logger,
        seq_length, batch_size, input_size,
        hidden_size_, direction_, input_forget_, faw,
        bias_1, peephole_weights_1, initial_hidden_1, initial_cell_1,
        activation_funcs_.Entries()[0],
        activation_funcs_.Entries()[1],
        activation_funcs_.Entries()[2],
        clip_, thread_pool);

    fw.Compute(input, sequence_lens_span, num_directions_, input_weights_1, recurrent_weights_1, output_1, hidden_output_1, last_cell_1);
  }

  if (!output.empty()) {
    DumpMatrix("Y", output.data(), seq_length * num_directions_ * batch_size, hidden_size_);
  }

  // these always get written to regardless of whether we're returning them as optional output or not
  DumpMatrix("Y_h", hidden_output.data(), num_directions_ * batch_size, hidden_size_);
  DumpMatrix("Y_c", last_cell.data(), num_directions_ * batch_size, hidden_size_);

  return Status::OK();
}

static Status ValidateRnnInputsWithExtraInputFromState(
    const Tensor& X,
    const Tensor& W,
    const Tensor& R,
    const Tensor* B,
    int WRB_dim_1_multipler,
    const Tensor* sequence_lens,
    const Tensor* initial_h,
    int64_t num_directions,
    int64_t hidden_size,
    int64_t extra_input_size) {
  auto& X_shape = X.Shape();
  auto& W_shape = W.Shape();
  auto& R_shape = R.Shape();

  int64_t seq_length = X_shape[0];
  int64_t batch_size = X_shape[1];
  int64_t input_size = X_shape[2] + extra_input_size;

  if (X_shape.NumDimensions() != 3)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input X must have 3 dimensions only. Actual:", X_shape);

  if (W_shape.NumDimensions() != 3 ||
      W_shape[0] != num_directions ||
      W_shape[1] != hidden_size * WRB_dim_1_multipler ||
      W_shape[2] != input_size)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input W must have shape {",
                           num_directions, ",", WRB_dim_1_multipler, "*", hidden_size, ",",
                           input_size, "}. Actual:", W_shape);

  if (R_shape.NumDimensions() != 3 ||
      R_shape[0] != num_directions ||
      R_shape[1] != hidden_size * WRB_dim_1_multipler ||
      R_shape[2] != hidden_size)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input R must have shape {",
                           num_directions, ",", WRB_dim_1_multipler, "*", hidden_size, ",",
                           hidden_size, "}. Actual:", R_shape);

  if (B != nullptr) {
    auto& B_shape = B->Shape();
    if (B_shape.NumDimensions() != 2 ||
        B_shape[0] != num_directions ||
        B_shape[1] != 2 * WRB_dim_1_multipler * hidden_size)
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input B must have shape {",
                             num_directions, ",", 2 * WRB_dim_1_multipler, "*", hidden_size, "}. Actual:", B_shape);
  }

  if (sequence_lens != nullptr) {
    auto& sequence_lens_shape = sequence_lens->Shape();
    if (sequence_lens_shape.NumDimensions() != 1 ||
        sequence_lens_shape[0] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input sequence_lens must have shape {",
                             batch_size, "}. Actual:", sequence_lens_shape);
    }

    auto sequence_len_entries = sequence_lens->DataAsSpan<int>();
    if (std::any_of(sequence_len_entries.cbegin(),
                    sequence_len_entries.cend(),
                    [seq_length](int len) { return len <= 0 || len > seq_length; })) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Invalid value/s in sequence_lens. All values must be > 0 and < seq_length. seq_length=", seq_length);
    }
  }

  if (initial_h != nullptr) {
    auto& initial_h_shape = initial_h->Shape();

    if (initial_h_shape.NumDimensions() != 3 ||
        initial_h_shape[0] != num_directions ||
        initial_h_shape[1] != batch_size ||
        initial_h_shape[2] != hidden_size)

      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input initial_h must have shape {",
                             num_directions, ",", batch_size, ",", hidden_size, "}. Actual:", initial_h_shape);
  }

  return Status::OK();
}  // namespace detail

Status DeepCpuAttnLstmOp::ValidateInputs(
    const Tensor& X, const Tensor& W, const Tensor& R, const Tensor* B,
    const Tensor* sequence_lens, const Tensor* initial_h, const Tensor* initial_c,
    const Tensor* P, int batch_size,
    const Tensor& am_query_layer_weights, const Tensor& am_memory_layer_weights, const Tensor& am_v_weights,
    const Tensor& attn_memory, const Tensor* attn_memory_seq_lens, const Tensor* attn_layer_weights) const {
  // Check memory of [batch_size, max_memory_step, memory_depth_], its sequence length of [batch_size]
  auto memory_shape = attn_memory.Shape();
  if (memory_shape.NumDimensions() != 3 || memory_shape[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Attention mechanism memory shape error! Expected: {", batch_size,
                           "}, actural: ", memory_shape);
  }
  const int max_memory_step = gsl::narrow<int>(memory_shape[1]);
  const int memory_depth = gsl::narrow<int>(memory_shape[2]);
  if (attn_memory_seq_lens != nullptr) {
    auto memory_seq_lens_shape = attn_memory_seq_lens->Shape();
    if (memory_seq_lens_shape.NumDimensions() != 1 || memory_seq_lens_shape[0] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Attention mechanism memory sequence lengths must have shape {", batch_size,
                             "}, actural: ", memory_seq_lens_shape);
    }
    const gsl::span<const int> mem_seq_lens_span = attn_memory_seq_lens->DataAsSpan<int>();
    auto item_not_in_range = std::find_if(
        mem_seq_lens_span.cbegin(), mem_seq_lens_span.cend(),
        [max_memory_step](int len) { return len <= 0 || len > max_memory_step; });
    if (item_not_in_range != mem_seq_lens_span.cend()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Attention mechanism memory sequence lengths value must in (0, ",
                             max_memory_step, "], while ", *item_not_in_range, " found!");
    }
  }

  // Check memory layer weights of [num_directions, memory_depth, am_attn_size]
  auto memory_layer_shape = am_memory_layer_weights.Shape();
  if (memory_layer_shape.NumDimensions() != 3 ||
      memory_layer_shape[0] != num_directions_ ||
      memory_layer_shape[1] != memory_depth) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Attention memory layer weight shape error! Expected:{",
                           num_directions_, ",", memory_depth, ", am_attn_size}, Got:", memory_layer_shape);
  }
  const int am_attn_size = gsl::narrow<int>(memory_layer_shape[2]);

  // check query layer weights of [num_directions, query_depth(hidden_size of lstm), am_attn_size]
  auto query_layer_shape = am_query_layer_weights.Shape();
  if (query_layer_shape.NumDimensions() != 3 ||
      query_layer_shape[0] != num_directions_ ||
      query_layer_shape[1] != hidden_size_ ||
      query_layer_shape[2] != am_attn_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Attention query layer weight shape error! Expected:{",
                           num_directions_, ", ", hidden_size_, ", ", am_attn_size, "}, Got: ", query_layer_shape);
  }

  // check attention v for [num_directions, am_attn_size]
  auto v_shape = am_v_weights.Shape();
  if (v_shape.NumDimensions() != 2 ||
      v_shape[0] != num_directions_ ||
      v_shape[1] != am_attn_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Attention v weight shape error! Expected:{", num_directions_, ", ", am_attn_size,
                           "}. Got: ", v_shape);
  }

  // Check attention layer weights for [num_directions, memory_depth+cell_hidden_size, aw_attn_size]
  const bool has_attention_layer = attn_layer_weights != nullptr;
  int aw_attn_size = memory_depth;
  if (has_attention_layer) {
    auto attn_layer_shape = attn_layer_weights->Shape();
    if (attn_layer_shape.NumDimensions() != 3 ||
        attn_layer_shape[0] != num_directions_ ||
        attn_layer_shape[1] != memory_depth + hidden_size_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Attention layer weight shape error! Expected: {", num_directions_, ", ",
                             memory_depth + hidden_size_, ", aw_attn_size}. Got:", attn_layer_shape);
    }
    aw_attn_size = gsl::narrow<int>(attn_layer_shape[2]);
  }

  auto status = ValidateRnnInputsWithExtraInputFromState(
      X, W, R, B, 4, sequence_lens, initial_h, num_directions_, hidden_size_, aw_attn_size);
  ORT_RETURN_IF_ERROR(status);

  if (initial_c != nullptr) {
    auto& initial_c_shape = initial_c->Shape();

    if (initial_c_shape.NumDimensions() != 3 ||
        initial_c_shape[0] != num_directions_ ||
        initial_c_shape[1] != batch_size ||
        initial_c_shape[2] != hidden_size_)

      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input initial_c must have shape {",
                             num_directions_, ",", batch_size, ",", hidden_size_, "}. Actual:", initial_c_shape);
  }

  if (P != nullptr) {
    auto& p_shape = P->Shape();

    if (p_shape.NumDimensions() != 2 ||
        p_shape[0] != num_directions_ ||
        p_shape[1] != 3 * hidden_size_)

      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input P must have shape {",
                             num_directions_, ",", 3 * hidden_size_, "}. Actual:", p_shape);
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
