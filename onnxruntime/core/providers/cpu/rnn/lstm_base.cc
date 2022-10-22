// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lstm_base.h"
#include "uni_directional_lstm.h"
#include "core/common/narrow.h"
//TODO: fix the warnings
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26451)
#endif
namespace onnxruntime {

using namespace rnn::detail;

// #define DUMP_MATRIXES to provide lots of diagnostic output
#if defined(DUMP_MATRIXES)
#define DumpMatrix(...) ::onnxruntime::rnn::detail::DumpMatrixImpl(__VA_ARGS__)
#else
#define DumpMatrix(...) ((void)0)
#endif

template <typename InputT, typename WeightT>
Status LSTMBase::ComputeImpl(OpKernelContext& context,
                             const rnn::detail::GemmWeights<WeightT>& W_1,
                             const rnn::detail::GemmWeights<WeightT>& W_2,
                             const rnn::detail::GemmWeights<WeightT>& R_1,
                             const rnn::detail::GemmWeights<WeightT>& R_2) const {
  concurrency::ThreadPool* thread_pool = context.GetOperatorThreadPool();

  auto& logger = context.Logger();

  const Tensor& X = *context.Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]

  // optional
  const Tensor* B = context.Input<Tensor>(3);              // bias. [num_directions, 8*hidden_size]
  const Tensor* sequence_lens = context.Input<Tensor>(4);  // [batch_size]
  const Tensor* initial_h = context.Input<Tensor>(5);      // initial hidden. [num_directions, batch_size, hidden_size]
  const Tensor* initial_c = context.Input<Tensor>(6);      // initial cell. [num_directions, batch_size, hidden_size]
  const Tensor* P = context.Input<Tensor>(7);              // peephole weights. [num_directions, 3*hidden_size]

  const auto& X_shape = X.Shape();

  int seq_length = narrow<int>(X_shape[0]);
  int batch_size = narrow<int>(X_shape[1]);
  int input_size = narrow<int>(X_shape[2]);

  Status status = ValidateInputs(X, B, sequence_lens, initial_h, initial_c, P);
  ORT_RETURN_IF_ERROR(status);

  // LSTM outputs are optional but must be in the same order
  TensorShape Y_dims{seq_length, num_directions_, batch_size, hidden_size_};
  Tensor* Y = context.Output(/*index*/ 0, Y_dims);

  TensorShape Y_h_dims{num_directions_, batch_size, hidden_size_};
  Tensor* Y_h = context.Output(/*index*/ 1, Y_h_dims);

  TensorShape Y_c_dims{num_directions_, batch_size, hidden_size_};
  Tensor* Y_c = context.Output(/*index*/ 2, Y_c_dims);

  // Reset output and return if max sequence length is 0
  if (sequence_lens != nullptr) {
    int32_t max_sequence_length = *std::max_element(sequence_lens->Data<int32_t>(),
                                                    sequence_lens->Data<int32_t>() + sequence_lens->Shape().Size());
    if (max_sequence_length == 0) {
      if (Y != nullptr)
        std::fill_n(Y->MutableData<InputT>(), Y_dims.Size(), InputT{});
      if (Y_h != nullptr)
        std::fill_n(Y_h->MutableData<InputT>(), Y_h_dims.Size(), InputT{});
      if (Y_c != nullptr)
        std::fill_n(Y_c->MutableData<InputT>(), Y_c_dims.Size(), InputT{});
      return Status::OK();
    }
  }

  AllocatorPtr alloc;
  status = context.GetTempSpaceAllocator(&alloc);
  ORT_RETURN_IF_ERROR(status);

  gsl::span<const InputT> bias = B != nullptr ? B->DataAsSpan<InputT>() : gsl::span<const InputT>();
  gsl::span<const InputT> peephole_weights = P != nullptr ? P->DataAsSpan<InputT>() : gsl::span<const InputT>();

  // spans for first direction
  const size_t bias_size_per_direction = 8 * hidden_size_;
  const size_t peephole_weights_size_per_direction = 3 * hidden_size_;

  gsl::span<const InputT> bias_1 = bias.empty() ? bias : bias.subspan(0, bias_size_per_direction);
  gsl::span<const InputT> peephole_weights_1 =
      peephole_weights.empty() ? peephole_weights : peephole_weights.subspan(0, peephole_weights_size_per_direction);

  gsl::span<const InputT> input = X.DataAsSpan<InputT>();
  gsl::span<const int> sequence_lens_span =
      sequence_lens != nullptr ? sequence_lens->DataAsSpan<int>() : gsl::span<const int>();

  const size_t initial_hidden_size_per_direction = batch_size * hidden_size_;
  gsl::span<const InputT> initial_hidden = initial_h != nullptr ? initial_h->DataAsSpan<InputT>() : gsl::span<const InputT>();
  gsl::span<const InputT> initial_hidden_1 =
      initial_hidden.empty() ? initial_hidden : initial_hidden.subspan(0, initial_hidden_size_per_direction);

  const size_t initial_cell_size_per_direction = batch_size * hidden_size_;
  gsl::span<const InputT> initial_cell = initial_c != nullptr ? initial_c->DataAsSpan<InputT>() : gsl::span<const InputT>();
  gsl::span<const InputT> initial_cell_1 =
      initial_cell.empty() ? initial_cell : initial_cell.subspan(0, initial_cell_size_per_direction);

  // output shape is [seq_length, num_directions, batch_size, hidden_size]
  // so it's not a case of all the output for one direction being first.
  // due to that we can only easily check that the end of the output for each direction is valid.
  const size_t output_size = Y != nullptr ? Y->Shape().Size() : 0;
  const size_t per_direction_offset = batch_size * hidden_size_;
  gsl::span<InputT> output = Y != nullptr ? Y->MutableDataAsSpan<InputT>() : gsl::span<InputT>();
  gsl::span<InputT> output_1 =
      output.empty() ? output : output.subspan(0, output_size - (num_directions_ - 1) * per_direction_offset);

  // UniDirectionalLstm needs somewhere to write output, so even if we aren't returning Y_h and Y_c
  // we provide an appropriately sized buffer for that purpose.
  const size_t hidden_output_size_per_direction = batch_size * hidden_size_;
  IAllocatorUniquePtr<InputT> local_hidden_output;
  gsl::span<InputT> hidden_output =
      Y_h ? Y_h->MutableDataAsSpan<InputT>()
          : Allocate(alloc, hidden_output_size_per_direction * num_directions_, local_hidden_output);

  gsl::span<InputT> hidden_output_1 = hidden_output.subspan(0, hidden_output_size_per_direction);

  const size_t last_cell_size_per_direction = batch_size * hidden_size_;
  IAllocatorUniquePtr<InputT> local_last_cell;
  gsl::span<InputT> last_cell = Y_c ? Y_c->MutableDataAsSpan<InputT>() : Allocate(alloc, last_cell_size_per_direction * num_directions_, local_last_cell);

  gsl::span<InputT> last_cell_1 = last_cell.subspan(0, last_cell_size_per_direction);

  if (direction_ == Direction::kBidirectional) {
    // spans for second direction
    gsl::span<const InputT> bias_2 = bias.empty() ? bias : bias.subspan(bias_size_per_direction, bias_size_per_direction);
    gsl::span<const InputT> peephole_weights_2 =
        peephole_weights.empty() ? peephole_weights : peephole_weights.subspan(peephole_weights_size_per_direction, peephole_weights_size_per_direction);

    gsl::span<const InputT> initial_hidden_2 =
        initial_hidden.empty() ? initial_hidden : initial_hidden.subspan(initial_hidden_size_per_direction, initial_hidden_size_per_direction);
    gsl::span<const InputT> initial_cell_2 =
        initial_cell.empty() ? initial_cell : initial_cell.subspan(initial_cell_size_per_direction, initial_cell_size_per_direction);
    gsl::span<InputT> output_2 =
        output.empty() ? output : output.subspan(per_direction_offset, output_size - per_direction_offset);

    gsl::span<InputT> hidden_output_2 =
        hidden_output.subspan(hidden_output_size_per_direction, hidden_output_size_per_direction);
    gsl::span<InputT> last_cell_2 = last_cell.subspan(last_cell_size_per_direction, last_cell_size_per_direction);

    lstm::UniDirectionalLstm<InputT> fw(alloc, logger, seq_length, batch_size, input_size, hidden_size_,
                                        Direction::kForward, input_forget_, bias_1, peephole_weights_1, initial_hidden_1,
                                        initial_cell_1, activation_funcs_.Entries()[0], activation_funcs_.Entries()[1],
                                        activation_funcs_.Entries()[2], clip_, thread_pool);

    lstm::UniDirectionalLstm<InputT> bw(alloc, logger, seq_length, batch_size, input_size, hidden_size_,
                                        Direction::kReverse, input_forget_, bias_2, peephole_weights_2, initial_hidden_2,
                                        initial_cell_2, activation_funcs_.Entries()[3], activation_funcs_.Entries()[4],
                                        activation_funcs_.Entries()[5], clip_, thread_pool);

    fw.Compute(input, sequence_lens_span, num_directions_, W_1, R_1, output_1,
               hidden_output_1, last_cell_1);
    bw.Compute(input, sequence_lens_span, num_directions_, W_2, R_2, output_2,
               hidden_output_2, last_cell_2);
  } else {
    lstm::UniDirectionalLstm<InputT> fw(alloc, logger, seq_length, batch_size, input_size, hidden_size_, direction_,
                                        input_forget_, bias_1, peephole_weights_1, initial_hidden_1, initial_cell_1,
                                        activation_funcs_.Entries()[0], activation_funcs_.Entries()[1],
                                        activation_funcs_.Entries()[2], clip_, thread_pool);

    fw.Compute(input, sequence_lens_span, num_directions_, W_1, R_1, output_1,
               hidden_output_1, last_cell_1);
  }

  if (!output.empty())
    DumpMatrix("Y", output.data(), seq_length * num_directions_ * batch_size, hidden_size_);

  // these always get written to regardless of whether we're returning them as optional output or not
  DumpMatrix("Y_h", hidden_output.data(), num_directions_ * batch_size, hidden_size_);
  DumpMatrix("Y_c", last_cell.data(), num_directions_ * batch_size, hidden_size_);

  return Status::OK();
}

Status LSTMBase::ValidateInputs(const Tensor& X,
                                const Tensor* B,
                                const Tensor* sequence_lens,
                                const Tensor* initial_h,
                                const Tensor* initial_c,
                                const Tensor* P) const {
  auto& X_shape = X.Shape();

  int64_t seq_length = X_shape[0];
  int64_t batch_size = X_shape[1];

  if (X_shape.NumDimensions() != 3)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input X must have 3 dimensions only. Actual:", X_shape);

  if (B != nullptr) {
    auto& B_shape = B->Shape();
    if (B_shape.NumDimensions() != 2 ||
        B_shape[0] != num_directions_ ||
        B_shape[1] != 8 * hidden_size_)
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input B must have shape {",
                             num_directions_, ",", 8, "*", hidden_size_, "}. Actual:", B_shape);
  }

  if (sequence_lens != nullptr) {
    auto& sequence_lens_shape = sequence_lens->Shape();
    if (sequence_lens_shape.NumDimensions() != 1 ||
        sequence_lens_shape[0] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input sequence_lens must have shape {",
                             batch_size, "}. Actual:", sequence_lens_shape);
    }

    auto sequence_len_entries = sequence_lens->DataAsSpan<int>();
    if (std::any_of(sequence_len_entries.begin(),
                    sequence_len_entries.end(),
                    [seq_length](int len) { return len < 0 || len > seq_length; })) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Invalid value/s in sequence_lens. All values must be > 0 and < seq_length. seq_length=", seq_length);
    }
  }

  if (initial_h != nullptr) {
    auto& initial_h_shape = initial_h->Shape();

    if (initial_h_shape.NumDimensions() != 3 ||
        initial_h_shape[0] != num_directions_ ||
        initial_h_shape[1] != batch_size ||
        initial_h_shape[2] != hidden_size_)

      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input initial_h must have shape {",
                             num_directions_, ",", batch_size, ",", hidden_size_, "}. Actual:", initial_h_shape);
  }

  if (initial_c != nullptr) {
    auto& initial_c_shape = initial_c->Shape();

    if (initial_c_shape.NumDimensions() != 3 || initial_c_shape[0] != num_directions_ ||
        initial_c_shape[1] != batch_size || initial_c_shape[2] != hidden_size_)

      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input initial_c must have shape {", num_directions_, ",", batch_size,
                             ",", hidden_size_, "}. Actual:", initial_c_shape);
  }

  if (P != nullptr) {
    auto& p_shape = P->Shape();

    if (p_shape.NumDimensions() != 2 || p_shape[0] != num_directions_ || p_shape[1] != 3 * hidden_size_)

      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input P must have shape {", num_directions_, ",", 3 * hidden_size_,
                             "}. Actual:", p_shape);
  }

  return Status::OK();
}

template Status LSTMBase::ComputeImpl<float, float>(OpKernelContext& context,
                                                    const rnn::detail::GemmWeights<float>& W_1,
                                                    const rnn::detail::GemmWeights<float>& W_2,
                                                    const rnn::detail::GemmWeights<float>& R_1,
                                                    const rnn::detail::GemmWeights<float>& R_2) const;

template Status LSTMBase::ComputeImpl<float, uint8_t>(OpKernelContext& context,
                                                      const rnn::detail::GemmWeights<uint8_t>& W_1,
                                                      const rnn::detail::GemmWeights<uint8_t>& W_2,
                                                      const rnn::detail::GemmWeights<uint8_t>& R_1,
                                                      const rnn::detail::GemmWeights<uint8_t>& R_2) const;

}  // namespace onnxruntime
