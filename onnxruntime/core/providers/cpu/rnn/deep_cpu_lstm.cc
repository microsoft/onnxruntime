// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "deep_cpu_lstm.h"
#include "uni_directional_lstm.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

/*
ONNX_OPERATOR_SCHEMA(LSTM)
    .SetDoc(R"DOC(
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

`X` - input tensor
`i` - input gate
`o` - output gate
`f` - forget gate
`c` - cell gate
`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates
`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates
`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates
`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates
`P[iof]`  - P peephole weight vector for input, output, and forget gates
`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates
`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates
`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates
`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates
`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

`H` - Hidden state
`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)
  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)
  Affine(x)              - alpha*x + beta
  LeakyRelu(x)           - x if x >= 0 else alpha * x
  ThresholdedRelu(x)     - x if x >= alpha else 0
  ScaledTanh(x)          - alpha*Tanh(beta*x)
  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
  Softsign(x)            - x/(1 + |x|)
  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):
  - it = f(Xt*(Wi^T) + Ht-1*Ri + Pi (.) Ct-1 + Wbi + Rbi)
  - ft = f(Xt*(Wf^T) + Ht-1*Rf + Pf (.) Ct-1 + Wbf + Rbf)
  - ct = g(Xt*(Wc^T) + Ht-1*Rc + Wbc + Rbc)
  - Ct = ft (.) Ct-1 + it (.) ct
  - ot = f(Xt*(Wo^T) + Ht-1*Ro + Po (.) Ct + Wbo + Rbo)
  - Ht = ot (.) h(Ct)
)DOC")
    .Attr("direction", "Specify if the RNN is forward, reverse, or bidirectional. "
               "Must be one of forward (default), reverse, or bidirectional.",
               AttributeProto::STRING,
               std::string("forward"))
    .Attr("hidden_size", "Number of neurons in the hidden layer", AttributeProto::INT, OPTIONAL)
    .Attr("activations", "A list of 3 (or 6 if bidirectional) activation functions "
               "for input, output, forget, cell, and hidden. The activation functions must "
               "be one of the activation functions specified above. Optional: See the equations "
               "for default if not specified.",
               AttributeProto::STRINGS,
               OPTIONAL)
    .Attr("activation_alpha",
               "Optional scaling values used by some activation functions. The values "
               "are consumed in the order of activation functions, for example (f, g, h) "
               "in LSTM.",
               AttributeProto::FLOATS,
               OPTIONAL)
    .Attr("activation_beta",
               "Optional scaling values used by some activation functions. The values "
               "are consumed in the order of activation functions, for example (f, g, h) "
               "in LSTM.",
               AttributeProto::FLOATS,
               OPTIONAL)
    .Attr("output_sequence",
               "The sequence output for the hidden is optional if 0. Default 0.",
               AttributeProto::INT,
               static_cast<int64_t>(0));
    .Attr("clip", "Cell clip threshold. Clipping bounds the elements of a tensor "
               "in the range of [-threshold, +threshold] and is applied to the input "
               "of activations. No clip if not specified.", AttributeProto::FLOAT, OPTIONAL)
    .Attr("input_forget", "Couple the input and forget gates if 1, default 0.",
               AttributeProto::INT,
               static_cast<int64_t>(0))
    .Input(0, "X",
               "The input sequences packed (and potentially padded) into one 3-D "
               "tensor with the shape of `[seq_length, batch_size, input_size]`.", "T")
    .Input(1, "W",
               "The weight tensor for the gates. Concatenation of `W[iofc]` and "
               "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
               "`[num_directions, 4*hidden_size, input_size]`.", "T")
    .Input(2, "R",
               "The recurrence weight tensor. Concatenation of `R[iofc]` and "
               "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
               "`[num_directions, 4*hidden_size, hidden_size]`.", "T")
    .Input(3, "B",
               "The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, "
               "and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. This "
               "tensor has shape `[num_directions, 8*hidden_size]`. Optional: If not "
               "specified - assumed to be 0.", "T",
               OpSchema::Optional)
    .Input(4, "sequence_lens",
               "Optional tensor specifying lengths of the sequences in a batch. "
               "If not specified - assumed all sequences in the batch to have "
               "length `seq_length`. It has shape `[batch_size]`.", "T1",
               OpSchema::Optional)
    .Input(5, "initial_h",
                "Optional initial value of the hidden. If not specified - assumed "
                "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
                "T", OpSchema::Optional)
    .Input(6, "initial_c",
                "Optional initial value of the cell. If not specified - assumed "
                "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
"T", OpSchema::Optional)
    .Input(7, "P",
                "The weight tensor for peepholes. Concatenation of `P[iof]` and "
                "`PB[iof]` (if bidirectional) along dimension 0. It has shape "
                "`[num_directions, 3*hidde_size]`. Optional: If not specified - "
                "assumed to be 0.", "T",
                OpSchema::Optional)
    .Output(0, "Y",
                "A tensor that concats all the intermediate output values of the hidden. "
                "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. ",
                "T", OpSchema::Optional);
    .Output(1, "Y_h",
                "The last output value of the hidden. It has shape "
                "`[num_directions, batch_size, hidden_size]`.", "T", OpSchema::Optional);
    .Output(2, "Y_c",
                "The last output value of the cell. It has shape "
                "`[num_directions, batch_size, hidden_size]`.", "T", OpSchema::Optional);
    .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                    "Constrain input and output types to float tensors.")
    .TypeConstraint("T1", { "tensor(int32)" }, "Constrain seq_lens to integer tensor.");

*/

namespace onnxruntime {

/* LSTM operator */
ONNX_CPU_OPERATOR_KERNEL(LSTM, 7,
                         KernelDefBuilder()
                             .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                   DataTypeImpl::GetTensorType<double>()})
                             .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
                         DeepCpuLstmOp);

using namespace rnn::detail;

// LSTM details

Status DeepCpuLstmOp::TryPackWeights(const Tensor& weights, PackedWeights& packed_weights, bool& is_packed) {
  const auto& shape = weights.Shape();
  if (shape.NumDimensions() != 3) {
    return Status::OK();
  }

  // weights: [num_directions, 4*hidden_size, input_size]
  // recurrence weights: [num_directions, 4*hidden_size, hidden_size]
  const size_t N = static_cast<size_t>(shape[1]);
  const size_t K = static_cast<size_t>(shape[2]);

  if ((shape[0] != num_directions_) || (N != static_cast<size_t>(hidden_size_ * 4))) {
    return Status::OK();
  }

  const size_t packed_weights_size = MlasGemmPackBSize(N, K);
  if (packed_weights_size == 0) {
    return Status::OK();
  }

  auto alloc = Info().GetAllocator(0, OrtMemTypeDefault);
  auto* packed_weights_data = alloc->Alloc(SafeInt<size_t>(packed_weights_size) * num_directions_);
  packed_weights.buffer_ = BufferUniquePtr(packed_weights_data, BufferDeleter(alloc));
  packed_weights.weights_size_ = packed_weights_size;
  packed_weights.shape_ = shape;

  const auto* weights_data = weights.Data<float>();
  for (int i = 0; i < num_directions_; i++) {
    MlasGemmPackB(CblasTrans, N, K, weights_data, K, packed_weights_data);
    packed_weights_data = static_cast<uint8_t*>(packed_weights_data) + packed_weights_size;
    weights_data += N * K;
  }

  is_packed = true;
  return Status::OK();
}

Status DeepCpuLstmOp::PrePack(const Tensor& tensor, int input_idx, bool& is_packed) {
  is_packed = false;

  if (tensor.IsDataType<float>()) {
    if (input_idx == 1) {
      return TryPackWeights(tensor, packed_W_, is_packed);
    } else if (input_idx == 2) {
      return TryPackWeights(tensor, packed_R_, is_packed);
    }
  }

  return Status::OK();
}

Status DeepCpuLstmOp::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]

  Status status;
  // auto& logger = context->Logger();

  if (X.IsDataType<float>()) {
    status = ComputeImpl<float>(*context);
  } else if (X.IsDataType<double>()) {
    /* Need to update all the helpers to support double...
    status = ComputeImpl<double>(*context); */
    ORT_NOT_IMPLEMENTED("LSTM operator does not support double yet");
  } else {
    ORT_THROW("Invalid data type for LSTM operator of ", X.DataType());
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
Status DeepCpuLstmOp::ComputeImpl(OpKernelContext& context) const {
  concurrency::ThreadPool* thread_pool = context.GetOperatorThreadPool();

  auto& logger = context.Logger();

  const Tensor& X = *context.Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]
  const Tensor* W = packed_W_.buffer_ ? nullptr : context.Input<Tensor>(1);
  // weights. [num_directions, 4*hidden_size, input_size]
  const Tensor* R = packed_R_.buffer_ ? nullptr : context.Input<Tensor>(2);
  // recurrence weights. [num_directions, 4*hidden_size, hidden_size]

  // optional
  const Tensor* B = context.Input<Tensor>(3);              // bias. [num_directions, 8*hidden_size]
  const Tensor* sequence_lens = context.Input<Tensor>(4);  // [batch_size]
  const Tensor* initial_h = context.Input<Tensor>(5);      // initial hidden. [num_directions, batch_size, hidden_size]
  const Tensor* initial_c = context.Input<Tensor>(6);      // initial cell. [num_directions, batch_size, hidden_size]
  const Tensor* P = context.Input<Tensor>(7);              // peephole weights. [num_directions, 3*hidden_size]

  const auto& X_shape = X.Shape();

  int seq_length = gsl::narrow<int>(X_shape[0]);
  int batch_size = gsl::narrow<int>(X_shape[1]);
  int input_size = gsl::narrow<int>(X_shape[2]);

  const auto& W_shape = (W != nullptr) ? W->Shape() : packed_W_.shape_;
  const auto& R_shape = (R != nullptr) ? R->Shape() : packed_R_.shape_;

  Status status = ValidateInputs(X, W_shape, R_shape, B, sequence_lens, initial_h, initial_c, P, batch_size);
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
        std::fill_n(Y->MutableData<T>(), Y_dims.Size(), T{});
      if (Y_h != nullptr)
        std::fill_n(Y_h->MutableData<T>(), Y_h_dims.Size(), T{});
      if (Y_c != nullptr)
        std::fill_n(Y_c->MutableData<T>(), Y_c_dims.Size(), T{});
      return Status::OK();
    }
  }

  AllocatorPtr alloc;
  status = context.GetTempSpaceAllocator(&alloc);
  ORT_RETURN_IF_ERROR(status);

  const auto* input_weights = (W != nullptr) ? W->Data<T>() : nullptr;
  const auto* recurrent_weights = (R != nullptr) ? R->Data<T>() : nullptr;

  gsl::span<const T> bias = B != nullptr ? B->DataAsSpan<T>() : gsl::span<const T>();
  gsl::span<const T> peephole_weights = P != nullptr ? P->DataAsSpan<T>() : gsl::span<const T>();

  // spans for first direction
  const size_t input_weights_size_per_direction = 4 * hidden_size_ * input_size;
  const size_t hidden_weights_size_per_direction = 4 * hidden_size_ * hidden_size_;
  const size_t bias_size_per_direction = 8 * hidden_size_;
  const size_t peephole_weights_size_per_direction = 3 * hidden_size_;

  GemmWeights<T> input_weights_1(0, input_weights, input_weights_size_per_direction, packed_W_);
  GemmWeights<T> recurrent_weights_1(0, recurrent_weights, hidden_weights_size_per_direction, packed_R_);

  gsl::span<const T> bias_1 = bias.empty() ? bias : bias.subspan(0, bias_size_per_direction);
  gsl::span<const T> peephole_weights_1 =
      peephole_weights.empty() ? peephole_weights : peephole_weights.subspan(0, peephole_weights_size_per_direction);

  gsl::span<const T> input = X.DataAsSpan<T>();
  gsl::span<const int> sequence_lens_span =
      sequence_lens != nullptr ? sequence_lens->DataAsSpan<int>() : gsl::span<const int>();

  const size_t initial_hidden_size_per_direction = batch_size * hidden_size_;
  gsl::span<const T> initial_hidden = initial_h != nullptr ? initial_h->DataAsSpan<T>() : gsl::span<const T>();
  gsl::span<const T> initial_hidden_1 =
      initial_hidden.empty() ? initial_hidden : initial_hidden.subspan(0, initial_hidden_size_per_direction);

  const size_t initial_cell_size_per_direction = batch_size * hidden_size_;
  gsl::span<const T> initial_cell = initial_c != nullptr ? initial_c->DataAsSpan<T>() : gsl::span<const T>();
  gsl::span<const T> initial_cell_1 =
      initial_cell.empty() ? initial_cell : initial_cell.subspan(0, initial_cell_size_per_direction);

  // output shape is [seq_length, num_directions, batch_size, hidden_size]
  // so it's not a case of all the output for one direction being first.
  // due to that we can only easily check that the end of the output for each direction is valid.
  const size_t output_size = Y != nullptr ? Y->Shape().Size() : 0;
  const size_t per_direction_offset = batch_size * hidden_size_;
  gsl::span<T> output = Y != nullptr ? Y->MutableDataAsSpan<T>() : gsl::span<T>();
  gsl::span<T> output_1 =
      output.empty() ? output : output.subspan(0, output_size - (num_directions_ - 1) * per_direction_offset);

  // UniDirectionalLstm needs somewhere to write output, so even if we aren't returning Y_h and Y_c
  // we provide an appropriately sized buffer for that purpose.
  const size_t hidden_output_size_per_direction = batch_size * hidden_size_;
  IAllocatorUniquePtr<T> local_hidden_output;
  gsl::span<T> hidden_output =
      Y_h ? Y_h->MutableDataAsSpan<T>()
          : Allocate(alloc, hidden_output_size_per_direction * num_directions_, local_hidden_output);

  gsl::span<T> hidden_output_1 = hidden_output.subspan(0, hidden_output_size_per_direction);

  const size_t last_cell_size_per_direction = batch_size * hidden_size_;
  IAllocatorUniquePtr<T> local_last_cell;
  gsl::span<T> last_cell = Y_c ? Y_c->MutableDataAsSpan<T>() : Allocate(alloc, last_cell_size_per_direction * num_directions_, local_last_cell);

  gsl::span<T> last_cell_1 = last_cell.subspan(0, last_cell_size_per_direction);

  if (direction_ == Direction::kBidirectional) {
    GemmWeights<T> input_weights_2(1, input_weights, input_weights_size_per_direction, packed_W_);
    GemmWeights<T> recurrent_weights_2(1, recurrent_weights, hidden_weights_size_per_direction, packed_R_);

    // spans for second direction
    gsl::span<const T> bias_2 = bias.empty() ? bias : bias.subspan(bias_size_per_direction, bias_size_per_direction);
    gsl::span<const T> peephole_weights_2 =
        peephole_weights.empty() ? peephole_weights : peephole_weights.subspan(peephole_weights_size_per_direction, peephole_weights_size_per_direction);

    gsl::span<const T> initial_hidden_2 =
        initial_hidden.empty() ? initial_hidden : initial_hidden.subspan(initial_hidden_size_per_direction, initial_hidden_size_per_direction);
    gsl::span<const T> initial_cell_2 =
        initial_cell.empty() ? initial_cell : initial_cell.subspan(initial_cell_size_per_direction, initial_cell_size_per_direction);
    gsl::span<T> output_2 =
        output.empty() ? output : output.subspan(per_direction_offset, output_size - per_direction_offset);

    gsl::span<T> hidden_output_2 =
        hidden_output.subspan(hidden_output_size_per_direction, hidden_output_size_per_direction);
    gsl::span<T> last_cell_2 = last_cell.subspan(last_cell_size_per_direction, last_cell_size_per_direction);

    lstm::UniDirectionalLstm<T> fw(alloc, logger, seq_length, batch_size, input_size, hidden_size_,
                                   Direction::kForward, input_forget_, bias_1, peephole_weights_1, initial_hidden_1,
                                   initial_cell_1, activation_funcs_.Entries()[0], activation_funcs_.Entries()[1],
                                   activation_funcs_.Entries()[2], clip_, thread_pool);

    lstm::UniDirectionalLstm<T> bw(alloc, logger, seq_length, batch_size, input_size, hidden_size_,
                                   Direction::kReverse, input_forget_, bias_2, peephole_weights_2, initial_hidden_2,
                                   initial_cell_2, activation_funcs_.Entries()[3], activation_funcs_.Entries()[4],
                                   activation_funcs_.Entries()[5], clip_, thread_pool);

    fw.Compute(input, sequence_lens_span, num_directions_, input_weights_1, recurrent_weights_1, output_1,
               hidden_output_1, last_cell_1);
    bw.Compute(input, sequence_lens_span, num_directions_, input_weights_2, recurrent_weights_2, output_2,
               hidden_output_2, last_cell_2);
  } else {
    lstm::UniDirectionalLstm<T> fw(alloc, logger, seq_length, batch_size, input_size, hidden_size_, direction_,
                                   input_forget_, bias_1, peephole_weights_1, initial_hidden_1, initial_cell_1,
                                   activation_funcs_.Entries()[0], activation_funcs_.Entries()[1],
                                   activation_funcs_.Entries()[2], clip_, thread_pool);

    fw.Compute(input, sequence_lens_span, num_directions_, input_weights_1, recurrent_weights_1, output_1,
               hidden_output_1, last_cell_1);
  }

  if (!output.empty())
    DumpMatrix("Y", output.data(), seq_length * num_directions_ * batch_size, hidden_size_);

  // these always get written to regardless of whether we're returning them as optional output or not
  DumpMatrix("Y_h", hidden_output.data(), num_directions_ * batch_size, hidden_size_);
  DumpMatrix("Y_c", last_cell.data(), num_directions_ * batch_size, hidden_size_);

  return Status::OK();
}
}  // namespace onnxruntime
