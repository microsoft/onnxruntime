// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "core/platform/threadpool.h"
#include "core/framework/op_kernel_context_internal.h"

#include "core/providers/cpu/rnn/deep_cpu_lstm.h"

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/allocator.h"

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
ONNX_CPU_OPERATOR_KERNEL(
    LSTM,
    7,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    DeepCpuLstmOp);

using namespace rnn::detail;

// LSTM details
namespace detail {

// Helper struct for an activation function call information
template <typename TFunc>
struct ActivationInfo {
  TFunc func;
  float alpha;
  float beta;
};

// copying the peephole values into UniDirectionalLstm seems unnecessary. don't do that until proven necessary
#define LSTM_NO_PEEPHOLE_COPY

template <typename T>
class UniDirectionalLstm {
 public:
  UniDirectionalLstm(AllocatorPtr allocator, const logging::Logger& logger, int seq_length, int batch_size,
                     int input_size, int hidden_size, Direction direction, bool input_forget,
                     const gsl::span<const T>& bias, const gsl::span<const T>& peephole_weights,
                     const gsl::span<const T>& initial_hidden_state, const gsl::span<const T>& initial_cell_state,
                     const ActivationFuncs::Entry& activation_func_f, const ActivationFuncs::Entry& activation_func_g,
                     const ActivationFuncs::Entry& activation_func_h, float clip,
                     concurrency::ThreadPool& lstm_tp_,
                     concurrency::ThreadPool* mlas_tp_);

  void Compute(const gsl::span<const T>& inputs, const gsl::span<const int>& sequence_lengths, int num_directions,
               const gsl::span<const T>& input_weights, const gsl::span<const T>& recurrent_weights,
               gsl::span<T>& outputs, gsl::span<T>& final_hidden_state, gsl::span<T>& final_cell_state);

  ~UniDirectionalLstm() = default;

 private:
  using span_T_const_iter = typename gsl::span<T>::const_iterator;
  using span_T_iter = typename gsl::span<T>::iterator;

  void SetNumThreads();

  void GateComputations(span_T_iter& out, span_T_iter& out_end, span_T_iter& C_prev,
                        const span_T_iter& C_prev_end,  // Ct-1 value not 'ct'. using 'C' for clarity
                        span_T_iter& C_prev_clipped, const span_T_iter& C_prev_clipped_end, span_T_iter& batched_output,
                        span_T_iter& batched_output_end, const gsl::span<const int>& seq_lengths,
                        int min_sequence_length, int step, int row, int local_fused_hidden_rows, bool output_sequence);

  void AllocateBuffers();

  void InitializeBuffers(const gsl::span<const T>& initial_hidden_state,
                         const gsl::span<const T>& initial_cell_state);

  void LoadPeepholeWeights(const gsl::span<const T>& peephole_weights);
  void LoadBias(const gsl::span<const T>& WbRb_values);

  AllocatorPtr allocator_;
  const logging::Logger& logger_;

  int seq_length_;
  int batch_size_;
  int input_size_;
  int hidden_size_;

  Direction direction_;
  bool input_forget_;
  float clip_;

  bool batch_parallel_;

  bool use_bias_;
  bool use_peepholes_;

  int hidden_num_threads_ = -1;

  IAllocatorUniquePtr<T> output_iofc_ptr_;
  IAllocatorUniquePtr<T> hidden0_ptr_, batched_hidden0_ptr_;
  gsl::span<T> output_iofc_;
  gsl::span<T> hidden0_, batched_hidden0_;

  IAllocatorUniquePtr<T> internal_memory_prev_ptr_, batched_internal_memory_prev_ptr_;
  IAllocatorUniquePtr<T> internal_memory_cur_ptr_, batched_internal_memory_cur_ptr_;
  IAllocatorUniquePtr<T> batched_internal_memory_clipped_ptr_;
  gsl::span<T> internal_memory_prev_, batched_internal_memory_prev_;
  gsl::span<T> internal_memory_cur_, batched_internal_memory_cur_;
  gsl::span<T> batched_internal_memory_clipped_;

  IAllocatorUniquePtr<T> bias_WRi_ptr_, bias_WRf_ptr_, bias_WRo_ptr_, bias_WRc_ptr_;
  IAllocatorUniquePtr<T> batched_bias_WRi_ptr_, batched_bias_WRf_ptr_, batched_bias_WRo_ptr_, batched_bias_WRc_ptr_;
  IAllocatorUniquePtr<T> peephole_i_ptr_, peephole_f_ptr_, peephole_o_ptr_;
  IAllocatorUniquePtr<T> inputs_reverse_ptr_, outputs_reverse_ptr_;
  gsl::span<T> bias_WRi_, bias_WRf_, bias_WRo_, bias_WRc_;
  gsl::span<T> batched_bias_WRi_, batched_bias_WRf_, batched_bias_WRo_, *batched_bias_WRc_;
  gsl::span<T> inputs_reverse_, outputs_reverse_;

#if defined(LSTM_NO_PEEPHOLE_COPY)
  gsl::span<const T> peephole_i_, peephole_f_, peephole_o_;
#else
  gsl::span<T> peephole_i_, peephole_f_, peephole_o_;
#endif

  IAllocatorUniquePtr<int> sequence_lengths_ptr_;
  gsl::span<int> sequence_lengths_;

  deepcpu::ClipWithBiasFuncPtr clip_with_bias_ptr_;

  ActivationInfo<deepcpu::ActivationFuncPtr> activation_f_;
  ActivationInfo<deepcpu::ActivationFuncPtr> activation_g_;
  ActivationInfo<deepcpu::LstmMergeGatesFuncPtr> activation_h_;

  concurrency::ThreadPool& lstm_tp_;
  concurrency::ThreadPool* mlas_tp_;
};

}  // namespace detail

Status
DeepCpuLstmOp::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]

  Status status;
  // auto& logger = context->Logger();

  auto data_type = X.DataType();
  if (utils::IsPrimitiveDataType<float>(data_type))
    status = ComputeImpl<float>(*context);
  else if (utils::IsPrimitiveDataType<double>(data_type)) {
    /* Need to update all the helpers to support double...
    status = ComputeImpl<double>(*context); */
    ORT_NOT_IMPLEMENTED("LSTM operator does not support double yet");
  } else
    ORT_THROW("Invalid data type for LSTM operator of ", data_type);

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
  concurrency::ThreadPool* mlas_thread_pool = context.GetOperatorThreadPool();

  auto& logger = context.Logger();

  const Tensor& X = *context.Input<Tensor>(0);  // inputs. [seq_length, batch_size, input_size]
  const Tensor& W = *context.Input<Tensor>(1);  // weights. [num_directions, 4*hidden_size, input_size]
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

  Status status = ValidateInputs(X, W, R, B, sequence_lens, initial_h, initial_c, P, batch_size);
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
    int32_t max_sequence_length = *std::max_element(sequence_lens->Data<int32_t>(), sequence_lens->Data<int32_t>() + sequence_lens->Shape().Size());
    if (max_sequence_length == 0) {
      if (Y != nullptr) std::fill_n(Y->MutableData<T>(), Y_dims.Size(), T{});
      if (Y_h != nullptr) std::fill_n(Y_h->MutableData<T>(), Y_h_dims.Size(), T{});
      if (Y_c != nullptr) std::fill_n(Y_c->MutableData<T>(), Y_c_dims.Size(), T{});
      return Status::OK();
    }
  }

  AllocatorPtr alloc;
  status = context.GetTempSpaceAllocator(&alloc);
  ORT_RETURN_IF_ERROR(status);

  gsl::span<const T> input_weights = W.DataAsSpan<T>();
  gsl::span<const T> recurrent_weights = R.DataAsSpan<T>();
  gsl::span<const T> bias = B != nullptr ? B->DataAsSpan<T>() : gsl::span<const T>();
  gsl::span<const T> peephole_weights = P != nullptr ? P->DataAsSpan<T>() : gsl::span<const T>();

  // spans for first direction
  const size_t input_weights_size_per_direction = 4 * hidden_size_ * input_size;
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
  const size_t output_size = Y != nullptr ? Y->Shape().Size() : 0;
  const size_t per_direction_offset = batch_size * hidden_size_;
  gsl::span<T> output = Y != nullptr ? Y->MutableDataAsSpan<T>() : gsl::span<T>();
  gsl::span<T> output_1 =
      output.empty() ? output
                     : output.subspan(0, output_size - (num_directions_ - 1) * per_direction_offset);

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
  gsl::span<T> last_cell =
      Y_c ? Y_c->MutableDataAsSpan<T>()
          : Allocate(alloc, last_cell_size_per_direction * num_directions_, local_last_cell);

  gsl::span<T> last_cell_1 = last_cell.subspan(0, last_cell_size_per_direction);

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

    detail::UniDirectionalLstm<T> fw(alloc, logger, seq_length, batch_size, input_size,
                                     hidden_size_, Direction::kForward, input_forget_,
                                     bias_1, peephole_weights_1, initial_hidden_1, initial_cell_1,
                                     activation_funcs_.Entries()[0],
                                     activation_funcs_.Entries()[1],
                                     activation_funcs_.Entries()[2],
                                     clip_, lstm_tp_, mlas_thread_pool);

    detail::UniDirectionalLstm<T> bw(alloc, logger, seq_length, batch_size, input_size,
                                     hidden_size_, Direction::kReverse, input_forget_,
                                     bias_2, peephole_weights_2, initial_hidden_2, initial_cell_2,
                                     activation_funcs_.Entries()[3],
                                     activation_funcs_.Entries()[4],
                                     activation_funcs_.Entries()[5],
                                     clip_, lstm_tp_, mlas_thread_pool);

    fw.Compute(input, sequence_lens_span, num_directions_, input_weights_1, recurrent_weights_1,
               output_1, hidden_output_1, last_cell_1);
    bw.Compute(input, sequence_lens_span, num_directions_, input_weights_2, hidden_weights_2,
               output_2, hidden_output_2, last_cell_2);
  } else {
    detail::UniDirectionalLstm<T> fw(alloc, logger, seq_length, batch_size, input_size,
                                     hidden_size_, direction_, input_forget_,
                                     bias_1, peephole_weights_1, initial_hidden_1, initial_cell_1,
                                     activation_funcs_.Entries()[0],
                                     activation_funcs_.Entries()[1],
                                     activation_funcs_.Entries()[2],
                                     clip_, lstm_tp_, mlas_thread_pool);

    fw.Compute(input, sequence_lens_span, num_directions_, input_weights_1, recurrent_weights_1,
               output_1, hidden_output_1, last_cell_1);
  }

  if (!output.empty())
    DumpMatrix("Y", output.data(), seq_length * num_directions_ * batch_size, hidden_size_);

  // these always get written to regardless of whether we're returning them as optional output or not
  DumpMatrix("Y_h", hidden_output.data(), num_directions_ * batch_size, hidden_size_);
  DumpMatrix("Y_c", last_cell.data(), num_directions_ * batch_size, hidden_size_);

  return Status::OK();
}

Status DeepCpuLstmOp::ValidateInputs(const Tensor& X, const Tensor& W, const Tensor& R, const Tensor* B,
                                     const Tensor* sequence_lens, const Tensor* initial_h, const Tensor* initial_c,
                                     const Tensor* P, int batch_size) const {
  auto status = rnn::detail::ValidateCommonRnnInputs(X, W, R, B, 4, sequence_lens, initial_h,
                                                     num_directions_, hidden_size_);
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

/*************************************
*
* Implementation of UniDirectionalLstm
*
*/
namespace detail {

template <typename T>
UniDirectionalLstm<T>::UniDirectionalLstm(AllocatorPtr allocator,
                                          const logging::Logger& logger,
                                          const int seq_length,
                                          const int batch_size,
                                          const int input_size,
                                          const int hidden_size,
                                          Direction direction,
                                          const bool input_forget,
                                          const gsl::span<const T>& bias,
                                          const gsl::span<const T>& peephole_weights,
                                          const gsl::span<const T>& initial_hidden_state,
                                          const gsl::span<const T>& initial_cell_state,
                                          const ActivationFuncs::Entry& activation_func_f,
                                          const ActivationFuncs::Entry& activation_func_g,
                                          const ActivationFuncs::Entry& activation_func_h,
                                          const float clip,
                                          concurrency::ThreadPool& lstm_tp,
                                          concurrency::ThreadPool* mlas_tp)
    : allocator_(allocator),
      logger_(logger),
      seq_length_(seq_length),
      batch_size_(batch_size),
      input_size_(input_size),
      hidden_size_(hidden_size),
      direction_(direction),
      input_forget_(input_forget),
      clip_(clip),
      use_bias_(!bias.empty()),
      use_peepholes_(!peephole_weights.empty()),
      lstm_tp_(lstm_tp),
      mlas_tp_(mlas_tp) {
  activation_f_ = {deepcpu::ActivationFuncByName(activation_func_f.name),
                   activation_func_f.alpha,
                   activation_func_f.beta};

  activation_g_ = {deepcpu::ActivationFuncByName(activation_func_g.name),
                   activation_func_g.alpha,
                   activation_func_g.beta};

  activation_h_ = {deepcpu::LstmMergeGatesFuncByName(activation_func_h.name),
                   activation_func_h.alpha,
                   activation_func_h.beta};

  clip_with_bias_ptr_ = use_bias_ ? deepcpu::clip_add_bias : deepcpu::clip_ignore_bias;

  SetNumThreads();
  AllocateBuffers();
  InitializeBuffers(initial_hidden_state, initial_cell_state);

  if (!peephole_weights.empty())
    LoadPeepholeWeights(peephole_weights);
  if (!bias.empty())
    LoadBias(bias);
}

template <typename T>
void UniDirectionalLstm<T>::AllocateBuffers() {
  // allocate and fill with 0's.
  const bool fill = true;
  hidden0_ = Allocate(allocator_, hidden_size_, hidden0_ptr_, fill);
  internal_memory_prev_ = Allocate(allocator_, hidden_size_, internal_memory_prev_ptr_, fill);
  internal_memory_cur_ = Allocate(allocator_, hidden_size_, internal_memory_cur_ptr_, fill);
  batched_hidden0_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_hidden0_ptr_, fill);

  batched_internal_memory_prev_ = Allocate(allocator_, batch_size_ * hidden_size_,
                                           batched_internal_memory_prev_ptr_, fill);
  batched_internal_memory_cur_ = Allocate(allocator_, batch_size_ * hidden_size_,
                                          batched_internal_memory_cur_ptr_, fill);
  batched_internal_memory_clipped_ = Allocate(allocator_, batch_size_ * hidden_size_,
                                              batched_internal_memory_clipped_ptr_, fill);

  output_iofc_ = Allocate(allocator_, hidden_size_ * 4 * batch_size_ * seq_length_, output_iofc_ptr_, fill);

  if (use_bias_) {
    bias_WRi_ = Allocate(allocator_, hidden_size_, bias_WRi_ptr_);
    bias_WRf_ = Allocate(allocator_, hidden_size_, bias_WRf_ptr_);
    bias_WRo_ = Allocate(allocator_, hidden_size_, bias_WRo_ptr_);
    bias_WRc_ = Allocate(allocator_, hidden_size_, bias_WRc_ptr_);
  }

  if (direction_ == kReverse) {
    inputs_reverse_ = Allocate(allocator_, seq_length_ * batch_size_ * input_size_, inputs_reverse_ptr_);
    outputs_reverse_ = Allocate(allocator_, seq_length_ * batch_size_ * hidden_size_, outputs_reverse_ptr_);
  }

#if !defined(LSTM_NO_PEEPHOLE_COPY)
  if (use_peepholes_) {
    peephole_i_ = Allocate(allocator_, hidden_size_, peephole_i_ptr_);
    peephole_f_ = Allocate(allocator_, hidden_size_, peephole_f_ptr_);
    peephole_o_ = Allocate(allocator_, hidden_size_, peephole_o_ptr_);
  }
#endif
}

template <typename T>
void UniDirectionalLstm<T>::InitializeBuffers(const gsl::span<const T>& initial_hidden_state,
                                              const gsl::span<const T>& initial_cell_state) {
  if (!initial_hidden_state.empty()) {
    gsl::copy(initial_hidden_state, batched_hidden0_);
  } else {
    std::fill_n(batched_hidden0_.data(), batched_hidden0_.size(), T{});
  }

  if (!initial_cell_state.empty()) {
    gsl::copy(initial_cell_state, batched_internal_memory_prev_);
  } else {
    std::fill_n(batched_internal_memory_prev_.data(), batched_internal_memory_prev_.size(), T{});
  }
}

template <typename T>
void UniDirectionalLstm<T>::LoadPeepholeWeights(const gsl::span<const T>& peephole_weights) {
  int i = 0;
#if defined(LSTM_NO_PEEPHOLE_COPY)

  // just use spans. we don't change these values so there's no point copying to them
  peephole_i_ = peephole_weights.subspan((i++ * hidden_size_), hidden_size_);
  peephole_o_ = peephole_weights.subspan((i++ * hidden_size_), hidden_size_);
  peephole_f_ = peephole_weights.subspan((i++ * hidden_size_), hidden_size_);

#else
  DumpMatrix("P[i]", peephole_weights.data() + (i++ * hidden_size_), 1, hidden_size_);
  DumpMatrix("P[o]", peephole_weights.data() + (i++ * hidden_size_), 1, hidden_size_);
  DumpMatrix("P[f]", peephole_weights.data() + (i++ * hidden_size_), 1, hidden_size_);

  auto copy_weight = [this, &peephole_weights](int offset, gsl::span<T>& out) {
    typename gsl::span<const T> src = peephole_weights.subspan(offset, hidden_size_);
    gsl::copy(src, out);
  };

  i = 0;
  copy_weight((i++ * hidden_size_), peephole_i_);
  copy_weight((i++ * hidden_size_), peephole_o_);
  copy_weight((i++ * hidden_size_), peephole_f_);
#endif

  /*
  DumpMatrix("peephole_i_", peephole_i_.data(), 1, hidden_size_);
  DumpMatrix("peephole_o_", peephole_o_.data(), 1, hidden_size_);
  DumpMatrix("peephole_f_", peephole_f_.data(), 1, hidden_size_);
  */
}

template <typename T>
void UniDirectionalLstm<T>::LoadBias(const gsl::span<const T>& WbRb_values) {
  // add Wb and Rb
  auto copy_fused_bias = [this, &WbRb_values](int offset, gsl::span<T>& out) {
    // gap between Wb and Wb value for an entry
    const int Wb_to_Rb_offset = 4 * hidden_size_;

    for (int j = 0; j < hidden_size_; ++j)
      out[j] = WbRb_values[j + offset] + WbRb_values[j + offset + Wb_to_Rb_offset];
  };

  int i = 0;
  copy_fused_bias((i++) * hidden_size_, bias_WRi_);
  copy_fused_bias((i++) * hidden_size_, bias_WRo_);
  copy_fused_bias((i++) * hidden_size_, bias_WRf_);
  copy_fused_bias((i++) * hidden_size_, bias_WRc_);

  /*
  i = 0;
  DumpMatrix("Wb[i]", WbRb_values.data() + (i++ * hidden_size_), 1, hidden_size_);
  DumpMatrix("Wb[o]", WbRb_values.data() + (i++ * hidden_size_), 1, hidden_size_);
  DumpMatrix("Wb[f]", WbRb_values.data() + (i++ * hidden_size_), 1, hidden_size_);
  DumpMatrix("Wb[c]", WbRb_values.data() + (i++ * hidden_size_), 1, hidden_size_);
  DumpMatrix("Rb[i]", WbRb_values.data() + (i++ * hidden_size_), 1, hidden_size_);
  DumpMatrix("Rb[o]", WbRb_values.data() + (i++ * hidden_size_), 1, hidden_size_);
  DumpMatrix("Rb[f]", WbRb_values.data() + (i++ * hidden_size_), 1, hidden_size_);
  DumpMatrix("Rb[c]", WbRb_values.data() + (i++ * hidden_size_), 1, hidden_size_);

  DumpMatrix("Wb[i]+Rb[i]", bias_WRi_.data(), 1, hidden_size_);
  DumpMatrix("Wb[o]+Rb[o]", bias_WRo_.data(), 1, hidden_size_);
  DumpMatrix("Wb[f]+Rb[f]", bias_WRf_.data(), 1, hidden_size_);
  DumpMatrix("Wb[c]+Rb[c]", bias_WRc_.data(), 1, hidden_size_);
  */
}

template <typename T>
void UniDirectionalLstm<T>::Compute(const gsl::span<const T>& inputs_arg,
                                    const gsl::span<const int>& sequence_lengths_arg,
                                    const int num_directions,
                                    const gsl::span<const T>& input_weights,
                                    const gsl::span<const T>& recurrent_weights,
                                    gsl::span<T>& outputs,
                                    gsl::span<T>& final_hidden_state,
                                    gsl::span<T>& final_cell_state) {
  // copy spans (just T* and size, not data in span) as we may change them
  gsl::span<const T> inputs = inputs_arg;
  gsl::span<const int> sequence_lengths = sequence_lengths_arg;

  // if sequence lengths weren't provided, use internal array and init all to seq_length
  if (sequence_lengths.empty()) {
    sequence_lengths_ = Allocate(allocator_, batch_size_, sequence_lengths_ptr_, true, seq_length_);
    sequence_lengths = sequence_lengths_;
  }

  // LSTM Layer
  gsl::span<T> batched_hidden_state_one_step = batched_hidden0_;
  gsl::span<T> batched_internal_state_prev_one_step = batched_internal_memory_prev_;
  gsl::span<T> batched_internal_state_clipped_one_step = batched_internal_memory_clipped_;

  int output_step_length = batch_size_ * hidden_size_;

  // The bidirectional LSTM wrapper wraps this LSTM class and produces bi-directional output
  // the output has layout [seq,num_direction,batch,neurons].
  // When num_direction is 2, then this class will compute forward or backward LSTM.
  // The outputs corresponds to either [seq,0,batch,neurons] or [seq,1,batch,neurons]
  // Setting output_step_length this way allows writing the output directly without requiring
  // additional memcpy. Note that if direction is kReverse, we write to output_reverse buffer
  // which is then copied to output buffer, and ReverseSequence method handles the step length.
  if (direction_ == kForward && num_directions == 2)
    output_step_length = 2 * batch_size_ * hidden_size_;

  gsl::span<T> original_outputs = outputs;
  const bool output_sequence = !outputs.empty();

  if (direction_ == kReverse) {
    ReverseSequence(inputs, inputs_reverse_, sequence_lengths, seq_length_, batch_size_, input_size_, 1);
    inputs = inputs_reverse_;

    if (output_sequence)
      outputs = outputs_reverse_;
  }

  // DumpMatrix("Input", inputs.data(), seq_length_, batch_size_ * input_size_);

  // Calculate the max and min length
  int32_t max_sequence_length = *std::max_element(sequence_lengths.cbegin(), sequence_lengths.cend());
  int32_t min_sequence_length = std::min(seq_length_, *std::min_element(sequence_lengths.cbegin(),
                                                                        sequence_lengths.cend()));

  ///**************************LSTM Calculations****************************/
  float alpha = 1.0f;
  float beta = 0.0f;  // first call to ComputeGemm zeros out any existing data

  const int hidden_size_x4 = 4 * hidden_size_;
  const int total_rows = max_sequence_length * batch_size_;

  // apply the weights to all the inputs and save to output_IOFC
  ComputeGemm(total_rows, hidden_size_x4, input_size_, alpha,
              inputs.cbegin(), inputs.cend(),
              input_size_,
              input_weights.cbegin(), input_weights.cend(),  // W[iofc]
              input_size_, beta,
              output_iofc_.begin(), output_iofc_.end(),
              hidden_size_x4, mlas_tp_);

  DumpMatrix("Xt*(W[iofc]^T)", output_iofc_.data(), total_rows, hidden_size_x4);

  beta = 1.0f;  // calls to ComputeGemm now add to existing data

  // NOTE: we could refine the bounds checking in the calls below that use these values to instead
  // explicitly check just the range for each iteration, however if it's going to run over
  // it should also run over on the last iteration, so this should be good enough to catch any
  // logic errors causing bounds violations.
  const span_T_iter C_prev_end = batched_internal_state_prev_one_step.end();
  const span_T_iter C_prev_clipped_end = batched_internal_state_clipped_one_step.end();

  if (batch_parallel_) {
    int fused_hidden_rows = batch_size_ / hidden_num_threads_;
    if (batch_size_ % hidden_num_threads_ != 0)
      fused_hidden_rows++;

    // lambda to do all processing on fused_hidden_rows rows
    auto hidden_gemm_and_activations = [&](int row) {
      span_T_const_iter previous_state_end = batched_hidden_state_one_step.cend();

      //handling boundaries
      int local_fused_hidden_rows = fused_hidden_rows;
      if ((row + fused_hidden_rows) > batch_size_)
        local_fused_hidden_rows = batch_size_ - row;

      // these are all batch * hidden_size_ and get updated in-place when running GateComputations so non-const iters
      span_T_iter c_prev = batched_internal_state_prev_one_step.begin() + row * hidden_size_;
      span_T_iter c_prev_clipped = batched_internal_state_clipped_one_step.begin() + row * hidden_size_;

      // hidden state can be provided as input for first step, so need to special case that.
      // after the first step this will switch to the output from the previous step
      span_T_const_iter previous_state = batched_hidden_state_one_step.cbegin() + row * hidden_size_;

      // run through steps sequentially
      for (int step = 0; step < max_sequence_length; step++) {
#if defined(DUMP_MATRIXES)
        const std::string row_str = " [row=" + std::to_string(row) + ",seqno=" + std::to_string(step) + "]";
#endif

        span_T_iter step_out_IOFC = output_iofc_.begin() + (step * batch_size_ + row) * hidden_size_x4;

        // calculate Xt*(W[iofc]^T) + Ht-t*R[iofc]
        ComputeGemm(local_fused_hidden_rows, hidden_size_x4, hidden_size_, alpha,
                    previous_state, previous_state_end,  // Ht-1
                    hidden_size_,
                    recurrent_weights.cbegin(), recurrent_weights.cend(),  // R[iofc]
                    hidden_size_, beta,
                    step_out_IOFC, output_iofc_.end(),  // input contains Xt*(W[iofc]^T)
                    hidden_size_x4, mlas_tp_);

        DumpMatrix("Xt*(W[iofc]^T) + Ht-t*R[iofc]" + row_str,
                   &*step_out_IOFC, local_fused_hidden_rows, hidden_size_x4);

        span_T_iter batched_output;
        span_T_iter batched_output_end;
        if (output_sequence) {
          batched_output = outputs.begin() + step * output_step_length;
          batched_output_end = outputs.end();

        } else {
          batched_output = final_hidden_state.begin();
          batched_output_end = final_hidden_state.end();
        }

        span_T_iter step_out_IOFC_end = step_out_IOFC + local_fused_hidden_rows * hidden_size_x4;
        GateComputations(step_out_IOFC, step_out_IOFC_end,
                         c_prev, C_prev_end,
                         c_prev_clipped, C_prev_clipped_end,
                         batched_output, batched_output_end,
                         sequence_lengths, min_sequence_length, step, row, local_fused_hidden_rows, output_sequence);

        // copy last row to final_cell_state
        for (int lrow = row; lrow < row + local_fused_hidden_rows; ++lrow) {
          if ((step + 1) == sequence_lengths[lrow]) {
            gsl::span<const T> src = batched_internal_memory_prev_.subspan(lrow * hidden_size_, hidden_size_);
            gsl::span<T> dst = final_cell_state.subspan(lrow * hidden_size_, hidden_size_);
            gsl::copy(src, dst);
          }
          if (step == 0 && sequence_lengths[lrow] == 0) {
            auto final_cell_state_dst = final_cell_state.begin() + lrow * hidden_size_;
            std::fill_n(final_cell_state_dst, hidden_size_, T{});
          }
        }

        if (output_sequence) {
          // set to 0 if step >= sequence_length
          for (int lrow = row; lrow < row + local_fused_hidden_rows; lrow++) {
            if (step >= min_sequence_length && step >= sequence_lengths[lrow]) {
              auto output_lrow = outputs.begin() + step * output_step_length + lrow * hidden_size_;
              std::fill_n(output_lrow, hidden_size_, (T)0);
            }
          }
        }

        previous_state = batched_output + row * hidden_size_;
        previous_state_end = batched_output_end;
      }
    };

    ExecuteLambdaInParallel("Processing batch", hidden_gemm_and_activations, batch_size_, fused_hidden_rows, lstm_tp_, logger_);

  } else {
    span_T_const_iter previous_state_end = batched_hidden_state_one_step.cend();

    span_T_iter c_prev = batched_internal_state_prev_one_step.begin();
    span_T_iter c_prev_clipped = batched_internal_state_clipped_one_step.begin();

    // hidden state can be provided as input for first step, so need to special case that.
    // after the first step this will switch to the output from the previous step
    span_T_const_iter previous_state = batched_hidden_state_one_step.cbegin();

    //run through steps sequentially
    for (int step = 0; step < max_sequence_length; step++) {
#if defined(DUMP_MATRIXES)
      const std::string seqno_str = " [seqno=" + std::to_string(step) + "]";
#endif

      DumpMatrix("previous_state" + seqno_str, &*previous_state, batch_size_, hidden_size_);

      span_T_iter step_out_IOFC = output_iofc_.begin() + (step * batch_size_) * hidden_size_x4;

      // calculate Xt*(W[iofc]^T) + Ht-t*R[iofc]
      ComputeGemm(batch_size_, hidden_size_x4, hidden_size_, alpha,
                  previous_state, previous_state_end,  // Ht-1
                  hidden_size_,
                  recurrent_weights.cbegin(), recurrent_weights.cend(),  // R[iofc]
                  hidden_size_, beta,
                  step_out_IOFC, output_iofc_.end(),  // input contains Xt*(W[iofc]^T)
                  hidden_size_x4, mlas_tp_);

      span_T_iter batched_output;
      span_T_iter batched_output_end;
      if (output_sequence) {
        batched_output = outputs.begin() + step * output_step_length;
        batched_output_end = outputs.end();

      } else {
        batched_output = final_hidden_state.begin();
        batched_output_end = final_hidden_state.end();
      }

      span_T_iter step_out_IOFC_end = step_out_IOFC + batch_size_ * hidden_size_x4;
      GateComputations(step_out_IOFC, step_out_IOFC_end,
                       c_prev, C_prev_end,
                       c_prev_clipped, C_prev_clipped_end,
                       batched_output, batched_output_end,
                       sequence_lengths, min_sequence_length, step, 0, batch_size_, output_sequence);

      // copy last row to final_cell_state
      for (int lrow = 0; lrow < batch_size_; lrow++) {
        if ((step + 1) == sequence_lengths[lrow]) {
          gsl::copy(batched_internal_memory_prev_.subspan(lrow * hidden_size_, hidden_size_),
                    final_cell_state.subspan(lrow * hidden_size_, hidden_size_));
        }
        if (step == 0 && sequence_lengths[lrow] == 0) {
          auto final_cell_state_dst = final_cell_state.begin() + lrow * hidden_size_;
          std::fill_n(final_cell_state_dst, hidden_size_, T{});
        }
      }

      if (output_sequence) {
        //set to 0 if step >= sequence_length
        for (int lrow = 0; lrow < batch_size_; lrow++) {
          if (step >= min_sequence_length && step >= sequence_lengths[lrow]) {
            auto dst = outputs.begin() + step * output_step_length + lrow * hidden_size_;
            std::fill_n(dst, hidden_size_, (T)0);
          }
        }
      }

      previous_state = batched_output;
      previous_state_end = batched_output_end;
    }
  }

  for (int i = 0; i < batch_size_; i++) {
    const int seq_len = sequence_lengths[i];
    if (seq_len == 0) {  // zero out final_hidden_state if seq_len == 0
      auto final_hidden_state_dst = final_hidden_state.begin() + i * hidden_size_;
      std::fill_n(final_hidden_state_dst, hidden_size_, T{});
      continue;
    }
    if (output_sequence) {  // copy last output to final_hidden_state
      auto src = outputs.subspan((seq_len - 1) * output_step_length + i * hidden_size_, hidden_size_);
      auto dest = final_hidden_state.subspan(i * hidden_size_, hidden_size_);
      gsl::copy(src, dest);
    }
  }

  // zero any values beyond the evaluated steps
  if (output_sequence && max_sequence_length < seq_length_) {
    if (output_step_length == batch_size_ * hidden_size_) {  // contiguous
      const auto span_to_zero = outputs.subspan(
          max_sequence_length * output_step_length, (seq_length_ - max_sequence_length) * output_step_length);
      std::fill_n(span_to_zero.begin(), span_to_zero.size(), T{});
    } else {
      for (int i = max_sequence_length; i < seq_length_; ++i) {  // non-contiguous
        const auto span_to_zero = outputs.subspan(i * output_step_length, batch_size_ * hidden_size_);
        std::fill_n(span_to_zero.begin(), span_to_zero.size(), T{});
      }
    }
  }

  if (output_sequence && direction_ == Direction::kReverse)
    ReverseSequence<T>(outputs, original_outputs, sequence_lengths, seq_length_,
                       batch_size_, hidden_size_, num_directions);
}

// #define PREVIOUS_BROKEN_VERSION

template <typename T>
void UniDirectionalLstm<T>::GateComputations(span_T_iter& out, span_T_iter& out_end,
                                             span_T_iter& C_prev, const span_T_iter& C_prev_end,  // Ct-1 value not 'ct'. using 'C' for clarity
                                             span_T_iter& C_prev_clipped, const span_T_iter& C_prev_clipped_end,
                                             span_T_iter& batched_output, span_T_iter& batched_output_end,
                                             const gsl::span<const int>& seq_lengths,
                                             const int min_sequence_length,
                                             const int step,
                                             const int row,
                                             const int local_fused_hidden_rows,
                                             bool output_sequence) {
  int hidden_size_x4 = 4 * hidden_size_;

  // Activation gates.
  for (int b = 0; b < local_fused_hidden_rows; b++) {
    if (step >= min_sequence_length && step >= seq_lengths[row + b]) {
      if (output_sequence) {
        auto fill_output = batched_output + (row + b) * hidden_size_;
        std::fill(fill_output, fill_output + hidden_size_, T{});
      }

      continue;
    }

    // std::string row_str = " row[" + std::to_string(row + b) + "]";

    // check that we have hidden_size_x4 left starting at cur_out + b * hidden_size_x4, and get a raw pointer to that
    float* pi = SafeRawPointer<T>(out + b * hidden_size_x4, out_end, hidden_size_x4);
    float* po = pi + hidden_size_;
    float* pf = po + hidden_size_;
    float* pc = pf + hidden_size_;

#ifdef PREVIOUS_BROKEN_VERSION
    float* pCprev_hidden_size = SafeRawPointer<T>(C_prev, C_prev_end, hidden_size_);
#else
    float* pCprev_hidden_size = SafeRawPointer<T>(C_prev + b * hidden_size_, C_prev_end, hidden_size_);
#endif

    // DumpMatrix("C_prev" + row_str, pCprev_hidden_size, 1, hidden_size_);

    // Input Gate
    if (use_peepholes_) {
      deepcpu::elementwise_product(pCprev_hidden_size, SafeRawConstPointer<const T>(peephole_i_, 0, hidden_size_),
                                   pi, hidden_size_);
    }

    const float* pBi = use_bias_ ? SafeRawConstPointer<T>(bias_WRi_, 0, hidden_size_) : nullptr;
    clip_with_bias_ptr_(clip_, pBi, pi, hidden_size_);  // post: pi has input to f() to calculate i
    activation_f_.func(pi, hidden_size_, activation_f_.alpha, activation_f_.beta);
    // DumpMatrix("i" + row_str, pi, 1, hidden_size_);

    // Forget Gate
    if (input_forget_) {
      for (int i = 0; i < hidden_size_; i++)
        pf[i] = 1.0f - pi[i];
    } else {
      if (use_peepholes_) {
        deepcpu::elementwise_product(pCprev_hidden_size, SafeRawConstPointer<const T>(peephole_f_, 0, hidden_size_),
                                     pf, hidden_size_);
      }

      const float* pBf = use_bias_ ? SafeRawConstPointer<T>(bias_WRf_, 0, hidden_size_) : nullptr;
      clip_with_bias_ptr_(clip_, pBf, pf, hidden_size_);
      activation_f_.func(pf, hidden_size_, activation_f_.alpha, activation_f_.beta);
    }

    // DumpMatrix("f" + row_str, pf, 1, hidden_size_);

    // Block Gate
    const float* pBc = use_bias_ ? SafeRawConstPointer<T>(bias_WRc_, 0, hidden_size_) : nullptr;
    clip_with_bias_ptr_(clip_, pBc, pc, hidden_size_);
    activation_g_.func(pc, hidden_size_, activation_g_.alpha, activation_g_.beta);

    // DumpMatrix("c" + row_str, pc, 1, hidden_size_);

    // C_current. use previous C value as input, and update in-place
    float* pC_cur = pCprev_hidden_size;
#ifdef PREVIOUS_BROKEN_VERSION
    deepcpu::merge_lstm_gates_to_memory(pCprev_hidden_size + b * hidden_size_, pi, pf, pc, pCprev_hidden_size + b * hidden_size_, hidden_size_);
    // DumpMatrix("C", pCprev_hidden_size + b * hidden_size_, 1, hidden_size_);
#else
    deepcpu::merge_lstm_gates_to_memory(pCprev_hidden_size, pi, pf, pc, pC_cur, hidden_size_);
    // DumpMatrix("C", pC_cur, 1, hidden_size_);
#endif

    // Output Gate
    if (use_peepholes_)
      deepcpu::elementwise_product(pCprev_hidden_size, SafeRawConstPointer<const T>(peephole_o_, 0, hidden_size_),
                                   po, hidden_size_);

    // calculate 'ot'
    const float* pBo = use_bias_ ? SafeRawConstPointer<T>(bias_WRo_, 0, hidden_size_) : nullptr;
    clip_with_bias_ptr_(clip_, pBo, po, hidden_size_);
    activation_f_.func(po, hidden_size_, activation_f_.alpha, activation_f_.beta);
    // DumpMatrix("o" + row_str, po, 1, hidden_size_);

    // calculate 'Ht'
    float* pH = SafeRawPointer<T>(batched_output + row * hidden_size_ + b * hidden_size_,
                                  batched_output_end, hidden_size_);

    // the C_prev_clipped location is not actually used as input - it's temporary storage for writing
    // the clipped Ct value to, before calling h(). As such a) it could just be a local variable
    // of std::vector<float> with size of hidden_size_, b) the previous version wasn't 'broken' by never
    // incrementing what C_prev_clipped pointed to.
#ifdef PREVIOUS_BROKEN_VERSION
    float* pC_prev_clipped = SafeRawPointer<T>(C_prev_clipped, C_prev_clipped_end, hidden_size_);
#else
    float* pC_prev_clipped = SafeRawPointer<T>(C_prev_clipped + b * hidden_size_, C_prev_clipped_end, hidden_size_);
#endif

    activation_h_.func(pC_cur, pC_prev_clipped, po, pH, hidden_size_, activation_h_.alpha, activation_h_.beta);

    // DumpMatrix("H" + row_str, pH, 1, hidden_size_);
  }

  auto num_rows = local_fused_hidden_rows - row;
  std::string rows_str = " rows[" + std::to_string(row) + ".." + std::to_string(num_rows) + "]";

  DumpMatrix("i" + rows_str, &*out, num_rows, hidden_size_, 0, hidden_size_x4);
  DumpMatrix("o" + rows_str, &*out, num_rows, hidden_size_, 1 * hidden_size_, hidden_size_x4);
  DumpMatrix("f" + rows_str, &*out, num_rows, hidden_size_, 2 * hidden_size_, hidden_size_x4);
  DumpMatrix("c" + rows_str, &*out, num_rows, hidden_size_, 3 * hidden_size_, hidden_size_x4);
  DumpMatrix("C" + rows_str, &*C_prev, num_rows, hidden_size_);  // Ct overwrites the input C_prev value
  DumpMatrix("H" + rows_str, &*batched_output, num_rows, hidden_size_);
}

template <typename T>
void UniDirectionalLstm<T>::SetNumThreads() {
  int threads = std::thread::hardware_concurrency() - 1;

  if (threads < 1)
    threads = 1;

  hidden_num_threads_ = threads;
  batch_parallel_ = false;

  // for readability of the below logic
  const auto num_rows = batch_size_;
  const auto num_columns = hidden_size_;

  // parallelize by partitioning the batch rows
  if (num_rows > 4 || (num_rows >= 2 && num_columns <= 256)) {
    batch_parallel_ = true;
    VLOGS(logger_, 1) << "Hidden Threads : " << hidden_num_threads_;
  }
}

}  // namespace detail
}  // namespace onnxruntime
