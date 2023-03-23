// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/rnn/lstm_io_utils.h"

namespace onnxruntime::lstm {

namespace {

template <typename T>
rnn::detail::GemmWeights<T> LoadWeights(const Tensor* weights, const int index) {
  // index represents the direction of the weight to be loaded.
  // For example,
  //   in a uni-directional lstm, index can only ever be 0.
  //   in a bi-directional lstm, index 0 represents forward weights and index 1 represents backward weights
  const auto& weights_shape = weights->Shape();
  const auto* weights_data = weights->Data<T>();
  const size_t weights_size_per_direction = SafeInt<size_t>(weights_shape[1]) * weights_shape[2];
  return rnn::detail::GemmWeights<T>(index, weights_data, weights_size_per_direction, rnn::detail::PackedWeights());
}

void ValidateLSTMInputs(const Tensor* X, const Tensor* W, const Tensor* R, const Tensor* B,
                        const Tensor* SL, const Tensor* H0, const Tensor* C0, const Tensor* P,
                        const int directions, const int hidden_size) {
  const auto& X_shape = X->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 3U, "Input X must have 3 dimensions only. Actual:", X_shape);
  const int batch_size = narrow<int>(X_shape[1]);
  const int input_size = narrow<int>(X_shape[2]);

  const auto& W_shape = W->Shape();
  ORT_ENFORCE(W_shape.NumDimensions() == 3U &&
                  narrow<int>(W_shape[0]) == directions &&
                  narrow<int>(W_shape[1]) == 4 * hidden_size &&
                  narrow<int>(W_shape[2]) == input_size,
              "Input W must have shape {", directions, ", ", 4 * hidden_size, ", ", input_size, "}. Actual:", W_shape);

  const auto& R_shape = R->Shape();
  ORT_ENFORCE(R_shape.NumDimensions() == 3U &&
                  narrow<int>(R_shape[0]) == directions &&
                  narrow<int>(R_shape[1]) == 4 * hidden_size &&
                  narrow<int>(R_shape[2]) == hidden_size,
              "Input R must have shape {", directions, ", ", 4 * hidden_size, ", ", hidden_size, "}. Actual:", R_shape);

  if (B != nullptr) {
    const auto& B_shape = B->Shape();
    ORT_ENFORCE(B_shape.NumDimensions() == 2U &&
                    narrow<int>(B_shape[0]) == directions &&
                    narrow<int>(B_shape[1]) == 8 * hidden_size,
                "Input B must have shape {", directions, ", ", 8 * hidden_size, "}. Actual:", B_shape);
  }

  ORT_ENFORCE(!SL,
              "Sequence lengths input tensor (implying varying length input sequence length) "
              "is not supported for LSTMTraining and LSTMGrad. Fixed sequence length can be inferred from the "
              "input tensor shape.");

  if (H0 != nullptr) {
    const auto& H0_shape = H0->Shape();
    ORT_ENFORCE(H0_shape.NumDimensions() == 3U &&
                    narrow<int>(H0_shape[0]) == directions &&
                    narrow<int>(H0_shape[1]) == batch_size &&
                    narrow<int>(H0_shape[2]) == hidden_size,
                "Input H0 must have shape {", directions, ", ", batch_size, ", ", hidden_size, "}. Actual:", H0_shape);
  }

  if (C0 != nullptr) {
    const auto& C0_shape = C0->Shape();
    ORT_ENFORCE(C0_shape.NumDimensions() == 3U &&
                    narrow<int>(C0_shape[0]) == directions &&
                    narrow<int>(C0_shape[1]) == batch_size &&
                    narrow<int>(C0_shape[2]) == hidden_size,
                "Input C0 must have shape {", directions, ", ", batch_size, ", ", hidden_size, "}. Actual:", C0_shape);
  }

  if (P != nullptr) {
    const auto& P_shape = P->Shape();
    ORT_ENFORCE(P_shape.NumDimensions() == 2U &&
                    narrow<int>(P_shape[0]) == directions &&
                    narrow<int>(P_shape[1]) == 3 * hidden_size,
                "Input P must have shape {", directions, ", ", 3 * hidden_size, "}. Actual:", P_shape);
  }
}

void ValidateLSTMGradInputs(const Tensor* X, const Tensor* HAll, const Tensor* CAll, const Tensor* IOFC,
                            const Tensor* grad_HAll, const Tensor* grad_Ht, const Tensor* grad_Ct,
                            const int directions, const int hidden_size) {
  const auto& X_shape = X->Shape();
  const int sequence_length = narrow<int>(X_shape[0]);
  const int batch_size = narrow<int>(X_shape[1]);

  if (HAll != nullptr) {
    const auto& HAll_shape = HAll->Shape();
    ORT_ENFORCE(HAll_shape.NumDimensions() == 4U &&
                    narrow<int>(HAll_shape[0]) == sequence_length &&
                    narrow<int>(HAll_shape[1]) == directions &&
                    narrow<int>(HAll_shape[2]) == batch_size &&
                    narrow<int>(HAll_shape[3]) == hidden_size,
                "Input HAll must have shape {", sequence_length, ", ", directions, ", ", batch_size, ", ", hidden_size, "}. Actual:", HAll_shape);
  }

  if (CAll != nullptr) {
    const auto& CAll_shape = CAll->Shape();
    ORT_ENFORCE(CAll_shape.NumDimensions() == 4U &&
                    narrow<int>(CAll_shape[0]) == sequence_length &&
                    narrow<int>(CAll_shape[1]) == directions &&
                    narrow<int>(CAll_shape[2]) == batch_size &&
                    narrow<int>(CAll_shape[3]) == hidden_size,
                "Input CAll must have shape {", sequence_length, ", ", directions, ", ", batch_size, ", ", hidden_size, "}. Actual:", CAll_shape);
  }

  if (IOFC != nullptr) {
    const auto& IOFC_shape = IOFC->Shape();
    ORT_ENFORCE(IOFC_shape.NumDimensions() == 4U &&
                    narrow<int>(IOFC_shape[0]) == sequence_length &&
                    narrow<int>(IOFC_shape[1]) == directions &&
                    narrow<int>(IOFC_shape[2]) == batch_size &&
                    narrow<int>(IOFC_shape[3]) == 4 * hidden_size,
                "Input IOFC must have shape {", sequence_length, ", ", directions, ", ", batch_size, ", ", 4 * hidden_size, "}. Actual:", IOFC_shape);
  }

  if (grad_HAll != nullptr) {
    const auto& grad_HAll_shape = grad_HAll->Shape();
    ORT_ENFORCE(grad_HAll_shape.NumDimensions() == 4U &&
                    narrow<int>(grad_HAll_shape[0]) == sequence_length &&
                    narrow<int>(grad_HAll_shape[1]) == directions &&
                    narrow<int>(grad_HAll_shape[2]) == batch_size &&
                    narrow<int>(grad_HAll_shape[3]) == hidden_size,
                "Input grad_HAll must have shape {", sequence_length, ", ", directions, ", ", batch_size,
                ", ", hidden_size, "}. Actual:", grad_HAll_shape);
  }

  if (grad_Ht != nullptr) {
    const auto& grad_Ht_shape = grad_Ht->Shape();
    ORT_ENFORCE(grad_Ht_shape.NumDimensions() == 3U &&
                    narrow<int>(grad_Ht_shape[0]) == directions &&
                    narrow<int>(grad_Ht_shape[1]) == batch_size &&
                    narrow<int>(grad_Ht_shape[2]) == hidden_size,
                "Input grad_Ht must have shape {", directions, ", ", batch_size, ", ", hidden_size, "}. Actual:", grad_Ht_shape);
  }

  if (grad_Ct != nullptr) {
    const auto& grad_Ct_shape = grad_Ct->Shape();
    ORT_ENFORCE(grad_Ct_shape.NumDimensions() == 3U &&
                    narrow<int>(grad_Ct_shape[0]) == directions &&
                    narrow<int>(grad_Ct_shape[1]) == batch_size &&
                    narrow<int>(grad_Ct_shape[2]) == hidden_size,
                "Input grad_Ct must have shape {", directions, ", ", batch_size, ", ", hidden_size, "}. Actual:", grad_Ct_shape);
  }
}

}  // namespace

LSTMAttributes::LSTMAttributes(const OpKernelInfo& info) {
  std::string direction_str = info.GetAttrOrDefault<std::string>("direction", "forward");
  direction = rnn::detail::MakeDirection(direction_str);
  ORT_ENFORCE(direction == rnn::detail::Direction::kForward,
              "LSTM and LSTMGrad kernel only supports the forward direction for now. Provided direction: ", direction);
  num_directions = 1;

  std::vector<std::string> activation_func_names = info.GetAttrsOrDefault<std::string>("activations");
  const std::vector<float> activation_func_alphas = info.GetAttrsOrDefault<float>("activation_alpha");
  const std::vector<float> activation_func_betas = info.GetAttrsOrDefault<float>("activation_beta");
  if (activation_func_names.empty()) {
    activation_func_names.emplace_back("sigmoid");
    activation_func_names.emplace_back("tanh");
    activation_func_names.emplace_back("tanh");
  }

  ORT_ENFORCE(activation_func_names.size() == static_cast<size_t>(num_directions) * 3U,
              "Unexpected number of activation function names provided. Expected: ", num_directions * 3, " Actual: ", activation_func_names.size());

  ORT_ENFORCE(activation_func_names[0] == "sigmoid",
              "LSTM and LSTMGrad only support the sigmoid function for the f activation parameter.");
  ORT_ENFORCE(activation_func_names[1] == "tanh",
              "LSTM and LSTMGrad only support the tanh function for the g activation parameter.");
  ORT_ENFORCE(activation_func_names[2] == "tanh",
              "LSTM and LSTMGrad only support the tanh function for the h activation parameter.");
  activation_funcs = rnn::detail::ActivationFuncs(activation_func_names,
                                                  activation_func_alphas,
                                                  activation_func_betas);

  clip = info.GetAttrOrDefault<float>("clip", std::numeric_limits<float>::max());
  ORT_ENFORCE(clip == std::numeric_limits<float>::max(), "Clip is not supported for LSTM and LSTMGrad yet.");

  input_forget = narrow<int>(info.GetAttrOrDefault<int64_t>("input_forget", 0));
  ORT_ENFORCE(input_forget == 0, "Combining the input and forget gates is not yet supported in the LSTMGrad kernel.");

  int64_t hidden_size_int64_t;
  ORT_ENFORCE(info.GetAttr("hidden_size", &hidden_size_int64_t).IsOK() && hidden_size_int64_t > 0);
  hidden_size = narrow<int>(hidden_size_int64_t);
}

template <typename T>
LSTMInputs<T>::LSTMInputs(OpKernelContext* context, const int directions, const int hidden_size) {
  const Tensor* X = context->Input<Tensor>(0);   // input sequence [seq_length, batch_size, input_size]
  const Tensor* W = context->Input<Tensor>(1);   // weights [directions, 4 * hidden_size, input_size]
  const Tensor* R = context->Input<Tensor>(2);   // recurrence weights [directions, 4 * hidden_size, hidden_size]
  const Tensor* B = context->Input<Tensor>(3);   // bias [directions, 8 * hidden_size]
  const Tensor* SL = context->Input<Tensor>(4);  // sequence lengths [batch_size]
  const Tensor* H0 = context->Input<Tensor>(5);  // initial hidden state [directions, batch_size, hidden_size]
  const Tensor* C0 = context->Input<Tensor>(6);  // initial cell state [directions, batch_size, hidden_size]
  const Tensor* P = context->Input<Tensor>(7);   // peephole weights [directions, 3 * hidden_size]

  ValidateLSTMInputs(X, W, R, B, SL, H0, C0, P, directions, hidden_size);

  input = X->DataAsSpan<T>();
  shape = InputShape{
      narrow<int>(X->Shape()[0]),  // sequence length
      narrow<int>(X->Shape()[1]),  // batch size
      narrow<int>(X->Shape()[2]),  // input size
  };

  weights = LoadWeights<T>(W, 0);
  recurrence_weights = LoadWeights<T>(R, 0);
  bias = B ? B->DataAsSpan<T>() : gsl::span<const T>();
  sequence_lengths = SL ? SL->DataAsSpan<int>() : gsl::span<const int>();
  initial_hidden_state = H0 ? H0->DataAsSpan<T>() : gsl::span<const T>();
  initial_cell_state = C0 ? C0->DataAsSpan<T>() : gsl::span<const T>();
  peephole_weights = P ? P->DataAsSpan<T>() : gsl::span<const T>();
}

template <typename T>
LSTMOutputs<T>::LSTMOutputs(OpKernelContext* context, const int directions, const int sequence_length,
                            const int batch_size, const int hidden_size) {
  const TensorShape HAll_shape{sequence_length, directions, batch_size, hidden_size};  // [seq_length, directions, batch_size, hidden_size]
  Tensor* HAll = context->Output(0, HAll_shape);                                       // all hidden states
  const TensorShape Ht_shape{directions, batch_size, hidden_size};                     // [directions, batch_size, hidden_size]
  Tensor* Ht = context->Output(1, Ht_shape);                                           // final hidden state
  const TensorShape Ct_shape{directions, batch_size, hidden_size};                     // [directions, batch_size, hidden_size]
  Tensor* Ct = context->Output(2, Ct_shape);                                           // final cell state
  const TensorShape CAll_shape{sequence_length, directions, batch_size, hidden_size};  // [seq_length, directions, batch_size, hidden_size]
  Tensor* CAll = context->Output(3, CAll_shape);                                       // all cell states
  const int64_t hidden_size_x4 = static_cast<int64_t>(4) * static_cast<int64_t>(hidden_size);
  const TensorShape IOFC_shape{sequence_length, directions, batch_size, hidden_size_x4};  // [seq_length, directions, batch_size, 4 * hidden_size]
  Tensor* IOFC = context->Output(4, IOFC_shape);                                          // iofc gate computations

  AllocatorPtr alloc;
  ORT_THROW_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  ORT_ENFORCE(HAll, "All hidden states output is required for LSTMTraining to compute gradients.");
  all_hidden_states = HAll->MutableDataAsSpan<T>();
  const size_t final_state_size = static_cast<size_t>(directions) * static_cast<size_t>(batch_size) *
                                  static_cast<size_t>(hidden_size);
  final_hidden_state = Ht ? Ht->MutableDataAsSpan<T>()
                          : rnn::detail::Allocate(alloc, final_state_size,
                                                  h_final_ptr_, true, static_cast<T>(0));
  final_cell_state = Ct ? Ct->MutableDataAsSpan<T>()
                        : rnn::detail::Allocate(alloc, final_state_size,
                                                c_final_ptr_, true, static_cast<T>(0));

  ORT_ENFORCE(CAll, "All cell states output is required for LSTMTraining to compute gradients.");
  all_cell_states = CAll->MutableDataAsSpan<T>();

  ORT_ENFORCE(IOFC, "i, o, f, c gate computation output is required for LSTMTraining to compute gradients.");
  iofc = IOFC->MutableDataAsSpan<T>();
}

template <typename T>
LSTMGradInputs<T>::LSTMGradInputs(OpKernelContext* context, const int directions, const int hidden_size) {
  const Tensor* X = context->Input<Tensor>(0);          // input sequence [seq_length, batch_size, input_size]
  const Tensor* W = context->Input<Tensor>(1);          // weights [directions, 4 * hidden_size, input_size]
  const Tensor* R = context->Input<Tensor>(2);          // recurrence weights [directions, 4 * hidden_size, hidden_size]
  const Tensor* SL = context->Input<Tensor>(3);         // sequence lengths [batch_size]
  const Tensor* H0 = context->Input<Tensor>(4);         // initial hidden state [directions, batch_size, hidden_size]
  const Tensor* C0 = context->Input<Tensor>(5);         // initial cell state [directions, batch_size, hidden_size]
  const Tensor* HAll = context->Input<Tensor>(6);       // all hidden states [seq_length, directions, batch_size, hidden_size]
  const Tensor* CAll = context->Input<Tensor>(7);       // all cell states [seq_length, directions, batch_size, hidden_size]
  const Tensor* IOFC = context->Input<Tensor>(8);       // iofc gate computations [seq_length, directions, batch_size, 4 * hidden_size]
  const Tensor* grad_HAll = context->Input<Tensor>(9);  // grad w.r.t all hidden states [seq_length, directions, batch_size, hidden_size]
  const Tensor* grad_Ht = context->Input<Tensor>(10);   // grad w.r.t final hidden state [directions, batch_size, hidden_size]
  const Tensor* grad_Ct = context->Input<Tensor>(11);   // grad w.r.t final cell state [directions, batch_size, hidden_size]

  ValidateLSTMInputs(X, W, R, nullptr, SL, H0, C0, nullptr, directions, hidden_size);
  ValidateLSTMGradInputs(X, HAll, CAll, IOFC, grad_HAll, grad_Ht, grad_Ct, directions, hidden_size);

  AllocatorPtr alloc;
  ORT_THROW_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  input = X->DataAsSpan<T>();
  shape = InputShape{
      narrow<int>(X->Shape()[0]),  // sequence length
      narrow<int>(X->Shape()[1]),  // batch size
      narrow<int>(X->Shape()[2]),  // input size
  };

  weights = W->DataAsSpan<T>();
  recurrence_weights = R->DataAsSpan<T>();
  sequence_lengths = SL ? SL->DataAsSpan<int>() : gsl::span<const int>();
  const size_t initial_state_size = static_cast<size_t>(directions) * static_cast<size_t>(shape.batch_size) *
                                    static_cast<size_t>(hidden_size);
  initial_hidden_state = H0 ? H0->DataAsSpan<T>()
                            : rnn::detail::Allocate(alloc, initial_state_size,
                                                    initial_hidden_state_ptr_, true, static_cast<T>(0));
  initial_cell_state = C0 ? C0->DataAsSpan<T>()
                          : rnn::detail::Allocate(alloc, initial_state_size,
                                                  initial_cell_state_ptr_, true, static_cast<T>(0));

  ORT_ENFORCE(HAll, "All hidden states input to LSTMGrad must exist to compute the gradients.");
  all_hidden_states = HAll->DataAsSpan<T>();

  ORT_ENFORCE(CAll, "All cell states input to LSTMGrad must exist to compute the gradients.");
  all_cell_states = CAll->DataAsSpan<T>();

  ORT_ENFORCE(IOFC, "i, o, f, c gate computation input to LSTMGrad must exist to compute the gradients.");
  iofc = IOFC->DataAsSpan<T>();

  grad_all_hidden_states = grad_HAll ? grad_HAll->DataAsSpan<T>() : gsl::span<const T>();
  grad_final_hidden_state = grad_Ht ? grad_Ht->DataAsSpan<T>() : gsl::span<const T>();
  grad_final_cell_state = grad_Ct ? grad_Ct->DataAsSpan<T>() : gsl::span<const T>();
}

template <typename T>
LSTMGradOutputs<T>::LSTMGradOutputs(OpKernelContext* context, const int directions, const int sequence_length,
                                    const int batch_size, const int hidden_size, const int input_size) {
  // Outputs of the gradient kernel
  const TensorShape dX_shape{sequence_length, batch_size, input_size};  // [seq_length, batch_size, input_size]
  Tensor* dX = context->Output(0, dX_shape);                            // gradient w.r.t to the input X
  const int64_t hidden_sizex4 = static_cast<int64_t>(4) * static_cast<int64_t>(hidden_size);
  const TensorShape dW_shape{directions, hidden_sizex4, input_size};   // [directions, 4 * hidden_size, input_size]
  Tensor* dW = context->Output(1, dW_shape);                           // gradient w.r.t to the input weights W
  const TensorShape dR_shape{directions, hidden_sizex4, hidden_size};  // [directions, 4 * hidden_size, hidden_size]
  Tensor* dR = context->Output(2, dR_shape);                           // gradient w.r.t to the recurrence weights R
  const int64_t hidden_sizex8 = static_cast<int64_t>(8) * static_cast<int64_t>(hidden_size);
  const TensorShape dB_shape{directions, hidden_sizex8};             // [directions, 8 * hidden_size]
  Tensor* dB = context->Output(3, dB_shape);                         // gradient w.r.t to the bias
  const TensorShape dH0_shape{directions, batch_size, hidden_size};  // [directions, batch_size, hidden_size]
  Tensor* dH0 = context->Output(4, dH0_shape);                       // gradient w.r.t to the initial hidden state
  const TensorShape dC0_shape{directions, batch_size, hidden_size};  // [directions, batch_size, hidden_size]
  Tensor* dC0 = context->Output(5, dC0_shape);                       // gradient w.r.t to the initial cell state
  const int64_t hidden_sizex3 = static_cast<int64_t>(3) * static_cast<int64_t>(hidden_size);
  const TensorShape dP_shape{directions, hidden_sizex3};  // [directions, 3 * hidden_size]
  Tensor* dP = context->Output(6, dP_shape);              // gradient w.r.t to the peephole weights

  AllocatorPtr alloc;
  ORT_THROW_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  grad_input = dX ? dX->MutableDataAsSpan<T>() : gsl::span<T>();
  grad_weights = dW->MutableDataAsSpan<T>();
  grad_recurrence_weights = dR->MutableDataAsSpan<T>();
  grad_bias = dB ? dB->MutableDataAsSpan<T>() : gsl::span<T>();
  const size_t initial_state_size = static_cast<size_t>(directions) * static_cast<size_t>(batch_size) *
                                    static_cast<size_t>(hidden_size);
  grad_initial_cell_state = dC0 ? dC0->MutableDataAsSpan<T>()
                                : rnn::detail::Allocate(alloc, initial_state_size,
                                                        grad_initial_cell_state_ptr_, true, static_cast<T>(0));
  grad_initial_hidden_state = dH0 ? dH0->MutableDataAsSpan<T>()
                                  : rnn::detail::Allocate(alloc, initial_state_size,
                                                          grad_initial_hidden_state_ptr_, true, static_cast<T>(0));
  grad_peephole_weights = dP ? dP->MutableDataAsSpan<T>() : gsl::span<T>();
}

template struct LSTMInputs<float>;
template struct LSTMOutputs<float>;
template struct LSTMGradInputs<float>;
template struct LSTMGradOutputs<float>;

}  // namespace onnxruntime::lstm
