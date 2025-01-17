// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/rnn/gru_io_utils.h"

namespace onnxruntime::gru {

namespace {

template <typename T>
rnn::detail::GemmWeights<T> LoadWeights(const Tensor* weights, const int index) {
  // index represents the direction of the weight to be loaded.
  // For example,
  //   in a uni-directional gru, index can only ever be 0.
  //   in a bi-directional gru, index 0 represents forward weights and index 1 represents backward weights
  const auto& weights_shape = weights->Shape();
  const auto* weights_data = weights->Data<T>();
  const size_t weights_size_per_direction = SafeInt<size_t>(weights_shape[1]) * weights_shape[2];
  return rnn::detail::GemmWeights<T>(index, weights_data, weights_size_per_direction, rnn::detail::PackedWeights());
}

void ValidateGRUInputs(const Tensor* X, const Tensor* W, const Tensor* R, const Tensor* B,
                       const Tensor* SL, const Tensor* H0, const int directions, const int hidden_size) {
  const auto& X_shape = X->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 3U, "Input X must have 3 dimensions only. Actual:", X_shape);
  const int batch_size = narrow<int>(X_shape[1]);
  const int input_size = narrow<int>(X_shape[2]);

  const auto& W_shape = W->Shape();
  ORT_ENFORCE(W_shape.NumDimensions() == 3U &&
                  narrow<int>(W_shape[0]) == directions &&
                  narrow<int>(W_shape[1]) == 3 * hidden_size &&
                  narrow<int>(W_shape[2]) == input_size,
              "Input W must have shape {", directions, ", ", 3 * hidden_size, ", ", input_size, "}. Actual:", W_shape);

  const auto& R_shape = R->Shape();
  ORT_ENFORCE(R_shape.NumDimensions() == 3U &&
                  narrow<int>(R_shape[0]) == directions &&
                  narrow<int>(R_shape[1]) == 3 * hidden_size &&
                  narrow<int>(R_shape[2]) == hidden_size,
              "Input R must have shape {", directions, ", ", 3 * hidden_size, ", ", hidden_size, "}. Actual:", R_shape);

  if (B != nullptr) {
    const auto& B_shape = B->Shape();
    ORT_ENFORCE(B_shape.NumDimensions() == 2U &&
                    narrow<int>(B_shape[0]) == directions &&
                    narrow<int>(B_shape[1]) == 6 * hidden_size,
                "Input B must have shape {", directions, ", ", 6 * hidden_size, "}. Actual:", B_shape);
  }

  ORT_ENFORCE(!SL,
              "Sequence lengths input tensor (implying varying length input sequence length) "
              "is not supported for GRUTraining and GRUGrad. Fixed sequence length can be inferred from the "
              "input tensor shape.");

  if (H0 != nullptr) {
    const auto& H0_shape = H0->Shape();
    ORT_ENFORCE(H0_shape.NumDimensions() == 3U &&
                    narrow<int>(H0_shape[0]) == directions &&
                    narrow<int>(H0_shape[1]) == batch_size &&
                    narrow<int>(H0_shape[2]) == hidden_size,
                "Input H0 must have shape {", directions, ", ", batch_size, ", ", hidden_size, "}. Actual:", H0_shape);
  }
}

void ValidateGRUGradInputs(const Tensor* X, const Tensor* HAll, const Tensor* ZRH,
                           const Tensor* grad_HAll, const Tensor* grad_Ht,
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

  if (ZRH != nullptr) {
    const auto& ZRH_shape = ZRH->Shape();
    ORT_ENFORCE(ZRH_shape.NumDimensions() == 4U &&
                    narrow<int>(ZRH_shape[0]) == sequence_length &&
                    narrow<int>(ZRH_shape[1]) == directions &&
                    narrow<int>(ZRH_shape[2]) == batch_size &&
                    narrow<int>(ZRH_shape[3]) == 3 * hidden_size,
                "Input ZRH must have shape {", sequence_length, ", ", directions, ", ", batch_size, ", ", 3 * hidden_size, "}. Actual:", ZRH_shape);
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
}

}  // namespace

GRUAttributes::GRUAttributes(const OpKernelInfo& info) {
  std::string direction_str = info.GetAttrOrDefault<std::string>("direction", "forward");
  direction = rnn::detail::MakeDirection(direction_str);
  ORT_ENFORCE(direction == rnn::detail::Direction::kForward,
              "GRUTraining and GRUGrad kernel only supports the forward direction for now. Provided direction: ",
              direction);
  num_directions = 1;

  std::vector<std::string> activation_func_names = info.GetAttrsOrDefault<std::string>("activations");
  const std::vector<float> activation_func_alphas = info.GetAttrsOrDefault<float>("activation_alpha");
  const std::vector<float> activation_func_betas = info.GetAttrsOrDefault<float>("activation_beta");
  if (activation_func_names.empty()) {
    activation_func_names.emplace_back("sigmoid");
    activation_func_names.emplace_back("tanh");
  }

  ORT_ENFORCE(activation_func_names.size() == static_cast<size_t>(num_directions) * 2U,
              "Unexpected number of activation function names provided. Expected: ", num_directions * 2, " Actual: ", activation_func_names.size());

  ORT_ENFORCE(activation_func_names[0] == "sigmoid",
              "GRUTraining and GRUGrad only support the sigmoid function for the f activation parameter.");
  ORT_ENFORCE(activation_func_names[1] == "tanh",
              "GRUTraining and GRUGrad only support the tanh function for the g activation parameter.");
  activation_funcs = rnn::detail::ActivationFuncs(activation_func_names,
                                                  activation_func_alphas,
                                                  activation_func_betas);

  clip = info.GetAttrOrDefault<float>("clip", std::numeric_limits<float>::max());
  ORT_ENFORCE(clip == std::numeric_limits<float>::max(), "Clip is not supported for GRUTraining and GRUGrad yet.");

  linear_before_reset = info.GetAttrOrDefault<int64_t>("linear_before_reset", 0) == 1;

  int64_t hidden_size_int64_t;
  ORT_ENFORCE(info.GetAttr("hidden_size", &hidden_size_int64_t).IsOK() && hidden_size_int64_t > 0);
  hidden_size = narrow<int>(hidden_size_int64_t);
}

template <typename T>
GRUInputs<T>::GRUInputs(OpKernelContext* context, const int directions, const int hidden_size) {
  const Tensor* X = context->Input<Tensor>(0);   // input sequence [seq_length, batch_size, input_size]
  const Tensor* W = context->Input<Tensor>(1);   // weights [directions, 3 * hidden_size, input_size]
  const Tensor* R = context->Input<Tensor>(2);   // recurrence weights [directions, 3 * hidden_size, hidden_size]
  const Tensor* B = context->Input<Tensor>(3);   // bias [directions, 6 * hidden_size]
  const Tensor* SL = context->Input<Tensor>(4);  // sequence lengths [batch_size]
  const Tensor* H0 = context->Input<Tensor>(5);  // initial hidden state [directions, batch_size, hidden_size]

  ValidateGRUInputs(X, W, R, B, SL, H0, directions, hidden_size);

  input = X->DataAsSpan<T>();
  shape = InputShape{
      narrow<int>(X->Shape()[0]),  // sequence length
      narrow<int>(X->Shape()[1]),  // batch size
      narrow<int>(X->Shape()[2]),  // input size
  };

  const size_t zr_size_per_direction = 2 * static_cast<size_t>(hidden_size) * static_cast<size_t>(hidden_size);
  const size_t h_size_per_direction = static_cast<size_t>(hidden_size) * static_cast<size_t>(hidden_size);

  weights = LoadWeights<T>(W, 0);
  auto recurrence_weights = R->DataAsSpan<T>();
  auto recurrence_weights_zr_span = recurrence_weights.subspan(0, zr_size_per_direction);
  auto recurrence_weights_h_span = recurrence_weights.subspan(zr_size_per_direction, h_size_per_direction);
  recurrence_weights_zr.Init(0, recurrence_weights_zr_span.data(),
                             recurrence_weights_zr_span.size(),
                             rnn::detail::PackedWeights(), nullptr);
  recurrence_weights_h.Init(0, recurrence_weights_h_span.data(),
                            recurrence_weights_h_span.size(),
                            rnn::detail::PackedWeights(), nullptr);
  bias = B ? B->DataAsSpan<T>() : gsl::span<const T>();
  sequence_lengths = SL ? SL->DataAsSpan<int>() : gsl::span<const int>();
  initial_hidden_state = H0 ? H0->DataAsSpan<T>() : gsl::span<const T>();
}

template <typename T>
GRUOutputs<T>::GRUOutputs(OpKernelContext* context, const int directions, const int sequence_length,
                          const int batch_size, const int hidden_size) {
  const TensorShape HAll_shape{sequence_length, directions, batch_size, hidden_size};  // [seq_length, directions, batch_size, hidden_size]
  Tensor* HAll = context->Output(0, HAll_shape);                                       // all hidden states
  const TensorShape Ht_shape{directions, batch_size, hidden_size};                     // [directions, batch_size, hidden_size]
  Tensor* Ht = context->Output(1, Ht_shape);                                           // final hidden state
  const int64_t hidden_size_x3 = static_cast<int64_t>(3) * static_cast<int64_t>(hidden_size);
  const TensorShape ZRH_shape{sequence_length, directions, batch_size, hidden_size_x3};  // [seq_length, directions, batch_size, 3 * hidden_size]
  Tensor* ZRH = context->Output(2, ZRH_shape);                                           // zrh gate computations

  AllocatorPtr alloc;
  ORT_THROW_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  ORT_ENFORCE(HAll, "All hidden states output is required for GRUTraining to compute gradients.");
  all_hidden_states = HAll->MutableDataAsSpan<T>();
  const size_t final_state_size = static_cast<size_t>(directions) * static_cast<size_t>(batch_size) *
                                  static_cast<size_t>(hidden_size);
  final_hidden_state = Ht ? Ht->MutableDataAsSpan<T>()
                          : rnn::detail::Allocate(alloc, final_state_size,
                                                  h_final_ptr_, true, static_cast<T>(0));

  ORT_ENFORCE(ZRH, "z, r, h gate computation output is required for GRUTraining to compute gradients.");
  zrh = ZRH->MutableDataAsSpan<T>();
}

template <typename T>
GRUGradInputs<T>::GRUGradInputs(OpKernelContext* context, const int directions, const int hidden_size) {
  const Tensor* X = context->Input<Tensor>(0);          // input sequence [seq_length, batch_size, input_size]
  const Tensor* W = context->Input<Tensor>(1);          // weights [directions, 3 * hidden_size, input_size]
  const Tensor* R = context->Input<Tensor>(2);          // recurrence weights [directions, 3 * hidden_size, hidden_size]
  const Tensor* B = context->Input<Tensor>(3);          // bias [directions, 6 * hidden_size]
  const Tensor* SL = context->Input<Tensor>(4);         // sequence lengths [batch_size]
  const Tensor* H0 = context->Input<Tensor>(5);         // initial hidden state [directions, batch_size, hidden_size]
  const Tensor* HAll = context->Input<Tensor>(6);       // all hidden states [seq_length, directions, batch_size, hidden_size]
  const Tensor* ZRH = context->Input<Tensor>(7);        // zrh gate computations [seq_length, directions, batch_size, 3 * hidden_size]
  const Tensor* grad_HAll = context->Input<Tensor>(8);  // grad w.r.t all hidden states [seq_length, directions, batch_size, hidden_size]
  const Tensor* grad_Ht = context->Input<Tensor>(9);    // grad w.r.t final hidden state [directions, batch_size, hidden_size]

  ValidateGRUInputs(X, W, R, B, SL, H0, directions, hidden_size);
  ValidateGRUGradInputs(X, HAll, ZRH, grad_HAll, grad_Ht, directions, hidden_size);

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
  bias = B ? B->DataAsSpan<T>() : gsl::span<const T>();
  sequence_lengths = SL ? SL->DataAsSpan<int>() : gsl::span<const int>();
  const size_t initial_state_size = static_cast<size_t>(directions) * static_cast<size_t>(shape.batch_size) *
                                    static_cast<size_t>(hidden_size);
  initial_hidden_state = H0 ? H0->DataAsSpan<T>()
                            : rnn::detail::Allocate(alloc, initial_state_size,
                                                    initial_hidden_state_ptr_, true, static_cast<T>(0));

  ORT_ENFORCE(HAll, "All hidden states input to GRUGrad must exist to compute the gradients.");
  all_hidden_states = HAll->DataAsSpan<T>();

  ORT_ENFORCE(ZRH, "z, r, h gate computation input to GRUGrad must exist to compute the gradients.");
  zrh = ZRH->DataAsSpan<T>();

  grad_all_hidden_states = grad_HAll ? grad_HAll->DataAsSpan<T>() : gsl::span<const T>();
  grad_final_hidden_state = grad_Ht ? grad_Ht->DataAsSpan<T>() : gsl::span<const T>();
}

template <typename T>
GRUGradOutputs<T>::GRUGradOutputs(OpKernelContext* context, const int directions, const int sequence_length,
                                  const int batch_size, const int hidden_size, const int input_size) {
  // Outputs of the gradient kernel
  const TensorShape dX_shape{sequence_length, batch_size, input_size};  // [seq_length, batch_size, input_size]
  Tensor* dX = context->Output(0, dX_shape);                            // gradient w.r.t to the input X
  const int64_t hidden_sizex3 = static_cast<int64_t>(3) * static_cast<int64_t>(hidden_size);
  const TensorShape dW_shape{directions, hidden_sizex3, input_size};   // [directions, 3 * hidden_size, input_size]
  Tensor* dW = context->Output(1, dW_shape);                           // gradient w.r.t to the input weights W
  const TensorShape dR_shape{directions, hidden_sizex3, hidden_size};  // [directions, 3 * hidden_size, hidden_size]
  Tensor* dR = context->Output(2, dR_shape);                           // gradient w.r.t to the recurrence weights R
  const int64_t hidden_sizex6 = static_cast<int64_t>(6) * static_cast<int64_t>(hidden_size);
  const TensorShape dB_shape{directions, hidden_sizex6};             // [directions, 6 * hidden_size]
  Tensor* dB = context->Output(3, dB_shape);                         // gradient w.r.t to the bias
  const TensorShape dH0_shape{directions, batch_size, hidden_size};  // [directions, batch_size, hidden_size]
  Tensor* dH0 = context->Output(4, dH0_shape);                       // gradient w.r.t to the initial hidden state

  AllocatorPtr alloc;
  ORT_THROW_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  grad_input = dX ? dX->MutableDataAsSpan<T>() : gsl::span<T>();
  grad_weights = dW->MutableDataAsSpan<T>();
  grad_recurrence_weights = dR->MutableDataAsSpan<T>();
  grad_bias = dB ? dB->MutableDataAsSpan<T>() : gsl::span<T>();
  const size_t initial_state_size = static_cast<size_t>(directions) * static_cast<size_t>(batch_size) *
                                    static_cast<size_t>(hidden_size);
  grad_initial_hidden_state = dH0 ? dH0->MutableDataAsSpan<T>()
                                  : rnn::detail::Allocate(alloc, initial_state_size,
                                                          grad_initial_hidden_state_ptr_, true, static_cast<T>(0));
}

template struct GRUInputs<float>;
template struct GRUOutputs<float>;
template struct GRUGradInputs<float>;
template struct GRUGradOutputs<float>;

}  // namespace onnxruntime::gru
