// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/rnn/lstm_grad_compute.h"

namespace onnxruntime::lstm {

namespace {

void ElementwiseProduct(const float* op1, const float* op2, float* dest, int size) {
  for (int i = 0; i < size; i++)
    dest[i] = op1[i] * op2[i];
}

}  // namespace

template <typename T>
LSTMGradImpl<T>::LSTMGradImpl(int sequence_length, int batch_size, int hidden_size, int input_size,
                              concurrency::ThreadPool* thread_pool, AllocatorPtr allocator)
    : sequence_length_(sequence_length),
      batch_size_(batch_size),
      hidden_size_(hidden_size),
      input_size_(input_size),
      thread_pool_(thread_pool),
      allocator_(allocator) {
  const size_t hidden_size_x4 = 4U * static_cast<size_t>(hidden_size_);
  const size_t weight_size = 4U * static_cast<size_t>(hidden_size_) * static_cast<size_t>(input_size_);
  const size_t recurrence_weight_size = 4U * static_cast<size_t>(hidden_size_) * static_cast<size_t>(hidden_size_);
  grad_a_span_ = rnn::detail::Allocate(allocator_, hidden_size_x4, grad_a_ptr_, true, static_cast<T>(0));
  grad_Ct2_span_ = rnn::detail::Allocate(allocator_, hidden_size_, grad_Ct2_ptr_, true, static_cast<T>(0));
  grad_W_span_ = rnn::detail::Allocate(allocator_, weight_size, grad_W_ptr_, true, static_cast<T>(0));
  grad_R_span_ = rnn::detail::Allocate(allocator_, recurrence_weight_size, grad_R_ptr_, true, static_cast<T>(0));
}

template <typename T>
void LSTMGradImpl<T>::ComputeGradient(const LSTMGradInputs<T>& inputs, LSTMGradOutputs<T>& outputs) {
  using namespace rnn::detail;

  // A note on the memory layout of buffers used in this function:
  // iofc buffer:
  //  ____________________________________________________________
  // | | | | | | | | |...| | | | |...| | | | | | | | |...| | | | |
  // |i|o|f|c|i|o|f|c|...|i|o|f|c|...|i|o|f|c|i|o|f|c|...|i|o|f|c|
  // |_|_|_|_|_|_|_|_|___|_|_|_|_|___|_|_|_|_|_|_|_|_|___|_|_|_|_|
  // ->|H|<- Each i, o, f, c block is of size hidden size
  // <--B1--><--B2-->....<--Bn-->....<--B1--><--B2-->....<--Bn--> Batch size is n
  // <------------S1------------>....<------------St-------------> Sequence length is t
  //
  // all hidden states buffer:
  //  ___________________________________________
  // |  |  |   |  |  |  |...|  |...|  |  |...|  |
  // |B1|B2|...|Bn|B1|B2|...|Bn|...|B1|B2|...|Bn|
  // |__|__|___|__|__|__|___|__|___|__|__|___|__|
  // -->|H |<-- Each block is of size hidden size. Each B represents an index of the batch. So batch size is n
  // <----S1-----><----S2----->....<----St-----> Sequence length is t
  // Every buffer having a sequnence length dimension are structured as the above two.
  //
  // Weight buffers:
  //  _______________________________________________
  // |         |           |            |           |
  // |   Wi    |     Wo    |     Wf     |     Wc    |
  // |_________|___________|____________|___________|
  // <---HxI---> Each block is hidden size x input size long
  // Each block represents either i, o, f, or c weights.
  // Every weight buffer is similarly structured

  const size_t hidden_size_x4 = 4U * static_cast<size_t>(hidden_size_);
  const size_t hidden_size_x8 = 8U * static_cast<size_t>(hidden_size_);
  const size_t weight_size = 4U * static_cast<size_t>(hidden_size_) * static_cast<size_t>(input_size_);
  const size_t recurrence_weight_size = 4U * static_cast<size_t>(hidden_size_) * static_cast<size_t>(hidden_size_);
  const size_t hidden_size_x3 = 3U * static_cast<size_t>(hidden_size_);
  const size_t hidden_sizexinput_size = static_cast<size_t>(hidden_size_) * static_cast<size_t>(input_size_);
  const size_t hidden_sizexhidden_size = static_cast<size_t>(hidden_size_) * static_cast<size_t>(hidden_size_);

  const gsl::span<const T>& W = inputs.weights;
  const float* Wi = SafeRawPointer<const T>(W.begin(), W.end(), weight_size);
  const float* Wo = Wi + hidden_sizexinput_size;
  const float* Wf = Wo + hidden_sizexinput_size;
  const float* Wc = Wf + hidden_sizexinput_size;

  const gsl::span<const T>& R = inputs.recurrence_weights;
  const float* Ri = SafeRawPointer<const T>(R.begin(), R.end(), recurrence_weight_size);
  const float* Ro = Ri + hidden_sizexhidden_size;
  const float* Rf = Ro + hidden_sizexhidden_size;
  const float* Rc = Rf + hidden_sizexhidden_size;

  const bool grad_input_required = !outputs.grad_input.empty();
  const bool grad_weights_required = !outputs.grad_weights.empty();
  const bool grad_recurrence_weights_required = !outputs.grad_recurrence_weights.empty();
  const bool grad_bias_required = !outputs.grad_bias.empty();
  const bool grad_peephole_weights_required = !outputs.grad_peephole_weights.empty();

  auto& grad_W = outputs.grad_weights;
  if (grad_weights_required) {
    std::fill_n(grad_W.data(), grad_W.size(), static_cast<T>(0));
  }
  float* grad_Wi = grad_weights_required
                       ? SafeRawPointer<T>(grad_W.begin(), grad_W.end(), weight_size)
                       : nullptr;
  float* grad_Wo = grad_weights_required ? grad_Wi + hidden_sizexinput_size : nullptr;
  float* grad_Wf = grad_weights_required ? grad_Wo + hidden_sizexinput_size : nullptr;
  float* grad_Wc = grad_weights_required ? grad_Wf + hidden_sizexinput_size : nullptr;

  auto& grad_R = outputs.grad_recurrence_weights;
  if (grad_recurrence_weights_required) {
    std::fill_n(grad_R.data(), grad_R.size(), static_cast<T>(0));
  }
  float* grad_Ri = grad_recurrence_weights_required
                       ? SafeRawPointer<T>(grad_R.begin(), grad_R.end(), recurrence_weight_size)
                       : nullptr;
  float* grad_Ro = grad_recurrence_weights_required ? grad_Ri + hidden_sizexhidden_size : nullptr;
  float* grad_Rf = grad_recurrence_weights_required ? grad_Ro + hidden_sizexhidden_size : nullptr;
  float* grad_Rc = grad_recurrence_weights_required ? grad_Rf + hidden_sizexhidden_size : nullptr;

  // Fill grad bias with 0s since they are used as accumulators
  auto& grad_B = outputs.grad_bias;
  if (grad_bias_required) {
    std::fill_n(grad_B.data(), grad_B.size(), static_cast<T>(0));
  }
  float* grad_Wbi = grad_bias_required ? SafeRawPointer<T>(grad_B.begin(), grad_B.end(), hidden_size_x8) : nullptr;
  float* grad_Wbo = grad_bias_required ? grad_Wbi + hidden_size_ : nullptr;
  float* grad_Wbf = grad_bias_required ? grad_Wbo + hidden_size_ : nullptr;
  float* grad_Wbc = grad_bias_required ? grad_Wbf + hidden_size_ : nullptr;
  float* grad_Rbi = grad_bias_required ? grad_Wbc + hidden_size_ : nullptr;
  float* grad_Rbo = grad_bias_required ? grad_Rbi + hidden_size_ : nullptr;
  float* grad_Rbf = grad_bias_required ? grad_Rbo + hidden_size_ : nullptr;
  float* grad_Rbc = grad_bias_required ? grad_Rbf + hidden_size_ : nullptr;

  // Fill grad peepholes with 0s since they are used as accumulators
  auto& grad_P = outputs.grad_peephole_weights;
  if (grad_peephole_weights_required) {
    std::fill_n(grad_P.data(), grad_P.size(), static_cast<T>(0));
  }
  float* grad_pi = grad_peephole_weights_required ? SafeRawPointer<T>(grad_P.begin(), grad_P.end(), hidden_size_x3)
                                                  : nullptr;
  float* grad_po = grad_peephole_weights_required ? grad_pi + hidden_size_ : nullptr;
  float* grad_pf = grad_peephole_weights_required ? grad_po + hidden_size_ : nullptr;

  constexpr float alpha = 1.0f;
  // Gemm accumulation results in incorrect values. For now, use custom accumulation logic.
  constexpr float weight_beta = 0.0f;

  float* grad_ai = SafeRawPointer<T>(grad_a_span_.begin(), grad_a_span_.end(), hidden_size_x4);
  float* grad_ao = grad_ai + hidden_size_;
  float* grad_af = grad_ao + hidden_size_;
  float* grad_ac = grad_af + hidden_size_;

  float* grad_Wi_local = SafeRawPointer<T>(grad_W_span_.begin(), grad_W_span_.end(), weight_size);
  float* grad_Wo_local = grad_Wi_local + hidden_sizexinput_size;
  float* grad_Wf_local = grad_Wo_local + hidden_sizexinput_size;
  float* grad_Wc_local = grad_Wf_local + hidden_sizexinput_size;

  float* grad_Ri_local = SafeRawPointer<T>(grad_R_span_.begin(), grad_R_span_.end(), recurrence_weight_size);
  float* grad_Ro_local = grad_Ri_local + hidden_sizexhidden_size;
  float* grad_Rf_local = grad_Ro_local + hidden_sizexhidden_size;
  float* grad_Rc_local = grad_Rf_local + hidden_sizexhidden_size;

  float* grad_Ct2 = SafeRawPointer<T>(grad_Ct2_span_.begin(), grad_Ct2_span_.end(), hidden_size_);

  std::fill_n(outputs.grad_initial_cell_state.data(), outputs.grad_initial_cell_state.size(), static_cast<T>(0));
  std::fill_n(outputs.grad_initial_hidden_state.data(), outputs.grad_initial_hidden_state.size(), static_cast<T>(0));

  for (int idx = 0; idx < batch_size_; ++idx) {
    const size_t hidden_sizexidx = static_cast<size_t>(idx) * static_cast<size_t>(hidden_size_);
    const float* grad_Cfinal = inputs.grad_final_cell_state.empty()
                                   ? nullptr
                                   : SafeRawPointer<const T>(inputs.grad_final_cell_state.begin() + hidden_sizexidx,
                                                             inputs.grad_final_cell_state.end(), hidden_size_);

    // Accumulate grad_Cfinal into grad_initial_cell_state
    float* grad_Ct = SafeRawPointer<T>(outputs.grad_initial_cell_state.begin() + hidden_sizexidx,
                                       outputs.grad_initial_cell_state.end(), hidden_size_);
    if (grad_Cfinal)
      deepcpu::elementwise_sum1(grad_Cfinal, grad_Ct, hidden_size_);

    const float* grad_Hfinal = inputs.grad_final_hidden_state.empty()
                                   ? nullptr
                                   : SafeRawPointer<const T>(inputs.grad_final_hidden_state.begin() + hidden_sizexidx,
                                                             inputs.grad_final_hidden_state.end(), hidden_size_);
    float* grad_Ht = SafeRawPointer<T>(outputs.grad_initial_hidden_state.begin() + hidden_sizexidx,
                                       outputs.grad_initial_hidden_state.end(), hidden_size_);
    // The LSTM outputs: all hidden states, final hidden state and final cell states
    // In addition to all hidden states being used, the final hidden state could also be used in the remainder of the
    // graph. So, accumulate the final hidden state gradient in gradHt.
    // Later, we will also accumulate from the gradient of all hideen states as that could also be used in the graph.
    if (grad_Hfinal)
      deepcpu::elementwise_sum1(grad_Hfinal, grad_Ht, hidden_size_);

    for (int t = sequence_length_ - 1; t >= 0; --t) {
      const size_t iofc_offset = (static_cast<size_t>(t) * static_cast<size_t>(batch_size_) +
                                  static_cast<size_t>(idx)) *
                                 hidden_size_x4;
      auto iofc = inputs.iofc.begin() + iofc_offset;
      const float* it = SafeRawPointer<const T>(iofc, inputs.iofc.end(), hidden_size_x4);
      const float* ot = it + hidden_size_;
      const float* ft = ot + hidden_size_;
      const float* ct = ft + hidden_size_;

      // Retrieve current C, previous H and previous C from the given inputs.
      // Assume these inputs always exist. Caller of this function must ensure that.
      const size_t CH_offset = (static_cast<size_t>(t) * static_cast<size_t>(batch_size_) +
                                static_cast<size_t>(idx)) *
                               static_cast<size_t>(hidden_size_);
      const size_t CHtminus1_offset = t > 0 ? ((static_cast<size_t>(t) - 1U) * static_cast<size_t>(batch_size_) +
                                               static_cast<size_t>(idx)) *
                                                  hidden_size_
                                            : 0U;
      const float* Ct = SafeRawPointer<const T>(inputs.all_cell_states.begin() + CH_offset,
                                                inputs.all_cell_states.end(), hidden_size_);
      const float* Ctminus1 = t > 0 ? SafeRawPointer<const T>(inputs.all_cell_states.begin() + CHtminus1_offset,
                                                              inputs.all_cell_states.end(), hidden_size_)
                                    : SafeRawPointer<const T>(inputs.initial_cell_state.begin() + hidden_sizexidx,
                                                              inputs.initial_cell_state.end(), hidden_size_);
      const float* grad_Ht2 = inputs.grad_all_hidden_states.empty()
                                  ? nullptr
                                  : SafeRawPointer<const T>(
                                        inputs.grad_all_hidden_states.begin() + CH_offset,
                                        inputs.grad_all_hidden_states.end(), hidden_size_);
      // Accumulate the gradient from the gradients of all hidden states for this sequence index and batch index.
      if (grad_Ht2)
        deepcpu::elementwise_sum1(grad_Ht2, grad_Ht, hidden_size_);

      // Ct2_tilde = tanh(Ct2)
      MlasComputeTanh(Ct, grad_Ct2, hidden_size_);

      // Ht = ot (.) Ct2_tilde
      // dL/dot = dL/dHt (.) Ct2_tilde ---------- (1)
      ElementwiseProduct(grad_Ht, grad_Ct2, grad_ao, hidden_size_);

      // Ht = ot (.) Ct2_tilde
      // dL/dCt2_tilde = dL/dHt (.) ot ---------- (2)
      // dL/dCt2 = dL/dCt2_tilde (.) (1 - (tanh(Ct))^2) ---------- (3)
      for (int h = 0; h < hidden_size_; ++h) {
        grad_Ct2[h] = grad_Ht[h] * ot[h] * (1 - grad_Ct2[h] * grad_Ct2[h]);
      }

      // Ct -> multiplex gate -> Ct1
      //                      -> Ct2
      // dL/dCt = dL/dCt1 + dL/dCt2 ---------- (4)
      deepcpu::elementwise_sum1(grad_Ct2, grad_Ct, hidden_size_);

      // Ct = ft (.) Ct-1 + it (.) ct
      // dL/dit = dL/dCt (.) ct ---------- (5)
      ElementwiseProduct(grad_Ct, ct, grad_ai, hidden_size_);

      // Ct = ft (.) Ct-1 + it (.) ct
      // dL/dct = dL/dCt (.) it ---------- (6)
      ElementwiseProduct(grad_Ct, it, grad_ac, hidden_size_);

      // Ct = ft (.) Ct-1 + it (.) ct
      // dL/dft = dL/dCt (.) Ct-1 ---------- (7)
      ElementwiseProduct(grad_Ct, Ctminus1, grad_af, hidden_size_);

      // Ct = ft (.) Ct-1 + it (.) ct
      // dL/dCt-1 = dL/dCt (.) ft ---------- (8)
      // Note that peephole weights do not impact the backward propagation to Ct-1 and Ct
      // as noted in the paper.
      ElementwiseProduct(grad_Ct, ft, grad_Ct, hidden_size_);

      // ct = tanh(ac)
      // dL/dac = dL/dct (.) (1 - (tanh(ac))^2) ---------- (9)
      for (int h = 0; h < hidden_size_; ++h) {
        grad_ac[h] = grad_ac[h] * (1 - ct[h] * ct[h]);
      }

      // it = sigmoid(ai)
      // dL/dai = dL/dit (.) (sigmoid(ai) * (1 - sigmoid(ai))) ---------- (10)
      for (int h = 0; h < hidden_size_; ++h) {
        grad_ai[h] = grad_ai[h] * (it[h] * (1 - it[h]));
      }

      // ft = sigmoid(af)
      // dL/daf = dL/dft (.) (sigmoid(af) * (1 - sigmoid(af))) ---------- (11)
      for (int h = 0; h < hidden_size_; ++h) {
        grad_af[h] = grad_af[h] * (ft[h] * (1 - ft[h]));
      }

      // ot = sigmoid(ao)
      // dL/dao = dL/dot (.) (sigmoid(ao) * (1 - sigmoid(ao))) ---------- (12)
      for (int h = 0; h < hidden_size_; ++h) {
        grad_ao[h] = grad_ao[h] * (ot[h] * (1 - ot[h]));
      }

      if (grad_input_required) {
        const size_t X_offset = (static_cast<size_t>(t) * static_cast<size_t>(batch_size_) +
                                 static_cast<size_t>(idx)) *
                                static_cast<size_t>(input_size_);
        // Xt -> multiplex gate -> Xti
        //                      -> Xto
        //                      -> Xtf
        //                      -> Xtc
        // dL/dXt = dL/dXti  + dL/dXto + dL/dXtf + dL/dXtc ---------- (13)
        float input_beta = 0.0f;

        // ai = Xti * Wi^T + Ht-1i * Ri^T + Pi (.) Ct-1 + Wbi + Rbi
        // dL/dXti = dL/dai * Wi ---------- (14)
        // [1, input_size_] = [1, hidden_size_] * [hidden_size_, input_size_]
        // M = 1, N = input_size_, K = hidden_size_
        float* grad_Xt = SafeRawPointer<T>(outputs.grad_input.begin() + X_offset,
                                           outputs.grad_input.end(), input_size_);
        ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, input_size_,
                                         hidden_size_, alpha, grad_ai, Wi, input_beta, grad_Xt, thread_pool_);

        // ao = Xto * Wo^T + Ht-1o * Ro^T + Po (.) Ct + Wbo + Rbo
        // dL/dXto = dL/dao * Wo ---------- (15)
        // [1, input_size_] = [1, hidden_size_] * [hidden_size_, input_size_]
        // M = 1, N = input_size_, K = hidden_size_
        input_beta = 1.0f;
        ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, input_size_,
                                         hidden_size_, alpha, grad_ao, Wo, input_beta, grad_Xt, thread_pool_);

        // af = Xtf * Wf^T + Ht-1f * Rf^T + Pf (.) Ct-1 + Wbf + Rbf
        // dL/dXtf = dL/daf * Wf ---------- (16)
        // [1, input_size_] = [1, hidden_size_] * [hidden_size_, input_size_]
        // M = 1, N = input_size_, K = hidden_size_
        ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, input_size_,
                                         hidden_size_, alpha, grad_af, Wf, input_beta, grad_Xt, thread_pool_);

        // ac = Xtc * Wc^T + Ht-1c * Rc^T + Wbc + Rbc
        // dL/dXtc = dL/dac * Wc ---------- (17)
        // [1, input_size_] = [1, hidden_size_] * [hidden_size_, input_size_]
        // M = 1, N = input_size_, K = hidden_size_
        ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, input_size_,
                                         hidden_size_, alpha, grad_ac, Wc, input_beta, grad_Xt, thread_pool_);
      }

      if (grad_weights_required) {
        const size_t X_offset = (static_cast<size_t>(t) * static_cast<size_t>(batch_size_) +
                                 static_cast<size_t>(idx)) *
                                static_cast<size_t>(input_size_);
        // ai = Xti * Wi^T + Ht-1i * Ri^T + Pi (.) Ct-1 + Wbi + Rbi
        // dL/dWi = dL/dai^T * Xti ---------- (18)
        // [hidden_size_, input_size_] = [1, hidden_size_]^T * [1, input_size_]
        // M = hidden_size_, N = input_size_, K = 1
        const float* Xt = SafeRawPointer<const T>(inputs.input.begin() + X_offset,
                                                  inputs.input.end(), input_size_);
        ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, input_size_,
                                         1, alpha, grad_ai, Xt, weight_beta, grad_Wi_local, thread_pool_);
        // Note that the weight beta is always 0. So, we must accumulate ourselves.
        deepcpu::elementwise_sum1(grad_Wi_local, grad_Wi, hidden_size_ * input_size_);

        // ao = Xto * Wo^T + Ht-1o * Ro^T + Po (.) Ct + Wbo + Rbo
        // dL/dWo = dL/dao^T * Xto ---------- (19)
        // [hidden_size_, input_size_] = [1, hidden_size_]^T * [1, input_size_]
        // M = hidden_size_, N = input_size_, K = 1
        ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, input_size_,
                                         1, alpha, grad_ao, Xt, weight_beta, grad_Wo_local, thread_pool_);
        // Note that the weight beta is always 0. So, we must accumulate ourselves.
        deepcpu::elementwise_sum1(grad_Wo_local, grad_Wo, hidden_size_ * input_size_);

        // af = Xtf * Wf^T + Ht-1f * Rf^T + Pf (.) Ct-1 + Wbf + Rbf
        // dL/dWf = dL/daf^T * Xtf ---------- (20)
        // [hidden_size_, input_size_] = [1, hidden_size_]^T * [1, input_size_]
        // M = hidden_size_, N = input_size_, K = 1
        ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, input_size_,
                                         1, alpha, grad_af, Xt, weight_beta, grad_Wf_local, thread_pool_);
        // Note that the weight beta is always 0. So, we must accumulate ourselves.
        deepcpu::elementwise_sum1(grad_Wf_local, grad_Wf, hidden_size_ * input_size_);

        // ac = Xtc * Wc^T + Ht-1c * Rc^T + Wbc + Rbc
        // dL/dWc = dL/dac^T * Xtc ---------- (21)
        // [hidden_size_, input_size_] = [1, hidden_size_]^T * [1, input_size_]
        // M = hidden_size_, N = input_size_, K = 1
        ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, input_size_,
                                         1, alpha, grad_ac, Xt, weight_beta, grad_Wc_local, thread_pool_);
        // Note that the weight beta is always 0. So, we must accumulate ourselves.
        deepcpu::elementwise_sum1(grad_Wc_local, grad_Wc, hidden_size_ * input_size_);
      }

      // all_hidden_states must always exist
      // initial_hidden_state may not exist
      if (grad_recurrence_weights_required) {
        const float* Htminus1 = t > 0 ? SafeRawPointer<const T>(
                                            inputs.all_hidden_states.begin() + CHtminus1_offset,
                                            inputs.all_hidden_states.end(), hidden_size_)
                                      : SafeRawPointer<const T>(
                                            inputs.initial_hidden_state.begin() + hidden_sizexidx,
                                            inputs.initial_hidden_state.end(), hidden_size_);

        // ai = Xti * Wi^T + Ht-1i * Ri^T + Pi (.) Ct-1 + Wbi + Rbi
        // dL/dRi = dL/dai^T * Ht-1i ---------- (22)
        // [hidden_size_, hidden_size_] = [1, hidden_size_]^T * [1, hidden_size_]
        // M = hidden_size_, N = hidden_size_, K = 1
        ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, hidden_size_,
                                         1, alpha, grad_ai, Htminus1, weight_beta, grad_Ri_local, thread_pool_);
        // Note that the weight beta is always 0. So, we must accumulate ourselves.
        deepcpu::elementwise_sum1(grad_Ri_local, grad_Ri, hidden_size_ * hidden_size_);

        // ao = Xto * Wo^T + Ht-1o * Ro^T + Po (.) Ct + Wbo + Rbo
        // dL/dRo = dL/dao^T * Ht-1o ---------- (23)
        // [hidden_size_, hidden_size_] = [1, hidden_size_]^T * [1, hidden_size_]
        // M = hidden_size_, N = hidden_size_, K = 1
        ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, hidden_size_,
                                         1, alpha, grad_ao, Htminus1, weight_beta, grad_Ro_local, thread_pool_);
        // Note that the weight beta is always 0. So, we must accumulate ourselves.
        deepcpu::elementwise_sum1(grad_Ro_local, grad_Ro, hidden_size_ * hidden_size_);

        // af = Xtf * Wf^T + Ht-1f * Rf^T + Pf (.) Ct-1 + Wbf + Rbf
        // dL/dRf = dL/daf^T * Ht-1f ---------- (24)
        // [hidden_size_, hidden_size_] = [1, hidden_size_]^T * [1, hidden_size_]
        // M = hidden_size_, N = hidden_size_, K = 1
        ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, hidden_size_,
                                         1, alpha, grad_af, Htminus1, weight_beta, grad_Rf_local, thread_pool_);
        // Note that the weight beta is always 0. So, we must accumulate ourselves.
        deepcpu::elementwise_sum1(grad_Rf_local, grad_Rf, hidden_size_ * hidden_size_);

        // ac = Xtc * Wc^T + Ht-1c * Rc^T + Wbc + Rbc
        // dL/dRc = dL/dac^T * Ht-1c ---------- (25)
        // [hidden_size_, hidden_size_] = [1, hidden_size_]^T * [1, hidden_size_]
        // M = hidden_size_, N = hidden_size_, K = 1
        ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, hidden_size_,
                                         1, alpha, grad_ac, Htminus1, weight_beta, grad_Rc_local, thread_pool_);
        // Note that the weight beta is always 0. So, we must accumulate ourselves.
        deepcpu::elementwise_sum1(grad_Rc_local, grad_Rc, hidden_size_ * hidden_size_);
      }

      if (grad_bias_required) {
        // ai = Xti * Wi^T + Ht-1i * Ri^T + Pi (.) Ct-1 + Wbi + Rbi
        // dL/dWbi = dL/dai ---------- (26)
        deepcpu::elementwise_sum1(grad_ai, grad_Wbi, hidden_size_);

        // ai = Xti * Wi^T + Ht-1i * Ri^T + Pi (.) Ct-1 + Wbi + Rbi
        // dL/dRbi = dL/dai ---------- (27)
        deepcpu::elementwise_sum1(grad_ai, grad_Rbi, hidden_size_);

        // ao = Xto * Wo^T + Ht-1o * Ro^T + Po (.) Ct + Wbo + Rbo
        // dL/dWbo = dL/dao ---------- (28)
        deepcpu::elementwise_sum1(grad_ao, grad_Wbo, hidden_size_);

        // ao = Xto * Wo^T + Ht-1o * Ro^T + Po (.) Ct + Wbo + Rbo
        // dL/dRbo = dL/dao ---------- (29)
        deepcpu::elementwise_sum1(grad_ao, grad_Rbo, hidden_size_);

        // af = Xtf * Wf^T + Ht-1f * Rf^T + Pf (.) Ct-1 + Wbf + Rbf
        // dL/dWbf = dL/daf ---------- (30)
        deepcpu::elementwise_sum1(grad_af, grad_Wbf, hidden_size_);

        // af = Xtf * Wf^T + Ht-1f * Rf^T + Pf (.) Ct-1 + Wbf + Rbf
        // dL/dRbf = dL/daf ---------- (31)
        deepcpu::elementwise_sum1(grad_af, grad_Rbf, hidden_size_);

        // ac = Xtc * Wc^T + Ht-1c * Rc^T + Wbc + Rbc
        // dL/dWbc = dL/dac ---------- (32)
        deepcpu::elementwise_sum1(grad_ac, grad_Wbc, hidden_size_);

        // ac = Xtc * Wc^T + Ht-1c * Rc^T + Wbc + Rbc
        // dL/dRbc = dL/dac ---------- (33)
        deepcpu::elementwise_sum1(grad_ac, grad_Rbc, hidden_size_);
      }

      if (grad_peephole_weights_required) {
        // ai = Xti * Wi^T + Ht-1i * Ri^T + Pi (.) Ct-1 + Wbi + Rbi
        // dL/dPi = dL/dai (.) Ct-1 ---------- (34)
        deepcpu::elementwise_product(grad_ai, Ctminus1, grad_pi, hidden_size_);

        // ao = Xto * Wo^T + Ht-1o * Ro^T + Po (.) Ct + Wbo + Rbo
        // dL/dPo = dL/dao (.) Ct ---------- (35)
        deepcpu::elementwise_product(grad_ao, Ct, grad_po, hidden_size_);

        // af = Xtf * Wf^T + Ht-1f * Rf^T + Pf (.) Ct-1 + Wbf + Rbf
        // dL/dPf = dL/daf (.) Ct-1 ---------- (36)
        deepcpu::elementwise_product(grad_af, Ctminus1, grad_pf, hidden_size_);
      }

      // Ht-1 -> multiplex gate -> Ht-1i
      //                        -> Ht-1o
      //                        -> Ht-1f
      //                        -> Ht-1c
      // dL/dHt-1 = dL/dHt-1i  + dL/dHt-1o + dL/dHt-1f + dL/dHt-1c ---------- (37)
      float recurrence_input_beta = 0.0f;

      // ai = Xti * Wi^T + Ht-1i * Ri^T + Pi (.) Ct-1 + Wbi + Rbi
      // dL/dHt-1i = dL/dai * Ri ---------- (38)
      // [1, hidden_size_] = [1, hidden_size_] * [hidden_size_, hidden_size_]
      // M = 1, N = hidden_size_, K = hidden_size_
      ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, hidden_size_,
                                       hidden_size_, alpha, grad_ai, Ri, recurrence_input_beta, grad_Ht, thread_pool_);

      recurrence_input_beta = 1.0f;

      // ao = Xto * Wo^T + Ht-1o * Ro^T + Po (.) Ct + Wbo + Rbo
      // dL/dHt-1o = dL/dao * Ro ---------- (39)
      // [1, hidden_size_] = [1, hidden_size_] * [hidden_size_, hidden_size_]
      // M = 1, N = hidden_size_, K = hidden_size_
      ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, hidden_size_,
                                       hidden_size_, alpha, grad_ao, Ro, recurrence_input_beta, grad_Ht, thread_pool_);

      // af = Xtf * Wf^T + Ht-1f * Rf^T + Pf (.) Ct-1 + Wbf + Rbf
      // dL/dHt-1f = dL/daf * Rf ---------- (40)
      // [1, hidden_size_] = [1, hidden_size_] * [hidden_size_, hidden_size_]
      // M = 1, N = hidden_size_, K = hidden_size_
      ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, hidden_size_,
                                       hidden_size_, alpha, grad_af, Rf, recurrence_input_beta, grad_Ht, thread_pool_);

      // ac = Xtc * Wc^T + Ht-1c * Rc^T + Wbc + Rbc
      // dL/dHt-1c = dL/dac * Rc ---------- (41)
      // [1, hidden_size_] = [1, hidden_size_] * [hidden_size_, hidden_size_]
      // M = 1, N = hidden_size_, K = hidden_size_
      ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, hidden_size_,
                                       hidden_size_, alpha, grad_ac, Rc, recurrence_input_beta, grad_Ht, thread_pool_);
    }
  }
}

template class LSTMGradImpl<float>;

}  // namespace onnxruntime::lstm
