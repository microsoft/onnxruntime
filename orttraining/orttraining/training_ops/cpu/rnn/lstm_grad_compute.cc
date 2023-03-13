#include "orttraining/training_ops/cpu/rnn/lstm_grad_compute.h"

namespace onnxruntime::lstm {

namespace {

void elementwise_product(const float* op1, const float* op2, float* dest, int size) {
  for (int i = 0; i < size; i++)
    dest[i] = op1[i] * op2[i];
}

}  // namespace

template <typename T>
LSTMGradImpl<T>::LSTMGradImpl(size_t sequence_length, size_t batch_size, size_t hidden_size, size_t input_size,
                              concurrency::ThreadPool* thread_pool, AllocatorPtr allocator)
    : sequence_length_(sequence_length),
      batch_size_(batch_size),
      hidden_size_(hidden_size),
      input_size_(input_size),
      thread_pool_(thread_pool),
      allocator_(allocator) {
  grad_a_span_ = rnn::detail::Allocate(allocator_, 4 * hidden_size_, grad_a_ptr_, true, static_cast<T>(0));
  grad_Ct2_span_ = rnn::detail::Allocate(allocator_, hidden_size_, grad_Ct2_ptr_, true, static_cast<T>(0));
  grad_W_span_ = rnn::detail::Allocate(allocator_, 4 * hidden_size_ * input_size_, grad_W_ptr_, true, static_cast<T>(0));
  grad_R_span_ = rnn::detail::Allocate(allocator_, 4 * hidden_size_ * hidden_size_, grad_R_ptr_, true, static_cast<T>(0));
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
  // ->|H|<- Each block is of size hidden size
  // Each B represents an index of the batch. So batch size is n
  // <----S1-----><----S2----->....<----St-----> Sequence length is t
  // Every buffer having a sequnence length dimension are structured as the above two.
  //
  // Weight buffers:
  //  _______________________________________________
  // |         |           |            |           |
  // |   Wi    |     Wo    |     Wf     |     Wc    |
  // |_________|___________|____________|___________|
  // <---HxI---> Each block is hidden size x input size long
  // Each blodk represents either i, o, f, c weioghts.
  // Every weight buffer is similarly structured

  const gsl::span<const T>& W = inputs.weights;
  const float* Wi = SafeRawPointer<const T>(W.begin(), W.end(), 4 * hidden_size_ * input_size_);
  const float* Wo = Wi + hidden_size_ * input_size_;
  const float* Wf = Wo + hidden_size_ * input_size_;
  const float* Wc = Wf + hidden_size_ * input_size_;

  const gsl::span<const T>& R = inputs.recurrence_weights;
  const float* Ri = SafeRawPointer<const T>(R.begin(), R.end(), 4 * hidden_size_ * hidden_size_);
  const float* Ro = Ri + hidden_size_ * hidden_size_;
  const float* Rf = Ro + hidden_size_ * hidden_size_;
  const float* Rc = Rf + hidden_size_ * hidden_size_;

  auto& grad_W = outputs.grad_weights;
  std::fill_n(grad_W.data(), grad_W.size(), static_cast<T>(0));
  float* grad_Wi = SafeRawPointer<T>(grad_W.begin(), grad_W.end(), 4 * hidden_size_ * input_size_);
  float* grad_Wo = grad_Wi + hidden_size_ * input_size_;
  float* grad_Wf = grad_Wo + hidden_size_ * input_size_;
  float* grad_Wc = grad_Wf + hidden_size_ * input_size_;

  auto& grad_R = outputs.grad_recurrence_weights;
  std::fill_n(grad_R.data(), grad_R.size(), static_cast<T>(0));
  float* grad_Ri = SafeRawPointer<T>(grad_R.begin(), grad_R.end(), 4 * hidden_size_ * hidden_size_);
  float* grad_Ro = grad_Ri + hidden_size_ * hidden_size_;
  float* grad_Rf = grad_Ro + hidden_size_ * hidden_size_;
  float* grad_Rc = grad_Rf + hidden_size_ * hidden_size_;

  // Fill grad bias with 0s since they are used as accumulators
  auto& grad_B = outputs.grad_bias;
  const bool use_bias = !grad_B.empty();
  if (use_bias) {
    std::fill_n(grad_B.data(), grad_B.size(), static_cast<T>(0));
  }
  float* grad_Wbi = use_bias ? SafeRawPointer<T>(grad_B.begin(), grad_B.end(), 8 * hidden_size_)
                             : nullptr;
  float* grad_Wbo = use_bias ? grad_Wbi + hidden_size_
                             : nullptr;
  float* grad_Wbf = use_bias ? grad_Wbo + hidden_size_
                             : nullptr;
  float* grad_Wbc = use_bias ? grad_Wbf + hidden_size_
                             : nullptr;
  float* grad_Rbi = use_bias ? grad_Wbc + hidden_size_
                             : nullptr;
  float* grad_Rbo = use_bias ? grad_Rbi + hidden_size_
                             : nullptr;
  float* grad_Rbf = use_bias ? grad_Rbo + hidden_size_
                             : nullptr;
  float* grad_Rbc = use_bias ? grad_Rbf + hidden_size_
                             : nullptr;

  // Fill grad peepholes with 0s since they are used as accumulators
  auto& grad_P = outputs.grad_peephole_weights;
  const bool use_peepholes = !grad_P.empty();
  if (use_peepholes) {
    std::fill_n(grad_P.data(), grad_P.size(), static_cast<T>(0));
  }
  float* grad_pi = use_peepholes ? SafeRawPointer<T>(grad_P.begin(), grad_P.end(), 3 * hidden_size_)
                                 : nullptr;
  float* grad_po = use_peepholes ? grad_pi + hidden_size_
                                 : nullptr;
  float* grad_pf = use_peepholes ? grad_po + hidden_size_
                                 : nullptr;

  constexpr float alpha = 1.0f;
  // Gemm accumulation results in incorrect values. For now, use custom accumulation logic.
  constexpr float weight_beta = 0.0f;

  float* grad_ai = SafeRawPointer<T>(grad_a_span_.begin(), grad_a_span_.end(), 4 * hidden_size_);
  float* grad_ao = grad_ai + hidden_size_;
  float* grad_af = grad_ao + hidden_size_;
  float* grad_ac = grad_af + hidden_size_;

  float* grad_Wi_local = SafeRawPointer<T>(grad_W_span_.begin(), grad_W_span_.end(), 4 * hidden_size_ * input_size_);
  float* grad_Wo_local = grad_Wi_local + hidden_size_ * input_size_;
  float* grad_Wf_local = grad_Wo_local + hidden_size_ * input_size_;
  float* grad_Wc_local = grad_Wf_local + hidden_size_ * input_size_;

  float* grad_Ri_local = SafeRawPointer<T>(grad_R_span_.begin(), grad_R_span_.end(), 4 * hidden_size_ * hidden_size_);
  float* grad_Ro_local = grad_Ri_local + hidden_size_ * hidden_size_;
  float* grad_Rf_local = grad_Ro_local + hidden_size_ * hidden_size_;
  float* grad_Rc_local = grad_Rf_local + hidden_size_ * hidden_size_;

  float* grad_Ct2 = SafeRawPointer<T>(grad_Ct2_span_.begin(), grad_Ct2_span_.end(), hidden_size_);

  std::fill_n(outputs.grad_initial_cell_state.data(), outputs.grad_initial_cell_state.size(), static_cast<T>(0));
  std::fill_n(outputs.grad_initial_hidden_state.data(), outputs.grad_initial_hidden_state.size(), static_cast<T>(0));

  const size_t hidden_size_x4 = 4 * hidden_size_;
  for (size_t idx = 0; idx < batch_size_; ++idx) {
    // maybe the first input is used and not the last.
    const float* grad_Cfinal = SafeRawPointer<const T>(inputs.grad_final_cell_state.begin() + idx * hidden_size_,
                                                       inputs.grad_final_cell_state.end(), hidden_size_);

    // Accumulate grad_Cfinal into grad_initial_cell_state
    float* grad_Ct = SafeRawPointer<T>(outputs.grad_initial_cell_state.begin() + idx * hidden_size_,
                                       outputs.grad_initial_cell_state.end(), hidden_size_);
    deepcpu::elementwise_sum1(grad_Cfinal, grad_Ct, hidden_size_);

    const float* grad_Hfinal = SafeRawPointer<const T>(inputs.grad_final_hidden_state.begin() + idx * hidden_size_,
                                                       inputs.grad_final_hidden_state.end(), hidden_size_);
    float* grad_Ht = SafeRawPointer<T>(outputs.grad_initial_hidden_state.begin() + idx * hidden_size_,
                                       outputs.grad_initial_hidden_state.end(), hidden_size_);
    // The LSTM outputs: all hidden states, final hidden state and final cell states
    // In addition of all hidden states being used, the final hidden state could also be used in the remainder of the
    // graph. So, accumulate the final hidden state gradient in gradHt.
    // Later, we will also accumulate from the gradient of all hideen states as that could also be used in the graph.
    deepcpu::elementwise_sum1(grad_Hfinal, grad_Ht, hidden_size_);

    for (int t = sequence_length_ - 1; t >= 0; --t) {
      auto iofc = inputs.iofc.begin() + (t * batch_size_ + idx) * hidden_size_x4;
      const float* it = SafeRawPointer<const T>(iofc, inputs.iofc.end(), hidden_size_x4);
      const float* ot = it + hidden_size_;
      const float* ft = ot + hidden_size_;
      const float* ct = ft + hidden_size_;

      // Retrieve current C, previous H and previous C from the given inputs.
      // Assume these inputs always exist. Caller of this function must ensure that.
      const float* Htminus1 = t > 0 ? SafeRawPointer<const T>(inputs.all_hidden_states.begin() + ((t - 1) * batch_size_ + idx) * hidden_size_,
                                                              inputs.all_hidden_states.end(), hidden_size_)
                                    : SafeRawPointer<const T>(inputs.initial_hidden_state.begin() + idx * hidden_size_,
                                                              inputs.initial_hidden_state.end(), hidden_size_);
      const float* Ct = SafeRawPointer<const T>(inputs.all_cell_states.begin() + (t * batch_size_ + idx) * hidden_size_,
                                                inputs.all_cell_states.end(), hidden_size_);
      const float* Ctminus1 = t > 0 ? SafeRawPointer<const T>(inputs.all_cell_states.begin() + ((t - 1) * batch_size_ + idx) * hidden_size_,
                                                              inputs.all_cell_states.end(), hidden_size_)
                                    : SafeRawPointer<const T>(inputs.initial_cell_state.begin() + idx * hidden_size_,
                                                              inputs.initial_cell_state.end(), hidden_size_);
      const float* grad_Ht2 = SafeRawPointer<const T>(inputs.grad_all_hidden_states.begin() + (t * batch_size_ + idx) * hidden_size_,
                                                      inputs.grad_all_hidden_states.end(), hidden_size_);
      // Accumulate the gradient from the gradients of all hidden states for this sequence index and batch index.
      deepcpu::elementwise_sum1(grad_Ht2, grad_Ht, hidden_size_);

      // Ct2_tilde = tanh(Ct2)
      MlasComputeTanh(Ct, grad_Ct2, hidden_size_);

      // Ht = ot (.) Ct2_tilde
      // dL/dot = dL/dHt (.) Ct2_tilde ---------- (1)
      elementwise_product(grad_Ht, grad_Ct2, grad_ao, hidden_size_);

      // Ht = ot (.) Ct2_tilde
      // dL/dCt2_tilde = dL/dHt (.) ot ---------- (2)
      // dL/dCt2 = dL/dCt2_tilde (.) (1 - (tanh(Ct))^2) ---------- (3)
      for (size_t h = 0; h < hidden_size_; ++h) {
        grad_Ct2[h] = grad_Ht[h] * ot[h] * (1 - grad_Ct2[h] * grad_Ct2[h]);
      }

      // Ct -> multiplex gate -> Ct1
      //                      -> Ct2
      // dL/dCt = dL/dCt1 + dL/dCt2 ---------- (4)
      deepcpu::elementwise_sum1(grad_Ct2, grad_Ct, hidden_size_);

      // Ct = ft (.) Ct-1 + it (.) ct
      // dL/dit = dL/dCt (.) ct ---------- (5)
      elementwise_product(grad_Ct, ct, grad_ai, hidden_size_);

      // Ct = ft (.) Ct-1 + it (.) ct
      // dL/dct = dL/dCt (.) it ---------- (6)
      elementwise_product(grad_Ct, it, grad_ac, hidden_size_);

      // Ct = ft (.) Ct-1 + it (.) ct
      // dL/dft = dL/dCt (.) Ct-1 ---------- (7)
      elementwise_product(grad_Ct, Ctminus1, grad_af, hidden_size_);

      // Ct = ft (.) Ct-1 + it (.) ct
      // dL/dCt-1 = dL/dCt (.) ft ---------- (8)
      elementwise_product(grad_Ct, ft, grad_Ct, hidden_size_);

      // ct = tanh(ac)
      // dL/dac = dL/dct (.) (1 - (tanh(ac))^2) ---------- (9)
      for (size_t h = 0; h < hidden_size_; ++h) {
        grad_ac[h] = grad_ac[h] * (1 - ct[h] * ct[h]);
      }

      // it = sigmoid(ai)
      // dL/dai = dL/dit (.) (sigmoid(ai) * (1 - sigmoid(ai))) ---------- (10)
      for (size_t h = 0; h < hidden_size_; ++h) {
        grad_ai[h] = grad_ai[h] * (it[h] * (1 - it[h]));
      }

      // ft = sigmoid(af)
      // dL/daf = dL/dft (.) (sigmoid(af) * (1 - sigmoid(af))) ---------- (11)
      for (size_t h = 0; h < hidden_size_; ++h) {
        grad_af[h] = grad_af[h] * (ft[h] * (1 - ft[h]));
      }

      // ot = sigmoid(ao)
      // dL/dao = dL/dot (.) (sigmoid(ao) * (1 - sigmoid(ao))) ---------- (12)
      for (size_t h = 0; h < hidden_size_; ++h) {
        grad_ao[h] = grad_ao[h] * (ot[h] * (1 - ot[h]));
      }

      // -----------------------------------------------------------
      // Input gate computations

      // ai = Xti * Wi^T + Ht-1i * Ri^T + Pi (.) Ct-1 + Wbi + Rbi
      // dL/dXti = dL/dai * Wi ---------- (13)
      // [1, input_size_] = [1, hidden_size_] * [hidden_size_, input_size_]
      // M = 1, N = input_size_, K = hidden_size_
      float input_beta = 0.0f;
      float* grad_Xt = SafeRawPointer<T>(outputs.grad_input.begin() + (t * batch_size_ + idx) * input_size_,
                                         outputs.grad_input.end(), input_size_);
      ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, input_size_,
                                       hidden_size_, alpha, grad_ai, Wi, input_beta, grad_Xt, thread_pool_);

      // ai = Xti * Wi^T + Ht-1i * Ri^T + Pi (.) Ct-1 + Wbi + Rbi
      // dL/dWi = dL/dai^T * Xti ---------- (14)
      // [hidden_size_, input_size_] = [1, hidden_size_]^T * [1, input_size_]
      // M = hidden_size_, N = input_size_, K = 1
      const float* Xt = SafeRawPointer<const T>(inputs.input.begin() + (t * batch_size_ + idx) * input_size_,
                                                inputs.input.end(), input_size_);
      ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, input_size_,
                                       1, alpha, grad_ai, Xt, weight_beta, grad_Wi_local, thread_pool_);
      // Note that the weight beta is always 0. So, we must accumulate ourselves.
      deepcpu::elementwise_sum1(grad_Wi_local, grad_Wi, hidden_size_ * input_size_);

      if (use_peepholes) {
        // ai = Xti * Wi^T + Ht-1i * Ri^T + Pi (.) Ct-1 + Wbi + Rbi
        // dL/dPi = dL/dai (.) Ct-1 ---------- (15)
        deepcpu::elementwise_product(grad_ai, Ctminus1, grad_pi, hidden_size_);
      }

      if (use_bias) {
        // ai = Xti * Wi^T + Ht-1i * Ri^T + Pi (.) Ct-1 + Wbi + Rbi
        // dL/dWbi = dL/dai ---------- (16)
        deepcpu::elementwise_sum1(grad_ai, grad_Wbi, hidden_size_);

        // ai = Xti * Wi^T + Ht-1i * Ri^T + Pi (.) Ct-1 + Wbi + Rbi
        // dL/dRbi = dL/dai ---------- (17)
        deepcpu::elementwise_sum1(grad_ai, grad_Rbi, hidden_size_);
      }

      // ai = Xti * Wi^T + Ht-1i * Ri^T + Pi (.) Ct-1 + Wbi + Rbi
      // dL/dHt-1i = dL/dai * Ri ---------- (18)
      // [1, hidden_size_] = [1, hidden_size_] * [hidden_size_, hidden_size_]
      // M = 1, N = hidden_size_, K = hidden_size_
      ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, hidden_size_,
                                       hidden_size_, alpha, grad_ai, Ri, input_beta, grad_Ht, thread_pool_);

      // ai = Xti * Wi^T + Ht-1i * Ri^T + Pi (.) Ct-1 + Wbi + Rbi
      // dL/dRi = dL/dai^T * Ht-1i ---------- (19)
      // [hidden_size_, hidden_size_] = [1, hidden_size_]^T * [1, hidden_size_]
      // M = hidden_size_, N = hidden_size_, K = 1
      ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, hidden_size_,
                                       1, alpha, grad_ai, Htminus1, weight_beta, grad_Ri_local, thread_pool_);
      // Note that the weight beta is always 0. So, we must accumulate ourselves.
      deepcpu::elementwise_sum1(grad_Ri_local, grad_Ri, hidden_size_ * hidden_size_);

      // -----------------------------------------------------------
      // Output gate computations

      // ao = Xto * Wo^T + Ht-1o * Ro^T + Po (.) Ct + Wbo + Rbo
      // dL/dXto = dL/dao * Wo ---------- (20)
      // [1, input_size_] = [1, hidden_size_] * [hidden_size_, input_size_]
      // M = 1, N = input_size_, K = hidden_size_
      input_beta = 1.0f;
      ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, input_size_,
                                       hidden_size_, alpha, grad_ao, Wo, input_beta, grad_Xt, thread_pool_);

      // ao = Xto * Wo^T + Ht-1o * Ro^T + Po (.) Ct + Wbo + Rbo
      // dL/dWo = dL/dao^T * Xto ---------- (21)
      // [hidden_size_, input_size_] = [1, hidden_size_]^T * [1, input_size_]
      // M = hidden_size_, N = input_size_, K = 1
      ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, input_size_,
                                       1, alpha, grad_ao, Xt, weight_beta, grad_Wo_local, thread_pool_);
      // Note that the weight beta is always 0. So, we must accumulate ourselves.
      deepcpu::elementwise_sum1(grad_Wo_local, grad_Wo, hidden_size_ * input_size_);

      if (use_peepholes) {
        // ao = Xto * Wo^T + Ht-1o * Ro^T + Po (.) Ct + Wbo + Rbo
        // dL/dPo = dL/dao (.) Ct ---------- (22)
        deepcpu::elementwise_product(grad_ao, Ct, grad_po, hidden_size_);
      }

      if (use_bias) {
        // ao = Xto * Wo^T + Ht-1o * Ro^T + Po (.) Ct + Wbo + Rbo
        // dL/dWbo = dL/dao ---------- (23)
        deepcpu::elementwise_sum1(grad_ao, grad_Wbo, hidden_size_);

        // ao = Xto * Wo^T + Ht-1o * Ro^T + Po (.) Ct + Wbo + Rbo
        // dL/dRbo = dL/dao ---------- (24)
        deepcpu::elementwise_sum1(grad_ao, grad_Rbo, hidden_size_);
      }

      // ao = Xto * Wo^T + Ht-1o * Ro^T + Po (.) Ct + Wbo + Rbo
      // dL/dHt-1o = dL/dao * Ro ---------- (25)
      // [1, hidden_size_] = [1, hidden_size_] * [hidden_size_, hidden_size_]
      // M = 1, N = hidden_size_, K = hidden_size_
      ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, hidden_size_,
                                       hidden_size_, alpha, grad_ao, Ro, input_beta, grad_Ht, thread_pool_);

      // ao = Xto * Wo^T + Ht-1o * Ro^T + Po (.) Ct + Wbo + Rbo
      // dL/dRo = dL/dao^T * Ht-1o ---------- (26)
      // [hidden_size_, hidden_size_] = [1, hidden_size_]^T * [1, hidden_size_]
      // M = hidden_size_, N = hidden_size_, K = 1
      ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, hidden_size_,
                                       1, alpha, grad_ao, Htminus1, weight_beta, grad_Ro_local, thread_pool_);
      // Note that the weight beta is always 0. So, we must accumulate ourselves.
      deepcpu::elementwise_sum1(grad_Ro_local, grad_Ro, hidden_size_ * hidden_size_);

      // -----------------------------------------------------------
      // Forget gate computations

      // af = Xtf * Wf^T + Ht-1f * Rf^T + Pf (.) Ct-1 + Wbf + Rbf
      // dL/dXtf = dL/daf * Wf ---------- (27)
      // [1, input_size_] = [1, hidden_size_] * [hidden_size_, input_size_]
      // M = 1, N = input_size_, K = hidden_size_
      ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, input_size_,
                                       hidden_size_, alpha, grad_af, Wf, input_beta, grad_Xt, thread_pool_);

      // af = Xtf * Wf^T + Ht-1f * Rf^T + Pf (.) Ct-1 + Wbf + Rbf
      // dL/dWf = dL/daf^T * Xtf ---------- (28)
      // [hidden_size_, input_size_] = [1, hidden_size_]^T * [1, input_size_]
      // M = hidden_size_, N = input_size_, K = 1
      ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, input_size_,
                                       1, alpha, grad_af, Xt, weight_beta, grad_Wf_local, thread_pool_);
      // Note that the weight beta is always 0. So, we must accumulate ourselves.
      deepcpu::elementwise_sum1(grad_Wf_local, grad_Wf, hidden_size_ * input_size_);

      if (use_peepholes) {
        // af = Xtf * Wf^T + Ht-1f * Rf^T + Pf (.) Ct-1 + Wbf + Rbf
        // dL/dPf = dL/daf (.) Ct-1 ---------- (29)
        deepcpu::elementwise_product(grad_af, Ctminus1, grad_pf, hidden_size_);
      }

      if (use_bias) {
        // af = Xtf * Wf^T + Ht-1f * Rf^T + Pf (.) Ct-1 + Wbf + Rbf
        // dL/dWbf = dL/daf ---------- (30)
        deepcpu::elementwise_sum1(grad_af, grad_Wbf, hidden_size_);

        // af = Xtf * Wf^T + Ht-1f * Rf^T + Pf (.) Ct-1 + Wbf + Rbf
        // dL/dRbf = dL/daf ---------- (31)
        deepcpu::elementwise_sum1(grad_af, grad_Rbf, hidden_size_);
      }

      // af = Xtf * Wf^T + Ht-1f * Rf^T + Pf (.) Ct-1 + Wbf + Rbf
      // dL/dHt-1f = dL/daf * Rf ---------- (32)
      // [1, hidden_size_] = [1, hidden_size_] * [hidden_size_, hidden_size_]
      // M = 1, N = hidden_size_, K = hidden_size_
      ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, hidden_size_,
                                       hidden_size_, alpha, grad_af, Rf, input_beta, grad_Ht, thread_pool_);

      // af = Xtf * Wf^T + Ht-1f * Rf^T + Pf (.) Ct-1 + Wbf + Rbf
      // dL/dRf = dL/daf^T * Ht-1f ---------- (33)
      // [hidden_size_, hidden_size_] = [1, hidden_size_]^T * [1, hidden_size_]
      // M = hidden_size_, N = hidden_size_, K = 1
      ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, hidden_size_,
                                       1, alpha, grad_af, Htminus1, weight_beta, grad_Rf_local, thread_pool_);
      // Note that the weight beta is always 0. So, we must accumulate ourselves.
      deepcpu::elementwise_sum1(grad_Rf_local, grad_Rf, hidden_size_ * hidden_size_);

      // -----------------------------------------------------------
      // Control gate computations

      // ac = Xtc * Wc^T + Ht-1c * Rc^T + Wbc + Rbc
      // dL/dXtc = dL/dac * Wc ---------- (34)
      // [1, input_size_] = [1, hidden_size_] * [hidden_size_, input_size_]
      // M = 1, N = input_size_, K = hidden_size_
      ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, input_size_,
                                       hidden_size_, alpha, grad_ac, Wc, input_beta, grad_Xt, thread_pool_);

      // ac = Xtc * Wc^T + Ht-1c * Rc^T + Wbc + Rbc
      // dL/dWc = dL/dac^T * Xtc ---------- (35)
      // [hidden_size_, input_size_] = [1, hidden_size_]^T * [1, input_size_]
      // M = hidden_size_, N = input_size_, K = 1
      ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, input_size_,
                                       1, alpha, grad_ac, Xt, weight_beta, grad_Wc_local, thread_pool_);
      // Note that the weight beta is always 0. So, we must accumulate ourselves.
      deepcpu::elementwise_sum1(grad_Wc_local, grad_Wc, hidden_size_ * input_size_);

      if (use_bias) {
        // ac = Xtc * Wc^T + Ht-1c * Rc^T + Wbc + Rbc
        // dL/dWbc = dL/dac ---------- (36)
        deepcpu::elementwise_sum1(grad_ac, grad_Wbc, hidden_size_);

        // ac = Xtc * Wc^T + Ht-1c * Rc^T + Wbc + Rbc
        // dL/dRbc = dL/dac ---------- (37)
        deepcpu::elementwise_sum1(grad_ac, grad_Rbc, hidden_size_);
      }

      // ac = Xtc * Wc^T + Ht-1c * Rc^T + Wbc + Rbc
      // dL/dHt-1c = dL/dac * Rc ---------- (38)
      // [1, hidden_size_] = [1, hidden_size_] * [hidden_size_, hidden_size_]
      // M = 1, N = hidden_size_, K = hidden_size_
      ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, hidden_size_,
                                       hidden_size_, alpha, grad_ac, Rc, input_beta, grad_Ht, thread_pool_);

      // ac = Xtc * Wc^T + Ht-1c * Rc^T + Wbc + Rbc
      // dL/dRc = dL/dac^T * Ht-1c ---------- (39)
      // [hidden_size_, hidden_size_] = [1, hidden_size_]^T * [1, hidden_size_]
      // M = hidden_size_, N = hidden_size_, K = 1
      ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, hidden_size_,
                                       1, alpha, grad_ac, Htminus1, weight_beta, grad_Rc_local, thread_pool_);
      // Note that the weight beta is always 0. So, we must accumulate ourselves.
      deepcpu::elementwise_sum1(grad_Rc_local, grad_Rc, hidden_size_ * hidden_size_);

      // -----------------------------------------------------------

      // Xt -> multiplex gate -> Xti
      //                      -> Xto
      //                      -> Xtf
      //                      -> Xtc
      // dL/dXt = dL/dXti  + dL/dXto + dL/dXtf + dL/dXtc ---------- (40)
      // The accumulation happens after each gate computation

      // Ht-1 -> multiplex gate -> Ht-1i
      //                        -> Ht-1o
      //                        -> Ht-1f
      //                        -> Ht-1c
      // dL/dHt-1 = dL/dHt-1i  + dL/dHt-1o + dL/dHt-1f + dL/dHt-1c ---------- (41)
      // The accumulation happens after each gate computation
    }
  }
}

template class LSTMGradImpl<float>;

}  // namespace onnxruntime::lstm
