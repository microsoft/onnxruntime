// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/rnn/gru_grad_compute.h"

namespace onnxruntime::gru {

namespace {

void ElementwiseSub(const float* op1, const float* op2, float* dest, int size) {
  for (int i = 0; i < size; i++)
    dest[i] = op1[i] - op2[i];
}

void ElementwiseProduct(const float* op1, const float* op2, float* dest, int size) {
  for (int i = 0; i < size; i++)
    dest[i] = op1[i] * op2[i];
}

}  // namespace

template <typename T>
GRUGradImpl<T>::GRUGradImpl(int sequence_length, int batch_size, int hidden_size, int input_size,
                            bool linear_before_reset, concurrency::ThreadPool* thread_pool,
                            AllocatorPtr allocator)
    : sequence_length_(sequence_length),
      batch_size_(batch_size),
      hidden_size_(hidden_size),
      input_size_(input_size),
      linear_before_reset_(linear_before_reset),
      thread_pool_(thread_pool),
      allocator_(allocator) {
  const size_t hidden_size_x3 = 3U * static_cast<size_t>(hidden_size_);
  const size_t weight_size = 3U * static_cast<size_t>(hidden_size_) * static_cast<size_t>(input_size_);
  const size_t recurrence_weight_size = 3U * static_cast<size_t>(hidden_size_) * static_cast<size_t>(hidden_size_);
  grad_a_span_ = rnn::detail::Allocate(allocator_, hidden_size_x3, grad_a_ptr_, true, static_cast<T>(0));
  rt_factor_span_ = rnn::detail::Allocate(allocator_, hidden_size_, rt_factor_ptr_, true, static_cast<T>(0));
  grad_W_span_ = rnn::detail::Allocate(allocator_, weight_size, grad_W_ptr_, true, static_cast<T>(0));
  grad_R_span_ = rnn::detail::Allocate(allocator_, recurrence_weight_size, grad_R_ptr_, true, static_cast<T>(0));
}

template <typename T>
void GRUGradImpl<T>::ComputeGradient(const GRUGradInputs<T>& inputs, GRUGradOutputs<T>& outputs) {
  using namespace rnn::detail;

  // A note on the memory layout of buffers used in this function:
  // zrh buffer:
  //  ________________________________________________
  // | | | | | | |   | | | |   | | | | | | |   | | | |
  // |z|r|h|z|r|h|...|z|r|h|...|z|r|h|z|r|h|...|z|r|h|
  // |_|_|_|_|_|_|___|_|_|_|___|_|_|_|_|_|_|___|_|_|_|
  // ->|H|<- Each z, t, h block is of size hidden size
  // <-B1-><-B2->....<-Bn->....<-B1-><-B2->....<-Bn-> Batch size is n
  // <---------S1--------->....<---------St----------> Sequence length is t
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
  //  ___________________________________
  // |         |           |            |
  // |   Wz    |     Wr    |     Wh     |
  // |_________|___________|____________|
  // <---HxI---> Each block is hidden size x input size long
  // Each block represents either z, r, or h weights.
  // Every weight buffer is similarly structured

  const size_t hidden_size_x3 = 3U * static_cast<size_t>(hidden_size_);
  const size_t hidden_size_x6 = 6U * static_cast<size_t>(hidden_size_);
  const size_t weight_size = 3U * static_cast<size_t>(hidden_size_) * static_cast<size_t>(input_size_);
  const size_t recurrence_weight_size = 3U * static_cast<size_t>(hidden_size_) * static_cast<size_t>(hidden_size_);
  const size_t hidden_sizexinput_size = static_cast<size_t>(hidden_size_) * static_cast<size_t>(input_size_);
  const size_t hidden_sizexhidden_size = static_cast<size_t>(hidden_size_) * static_cast<size_t>(hidden_size_);

  const gsl::span<const T>& W = inputs.weights;
  const float* Wz = SafeRawPointer<const T>(W.begin(), W.end(), weight_size);
  const float* Wr = Wz + hidden_sizexinput_size;
  const float* Wh = Wr + hidden_sizexinput_size;

  const gsl::span<const T>& R = inputs.recurrence_weights;
  const float* Rz = SafeRawPointer<const T>(R.begin(), R.end(), recurrence_weight_size);
  const float* Rr = Rz + hidden_sizexhidden_size;
  const float* Rh = Rr + hidden_sizexhidden_size;

  const gsl::span<const T>& B = inputs.bias;
  const float* Rbh = B.empty()
                         ? nullptr
                         : SafeRawPointer<const T>(B.begin() + 5U * static_cast<size_t>(hidden_size_),
                                                   B.end(), hidden_size_);

  const bool grad_input_required = !outputs.grad_input.empty();
  const bool grad_weights_required = !outputs.grad_weights.empty();
  const bool grad_recurrence_weights_required = !outputs.grad_recurrence_weights.empty();
  const bool grad_bias_required = !outputs.grad_bias.empty();

  auto& grad_W = outputs.grad_weights;
  if (grad_weights_required) {
    std::fill_n(grad_W.data(), grad_W.size(), static_cast<T>(0));
  }
  float* grad_Wz = grad_weights_required
                       ? SafeRawPointer<T>(grad_W.begin(), grad_W.end(), weight_size)
                       : nullptr;
  float* grad_Wr = grad_weights_required ? grad_Wz + hidden_sizexinput_size : nullptr;
  float* grad_Wh = grad_weights_required ? grad_Wr + hidden_sizexinput_size : nullptr;

  auto& grad_R = outputs.grad_recurrence_weights;
  if (grad_recurrence_weights_required) {
    std::fill_n(grad_R.data(), grad_R.size(), static_cast<T>(0));
  }
  float* grad_Rz = grad_recurrence_weights_required
                       ? SafeRawPointer<T>(grad_R.begin(), grad_R.end(), recurrence_weight_size)
                       : nullptr;
  float* grad_Rr = grad_recurrence_weights_required ? grad_Rz + hidden_sizexhidden_size : nullptr;
  float* grad_Rh = grad_recurrence_weights_required ? grad_Rr + hidden_sizexhidden_size : nullptr;

  // Fill grad bias with 0s since they are used as accumulators
  auto& grad_B = outputs.grad_bias;
  if (grad_bias_required) {
    std::fill_n(grad_B.data(), grad_B.size(), static_cast<T>(0));
  }
  float* grad_Wbz = grad_bias_required ? SafeRawPointer<T>(grad_B.begin(), grad_B.end(), hidden_size_x6) : nullptr;
  float* grad_Wbr = grad_bias_required ? grad_Wbz + hidden_size_ : nullptr;
  float* grad_Wbh = grad_bias_required ? grad_Wbr + hidden_size_ : nullptr;
  float* grad_Rbz = grad_bias_required ? grad_Wbh + hidden_size_ : nullptr;
  float* grad_Rbr = grad_bias_required ? grad_Rbz + hidden_size_ : nullptr;
  float* grad_Rbh = grad_bias_required ? grad_Rbr + hidden_size_ : nullptr;

  constexpr float alpha = 1.0f;
  constexpr float weight_beta = 0.0f;

  float* grad_az = SafeRawPointer<T>(grad_a_span_.begin(), grad_a_span_.end(), hidden_size_x3);
  float* grad_ar = grad_az + hidden_size_;
  float* grad_ah = grad_ar + hidden_size_;

  float* grad_Wz_local = SafeRawPointer<T>(grad_W_span_.begin(), grad_W_span_.end(), weight_size);
  float* grad_Wr_local = grad_Wz_local + hidden_sizexinput_size;
  float* grad_Wh_local = grad_Wr_local + hidden_sizexinput_size;

  float* grad_Rz_local = SafeRawPointer<T>(grad_R_span_.begin(), grad_R_span_.end(), recurrence_weight_size);
  float* grad_Rr_local = grad_Rz_local + hidden_sizexhidden_size;
  float* grad_Rh_local = grad_Rr_local + hidden_sizexhidden_size;

  float* rt_factor = SafeRawPointer<T>(rt_factor_span_.begin(), rt_factor_span_.end(), hidden_size_);

  std::fill_n(outputs.grad_initial_hidden_state.data(), outputs.grad_initial_hidden_state.size(), static_cast<T>(0));

  for (int idx = 0; idx < batch_size_; ++idx) {
    const size_t hidden_sizexidx = static_cast<size_t>(idx) * static_cast<size_t>(hidden_size_);
    float* grad_Ht = SafeRawPointer<T>(outputs.grad_initial_hidden_state.begin() + hidden_sizexidx,
                                       outputs.grad_initial_hidden_state.end(), hidden_size_);
    const float* grad_Hfinal = inputs.grad_final_hidden_state.empty()
                                   ? nullptr
                                   : SafeRawPointer<const T>(inputs.grad_final_hidden_state.begin() + hidden_sizexidx,
                                                             inputs.grad_final_hidden_state.end(), hidden_size_);

    if (grad_Hfinal)
      deepcpu::elementwise_sum1(grad_Hfinal, grad_Ht, hidden_size_);

    for (int t = sequence_length_ - 1; t >= 0; --t) {
      const size_t zrh_offset = (static_cast<size_t>(t) * static_cast<size_t>(batch_size_) +
                                 static_cast<size_t>(idx)) *
                                hidden_size_x3;
      auto zrh = inputs.zrh.begin() + zrh_offset;
      const float* zt = SafeRawPointer<const T>(zrh, inputs.zrh.end(), hidden_size_x3);
      const float* rt = zt + hidden_size_;
      const float* ht = rt + hidden_size_;

      const size_t H_offset = (static_cast<size_t>(t) * static_cast<size_t>(batch_size_) +
                               static_cast<size_t>(idx)) *
                              static_cast<size_t>(hidden_size_);
      const size_t Htminus1_offset = t > 0 ? ((static_cast<size_t>(t) - 1U) * static_cast<size_t>(batch_size_) +
                                              static_cast<size_t>(idx)) *
                                                 hidden_size_
                                           : 0U;

      const float* grad_Ht2 = inputs.grad_all_hidden_states.empty()
                                  ? nullptr
                                  : SafeRawPointer<const T>(
                                        inputs.grad_all_hidden_states.begin() + H_offset,
                                        inputs.grad_all_hidden_states.end(), hidden_size_);

      const float* Htminus1 = t > 0 ? SafeRawPointer<const T>(
                                          inputs.all_hidden_states.begin() + Htminus1_offset,
                                          inputs.all_hidden_states.end(), hidden_size_)
                                    : SafeRawPointer<const T>(
                                          inputs.initial_hidden_state.begin() + hidden_sizexidx,
                                          inputs.initial_hidden_state.end(), hidden_size_);

      // Accumulate the gradient from the gradients of all hidden states for this sequence index and batch index.
      if (grad_Ht2)
        deepcpu::elementwise_sum1(grad_Ht2, grad_Ht, hidden_size_);

      // Ht = (1 - zt) (.) ht + zt (.) Ht-1H
      // dL/dzt = dL/dHt (.) (Ht-1h - ht) ---------- (1)
      ElementwiseSub(Htminus1, ht, grad_az, hidden_size_);
      ElementwiseProduct(grad_az, grad_Ht, grad_az, hidden_size_);

      // zt = sigmoid(az)
      // dL/daz = dL/dzt (.) zt (.) (1 - zt) ---------- (2)
      for (int h = 0; h < hidden_size_; ++h) {
        grad_az[h] = grad_az[h] * (zt[h] * (1 - zt[h]));
      }

      // Ht = (1 - zt) (.) ht + zt (.) Ht-1H
      // dL/dht = dL/dHt (.) (1 - zt) ---------- (3)
      for (int h = 0; h < hidden_size_; ++h) {
        grad_ah[h] = grad_Ht[h] * (1 - zt[h]);
      }

      // ht = tanh(ah)
      // dL/dah = dL/dht (.) (1 - ht^2) ---------- (4)
      for (int h = 0; h < hidden_size_; ++h) {
        grad_ah[h] = grad_ah[h] * (1 - ht[h] * ht[h]);
      }

      if (!linear_before_reset_) {
        // ah = Xth * Wh^T + (rt (.) Ht-1h) * Rh^T + Wbh + Rbh
        // dL/drt = (dL/dah * Rh) (.) (Ht-1h) ---------- (5)
        ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, hidden_size_,
                                         hidden_size_, alpha, grad_ah, Rh, weight_beta, grad_ar, thread_pool_);
        ElementwiseProduct(grad_ar, Htminus1, grad_ar, hidden_size_);
      } else {
        // ah = Xth * Wh^T + rt (.) (Ht-1h * Rh^T + Rbh) + Wbh
        // dL/drt = dL/dah (.) (Ht-1h * Rh^T + Rbh) ---------- (5)
        ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasTrans, 1, hidden_size_,
                                         hidden_size_, alpha, Htminus1, Rh, weight_beta, grad_ar, thread_pool_);
        if (Rbh != nullptr)
          deepcpu::elementwise_sum1(Rbh, grad_ar, hidden_size_);
        ElementwiseProduct(grad_ar, grad_ah, grad_ar, hidden_size_);
      }

      // rt = sigmoid(ar)
      // dL/dar = dL/drt (.) rt (.) (1 - rt) ---------- (6)
      for (int h = 0; h < hidden_size_; ++h) {
        grad_ar[h] = grad_ar[h] * (rt[h] * (1 - rt[h]));
      }

      if (grad_input_required) {
        const size_t X_offset = (static_cast<size_t>(t) * static_cast<size_t>(batch_size_) +
                                 static_cast<size_t>(idx)) *
                                static_cast<size_t>(input_size_);
        // Xt -> multiplex gate -> Xtz
        //                      -> Xtr
        //                      -> Xth
        // dL/dXt = dL/dXtz  + dL/dXtr + dL/dXth ---------- (7)
        float input_beta = 0.0f;

        // az = Xtz * Wz^T + Ht-1z * Rz^T + Wbz + Rbz
        // dL/dXtz = dL/daz * Wz ---------- (8)
        // [1, input_size_] = [1, hidden_size_] * [hidden_size_, input_size_]
        // M = 1, N = input_size_, K = hidden_size_
        float* grad_Xt = SafeRawPointer<T>(outputs.grad_input.begin() + X_offset,
                                           outputs.grad_input.end(), input_size_);
        ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, input_size_,
                                         hidden_size_, alpha, grad_az, Wz, input_beta, grad_Xt, thread_pool_);

        // ar = Xtr * Wr^T + Ht-1r * Rr^T + Wbr + Rbr
        // dL/dXtr = dL/dar * Wr ---------- (9)
        // [1, input_size_] = [1, hidden_size_] * [hidden_size_, input_size_]
        // M = 1, N = input_size_, K = hidden_size_
        input_beta = 1.0f;
        ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, input_size_,
                                         hidden_size_, alpha, grad_ar, Wr, input_beta, grad_Xt, thread_pool_);

        // ah = Xth * Wh^T + (rt (.) Ht-1h) * Rh^T + Wbh + Rbh
        // dL/dXth = dL/dah * Wh ---------- (10)
        // [1, input_size_] = [1, hidden_size_] * [hidden_size_, input_size_]
        // M = 1, N = input_size_, K = hidden_size_
        ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, input_size_,
                                         hidden_size_, alpha, grad_ah, Wh, input_beta, grad_Xt, thread_pool_);
      }

      if (grad_weights_required) {
        const size_t X_offset = (static_cast<size_t>(t) * static_cast<size_t>(batch_size_) +
                                 static_cast<size_t>(idx)) *
                                static_cast<size_t>(input_size_);
        // az = Xtz * Wz^T + Ht-1z * Rz^T + Wbz + Rbz
        // dL/dWz = dL/daz^T * Xtz ---------- (11)
        // [hidden_size_, input_size_] = [1, hidden_size_]^T * [1, input_size_]
        // M = hidden_size_, N = input_size_, K = 1
        const float* Xt = SafeRawPointer<const T>(inputs.input.begin() + X_offset,
                                                  inputs.input.end(), input_size_);
        ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, input_size_,
                                         1, alpha, grad_az, Xt, weight_beta, grad_Wz_local, thread_pool_);
        // Note that the weight beta is always 0. So, we must accumulate ourselves.
        deepcpu::elementwise_sum1(grad_Wz_local, grad_Wz, hidden_size_ * input_size_);

        // ar = Xtr * Wr^T + Ht-1r * Rr^T + Wbr + Rbr
        // dL/dWr = dL/dar^T * Xtr ---------- (12)
        // [hidden_size_, input_size_] = [1, hidden_size_]^T * [1, input_size_]
        // M = hidden_size_, N = input_size_, K = 1
        ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, input_size_,
                                         1, alpha, grad_ar, Xt, weight_beta, grad_Wr_local, thread_pool_);
        // Note that the weight beta is always 0. So, we must accumulate ourselves.
        deepcpu::elementwise_sum1(grad_Wr_local, grad_Wr, hidden_size_ * input_size_);

        // ah = Xth * Wh^T + Ht-1h * Rh^T + Wbh + Rbh
        // dL/dWh = dL/dah^T * Xth ---------- (13)
        // [hidden_size_, input_size_] = [1, hidden_size_]^T * [1, input_size_]
        // M = hidden_size_, N = input_size_, K = 1
        ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, input_size_,
                                         1, alpha, grad_ah, Xt, weight_beta, grad_Wh_local, thread_pool_);
        // Note that the weight beta is always 0. So, we must accumulate ourselves.
        deepcpu::elementwise_sum1(grad_Wh_local, grad_Wh, hidden_size_ * input_size_);
      }

      if (grad_recurrence_weights_required) {
        // az = Xtz * Wz^T + Ht-1z * Rz^T + Wbz + Rbz
        // dL/dRz = dL/daz^T * Ht-1z ---------- (14)
        // [hidden_size_, hidden_size_] = [1, hidden_size_]^T * [1, hidden_size_]
        // M = hidden_size_, N = hidden_size_, K = 1
        ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, hidden_size_,
                                         1, alpha, grad_az, Htminus1, weight_beta, grad_Rz_local, thread_pool_);
        // Note that the weight beta is always 0. So, we must accumulate ourselves.
        deepcpu::elementwise_sum1(grad_Rz_local, grad_Rz, hidden_size_ * hidden_size_);

        // ar = Xtr * Wr^T + Ht-1r * Rr^T + Wbr + Rbr
        // dL/dRr = dL/dar^T * Ht-1r ---------- (15)
        // [hidden_size_, hidden_size_] = [1, hidden_size_]^T * [1, hidden_size_]
        // M = hidden_size_, N = hidden_size_, K = 1
        ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, hidden_size_,
                                         1, alpha, grad_ar, Htminus1, weight_beta, grad_Rr_local, thread_pool_);
        // Note that the weight beta is always 0. So, we must accumulate ourselves.
        deepcpu::elementwise_sum1(grad_Rr_local, grad_Rr, hidden_size_ * hidden_size_);

        if (!linear_before_reset_) {
          // ah = Xth * Wh^T + (rt (.) Ht-1h) * Rh^T + Wbh + Rbh
          // dL/dRh = dL/dah^T * (rt (.) Ht-1h) ---------- (16)
          // [hidden_size_, hidden_size_] = [1, hidden_size_]^T * [1, hidden_size_]
          // M = hidden_size_, N = hidden_size_, K = 1
          ElementwiseProduct(rt, Htminus1, rt_factor, hidden_size_);
          ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, hidden_size_,
                                           1, alpha, grad_ah, rt_factor, weight_beta, grad_Rh_local, thread_pool_);
          // Note that the weight beta is always 0. So, we must accumulate ourselves.
          deepcpu::elementwise_sum1(grad_Rh_local, grad_Rh, hidden_size_ * hidden_size_);
        } else {
          // ah = Xth * Wh^T + rt (.) (Ht-1h * Rh^T + Rbh) + Wbh
          // dL/dah = G -> dL = G : dah -> dL = G (.) rt : d (Ht-1 * Rh^T)
          // dL/dRh = (dL/dah (.) rt)^T * Ht-1h ---------- (16)
          // [hidden_size_, hidden_size_] = [1, hidden_size_]^T * [1, hidden_size_]
          // M = hidden_size_, N = hidden_size_, K = 1
          ElementwiseProduct(grad_ah, rt, rt_factor, hidden_size_);
          ::onnxruntime::math::Gemm<float>(CblasTrans, CblasNoTrans, hidden_size_, hidden_size_,
                                           1, alpha, rt_factor, Htminus1, weight_beta, grad_Rh_local, thread_pool_);
          // Note that the weight beta is always 0. So, we must accumulate ourselves.
          deepcpu::elementwise_sum1(grad_Rh_local, grad_Rh, hidden_size_ * hidden_size_);
        }
      }

      if (grad_bias_required) {
        // az = Xtz * Wz^T + Ht-1z * Rz^T + Wbz + Rbz
        // dL/dWbz = dL/daz ---------- (17)
        deepcpu::elementwise_sum1(grad_az, grad_Wbz, hidden_size_);

        // az = Xtz * Wz^T + Ht-1z * Rz^T + Wbz + Rbz
        // dL/dRbz = dL/daz ---------- (18)
        deepcpu::elementwise_sum1(grad_az, grad_Rbz, hidden_size_);

        // ar = Xtr * Wr^T + Ht-1r * Rr^T + Wbr + Rbr
        // dL/dWbr = dL/dar ---------- (19)
        deepcpu::elementwise_sum1(grad_ar, grad_Wbr, hidden_size_);

        // ar = Xtr * Wr^T + Ht-1r * Rr^T + Wbr + Rbr
        // dL/dRbr = dL/dar ---------- (20)
        deepcpu::elementwise_sum1(grad_ar, grad_Rbr, hidden_size_);

        // ah = Xth * Wh^T + (rt (.) Ht-1h) * Rh^T + Wbh + Rbh
        // dL/dWbh = dL/dah ---------- (21)
        deepcpu::elementwise_sum1(grad_ah, grad_Wbh, hidden_size_);

        if (!linear_before_reset_) {
          // ah = Xth * Wh^T + (rt (.) Ht-1h) * Rh^T + Wbh + Rbh
          // dL/dRbh = dL/dah ---------- (22)
          deepcpu::elementwise_sum1(grad_ah, grad_Rbh, hidden_size_);
        } else {
          // ah = Xth * Wh^T + rt (.) (Ht-1h * Rh^T + Rbh) + Wbh
          // dL/dRbh = dL/dah (.) rt ---------- (22)
          deepcpu::elementwise_product(grad_ah, rt, grad_Rbh, hidden_size_);
        }
      }

      // dL/dHt-1
      // Ht-1 -> multiplex gate -> Ht-1z
      //                        -> Ht-1r
      //                        -> Ht-1h
      //                        -> Ht-1H
      // dL/dHt-1 = dL/dHt-1z  + dL/dHt-1r + dL/dHt-1h + dL/dHt-1H ---------- (23)
      float recurrence_input_beta = 1.0f;

      // Ht = (1 - zt) (.) ht + zt (.) Ht-1H
      // dL/dHt-1H = dL/dHt (.) zt ---------- (24)
      ElementwiseProduct(grad_Ht, zt, grad_Ht, hidden_size_);

      // az = Xtz * Wz^T + Ht-1z * Rz^T +  Wbz + Rbz
      // dL/dHt-1z = dL/daz * Rz ---------- (25)
      // [1, hidden_size_] = [1, hidden_size_] * [hidden_size_, hidden_size_]
      // M = 1, N = hidden_size_, K = hidden_size_
      ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, hidden_size_,
                                       hidden_size_, alpha, grad_az, Rz, recurrence_input_beta, grad_Ht, thread_pool_);

      // ar = Xtr * Wr^T + Ht-1r * Rr^T +  Wbr + Rbr
      // dL/dHt-1r = dL/dar * Rr ---------- (26)
      // [1, hidden_size_] = [1, hidden_size_] * [hidden_size_, hidden_size_]
      // M = 1, N = hidden_size_, K = hidden_size_
      ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, hidden_size_,
                                       hidden_size_, alpha, grad_ar, Rr, recurrence_input_beta, grad_Ht, thread_pool_);

      if (!linear_before_reset_) {
        // ah = Xth * Wh^T + (rt (.) Ht-1h) * Rh^T + Wbh + Rbh
        // dL/dHt-1h = (dL/dah * Rh) (.) (rt) ---------- (27)
        // [1, hidden_size_] = [1, hidden_size_] * [hidden_size_, hidden_size_]
        // M = 1, N = hidden_size_, K = hidden_size_
        // We need a temporary buffer to store the result of (dL/dah * Rh).
        // Since this is the last step, we can pick any buffer that is not needed anymore (for example grad_ar)
        // to store the intermediate result (making sure to clear the results in grad_ar before writing to it).
        recurrence_input_beta = 0.0f;
        ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, hidden_size_,
                                         hidden_size_, alpha, grad_ah, Rh, recurrence_input_beta, grad_ar, thread_pool_);
        deepcpu::elementwise_product(grad_ar, rt, grad_Ht, hidden_size_);
      } else {
        // ah = Xth * Wh^T + rt (.) (Ht-1h * Rh^T + Rbh) + Wbh
        // dL/dah = G -> dL = G : dah -> dL = G (.) rt : d (Ht-1h * Rh^T)
        // dL/dHt-1h = (dL/dah (.) rt) * Rh ---------- (27)
        // [1, hidden_size_] = [1, hidden_size_] * [hidden_size_, hidden_size_]
        // M = 1, N = hidden_size_, K = hidden_size_
        recurrence_input_beta = 1.0f;
        ElementwiseProduct(grad_ah, rt, rt_factor, hidden_size_);
        ::onnxruntime::math::Gemm<float>(CblasNoTrans, CblasNoTrans, 1, hidden_size_,
                                         hidden_size_, alpha, rt_factor, Rh, recurrence_input_beta, grad_Ht, thread_pool_);
      }
    }
  }
}

template class GRUGradImpl<float>;

}  // namespace onnxruntime::gru
