// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/rnn/rnn_helpers.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <unordered_map>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/cpu/rnn/rnn_activation_functors.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
// TODO: fix the warnings
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26451)
#endif
namespace onnxruntime {
namespace rnn {
namespace detail {

using namespace ::onnxruntime::common;

Status ValidateCommonRnnInputs(const Tensor& X,
                               const TensorShape& W_shape,
                               const TensorShape& R_shape,
                               const Tensor* B,
                               int WRB_dim_1_multipler,
                               const Tensor* sequence_lens,
                               const Tensor* initial_h,
                               int64_t num_directions,
                               int64_t hidden_size) {
  auto& X_shape = X.Shape();

  int64_t seq_length = X_shape[0];
  int64_t batch_size = X_shape[1];
  int64_t input_size = X_shape[2];

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
        initial_h_shape[0] != num_directions ||
        initial_h_shape[1] != batch_size ||
        initial_h_shape[2] != hidden_size)

      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input initial_h must have shape {",
                             num_directions, ",", batch_size, ",", hidden_size, "}. Actual:", initial_h_shape);
  }

  return Status::OK();
}  // namespace detail

// map of arg name and whether the alpha and/or beta arguments are required
static std::unordered_map<std::string, std::pair<bool, bool>> NameToArgUsageMap{
    {"affine", {true, true}},
    {"relu", {false, false}},
    {"leakyrelu", {true, false}},
    {"thresholdedrelu", {true, false}},
    {"tanh", {false, false}},
    {"scaledtanh", {true, true}},
    {"sigmoid", {false, false}},
    {"hardsigmoid", {true, true}},
    {"elu", {true, false}},
    {"softsign", {false, false}},
    {"softplus", {false, false}}};

// map of alpha/beta defaults
static std::unordered_map<std::string, std::pair<float, float>>
    NameToArgDefaultsMap{{"leakyrelu", {0.01f, 0.f}},
                         {"hardsigmoid", {0.2f, 0.5f}},
                         {"elu", {1.0f, 0.f}}};

std::string NormalizeActivationArgumentAndGetAlphaBetaCount(const std::string& activation,
                                                            std::vector<float>::const_iterator& cur_alpha,
                                                            const std::vector<float>::const_iterator& end_alpha,
                                                            std::vector<float>::const_iterator& cur_beta,
                                                            const std::vector<float>::const_iterator& end_beta,
                                                            float& alpha, float& beta) {
  std::string name(activation);
  std::transform(name.begin(), name.end(), name.begin(),
                 [](const unsigned char i) { return static_cast<char>(::tolower(i)); });

  auto usage_entry = NameToArgUsageMap.find(name);
  if (usage_entry == NameToArgUsageMap.end()) {
    ORT_THROW(
        "Expecting activation to be one of Affine, Relu, LeakyRelu, "
        "ThresholdedRelu, Tanh, ScaledTanh, Sigmoid, HardSigmoid, "
        "Elu, Softsign, Softplus. Got " +
        activation);
  }

  bool needs_alpha = usage_entry->second.first;
  bool needs_beta = usage_entry->second.second;

  auto set_if_needed = [](bool needed,
                          std::vector<float>::const_iterator& in,
                          const std::vector<float>::const_iterator& in_end,
                          const float default_val,
                          float& out) {
    if (needed) {
      if (in != in_end) {
        out = *in;
        ++in;
      } else {
        out = default_val;
      }
    }
  };

  auto defaults_entry = NameToArgDefaultsMap.find(name);
  if (defaults_entry != NameToArgDefaultsMap.end()) {
    set_if_needed(needs_alpha, cur_alpha, end_alpha, defaults_entry->second.first, alpha);
    set_if_needed(needs_beta, cur_beta, end_beta, defaults_entry->second.second, beta);
  } else {
    set_if_needed(needs_alpha, cur_alpha, end_alpha, 0.f, alpha);
    set_if_needed(needs_beta, cur_beta, end_beta, 0.f, beta);
  }

  return name;
}

ActivationFuncs::ActivationFuncs(const std::vector<std::string>& funcs,
                                 const std::vector<float>& alphas,
                                 const std::vector<float>& betas) {
  auto cur_alpha = alphas.cbegin();
  auto end_alpha = alphas.cend();
  auto cur_beta = betas.cbegin();
  auto end_beta = betas.cend();

  for (const auto& input_func : funcs) {
    float alpha = 0.f;
    float beta = 0.f;
    std::string func = detail::NormalizeActivationArgumentAndGetAlphaBetaCount(
        input_func, cur_alpha, end_alpha, cur_beta, end_beta, alpha, beta);

    entries_.push_back(Entry{func, alpha, beta});
  }
}

#if defined(DUMP_MATRIXES)
void DumpMatrixImpl(const std::string& name, const float* src, int row, int col, int offset, int col_width) {
  std::cout << "Dump matrix: " << name << std::endl;

  if (col_width == -1) col_width = col;

  for (int r = 0; r < row; r++) {
    for (int c = 0; c < col; c++) {
      int index = r * col_width + offset + c;
      std::cout << std::setw(12) << std::setprecision(8) << src[index];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
#endif

void ComputeGemm(const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const float* A,
                 const float* A_end,
                 const GemmWeights<float>& weights,
                 const float beta,
                 float* C,
                 float* C_end,
                 const int ldc,
                 uint8_t* /* quantized_A_buffer */,
                 int32_t* /* quantize_agg_C_buffer */,
                 concurrency::ThreadPool* thread_pool) {
  // validate all the inputs
  // need to use the lda/ldb/ldc strides which should be >= the columns for the span
  ORT_ENFORCE(A + (M * K) <= A_end);
  ORT_ENFORCE(C + (M * ldc - (ldc - N)) <= C_end);

  if (weights.is_prepacked_) {
    MlasGemm(
        CblasNoTrans,
        M, N, K, alpha,
        A, K,
        weights.buffer_, beta,
        C, ldc, thread_pool);
  } else {
    ::onnxruntime::math::GemmEx<float>(
        CblasNoTrans, CblasTrans,
        M, N, K, alpha,
        A, K,
        static_cast<const float*>(weights.buffer_), K, beta,
        C, ldc, thread_pool);
  }
}

void ComputeGemm(const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const float* A,
                 const float* A_end,
                 const GemmWeights<uint8_t>& weights,
                 const float beta,
                 float* C,
                 float* C_end,
                 const int ldc,
                 uint8_t* quantized_A_buffer,
                 int32_t* quantize_agg_C_buffer,
                 concurrency::ThreadPool* thread_pool) {
  // validate all the inputs
  // need to use the lda/ldb/ldc strides which should be >= the columns for the span
  ORT_ENFORCE(A + (M * K) <= A_end);
  ORT_ENFORCE(C + (M * ldc - (ldc - N)) <= C_end);
  ORT_ENFORCE(weights.quant_para_);
  ORT_ENFORCE(alpha == 1.0f && (beta == 0.0f || beta == 1.0f), "Quantized GEMM only support alpha equal to 1.0f and beta equal to 0.0f or 1.0f");

  float a_scale;
  uint8_t a_zero_point;
  GetQuantizationParameter(A, M * K, a_scale, a_zero_point, thread_pool);

  // quantize the data
  ParQuantizeLinearStd(A, quantized_A_buffer, M * K, a_scale, a_zero_point, thread_pool);

  bool b_is_signed = weights.quant_para_->is_signed;
  uint8_t b_zero_point = weights.quant_para_->zero_point ? *static_cast<const uint8_t*>(weights.quant_para_->zero_point) : 0;

  std::vector<float> scale_multiplier(weights.quant_para_->scale_size);
  for (size_t s = 0; s < weights.quant_para_->scale_size; s++) {
    scale_multiplier[s] = a_scale * (weights.quant_para_->scale[s]);
  }

  size_t ld_C_buffer = ldc;
  int32_t* C_buffer = reinterpret_cast<int32_t*>(C);
  if (beta == 1.0f) {
    C_buffer = quantize_agg_C_buffer;
    ld_C_buffer = static_cast<size_t>(N);
  }

  MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR output_processor(
      C, ldc, scale_multiplier.data(), nullptr,
      beta == 1.0f ? MLAS_QGEMM_OUTPUT_MODE::AccumulateMode : MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
      scale_multiplier.size() == 1 ? MLAS_QUANTIZATION_GRANULARITY::PerMatrix : MLAS_QUANTIZATION_GRANULARITY::PerColumn);

  MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = static_cast<size_t>(M);
  gemm_shape.N = static_cast<size_t>(N);
  gemm_shape.K = static_cast<size_t>(K);
  gemm_shape.BIsSigned = b_is_signed;

  MLAS_GEMM_QUANT_DATA_PARAMS gemm_params;
  gemm_params.A = quantized_A_buffer;
  gemm_params.lda = static_cast<size_t>(K);
  gemm_params.ZeroPointA = a_zero_point;
  gemm_params.B = weights.buffer_;
  gemm_params.ldb = static_cast<size_t>(N);
  gemm_params.ZeroPointB = &b_zero_point;
  gemm_params.BIsPacked = weights.is_prepacked_;
  gemm_params.C = C_buffer;
  gemm_params.ldc = ld_C_buffer;
  gemm_params.OutputProcessor = &output_processor;

  MlasGemm(gemm_shape, gemm_params, thread_pool);
}

namespace deepcpu {

constexpr float alpha_1 = 4.89352455891786e-03f;
constexpr float alpha_3 = 6.37261928875436e-04f;
constexpr float alpha_5 = 1.48572235717979e-05f;
constexpr float alpha_7 = 5.12229709037114e-08f;
constexpr float alpha_9 = -8.60467152213735e-11f;
constexpr float alpha_11 = 2.00018790482477e-13f;
constexpr float alpha_13 = -2.76076847742355e-16f;

constexpr float beta_0 = 4.89352518554385e-03f;
constexpr float beta_2 = 2.26843463243900e-03f;
constexpr float beta_4 = 1.18534705686654e-04f;
constexpr float beta_6 = 1.19825839466702e-06f;

constexpr float sigmoid_bound = 20.0f;
constexpr float tanh_bound = 10.0f;

#if defined(__GNUC__) && !defined(__wasm__)
#define restrict __restrict__
#elif defined(_MSC_VER)
#define restrict __restrict
#else
#define restrict
#endif

inline void clip_for_sigmoid_in_place(float* ps, int c) {
  for (int i = 0; i < c; i++) {
    if (ps[i] < -sigmoid_bound)
      ps[i] = -sigmoid_bound;
    else if (ps[i] > sigmoid_bound)
      ps[i] = sigmoid_bound;
  }
}

inline void clip_for_tanh_in_place(float* ps, int c) {
  for (int i = 0; i < c; i++) {
    if (ps[i] < -tanh_bound)
      ps[i] = -tanh_bound;
    else if (ps[i] > tanh_bound)
      ps[i] = tanh_bound;
  }
}

inline void clip_for_sigmoid(const float* ps, float* pd, int c) {
  for (int i = 0; i < c; i++) {
    if (ps[i] < -sigmoid_bound)
      pd[i] = -sigmoid_bound;
    else if (ps[i] > sigmoid_bound)
      pd[i] = sigmoid_bound;
    else
      pd[i] = ps[i];
  }
}

inline void clip_for_tanh(const float* ps, float* pd, int c) {
  for (int i = 0; i < c; i++) {
    if (ps[i] < -tanh_bound)
      pd[i] = -tanh_bound;
    else if (ps[i] > tanh_bound)
      pd[i] = tanh_bound;
    else
      pd[i] = ps[i];
  }
}

void add_bias_into_ignore(const float* ps, const float* pd, int c) {
  ORT_UNUSED_PARAMETER(ps);
  ORT_UNUSED_PARAMETER(pd);
  ORT_UNUSED_PARAMETER(c);
}

void add_bias_into(const float* ps, float* pd, int c) {
  for (int i = 0; i < c; i++) {
    pd[i] += ps[i];
  }
}

void clip(const float b, float* pd, int c) {
  for (int i = 0; i < c; i++) {
    float x = pd[i];
    if (x > b)
      pd[i] = b;
    else if (x < -b)
      pd[i] = -b;
  }
}

void clip_ignore_bias(const float b, const float* pb, float* pd, int c) {
  ORT_UNUSED_PARAMETER(pb);

  for (int i = 0; i < c; i++) {
    float x = pd[i];
    if (x > b)
      pd[i] = b;
    else if (x < -b)
      pd[i] = -b;
    else
      pd[i] = x;
  }
}

void clip_add_bias(const float b, const float* restrict pb, float* restrict pd, int c) {
  for (int i = 0; i < c; i++) {
    float x = pd[i] + pb[i];
    x = std::min(b, x);
    x = std::max(-b, x);
    pd[i] = x;
  }
}

void sigmoid_m(const float* restrict ps1, float* restrict ps1_c, const float* restrict ps2, float* restrict pd, int c,
               const float alpha, const float beta) {
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);
  ORT_UNUSED_PARAMETER(ps1_c);

  MlasComputeLogistic(ps1, pd, c);
  for (int i = 0; i < c; i++) {
    pd[i] *= ps2[i];
  }
}

void tanh_m(const float* restrict ps1, float* restrict ps1_c, const float* restrict ps2, float* restrict pd, int c,
            const float alpha, const float beta) {
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);
  ORT_UNUSED_PARAMETER(ps1_c);

  MlasComputeTanh(ps1, pd, c);
  for (int i = 0; i < c; i++) {
    pd[i] *= ps2[i];
  }
}

void relu_m(const float* restrict ps1, float* restrict ps1_c, const float* restrict ps2, float* restrict pd, int c,
            const float alpha, const float beta) {
  ORT_UNUSED_PARAMETER(ps1_c);
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);

  for (int i = 0; i < c; i++) {
    const float max = ps1[i] > 0 ? ps1[i] : 0.0f;
    pd[i] = ps2[i] * max;
  }
}

void composed_m(const float* ps1, const float* ps1_c, const float* ps2, float* pd, int c,
                std::function<float(float, float, float)> func, float alpha, float beta) {
  ORT_UNUSED_PARAMETER(ps1_c);
  for (int i = 0; i < c; i++) {
    pd[i] = ps2[i] * func(ps1[i], alpha, beta);
  }
}

void sigmoid_exact_m(const float* ps1, const float* ps1_c, const float* ps2, float* pd, int c, float alpha,
                     float beta) {
  ORT_UNUSED_PARAMETER(ps1_c);
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);

  for (int i = 0; i < c; i++) {
    float x = ps1[i];
    pd[i] = ps2[i] / (1 + ::std::exp(-x));
  }
}

void tanh_exact_m(const float* ps1, const float* ps1_c, const float* ps2, float* pd, int c, float alpha, float beta) {
  ORT_UNUSED_PARAMETER(ps1_c);
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);

  for (int i = 0; i < c; i++) {
    pd[i] = ::std::tanh(ps1[i]) * ps2[i];
  }
}

void sigmoid(float* pd, int c, float alpha, float beta) {
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);

  MlasComputeLogistic(pd, pd, c);
}

void tanh(float* pd, int c, float alpha, float beta) {
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);

  MlasComputeTanh(pd, pd, c);
}

void relu(float* pd, int c, float alpha, float beta) {
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);

  MLAS_ACTIVATION activation;
  activation.ActivationKind = MlasReluActivation;
  MlasActivation(&activation, pd, nullptr, 1, c, c);
}

void sigmoid_exact(float* pd, int c, float alpha, float beta) {
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);

  for (int i = 0; i < c; i++) {
    float x = pd[i];
    pd[i] = 1.0f / (1 + ::std::exp(-x));
  }
}

void tanh_exact(float* pd, int c, float alpha, float beta) {
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);

  for (int i = 0; i < c; i++) {
    float x = pd[i];
    pd[i] = ::std::tanh(x);
  }
}

// Help compiler simply and correctly optimize for pcurr == pprev case.
// Although without this in_place(), if restrict pprev and pcur, compiler could also work.
// Yet this in_place() follow the restrict semantic better.
static void merge_lstm_gates_to_memory_in_place(const float* restrict pi, const float* restrict pf,
                                                const float* restrict pg, float* restrict pcurr, int c) {
  for (int i = 0; i < c; i++) {
    pcurr[i] = pcurr[i] * pf[i] + pi[i] * pg[i];
  }
}

void merge_lstm_gates_to_memory(const float* pprev, const float* restrict pi, const float* restrict pf,
                                const float* restrict pg, float* pcurr, int c) {
  if (pprev == pcurr) {
    merge_lstm_gates_to_memory_in_place(pi, pf, pg, pcurr, c);
    return;
  }

  for (int i = 0; i < c; i++) {
    pcurr[i] = pprev[i] * pf[i] + pi[i] * pg[i];
  }
}

void gru_reset_gate_tanh(const float* ps1, float* ps2, float* pd, int c, float alpha, float beta) {
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);

  clip_for_tanh_in_place(ps2, c);

  for (int i = 0; i < c; i++) {
    float x = ps2[i];
    float x2 = x * x;
    float p = x2 * alpha_13 + alpha_11;
    p = x2 * p + alpha_9;
    p = x2 * p + alpha_7;
    p = x2 * p + alpha_5;
    p = x2 * p + alpha_3;
    p = x2 * p + alpha_1;
    p = x * p;
    float q = x2 * beta_6 + beta_4;
    q = x2 * q + beta_2;
    q = x2 * q + beta_0;
    pd[i] = ps1[i] * p / q;
  }
}

void gru_reset_gate_sigmoid(const float* ps1, float* ps2, float* pd, int c, float alpha, float beta) {
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);

  clip_for_sigmoid_in_place(ps2, c);

  for (int i = 0; i < c; i++) {
    float x = 0.5f * ps2[i];
    float x2 = x * x;
    float p = x2 * alpha_13 + alpha_11;
    p = x2 * p + alpha_9;
    p = x2 * p + alpha_7;
    p = x2 * p + alpha_5;
    p = x2 * p + alpha_3;
    p = x2 * p + alpha_1;
    p = x * p;
    float q = x2 * beta_6 + beta_4;
    q = x2 * q + beta_2;
    q = x2 * q + beta_0;
    pd[i] = ps1[i] * 0.5f * (1 + p / q);
  }
}

void gru_reset_gate_relu(const float* ps1, float* ps2, float* pd, int c, float alpha, float beta) {
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);

  for (int i = 0; i < c; i++) {
    const auto max = ps2[i] > 0 ? ps2[i] : 0.0f;
    pd[i] = ps1[i] * max;
  }
}

void gru_reset_gate_composed(const float* ps1, float* ps2, float* pd, int c,
                             std::function<float(float, float, float)> func, float alpha, float beta) {
  for (int i = 0; i < c; i++) {
    pd[i] = ps1[i] * func(ps2[i], alpha, beta);
  }
}

void gru_output_gate_tanh(float* ph, const float* pz, const float* ps, float* po, int c, float alpha, float beta) {
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);

  clip_for_tanh_in_place(ph, c);

  for (int i = 0; i < c; i++) {
    float x = ph[i];
    float x2 = x * x;
    float p = x2 * alpha_13 + alpha_11;
    p = x2 * p + alpha_9;
    p = x2 * p + alpha_7;
    p = x2 * p + alpha_5;
    p = x2 * p + alpha_3;
    p = x2 * p + alpha_1;
    p = x * p;
    float q = x2 * beta_6 + beta_4;
    q = x2 * q + beta_2;
    q = x2 * q + beta_0;
    po[i] = (1 - pz[i]) * (p / q) + pz[i] * ps[i];
  }
}

void gru_output_gate_relu(float* ph, const float* pz, const float* ps, float* po, int c, float alpha, float beta) {
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);

  for (int i = 0; i < c; i++) {
    float max = ph[i] > 0 ? ph[i] : 0.0f;
    po[i] = (1 - pz[i]) * max + pz[i] * ps[i];
  }
}

void gru_output_gate_composed(float* ph, const float* pz, const float* ps, float* po, int c,
                              std::function<float(float, float, float)> func, float alpha, float beta) {
  for (int i = 0; i < c; i++) {
    po[i] = (1 - pz[i]) * func(ph[i], alpha, beta) + pz[i] * ps[i];
  }
}

void gru_output_gate_sigmoid(float* ph, const float* pz, const float* ps, float* po, int c, float alpha, float beta) {
  ORT_UNUSED_PARAMETER(alpha);
  ORT_UNUSED_PARAMETER(beta);

  clip_for_sigmoid_in_place(ph, c);

  for (int i = 0; i < c; i++) {
    float x = 0.5f * (ph[i]);
    float x2 = x * x;
    float p = x2 * alpha_13 + alpha_11;
    p = x2 * p + alpha_9;
    p = x2 * p + alpha_7;
    p = x2 * p + alpha_5;
    p = x2 * p + alpha_3;
    p = x2 * p + alpha_1;
    p = x * p;
    float q = x2 * beta_6 + beta_4;
    q = x2 * q + beta_2;
    q = x2 * q + beta_0;
    po[i] = (1 - pz[i]) * 0.5f * (1 + (p / q)) + pz[i] * ps[i];
  }
}

void composed_activation_func(float* ps, int c, std::function<float(float, float, float)> func, float alpha,
                              float beta) {
  for (int i = 0; i < c; i++) {
    ps[i] = func(ps[i], alpha, beta);
  }
}

void composed_lstm_merge_gates_func(float* ps, int c, std::function<float(float, float, float)> func, float alpha,
                                    float beta) {
  for (int i = 0; i < c; i++) {
    ps[i] = func(ps[i], alpha, beta);
  }
}

void composed_gru_reset_gate_func(float* ps, int c, std::function<float(float, float, float)> func, float alpha,
                                  float beta) {
  for (int i = 0; i < c; i++) {
    ps[i] = func(ps[i], alpha, beta);
  }
}

void composed_gru_output_gate_func(float* ps, int c, std::function<float(float, float, float)> func, float alpha,
                                   float beta) {
  for (int i = 0; i < c; i++) {
    ps[i] = func(ps[i], alpha, beta);
  }
}

ActivationFuncPtr ActivationFuncByName(const std::string& func) {
  if (func == "sigmoid")
    return sigmoid;

  if (func == "tanh")
    return tanh;

  if (func == "relu")
    return relu;

  if (func == "affine")
    return
        [](float* ps, int c, float alpha, float beta) { composed_activation_func(ps, c, Affine<float>, alpha, beta); };

  if (func == "leakyrelu")
    return [](float* ps, int c, float alpha, float beta) {
      composed_activation_func(ps, c, LeakyRelu<float>, alpha, beta);
    };

  if (func == "thresholdedrelu")
    return [](float* ps, int c, float alpha, float beta) {
      composed_activation_func(ps, c, ThresholdedRelu<float>, alpha, beta);
    };

  if (func == "scaledtanh")
    return [](float* ps, int c, float alpha, float beta) {
      composed_activation_func(ps, c, ScaledTanh<float>, alpha, beta);
    };

  if (func == "hardsigmoid")
    return [](float* ps, int c, float alpha, float beta) {
      composed_activation_func(ps, c, HardSigmoid<float>, alpha, beta);
    };

  if (func == "elu")
    return [](float* ps, int c, float alpha, float beta) { composed_activation_func(ps, c, Elu<float>, alpha, beta); };

  if (func == "softsign")
    return [](float* ps, int c, float alpha, float beta) {
      composed_activation_func(ps, c, Softsign<float>, alpha, beta);
    };

  if (func == "softplus")
    return [](float* ps, int c, float alpha, float beta) {
      composed_activation_func(ps, c, Softplus<float>, alpha, beta);
    };

  ORT_THROW("Invalid activation function of ", func);
}

LstmMergeGatesFuncPtr LstmMergeGatesFuncByName(const std::string& func) {
  if (func == "sigmoid")
    return sigmoid_m;

  if (func == "tanh")
    return tanh_m;

  if (func == "relu")
    return relu_m;

  if (func == "affine")
    return [](const float* ps1, float* ps1_c, const float* ps2, float* ps3, int c, float alpha, float beta) {
      composed_m(ps1, ps1_c, ps2, ps3, c, Affine<float>, alpha, beta);
    };

  if (func == "leakyrelu")
    return [](const float* ps1, float* ps1_c, const float* ps2, float* ps3, int c, float alpha, float beta) {
      composed_m(ps1, ps1_c, ps2, ps3, c, LeakyRelu<float>, alpha, beta);
    };

  if (func == "thresholdedrelu")
    return [](const float* ps1, float* ps1_c, const float* ps2, float* ps3, int c, float alpha, float beta) {
      composed_m(ps1, ps1_c, ps2, ps3, c, ThresholdedRelu<float>, alpha, beta);
    };

  if (func == "scaledtanh")
    return [](const float* ps1, float* ps1_c, const float* ps2, float* ps3, int c, float alpha, float beta) {
      composed_m(ps1, ps1_c, ps2, ps3, c, ScaledTanh<float>, alpha, beta);
    };

  if (func == "hardsigmoid")
    return [](const float* ps1, float* ps1_c, const float* ps2, float* ps3, int c, float alpha, float beta) {
      composed_m(ps1, ps1_c, ps2, ps3, c, HardSigmoid<float>, alpha, beta);
    };

  if (func == "elu")
    return [](const float* ps1, float* ps1_c, const float* ps2, float* ps3, int c, float alpha, float beta) {
      composed_m(ps1, ps1_c, ps2, ps3, c, Elu<float>, alpha, beta);
    };

  if (func == "softsign")
    return [](const float* ps1, float* ps1_c, const float* ps2, float* ps3, int c, float alpha, float beta) {
      composed_m(ps1, ps1_c, ps2, ps3, c, Softsign<float>, alpha, beta);
    };

  if (func == "softplus")
    return [](const float* ps1, float* ps1_c, const float* ps2, float* ps3, int c, float alpha, float beta) {
      composed_m(ps1, ps1_c, ps2, ps3, c, Softplus<float>, alpha, beta);
    };

  ORT_THROW("Invalid LSTM merge activation function of ", func);
}

GruResetGateFuncPtr GruResetGateFuncByName(const std::string& func) {
  if (func == "sigmoid")
    return gru_reset_gate_sigmoid;

  if (func == "tanh")
    return gru_reset_gate_tanh;

  if (func == "relu")
    return gru_reset_gate_relu;

  if (func == "affine")
    return [](const float* ps1, float* ps2, float* ps3, int c, float alpha, float beta) {
      gru_reset_gate_composed(ps1, ps2, ps3, c, Affine<float>, alpha, beta);
    };

  if (func == "leakyrelu")
    return [](const float* ps1, float* ps2, float* ps3, int c, float alpha, float beta) {
      gru_reset_gate_composed(ps1, ps2, ps3, c, LeakyRelu<float>, alpha, beta);
    };

  if (func == "thresholdedrelu")
    return [](const float* ps1, float* ps2, float* ps3, int c, float alpha, float beta) {
      gru_reset_gate_composed(ps1, ps2, ps3, c, ThresholdedRelu<float>, alpha, beta);
    };

  if (func == "scaledtanh")
    return [](const float* ps1, float* ps2, float* ps3, int c, float alpha, float beta) {
      gru_reset_gate_composed(ps1, ps2, ps3, c, ScaledTanh<float>, alpha, beta);
    };

  if (func == "hardsigmoid")
    return [](const float* ps1, float* ps2, float* ps3, int c, float alpha, float beta) {
      gru_reset_gate_composed(ps1, ps2, ps3, c, HardSigmoid<float>, alpha, beta);
    };

  if (func == "elu")
    return [](const float* ps1, float* ps2, float* ps3, int c, float alpha, float beta) {
      gru_reset_gate_composed(ps1, ps2, ps3, c, Elu<float>, alpha, beta);
    };

  if (func == "softsign")
    return [](const float* ps1, float* ps2, float* ps3, int c, float alpha, float beta) {
      gru_reset_gate_composed(ps1, ps2, ps3, c, Softsign<float>, alpha, beta);
    };

  if (func == "softplus")
    return [](const float* ps1, float* ps2, float* ps3, int c, float alpha, float beta) {
      gru_reset_gate_composed(ps1, ps2, ps3, c, Softplus<float>, alpha, beta);
    };

  ORT_THROW("Invalid GRU reset gate activation function: ", func);
}

GruOutputGateFuncPtr GruOutputGateFuncByName(const std::string& func) {
  if (func == "sigmoid")
    return gru_output_gate_sigmoid;

  if (func == "tanh")
    return gru_output_gate_tanh;

  if (func == "relu")
    return gru_output_gate_relu;

  if (func == "affine")
    return [](float* ps1, const float* ps2, const float* ph, float* ps3, int c, float alpha, float beta) {
      gru_output_gate_composed(ps1, ps2, ph, ps3, c, Affine<float>, alpha, beta);
    };

  if (func == "leakyrelu")
    return [](float* ps1, const float* ps2, const float* ph, float* ps3, int c, float alpha, float beta) {
      gru_output_gate_composed(ps1, ps2, ph, ps3, c, LeakyRelu<float>, alpha, beta);
    };

  if (func == "thresholdedrelu")
    return [](float* ps1, const float* ps2, const float* ph, float* ps3, int c, float alpha, float beta) {
      gru_output_gate_composed(ps1, ps2, ph, ps3, c, ThresholdedRelu<float>, alpha, beta);
    };

  if (func == "scaledtanh")
    return [](float* ps1, const float* ps2, const float* ph, float* ps3, int c, float alpha, float beta) {
      gru_output_gate_composed(ps1, ps2, ph, ps3, c, ScaledTanh<float>, alpha, beta);
    };

  if (func == "hardsigmoid")
    return [](float* ps1, const float* ps2, const float* ph, float* ps3, int c, float alpha, float beta) {
      gru_output_gate_composed(ps1, ps2, ph, ps3, c, HardSigmoid<float>, alpha, beta);
    };

  if (func == "elu")
    return [](float* ps1, const float* ps2, const float* ph, float* ps3, int c, float alpha, float beta) {
      gru_output_gate_composed(ps1, ps2, ph, ps3, c, Elu<float>, alpha, beta);
    };

  if (func == "softsign")
    return [](float* ps1, const float* ps2, const float* ph, float* ps3, int c, float alpha, float beta) {
      gru_output_gate_composed(ps1, ps2, ph, ps3, c, Softsign<float>, alpha, beta);
    };

  if (func == "softplus")
    return [](float* ps1, const float* ps2, const float* ph, float* ps3, int c, float alpha, float beta) {
      gru_output_gate_composed(ps1, ps2, ph, ps3, c, Softplus<float>, alpha, beta);
    };

  ORT_THROW("Invalid GRU hidden gate activation function: ", func);
}

}  // namespace deepcpu
}  // namespace detail
}  // namespace rnn
}  // namespace onnxruntime
