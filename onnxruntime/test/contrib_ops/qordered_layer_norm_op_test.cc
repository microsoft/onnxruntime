// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qordered_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>  // MLFloat16 or float
static void RunQOrdered_LayerNorm_RowMajor(std::vector<int64_t> const& shape, int axis, float epsilon,
                                           const std::vector<int8_t>& vec_x, float scale_x,
                                           const std::vector<T>& gamma, const std::vector<T>* beta,
                                           float scale_y, const std::vector<int8_t>& vec_y) {
  std::vector<int64_t> bias_shape = {shape.back()};
  OpTester test_qorder("QOrderedLayerNormalization", 1, onnxruntime::kMSDomain);
  test_qorder.AddAttribute("axis", (int64_t)axis);
  test_qorder.AddAttribute("epsilon", epsilon);
  test_qorder.AddInput<int8_t>("X", shape, vec_x);
  test_qorder.AddInput<float>("scale_X", {}, {scale_x});
  test_qorder.AddInput<T>("scale", bias_shape, gamma);
  if (beta) {
    test_qorder.AddInput<T>("B", bias_shape, *beta);
  } else {
    test_qorder.AddOptionalInputEdge<T>();
  }
  test_qorder.AddInput<float>("scale_Y", {}, {scale_y});
  test_qorder.AddOutput<int8_t>("Y", shape, vec_y, false, 0.0f, 0.0f /* abs error */);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test_qorder.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

static void RunQOrdered_LayerNorm_Test(std::vector<int64_t> const& shape,
                                       int axis, bool has_bias) {
  int64_t N = std::accumulate(shape.begin(), shape.end(), int64_t{1LL}, std::multiplies<int64_t>());
  int64_t cols = 0, rows = 0, batch = 0;
  BatchRowColFromShape(shape, batch, rows, cols);

  std::vector<int64_t> bias_shape = {cols};
  std::vector<int8_t> vec_x = GenData<int8_t>(shape, 1.0f);
  RandomValueGenerator random{};
  std::vector<MLFloat16> gamma = ToFloat16(random.Uniform<float>(bias_shape, -1.0f, 1.0f));
  std::vector<MLFloat16> beta(cols);

  if (has_bias) {
    beta = ToFloat16(random.Uniform<float>(bias_shape, -1.0f / 16.0f, -1.0f / 16.0f));
  }

  std::vector<int8_t> vec_y(N);
  float scale_x = 2.0f;
  float scale_y = 2.0f / 128.0f;
  float epsilon = 0.00001f;

  const int8_t* bsrc = vec_x.data();
  int8_t* bdst = vec_y.data();
  for (int b = 0; b < batch; b++) {
    for (int r = 0; r < rows; r++) {
      // Var(X)=E[X*X]− E[X]* E[X]
      int64_t sum_x = std::accumulate(bsrc, bsrc + cols, 0LL, std::plus<int64_t>());
      int64_t sum_x2 = std::accumulate(bsrc, bsrc + cols, 0LL, [](int64_t s, int64_t v) { return s + v * v; });
      float u_x_scaled = scale_x * static_cast<float>(sum_x) / cols;  // no precision lost in static_cast<float>(sum_x)
      float var_x_scaled = static_cast<float>((static_cast<double>(sum_x2) - static_cast<double>(sum_x) * sum_x / cols) / cols) * scale_x * scale_x;
      float var_episilon = var_x_scaled + epsilon;
      float rsqrt_var = 1.0f / ::sqrtf(var_episilon);
      for (int c = 0; c < cols; c++) {
        float v = (scale_x * static_cast<float>(bsrc[c])) - u_x_scaled;
        v = (v * rsqrt_var * gamma[c].ToFloat()) + (has_bias ? beta[c].ToFloat() : 0.0f);
        v = v / scale_y;
        v = std::max(-128.0f, std::min(v, 127.0f));
        bdst[c] = static_cast<int8_t>((int)std::nearbyintf(v));
      }
      bsrc += cols;
      bdst += cols;
    }
  }

  RunQOrdered_LayerNorm_RowMajor(shape, axis, epsilon, vec_x, scale_x, gamma,
                                 has_bias ? &beta : nullptr, scale_y, vec_y);
}

TEST(QOrderedTest, LayerNormalization_RowMajor) {
  float scale_x = 1.0;

  int64_t batch = 2;
  int64_t sequence = 2;
  int64_t hidden = 32;

  std::vector<int8_t> vec_x = {
      -103, -24, -11, 125, 103, -117, 44, -109, 27, 122, 113,
      0, -7, 26, 110, -34, 28, -61, -107, -35, -54, 69,
      -3, -58, -119, 61, -19, -115, 35, 76, 16, 54,

      -103, -24, -11, 125, 103, -117, 44, -109, 27, 122, 113,
      0, -7, 26, 110, -34, 28, -61, -107, -35, -54, 69,
      -3, -58, -119, 61, -19, -115, 35, 76, 16, 54,

      -103, -24, -11, 125, 103, -117, 44, -109, 27, 122, 113,
      0, -7, 26, 110, -34, 28, -61, -107, -35, -54, 69,
      -3, -58, -119, 61, -19, -115, 35, 76, 16, 54,

      -103, -24, -11, 125, 103, -117, 44, -109, 27, 122, 113,
      0, -7, 26, 110, -34, 28, -61, -107, -35, -54, 69,
      -3, -58, -119, 61, -19, -115, 35, 76, 16, 54};

  std::vector<float> gamma_fp32 = {
      0.9536133f, 0.9160156f, 1.3310547f, 1.0068359f, 0.8095703f,
      1.0126953f, 0.8876953f, 1.4316406f, 1.0947266f, 1.0498047f,
      0.7373047f, 1.0615234f, 1.015625f, 1.0751953f, 1.0068359f,
      1.0908203f, 1.2011719f, 1.1962891f, 0.91796875f, 1.0947266f,
      1.3183594f, 1.0185547f, 1.0791016f, 1.0273438f, 0.8364258f,
      0.94873047f, 1.0292969f, 1.09375f, 1.0371094f, 1.1240234f,
      1.4384766f, 1.0068359f};

  auto gamma_fp16 = ToFloat16(gamma_fp32);

  std::vector<float> beta_fp32 = {
      0.01411438f, 0.18273926f, -0.12414551f, 0.09887695f, -0.1114502f,
      -0.08227539f, -0.04598999f, -0.11322021f, 0.12731934f, -0.06591797f,
      -0.00662994f, 0.04962158f, -0.04281616f, 0.07476807f, 0.23010254f,
      0.1036377f, 0.10852051f, 0.10919189f, -0.02905273f, -0.0512085f,
      -0.1194458f, 0.02661133f, 0.05789185f, -0.05239868f, 0.17907715f,
      -0.01765442f, -0.12255859f, -0.09729004f, 0.06591797f, 0.02258301f,
      -0.01844788f, -0.11999512f};

  auto beta_fp16 = ToFloat16(beta_fp32);

  std::vector<int8_t> vec_y = {
      -84, -8, -22, 114, 64, -108, 30, -128, 33, 105, 71,
      2, -10, 28, 109, -26, 35, -57, -87, -37, -70, 61,
      0, -56, -75, 48, -26, -115, 35, 74, 17, 38,

      -84, -8, -22, 114, 64, -108, 30, -128, 33, 105, 71,
      2, -10, 28, 109, -26, 35, -57, -87, -37, -70, 61,
      0, -56, -75, 48, -26, -115, 35, 74, 17, 38,

      -84, -8, -22, 114, 64, -108, 30, -128, 33, 105, 71,
      2, -10, 28, 109, -26, 35, -57, -87, -37, -70, 61,
      0, -56, -75, 48, -26, -115, 35, 74, 17, 38,

      -84, -8, -22, 114, 64, -108, 30, -128, 33, 105, 71,
      2, -10, 28, 109, -26, 35, -57, -87, -37, -70, 61,
      0, -56, -75, 48, -26, -115, 35, 74, 17, 38};

  float scale_y = 1.0 / 64.0f;

  RunQOrdered_LayerNorm_RowMajor({2, 2, 32}, -1, 0.00001f, vec_x, scale_x,
                                 gamma_fp16, &beta_fp16, scale_y, vec_y);

  RunQOrdered_LayerNorm_RowMajor({2, 2, 32}, -1, 0.00001f, vec_x, scale_x,
                                 gamma_fp32, &beta_fp32, scale_y, vec_y);
}
}  // namespace test
}  // namespace onnxruntime
