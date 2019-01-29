// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
using namespace std;
namespace onnxruntime {
namespace test {

namespace {

void FindMinMax(const vector<float>& vec, float* min,
                float* max) {
  *min = *max = 0;
  *min = *std::min_element(vec.begin(), vec.end());
  *max = *std::max_element(vec.begin(), vec.end());
}

// uses quantization range 0-255
void FindScaleAndZeroPoint(float min, float max, float* scale, uint8_t* zero_point) {
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);
  float qmin = 0;
  float qmax = 255;

  *scale = (max - min) / (qmax - qmin);
  const auto initial_zero_point = qmin - min / *scale;
  *zero_point = static_cast<uint8_t>(std::round(std::max(0.f, std::min(255.f, initial_zero_point))));
}

void Quantize(float scale, uint8_t zero_point,
              const std::vector<float>& input, std::vector<uint8_t>* input_quantized) {
  for (size_t i = 0; i < input.size(); i++) {
    const float clamped_val = std::max(0.f, std::min(255.f, std::round(static_cast<float>(input[i]) / scale) + zero_point));
    (*input_quantized)[i] = static_cast<uint8_t>(clamped_val);
  }
}

TEST(ConvTest, QLinearConv2DTest) {
  OpTester test("QLinearConv", 1, onnxruntime::kMSDomain);

  vector<float> X = {0.45246148109436035f, 0.15498268604278564f, 0.11199361085891724f, -0.39421093463897705f,
                     0.2626858949661255f, 0.13414543867111206f, -0.27184486389160156f, -0.43028733134269714f,
                     -0.26825493574142456f, 0.3893144130706787f, -0.13631996512413025f, -0.009590476751327515f,
                     -0.48771554231643677f, -0.25256502628326416f, -0.2812897562980652f, 0.4043201804161072f,
                     0.07795023918151855f, 0.326981782913208f, 0.13114392757415771f, -0.4416425824165344f,
                     0.12446999549865723f, 0.36739975214004517f, 0.1698915958404541f, 0.2008744478225708f,
                     0.23339951038360596f, 0.38613730669021606f, 0.11117297410964966f, 0.3877097964286804f,
                     0.20812749862670898f, -0.34297940135002136f, -0.029246658086776733f, -0.20483523607254028f,
                     -0.19244328141212463f, -0.11104947328567505f, -0.32830488681793213f, -0.01800677180290222f,
                     0.3618946671485901f, -0.40949052572250366f, -0.18248388171195984f, -0.3349453806877136f,
                     -0.34091079235076904f, 0.006497859954833984f, 0.4537564516067505f, 0.08006560802459717f,
                     -0.14788749814033508f, 0.034442365169525146f, -0.33322954177856445f, 0.06049239635467529f,
                     0.42619407176971436f};
  vector<int64_t> X_shape = {1, 1, 7, 7};

  vector<float> W = {-0.4406261742115021f};
  vector<int64_t> W_shape = {1, 1, 1, 1};

  auto expected_vals = {-0.19936637580394745f, -0.06828942894935608f, -0.04934731498360634f, 0.17369966208934784f,
                        -0.11574628204107285f, -0.05910799279808998f, 0.1197819635272026f, 0.18959586322307587f,
                        0.1182001456618309f, -0.17154212296009064f, 0.06006614491343498f, 0.0042258151806890965f,
                        0.21490024030208588f, 0.11128675937652588f, 0.12394362688064575f, -0.17815405130386353f,
                        -0.034346915781497955f, -0.14407673478126526f, -0.05778544768691063f, 0.19459928572177887f,
                        -0.05484473705291748f, -0.16188594698905945f, -0.07485868036746979f, -0.08851054310798645f,
                        -0.10284193605184555f, -0.17014220356941223f, -0.04898572340607643f, -0.17083507776260376f,
                        -0.09170642495155334f, 0.1511256992816925f, 0.012886842712759972f, 0.09025576710700989f,
                        0.08479554951190948f, 0.0489313043653965f, 0.14465972781181335f, 0.007934254594147205f,
                        -0.15946026146411896f, 0.1804322451353073f, 0.08040717244148254f, 0.1475857049226761f,
                        0.15021422505378723f, -0.0028631272725760937f, -0.19993697106838226f, -0.03527900204062462f,
                        0.06516310572624207f, -0.015176207758486271f, 0.14682966470718384f, -0.02665453404188156f,
                        -0.18779225647449493f};
  vector<int64_t> Y_shape = {1, 1, 7, 7};

  // Calculate quantization params and quantize the inputs and expected output
  float lhs_min, lhs_max, rhs_min, rhs_max, result_min, result_max;
  FindMinMax(X, &lhs_min, &lhs_max);
  FindMinMax(W, &rhs_min, &rhs_max);
  FindMinMax(expected_vals, &result_min, &result_max);

  float lhs_scale, rhs_scale, result_scale;
  uint8_t lhs_zero_point, rhs_zero_point, result_zero_point;
  FindScaleAndZeroPoint(lhs_min, lhs_max, &lhs_scale, &lhs_zero_point);
  FindScaleAndZeroPoint(rhs_min, rhs_max, &rhs_scale, &rhs_zero_point);
  FindScaleAndZeroPoint(result_min, result_max, &result_scale, &result_zero_point);

  vector<uint8_t> x_quantized(X.size()), w_quantized(W.size()), result_quantized(expected_vals.size());
  Quantize(lhs_scale, lhs_zero_point, X, &x_quantized);
  Quantize(rhs_scale, rhs_zero_point, W, &w_quantized);
  Quantize(result_scale, result_zero_point, expected_vals, &result_quantized);

  test.AddInput<uint8_t>("x", X_shape, x_quantized);
  test.AddInput<float>("x_scale", {}, {lhs_scale});
  test.AddInput<uint8_t>("x_zero_point", {}, {lhs_zero_point});

  test.AddInput<uint8_t>("w", W_shape, w_quantized);
  test.AddInput<float>("w_scale", {}, {rhs_scale});
  test.AddInput<uint8_t>("w_zero_point", {}, {rhs_zero_point});

  test.AddInput<float>("y_scale", {}, {result_scale});
  test.AddInput<uint8_t>("y_zero_point", {}, {result_zero_point});

  test.AddOutput<uint8_t>("y", Y_shape, result_quantized);

  test.Run();
}

TEST(ConvTest, QLinearConv3DTest) {
  OpTester test("QLinearConv", 1, onnxruntime::kMSDomain);

  vector<float> X = {0.010772407054901123f, -0.43806642293930054f, 0.455391526222229f, -0.28657248616218567f,
                     0.45676887035369873f, -0.0320507287979126f, 0.4229400157928467f, -0.18730869889259338f,
                     -0.45851585268974304f, 0.042054951190948486f, -0.13332295417785645f, -0.25374430418014526f,
                     -0.23845627903938293f, 0.12214112281799316f, -0.1778157651424408f, 0.1891845464706421f,
                     0.37962496280670166f, -0.033982306718826294f, 0.12737131118774414f, -0.040284961462020874f,
                     0.46427029371261597f, -0.22687292098999023f, 0.17398333549499512f, -0.3014046251773834f,
                     -0.4043419063091278f, -0.33206477761268616f, 0.04655301570892334f, -0.4947906732559204f,
                     0.0755157470703125f, 0.1173025369644165f, 0.47043120861053467f, 0.4824737310409546f,
                     -0.37734976410865784f, -0.056491583585739136f, -0.10790631175041199f, 0.043476223945617676f,
                     0.24469023942947388f, -0.4100031852722168f, 0.0616222620010376f, 0.2296960949897766f,
                     0.27883386611938477f, 0.08150351047515869f, 0.2453773021697998f, 0.08250969648361206f,
                     -0.1471814215183258f, -0.43011274933815f, 0.027180075645446777f, 0.3605625033378601f,
                     0.24954384565353394f, -0.22505927085876465f, -0.36272895336151123f, -0.47674262523651123f,
                     0.11275297403335571f, 0.49773406982421875f, 0.2686365246772766f, 0.025525271892547607f,
                     -0.3037869930267334f, 0.41126757860183716f, 0.36149072647094727f, 0.00883406400680542f,
                     -0.07959523797035217f, 0.3601323366165161f, 0.17322391271591187f, -0.012007325887680054f};
  vector<int64_t> X_shape = {1, 1, 4, 4, 4};
  vector<float> W = {0.32824617624282837f};
  vector<int64_t> W_shape = {1, 1, 1, 1, 1};
  vector<int64_t> Y_shape = {1, 1, 4, 4, 4};
  auto expected_vals = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0035360013134777546f, 0.14948052167892456f, 0.0f,
                        0.0f, -0.15050607919692993f, -0.043762750923633575f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.12386361509561539f, -0.03541983291506767f, 0.0f,
                        0.0f, 0.09152615070343018f, 0.08054415881633759f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  vector<int64_t> pads = {2, 2, 2, 2, 2, 2};
  vector<int64_t>strides = {2, 2, 2};

  // Calculate quantization params and quantize the inputs and expected output
  float lhs_min, lhs_max, rhs_min, rhs_max, result_min, result_max;
  FindMinMax(X, &lhs_min, &lhs_max);
  FindMinMax(W, &rhs_min, &rhs_max);
  FindMinMax(expected_vals, &result_min, &result_max);

  float lhs_scale, rhs_scale, result_scale;
  uint8_t lhs_zero_point, rhs_zero_point, result_zero_point;
  FindScaleAndZeroPoint(lhs_min, lhs_max, &lhs_scale, &lhs_zero_point);
  FindScaleAndZeroPoint(rhs_min, rhs_max, &rhs_scale, &rhs_zero_point);
  FindScaleAndZeroPoint(result_min, result_max, &result_scale, &result_zero_point);

  vector<uint8_t> x_quantized(X.size()), w_quantized(W.size()), result_quantized(expected_vals.size());
  Quantize(lhs_scale, lhs_zero_point, X, &x_quantized);
  Quantize(rhs_scale, rhs_zero_point, W, &w_quantized);
  Quantize(result_scale, result_zero_point, expected_vals, &result_quantized);

  test.AddAttribute("pads", pads);
  test.AddAttribute("strides", strides);

  test.AddInput<uint8_t>("x", X_shape, x_quantized);
  test.AddInput<float>("x_scale", {}, {lhs_scale});
  test.AddInput<uint8_t>("x_zero_point", {}, {lhs_zero_point});

  test.AddInput<uint8_t>("w", W_shape, w_quantized);
  test.AddInput<float>("w_scale", {}, {rhs_scale});
  test.AddInput<uint8_t>("w_zero_point", {}, {rhs_zero_point});

  test.AddInput<float>("y_scale", {}, {result_scale});
  test.AddInput<uint8_t>("y_zero_point", {}, {result_zero_point});

  test.AddOutput<uint8_t>("y", Y_shape, result_quantized);

  test.Run();      
}
}  // namespace
}  // namespace test
}  // namespace onnxruntime
