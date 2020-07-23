// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

struct QuantizedTensor {
  QuantizedTensor(const std::vector<float>& data) {
    // find input range min and max
    auto min = *std::min_element(data.begin(), data.end());
    auto max = *std::max_element(data.begin(), data.end());

    // ensure the data range includes zero
    min = std::min(min, 0.f);
    max = std::max(max, 0.f);

    float qmin = std::numeric_limits<uint8_t>::min();
    float qmax = std::numeric_limits<uint8_t>::max();

    // compute scale and zero point
    scale_ = (max - min) / (qmax - qmin);
    const auto initial_zero_point = qmin - min / scale_;
    zero_point_ = static_cast<uint8_t>(std::round(std::max(qmin, std::min(qmax, initial_zero_point))));

    // quantize the data
    quantized_.resize(data.size());
    for (size_t i = 0; i < data.size(); i++) {
      const float clamped_val = std::max(qmin, std::min(qmax, std::round(data[i] / scale_) + zero_point_));
      quantized_[i] = static_cast<uint8_t>(clamped_val);
    }
  }

  QuantizedTensor(const std::vector<uint8_t>& data, float scale, uint8_t zero_point)
      : quantized_(data), scale_(scale), zero_point_(zero_point) {
  }

  std::vector<uint8_t> quantized_;
  float scale_;
  uint8_t zero_point_;
};

struct QuantizedBiasTensor {
  QuantizedBiasTensor(const std::vector<float>& data,
                      const QuantizedTensor& X,
                      const QuantizedTensor& W) {
    scale_ = X.scale_ * W.scale_;

    // quantize the data
    quantized_.resize(data.size());
    for (size_t i = 0; i < data.size(); i++) {
      quantized_[i] = static_cast<int32_t>(std::floor(data[i] / (X.scale_ * W.scale_)));
    }
  }

  QuantizedBiasTensor(const std::vector<int32_t>& data, float scale)
      : quantized_(data), scale_(scale) {
  }

  std::vector<int32_t> quantized_;
  float scale_;
};

void TestQLinearConvOp(OpTester& test,
                       const QuantizedTensor& X,
                       const std::vector<int64_t>& X_shape,
                       const QuantizedTensor& W,
                       const std::vector<int64_t>& W_shape,
                       const QuantizedBiasTensor* B,
                       const QuantizedTensor& Y,
                       const std::vector<int64_t>& Y_shape,
                       bool all_input_initializer_except_x = false,
                       const std::unordered_set<std::string>& excluded_provider_types = {}) {
  test.AddInput<uint8_t>("x", X_shape, X.quantized_);
  test.AddInput<float>("x_scale", {}, {X.scale_}, all_input_initializer_except_x);
  test.AddInput<uint8_t>("x_zero_point", {}, {X.zero_point_}, all_input_initializer_except_x);

  test.AddInput<uint8_t>("w", W_shape, W.quantized_, all_input_initializer_except_x);
  test.AddInput<float>("w_scale", {}, {W.scale_}, all_input_initializer_except_x);
  test.AddInput<uint8_t>("w_zero_point", {}, {W.zero_point_}, all_input_initializer_except_x);

  test.AddInput<float>("y_scale", {}, {Y.scale_}, all_input_initializer_except_x);
  test.AddInput<uint8_t>("y_zero_point", {}, {Y.zero_point_}, all_input_initializer_except_x);

  if (B != nullptr) {
    const std::vector<int64_t> B_shape{static_cast<int64_t>(B->quantized_.size())};
    test.AddInput<int32_t>("b", B_shape, B->quantized_, all_input_initializer_except_x);
  }

  test.AddOutput<uint8_t>("y", Y_shape, Y.quantized_);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_provider_types);
}

void RunConv2DTest(bool all_input_initializer_except_x) {
  QuantizedTensor X({0.45246148109436035f, 0.15498268604278564f, 0.11199361085891724f, -0.39421093463897705f,
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
                     0.42619407176971436f});
  QuantizedTensor W({-0.4406261742115021f});
  QuantizedTensor Y({-0.19936637580394745f, -0.06828942894935608f, -0.04934731498360634f, 0.17369966208934784f,
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
                     -0.18779225647449493f});

  OpTester test("QLinearConv", 10);

  TestQLinearConvOp(test,
                    X, {1, 1, 7, 7},
                    W, {1, 1, 1, 1},
                    nullptr,
                    Y, {1, 1, 7, 7},
                    all_input_initializer_except_x);
}

TEST(QLinearConvTest, Conv2DTest) {
  RunConv2DTest(false);
}

TEST(QLinearConvTest, Conv2DTestAllInputInitializerExceptX) {
  RunConv2DTest(true);
}

TEST(QLinearConvTest, Conv3DTest) {
  QuantizedTensor X({0.010772407054901123f, -0.43806642293930054f, 0.455391526222229f, -0.28657248616218567f,
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
                     -0.07959523797035217f, 0.3601323366165161f, 0.17322391271591187f, -0.012007325887680054f});
  QuantizedTensor W({0.32824617624282837f});
  QuantizedTensor Y({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0035360013134777546f, 0.14948052167892456f, 0.0f,
                     0.0f, -0.15050607919692993f, -0.043762750923633575f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.12386361509561539f, -0.03541983291506767f, 0.0f,
                     0.0f, 0.09152615070343018f, 0.08054415881633759f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  OpTester test("QLinearConv", 10);
  test.AddAttribute("pads", std::vector<int64_t>{2, 2, 2, 2, 2, 2});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2, 2});

  TestQLinearConvOp(test,
                    X, {1, 1, 4, 4, 4},
                    W, {1, 1, 1, 1, 1},
                    nullptr,
                    Y, {1, 1, 4, 4, 4});
}

void RunConv2DWithBiasTest(bool all_input_initializer_except_x) {
  QuantizedTensor X({6, 81, 214, 151, 234, 42, 50, 89, 30, 91, 125, 141, 52, 31, 58, 224, 84, 251, 67, 137,
                     223, 119, 79, 220, 249, 75, 131, 246, 113, 56, 54, 197, 110, 142, 126, 171, 53, 228,
                     240, 83, 229, 218, 185, 9, 80, 116, 176, 193, 175, 253},
                    0.01f,
                    135);
  QuantizedTensor W({234, 229, 13, 187, 98, 161, 246, 188, 252, 107, 49, 72, 53, 212, 175, 47, 21, 14, 86,
                     230, 16, 177, 82, 166, 75, 220, 169, 119, 34, 205, 27, 9, 44, 74, 40, 8, 28, 139, 240,
                     106, 63, 2, 255, 156, 128, 222, 73, 51, 66, 48, 81, 247, 180, 91, 206, 239, 190, 146,
                     227, 235, 10, 130, 95, 232, 121, 133, 231, 162, 108, 105, 254, 143},
                    0.15f,
                    110);
  QuantizedBiasTensor B({-1123, 3212, 1723, -621}, X.scale_ * W.scale_);
  QuantizedTensor Y({67, 81, 66, 75, 71, 101, 20, 8, 44, 94, 83, 73, 133, 125, 54, 144, 165, 56, 53, 88,
                     130, 118, 170, 168, 140, 109, 103, 80, 122, 142, 129, 100, 39, 61, 141, 133, 59, 155,
                     68, 129, 74, 132, 83, 143, 146, 152, 81, 127, 82, 112, 131, 64, 82, 68, 93, 149, 146,
                     137, 201, 118, 112, 183, 171, 144, 85, 122, 86, 63, 163, 245, 95, 152, 126, 80, 82,
                     49, 136, 160, 187, 147, 29, 20, 135, 174, 126, 124, 36, 56, 0, 83, 134, 171, 119, 109,
                     85, 155, 157, 167, 194, 130},
                    0.75f,
                    121);

  OpTester test("QLinearConv", 10);
  test.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});

  // TODO: nGraph fails to handle the optional bias vector correctly.
  TestQLinearConvOp(test,
                    X, {1, 2, 5, 5},
                    W, {4, 2, 3, 3},
                    &B,
                    Y, {1, 4, 5, 5},
                    all_input_initializer_except_x,
                    {kNGraphExecutionProvider});
}

TEST(QLinearConvTest, WithBias_2D) {
  RunConv2DWithBiasTest(false);
}

TEST(QLinearConvTest, WithBias_2D_AllInputInitializerExceptX) {
  RunConv2DWithBiasTest(true);
}

TEST(QLinearConvTest, WithGroup_2D) {
  QuantizedTensor X({98, 166, 219, 195, 46, 97, 27, 211, 239, 1, 28, 208, 143, 144, 215, 252, 79, 5, 154,
                     56, 122, 191, 94, 25, 221, 48, 37, 182, 68, 245, 210, 206, 183, 22, 163, 104, 242,
                     112, 161, 66, 181, 235, 117, 75, 236, 61, 115, 36, 120, 253, 165, 214, 159, 132, 11,
                     201, 30, 249, 89, 171, 186, 67, 225, 197, 135, 142, 241, 169, 170, 164, 178, 58, 50,
                     51, 200, 43, 199, 126, 222, 123, 227, 42, 3, 21, 124, 220, 24, 47, 63, 110},
                    0.01f,
                    135);
  QuantizedTensor W({220, 111, 73, 254, 235, 151, 6, 156, 129, 204, 234, 198, 44, 89, 202, 82, 118, 189,
                     71, 120, 123, 121, 110, 83, 173, 248, 108, 229, 124, 68, 85, 239, 133, 213, 112, 122,
                     170, 231, 225, 195, 192, 9, 232, 97, 160, 227, 67, 137},
                    0.15f,
                    110);
  QuantizedBiasTensor B({-1853, 598, -17854, 14592, 42, -366}, X.scale_ * W.scale_);
  QuantizedTensor Y({113, 128, 70, 64, 125, 162, 80, 189, 112, 147, 121, 111, 96, 68, 94, 101, 77, 88, 223,
                     128, 163, 194, 138, 164, 122, 109, 117, 91, 72, 121, 134, 155, 127, 125, 98, 128},
                    0.75f,
                    121);

  OpTester test("QLinearConv", 10);
  test.AddAttribute("group", static_cast<int64_t>(3));
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 1, 1});
  test.AddAttribute("strides", std::vector<int64_t>{2, 2});

  // TODO: nGraph rejects grouped convolutions with bias.
  TestQLinearConvOp(test,
                    X, {1, 6, 3, 5},
                    W, {6, 2, 2, 2},
                    &B,
                    Y, {1, 6, 2, 3},
                    false,
                    {kNGraphExecutionProvider});
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
