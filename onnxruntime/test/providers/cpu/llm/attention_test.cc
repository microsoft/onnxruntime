// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include "gtest/gtest.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {
enum class TensorType {
  kFloat,
  kFloat16,
  kBFloat16
};
}  // anonymous namespace

static void RunTest4D(
    int batch_size,
    int q_num_heads,
    int q_sequence_length,
    int head_size,
    int kv_sequence_length,
    int kv_num_heads,
    int v_head_size,
    int past_sequence_length,
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<int32_t>& attn_mask,
    const std::vector<float>& past_key,
    const std::vector<float>& past_value,
    int is_causal,  // 0
    // int kv_num_heads, // not needed for 3D
    // int q_num_heads,  // not needed for 3D
    int qk_matmul_output_mode,  // 0
    float scale,
    float softcap,  // 0.0,
    int softmax_precision,
    TensorType tensor_type,
    const std::vector<float>& y,
    const std::vector<float>& present_key,
    const std::vector<float>& present_value,
    const std::vector<float>& qk_matmul_output,
    bool disable_cpu,
    bool disable_cuda,
    bool disable_dml) {
  int total_sequence_length = past_sequence_length + kv_sequence_length;
  // inputs
  std::vector<int64_t> q_shape = {batch_size, q_num_heads, q_sequence_length, head_size};
  std::vector<int64_t> k_shape = {batch_size, kv_num_heads, kv_sequence_length, head_size};
  std::vector<int64_t> v_shape = {batch_size, kv_num_heads, kv_sequence_length, v_head_size};
  std::vector<int64_t> attn_mask_shape = {batch_size, q_num_heads, q_sequence_length, total_sequence_length};
  std::vector<int64_t> past_key_shape = {batch_size, kv_num_heads, past_sequence_length, head_size};
  std::vector<int64_t> past_value_shape = {batch_size, kv_num_heads, past_sequence_length, head_size};
  // outputs
  std::vector<int64_t> y_shape = {batch_size, q_num_heads, q_sequence_length, v_head_size};
  std::vector<int64_t> present_key_shape = {batch_size, kv_num_heads, total_sequence_length, head_size};
  std::vector<int64_t> present_value_shape = {batch_size, kv_num_heads, total_sequence_length, v_head_size};
  std::vector<int64_t> qk_matmul_output_shape = {batch_size, q_num_heads, q_sequence_length, total_sequence_length};

  std::string op_type = "Attention";

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;

  int min_cuda_architecture = (tensor_type == TensorType::kBFloat16)
                                  ? 800
                              : (tensor_type == TensorType::kFloat16) ? 530
                                                                      : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_dml = (nullptr != DefaultDmlExecutionProvider().get()) && !disable_dml;
  bool enable_webgpu = nullptr != DefaultWebGpuExecutionProvider().get();

  if (enable_cuda && !disable_cuda) {
    execution_providers.push_back(DefaultCudaExecutionProvider());
  }
  if (enable_dml && !disable_dml) {
    execution_providers.push_back(DefaultDmlExecutionProvider());
  }
  if ((tensor_type == TensorType::kFloat || tensor_type == TensorType::kFloat16) && !disable_cpu) {
    execution_providers.push_back(DefaultCpuExecutionProvider());
  }
  if (enable_webgpu) {
    execution_providers.push_back(DefaultWebGpuExecutionProvider());
  }
  if (execution_providers.size() == 0) {
    // Return early if CI pipeline does not support EP (e.g. CUDA EP for CPU CI pipeline)
    return;
  }

  for (auto& ep : execution_providers) {
    OpTester test(op_type.c_str(), 23, onnxruntime::kOnnxDomain);

    test.AddAttribute<int64_t>("is_causal", is_causal);
    test.AddAttribute<int64_t>("kv_num_heads", kv_num_heads);
    test.AddAttribute<int64_t>("q_num_heads", q_num_heads);
    test.AddAttribute<int64_t>("qk_matmul_output_mode", qk_matmul_output_mode);
    test.AddAttribute<float>("scale", scale);
    test.AddAttribute<float>("softcap", softcap);
    test.AddAttribute<int64_t>("softmax_precision", softmax_precision);

    if (tensor_type == TensorType::kFloat) {
      // inputs
      test.AddInput<float>("Q", q_shape, q);
      test.AddInput<float>("K", k_shape, k);
      test.AddInput<float>("V", v_shape, v);
      if (!attn_mask.empty())
        test.AddInput<int32_t>("attn_mask", attn_mask_shape, attn_mask);
      else
        test.AddOptionalInputEdge<int32_t>();

      if (!past_key.empty())
        test.AddInput<float>("past_key", past_key_shape, past_key);
      else
        test.AddOptionalInputEdge<float>();

      if (!past_value.empty())
        test.AddInput<float>("past_value", past_value_shape, past_value);
      else
        test.AddOptionalInputEdge<float>();
      // outputs
      test.AddOutput<float>("Y", y_shape, y);
      if (!present_key.empty())
        test.AddOutput<float>("present_key", present_key_shape, present_key);
      if (!present_value.empty())
        test.AddOutput<float>("present_value", present_value_shape, present_value);
      if (!qk_matmul_output.empty())
        test.AddOutput<float>("qk_matmul_output", qk_matmul_output_shape, qk_matmul_output);
    } else if (tensor_type == TensorType::kFloat16) {
      // inputs
      test.AddInput<MLFloat16>("Q", q_shape, ToFloat16(q));
      test.AddInput<MLFloat16>("K", k_shape, ToFloat16(k));
      test.AddInput<MLFloat16>("V", v_shape, ToFloat16(v));
      if (!attn_mask.empty())
        test.AddInput<int32_t>("attn_mask", attn_mask_shape, attn_mask);
      else
        test.AddOptionalInputEdge<int32_t>();

      if (!past_key.empty())
        test.AddInput<MLFloat16>("past_key", past_key_shape, ToFloat16(past_key));
      else
        test.AddOptionalInputEdge<MLFloat16>();

      if (!past_value.empty())
        test.AddInput<MLFloat16>("past_value", past_value_shape, ToFloat16(past_value));
      else
        test.AddOptionalInputEdge<MLFloat16>();
      // outputs
      test.AddOutput<MLFloat16>("Y", y_shape, ToFloat16(y));
      if (!present_key.empty())
        test.AddOutput<MLFloat16>("present_key", present_key_shape, ToFloat16(present_key));
      if (!present_value.empty())
        test.AddOutput<MLFloat16>("present_value", present_value_shape, ToFloat16(present_value));
      if (!qk_matmul_output.empty())
        test.AddOutput<MLFloat16>("qk_matmul_output", qk_matmul_output_shape, ToFloat16(qk_matmul_output));
    } else {
      // inputs
      test.AddInput<BFloat16>("Q", q_shape, FloatsToBFloat16s(q));
      test.AddInput<BFloat16>("K", k_shape, FloatsToBFloat16s(k));
      test.AddInput<BFloat16>("V", v_shape, FloatsToBFloat16s(v));
      if (!attn_mask.empty())
        test.AddInput<int32_t>("attn_mask", attn_mask_shape, attn_mask);
      else
        test.AddOptionalInputEdge<int32_t>();

      if (!past_key.empty())
        test.AddInput<BFloat16>("past_key", past_key_shape, FloatsToBFloat16s(past_key));
      else
        test.AddOptionalInputEdge<BFloat16>();

      if (!past_value.empty())
        test.AddInput<BFloat16>("past_value", past_value_shape, FloatsToBFloat16s(past_value));
      else
        test.AddOptionalInputEdge<BFloat16>();
      // outputs
      test.AddOutput<BFloat16>("Y", y_shape, FloatsToBFloat16s(y));
      if (!present_key.empty())
        test.AddOutput<BFloat16>("present_key", present_key_shape, FloatsToBFloat16s(present_key));
      if (!present_value.empty())
        test.AddOutput<BFloat16>("present_value", present_value_shape, FloatsToBFloat16s(present_value));
      if (!qk_matmul_output.empty())
        test.AddOutput<BFloat16>("qk_matmul_output", qk_matmul_output_shape, FloatsToBFloat16s(qk_matmul_output));
    }

    std::vector<std::unique_ptr<IExecutionProvider>> test_execution_providers;
    test_execution_providers.push_back(std::move(ep));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &test_execution_providers);
  }
}

// Interleaved = true, pos ids shape = (batch_size, sequence_length)
TEST(AttentionTest, AttentionDefault4D) {
  int batch_size = 2;            // Q.shape[0]
  int q_num_heads = 3;           // Q.shape[1]
  int q_sequence_length = 4;     // Q.shape[2]
  int head_size = 8;             // Q.shape[3]
  int kv_sequence_length = 6;    // K.shape[2] and V.shape[2]
  int kv_num_heads = 3;          // K.shape[1] and V.shape[1]
  int v_head_size = 8;           // V.shape[3]
  int past_sequence_length = 5;  // past_key.shape[2] and past_value.shape[2]

  std::vector<float> q = {0.548814f, 0.715189f, 0.602763f, 0.544883f, 0.423655f, 0.645894f, 0.437587f, 0.891773f, 0.963663f, 0.383442f, 0.791725f, 0.528895f, 0.568045f, 0.925597f, 0.071036f, 0.087129f, 0.020218f, 0.832620f, 0.778157f, 0.870012f, 0.978618f, 0.799159f, 0.461479f, 0.780529f, 0.118274f, 0.639921f, 0.143353f, 0.944669f, 0.521848f, 0.414662f, 0.264556f, 0.774234f, 0.456150f, 0.568434f, 0.018790f, 0.617635f, 0.612096f, 0.616934f, 0.943748f, 0.681820f, 0.359508f, 0.437032f, 0.697631f, 0.060225f, 0.666767f, 0.670638f, 0.210383f, 0.128926f, 0.315428f, 0.363711f, 0.570197f, 0.438602f, 0.988374f, 0.102045f, 0.208877f, 0.161310f, 0.653108f, 0.253292f, 0.466311f, 0.244426f, 0.158970f, 0.110375f, 0.656330f, 0.138183f, 0.196582f, 0.368725f, 0.820993f, 0.097101f, 0.837945f, 0.096098f, 0.976459f, 0.468651f, 0.976761f, 0.604846f, 0.739264f, 0.039188f, 0.282807f, 0.120197f, 0.296140f, 0.118728f, 0.317983f, 0.414263f, 0.064147f, 0.692472f, 0.566601f, 0.265390f, 0.523248f, 0.093941f, 0.575947f, 0.929296f, 0.318569f, 0.667410f, 0.131798f, 0.716327f, 0.289406f, 0.183191f, 0.586513f, 0.020108f, 0.828940f, 0.004695f, 0.677817f, 0.270008f, 0.735194f, 0.962189f, 0.248753f, 0.576157f, 0.592042f, 0.572252f, 0.223082f, 0.952749f, 0.447125f, 0.846409f, 0.699479f, 0.297437f, 0.813798f, 0.396506f, 0.881103f, 0.581273f, 0.881735f, 0.692532f, 0.725254f, 0.501324f, 0.956084f, 0.643990f, 0.423855f, 0.606393f, 0.019193f, 0.301575f, 0.660174f, 0.290078f, 0.618015f, 0.428769f, 0.135474f, 0.298282f, 0.569965f, 0.590873f, 0.574325f, 0.653201f, 0.652103f, 0.431418f, 0.896547f, 0.367562f, 0.435865f, 0.891923f, 0.806194f, 0.703889f, 0.100227f, 0.919483f, 0.714241f, 0.998847f, 0.149448f, 0.868126f, 0.162493f, 0.615560f, 0.123820f, 0.848008f, 0.807319f, 0.569101f, 0.407183f, 0.069167f, 0.697429f, 0.453543f, 0.722056f, 0.866382f, 0.975522f, 0.855803f, 0.011714f, 0.359978f, 0.729991f, 0.171630f, 0.521037f, 0.054338f, 0.199997f, 0.018522f, 0.793698f, 0.223925f, 0.345352f, 0.928081f, 0.704414f, 0.031839f, 0.164694f, 0.621478f, 0.577229f, 0.237893f, 0.934214f, 0.613966f, 0.535633f, 0.589910f, 0.730122f, 0.311945f, 0.398221f, 0.209844f};
  std::vector<float> k = {0.070870f, 0.292794f, 0.152355f, 0.417486f, 0.131289f, 0.604118f, 0.382808f, 0.895386f, 0.967795f, 0.546885f, 0.274824f, 0.592230f, 0.896761f, 0.406733f, 0.552078f, 0.271653f, 0.455444f, 0.401714f, 0.248413f, 0.505866f, 0.310381f, 0.373035f, 0.524970f, 0.750595f, 0.333507f, 0.924159f, 0.862319f, 0.048690f, 0.253643f, 0.446136f, 0.104628f, 0.348476f, 0.740098f, 0.680514f, 0.622384f, 0.710528f, 0.204924f, 0.341698f, 0.676242f, 0.879235f, 0.543678f, 0.282700f, 0.030235f, 0.710337f, 0.007884f, 0.372679f, 0.530537f, 0.922111f, 0.089495f, 0.405942f, 0.024313f, 0.342611f, 0.622231f, 0.279068f, 0.209750f, 0.115703f, 0.577140f, 0.695270f, 0.671957f, 0.948861f, 0.002703f, 0.647197f, 0.600392f, 0.588740f, 0.962770f, 0.016872f, 0.696482f, 0.813679f, 0.509807f, 0.333965f, 0.790840f, 0.097243f, 0.442036f, 0.519952f, 0.693956f, 0.090886f, 0.227759f, 0.410302f, 0.623295f, 0.886961f, 0.618826f, 0.133461f, 0.980580f, 0.871786f, 0.502721f, 0.922348f, 0.541381f, 0.923306f, 0.829897f, 0.968286f, 0.919783f, 0.036034f, 0.174772f, 0.389135f, 0.952143f, 0.300029f, 0.160468f, 0.886305f, 0.446394f, 0.907876f, 0.160230f, 0.661117f, 0.440264f, 0.076487f, 0.696463f, 0.247399f, 0.039616f, 0.059944f, 0.061079f, 0.907733f, 0.739884f, 0.898062f, 0.672582f, 0.528940f, 0.304446f, 0.997962f, 0.362189f, 0.470649f, 0.378245f, 0.979527f, 0.174658f, 0.327988f, 0.680349f, 0.063208f, 0.607249f, 0.477646f, 0.284000f, 0.238413f, 0.514513f, 0.367928f, 0.456520f, 0.337477f, 0.970494f, 0.133439f, 0.096804f, 0.343392f, 0.591027f, 0.659176f, 0.397257f, 0.999278f, 0.351893f, 0.721407f, 0.637583f, 0.813054f, 0.976226f, 0.889794f, 0.764562f, 0.698249f, 0.335498f, 0.147686f, 0.062636f, 0.241902f, 0.432281f, 0.521996f, 0.773084f, 0.958741f, 0.117320f, 0.107004f, 0.589695f, 0.745398f, 0.848150f, 0.935832f, 0.983426f, 0.399802f, 0.380335f, 0.147809f, 0.684934f, 0.656762f, 0.862063f, 0.097258f, 0.497777f, 0.581082f, 0.241557f, 0.169025f, 0.859581f, 0.058535f, 0.470621f, 0.115834f, 0.457059f, 0.979962f, 0.423706f, 0.857125f, 0.117316f, 0.271252f, 0.403793f, 0.399812f, 0.671384f, 0.344718f, 0.713767f, 0.639187f, 0.399161f, 0.431760f, 0.614528f, 0.070042f, 0.822407f, 0.653421f, 0.726342f, 0.536923f, 0.110477f, 0.405036f, 0.405374f, 0.321043f, 0.029950f, 0.737254f, 0.109784f, 0.606308f, 0.703218f, 0.634786f, 0.959142f, 0.103298f, 0.867167f, 0.029190f, 0.534917f, 0.404244f, 0.524184f, 0.365100f, 0.190567f, 0.019123f, 0.518150f, 0.842777f, 0.373216f, 0.222864f, 0.080532f, 0.085311f, 0.221396f, 0.100014f, 0.265040f, 0.066149f, 0.065605f, 0.856276f, 0.162120f, 0.559682f, 0.773456f, 0.456410f, 0.153369f, 0.199596f, 0.432984f, 0.528234f, 0.349440f, 0.781480f, 0.751022f, 0.927212f, 0.028953f, 0.895691f, 0.392569f, 0.878372f, 0.690785f, 0.987349f, 0.759282f, 0.364545f, 0.501063f, 0.376389f, 0.364912f, 0.260904f, 0.495970f, 0.681740f, 0.277340f, 0.524380f, 0.117380f, 0.159845f, 0.046806f, 0.970731f, 0.003860f, 0.178580f, 0.612867f, 0.081370f, 0.881896f, 0.719620f, 0.966390f, 0.507636f, 0.300404f, 0.549501f, 0.930819f, 0.520761f, 0.267207f, 0.877399f, 0.371919f, 0.001383f, 0.247685f, 0.318234f, 0.858777f, 0.458503f, 0.444587f, 0.336102f, 0.880678f, 0.945027f, 0.991890f, 0.376741f};
  std::vector<float> v = {0.186193f, 0.944372f, 0.739551f, 0.490459f, 0.227415f, 0.254356f, 0.058029f, 0.434417f, 0.311796f, 0.696343f, 0.377752f, 0.179604f, 0.024679f, 0.067250f, 0.679393f, 0.453697f, 0.536579f, 0.896671f, 0.990339f, 0.216897f, 0.663078f, 0.263322f, 0.020651f, 0.758379f, 0.320017f, 0.383464f, 0.588317f, 0.831048f, 0.628982f, 0.872651f, 0.273542f, 0.798047f, 0.185636f, 0.952792f, 0.687488f, 0.215508f, 0.947371f, 0.730856f, 0.253942f, 0.213312f, 0.518201f, 0.025663f, 0.207470f, 0.424685f, 0.374170f, 0.463575f, 0.277629f, 0.586784f, 0.863856f, 0.117532f, 0.517379f, 0.132068f, 0.716860f, 0.396060f, 0.565421f, 0.183280f, 0.144848f, 0.488056f, 0.355613f, 0.940432f, 0.765325f, 0.748664f, 0.903720f, 0.083422f, 0.552192f, 0.584476f, 0.961936f, 0.292148f, 0.240829f, 0.100294f, 0.016430f, 0.929529f, 0.669917f, 0.785153f, 0.281730f, 0.586410f, 0.063955f, 0.485628f, 0.977495f, 0.876505f, 0.338159f, 0.961570f, 0.231702f, 0.949319f, 0.941378f, 0.799203f, 0.630448f, 0.874288f, 0.293020f, 0.848944f, 0.617877f, 0.013237f, 0.347234f, 0.148141f, 0.981829f, 0.478370f, 0.497391f, 0.639473f, 0.368585f, 0.136900f, 0.822118f, 0.189848f, 0.511319f, 0.224317f, 0.097844f, 0.862191f, 0.972919f, 0.960835f, 0.906555f, 0.774047f, 0.333145f, 0.081101f, 0.407241f, 0.232234f, 0.132488f, 0.053427f, 0.725594f, 0.011427f, 0.770581f, 0.146947f, 0.079522f, 0.089603f, 0.672048f, 0.245367f, 0.420539f, 0.557369f, 0.860551f, 0.727044f, 0.270328f, 0.131483f, 0.055374f, 0.301599f, 0.262118f, 0.456141f, 0.683281f, 0.695625f, 0.283519f, 0.379927f, 0.181151f, 0.788545f, 0.056848f, 0.696997f, 0.778695f, 0.777408f, 0.259423f, 0.373813f, 0.587600f, 0.272822f, 0.370853f, 0.197054f, 0.459856f, 0.044612f, 0.799796f, 0.076956f, 0.518835f, 0.306810f, 0.577543f, 0.959433f, 0.645570f, 0.035362f, 0.430402f, 0.510017f, 0.536178f, 0.681392f, 0.277596f, 0.128861f, 0.392676f, 0.956406f, 0.187131f, 0.903984f, 0.543806f, 0.456911f, 0.882041f, 0.458604f, 0.724168f, 0.399025f, 0.904044f, 0.690025f, 0.699622f, 0.327720f, 0.756779f, 0.636061f, 0.240020f, 0.160539f, 0.796391f, 0.959167f, 0.458139f, 0.590984f, 0.857723f, 0.457223f, 0.951874f, 0.575751f, 0.820767f, 0.908844f, 0.815524f, 0.159414f, 0.628898f, 0.398434f, 0.062713f, 0.424032f, 0.258684f, 0.849038f, 0.033305f, 0.958983f, 0.355369f, 0.356707f, 0.016329f, 0.185232f, 0.401260f, 0.929291f, 0.099615f, 0.945302f, 0.869489f, 0.454162f, 0.326701f, 0.232744f, 0.614465f, 0.033075f, 0.015606f, 0.428796f, 0.068074f, 0.251941f, 0.221161f, 0.253191f, 0.131055f, 0.012036f, 0.115484f, 0.618480f, 0.974256f, 0.990345f, 0.409054f, 0.162954f, 0.638762f, 0.490305f, 0.989410f, 0.065304f, 0.783234f, 0.288399f, 0.241419f, 0.662505f, 0.246063f, 0.665859f, 0.517309f, 0.424089f, 0.554688f, 0.287052f, 0.706575f, 0.414857f, 0.360546f, 0.828657f, 0.924967f, 0.046007f, 0.232627f, 0.348519f, 0.814966f, 0.985491f, 0.968972f, 0.904948f, 0.296556f, 0.992011f, 0.249420f, 0.105906f, 0.950953f, 0.233420f, 0.689768f, 0.058356f, 0.730709f, 0.881720f, 0.272437f, 0.379057f, 0.374296f, 0.748788f, 0.237807f, 0.171853f, 0.449292f, 0.304468f, 0.839189f, 0.237742f, 0.502389f, 0.942584f, 0.633998f, 0.867289f, 0.940210f, 0.750765f, 0.699575f, 0.967966f, 0.994401f, 0.451822f};
  ASSERT_EQ(q.size(), batch_size * q_num_heads * q_sequence_length * head_size);
  ASSERT_EQ(k.size(), batch_size * kv_num_heads * kv_sequence_length * head_size);
  ASSERT_EQ(v.size(), batch_size * kv_num_heads * kv_sequence_length * v_head_size);

  std::vector<float> y = {0.501465f, 0.543511f, 0.398088f, 0.474061f, 0.290507f, 0.423018f, 0.447999f, 0.672390f, 0.500878f, 0.545140f, 0.402253f, 0.478354f, 0.278711f, 0.420929f, 0.451124f, 0.682613f, 0.496502f, 0.557356f, 0.419293f, 0.467867f, 0.280946f, 0.422295f, 0.445183f, 0.675748f, 0.498804f, 0.545264f, 0.399543f, 0.471287f, 0.287601f, 0.424845f, 0.443877f, 0.670841f, 0.580098f, 0.450536f, 0.702941f, 0.538382f, 0.329768f, 0.543394f, 0.613723f, 0.562010f, 0.584549f, 0.447129f, 0.673676f, 0.537643f, 0.342950f, 0.515742f, 0.613437f, 0.502951f, 0.585248f, 0.443070f, 0.676620f, 0.549025f, 0.343112f, 0.522440f, 0.611621f, 0.507324f, 0.580745f, 0.461632f, 0.668496f, 0.507376f, 0.336816f, 0.500750f, 0.618162f, 0.500909f, 0.464240f, 0.493342f, 0.380525f, 0.530712f, 0.397056f, 0.582067f, 0.443341f, 0.559227f, 0.467916f, 0.503694f, 0.373170f, 0.549178f, 0.387171f, 0.587037f, 0.448581f, 0.561591f, 0.478681f, 0.496704f, 0.369457f, 0.545459f, 0.392339f, 0.587842f, 0.452645f, 0.576330f, 0.483897f, 0.491793f, 0.360676f, 0.530990f, 0.380686f, 0.603393f, 0.467172f, 0.583590f, 0.642787f, 0.470883f, 0.686034f, 0.642719f, 0.386365f, 0.366454f, 0.467120f, 0.405736f, 0.644347f, 0.466390f, 0.684379f, 0.640710f, 0.385963f, 0.366271f, 0.472645f, 0.403025f, 0.631421f, 0.453237f, 0.677676f, 0.643979f, 0.390879f, 0.377663f, 0.467158f, 0.401772f, 0.637457f, 0.459313f, 0.677889f, 0.659685f, 0.383362f, 0.379251f, 0.453763f, 0.401437f, 0.555998f, 0.186013f, 0.455395f, 0.406430f, 0.395553f, 0.526708f, 0.320193f, 0.484448f, 0.577368f, 0.190770f, 0.462801f, 0.384114f, 0.403607f, 0.534057f, 0.326255f, 0.496504f, 0.563586f, 0.180264f, 0.464196f, 0.384055f, 0.385514f, 0.537212f, 0.338047f, 0.485235f, 0.555800f, 0.177971f, 0.457827f, 0.377928f, 0.372441f, 0.541035f, 0.343750f, 0.483692f, 0.705313f, 0.467049f, 0.389698f, 0.530555f, 0.548003f, 0.637789f, 0.501241f, 0.493046f, 0.692096f, 0.474284f, 0.375588f, 0.530258f, 0.507811f, 0.618987f, 0.468782f, 0.502795f, 0.703758f, 0.479856f, 0.374269f, 0.518477f, 0.518286f, 0.631821f, 0.502535f, 0.509264f, 0.689539f, 0.474638f, 0.374363f, 0.519131f, 0.519441f, 0.644891f, 0.480984f, 0.490645f};
  RunTest4D(batch_size, q_num_heads, q_sequence_length, head_size, kv_sequence_length, kv_num_heads, v_head_size, past_sequence_length,
            q, k, v, std::vector<int32_t>(), std::vector<float>(), std::vector<float>(),
            0, 0, 1.f, 0.f, 1, TensorType::kFloat, y, std::vector<float>(), std::vector<float>(), std::vector<float>(),
            false, true, true);
}

}  // namespace test
}  // namespace onnxruntime
