// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "contrib_ops/cpu/bert/longformer_attention_base.h"

namespace onnxruntime {
namespace test {

static void RunAttentionTest(
    const std::vector<float>& input_data,           // input:          [batch_size, sequence_length, hidden_size]
    const std::vector<float>& weights_data,         // weights:        [hidden_size, 3 * hidden_size]
    const std::vector<float>& bias_data,            // bias:           [3 * hidden_size]
    const std::vector<float>& mask_data,            // mask:           [batch_size, sequence_length]
    const std::vector<float>& global_weights_data,  // global_weights: [hidden_size, 3 * hidden_size]
    const std::vector<float>& global_bias_data,     // global_bias:    [3 * hidden_size]
    const std::vector<int>& global_data,            // global:         [batch_size, sequence_length]
    const std::vector<float>& output_data,          // output:         [batch_size, sequence_length, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    int window,
    bool use_float16 = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;

  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get());
  bool enable_cpu = false;
  if (enable_cpu || enable_cuda) {
    OpTester tester("LongformerAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
    tester.AddAttribute<int64_t>("window", static_cast<int64_t>(window));

    std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
    std::vector<int64_t> weights_dims = {hidden_size, 3 * hidden_size};
    std::vector<int64_t> bias_dims = {3 * hidden_size};
    std::vector<int64_t> mask_dims = {batch_size, sequence_length};
    std::vector<int64_t> global_weights_dims = {hidden_size, 3 * hidden_size};
    std::vector<int64_t> global_bias_dims = {3 * hidden_size};
    std::vector<int64_t> global_dims = {batch_size, sequence_length};
    std::vector<int64_t> output_dims = input_dims;

    if (use_float16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
      tester.AddInput<MLFloat16>("weight", weights_dims, ToFloat16(weights_data));
      tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
      tester.AddInput<MLFloat16>("mask", mask_dims, ToFloat16(mask_data));
      tester.AddInput<MLFloat16>("global_weight", global_weights_dims, ToFloat16(global_weights_data));
      tester.AddInput<MLFloat16>("global_bias", global_bias_dims, ToFloat16(global_bias_data));
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("input", input_dims, input_data);
      tester.AddInput<float>("weight", weights_dims, weights_data);
      tester.AddInput<float>("bias", bias_dims, bias_data);
      tester.AddInput<float>("mask", mask_dims, mask_data);
      tester.AddInput<float>("global_weight", global_weights_dims, global_weights_data);
      tester.AddInput<float>("global_bias", global_bias_dims, global_bias_data);
      tester.AddOutput<float>("output", output_dims, output_data);
    }

    tester.AddInput<int32_t>("global", global_dims, global_data);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    if (enable_cuda) {
      execution_providers.push_back(DefaultCudaExecutionProvider());
    }
    if (enable_rocm) {
      execution_providers.push_back(DefaultRocmExecutionProvider());
    }
    if (enable_cpu) {
      execution_providers.push_back(DefaultCpuExecutionProvider());
    }

    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

static void GetTinyLongformerData(
    std::vector<float>& weight_data,
    std::vector<float>& bias_data,
    std::vector<float>& global_weight_data,
    std::vector<float>& global_bias_data) {
  weight_data = {
      -0.03736265f, -0.03700203f, -0.01265421f, -0.00005944f, 0.03058976f, -0.00278089f, -0.00340891f, -0.00879795f,
      -0.02133591f, 0.00784427f, 0.03478181f, -0.03402156f, 0.02373371f, 0.0063644f, -0.01468624f, 0.0504882f,
      0.00077419f, 0.00028939f, -0.02853666f, -0.01804554f, 0.00576997f, 0.0259758f, -0.0051732f, 0.01098385f,
      -0.03283566f, 0.0021469f, 0.00610109f, -0.03178682f, 0.00858954f, 0.0257727f, -0.02472394f, -0.02258367f,
      -0.0397051f, 0.00637861f, -0.00492215f, 0.00614283f, 0.01885833f, 0.0072404f, -0.02029094f, 0.00384926f,
      -0.01666811f, 0.01191303f, 0.00760845f, -0.01181425f, -0.00996346f, 0.00950979f, -0.01039405f, 0.01063836f,
      -0.0038129f, -0.00482849f, -0.00006185f, 0.01160069f, 0.00319028f, 0.0009577f, 0.03295509f, 0.00164638f,
      -0.0020025f, 0.01478013f, -0.01177825f, 0.04226499f, -0.0057663f, -0.00160333f, 0.00270868f, -0.00655655f,
      0.02530637f, 0.00078956f, -0.00534409f, 0.01388706f, -0.01226765f, 0.02728544f, 0.01600054f, -0.01379196f,
      0.01883291f, -0.01875546f, -0.01316095f, 0.01392878f, 0.0286913f, 0.02039422f, 0.0164386f, -0.01549883f,
      -0.01735487f, -0.04088231f, -0.00565702f, 0.00084416f, 0.03280086f, -0.02137024f, 0.00550848f, 0.0178341f,
      0.01426786f, -0.03154857f, -0.02185805f, -0.00808465f, -0.01845198f, 0.03087913f, 0.01739769f, 0.01742294f,
      -0.02740332f, -0.00040067f, -0.01135285f, 0.01528029f, -0.02968147f, 0.02274286f, -0.03160891f, 0.02072382f,
      0.01509116f, 0.03541054f, 0.01261792f, 0.02072236f, 0.00617444f, 0.01808203f, 0.00342582f, 0.03231942f,
      0.04922125f, 0.00663754f, -0.01104379f, 0.00624478f, -0.00897994f, 0.01370005f, -0.01877974f, -0.0193537f,
      0.02327156f, -0.0071105f, -0.01520648f, 0.00595162f, 0.00329103f, 0.00786545f, 0.0109383f, 0.0170538f,
      -0.01224139f, -0.00105941f, 0.01048978f, -0.00976777f, -0.02140773f, 0.00742492f, 0.01248992f, 0.0075539f,
      0.00286943f, -0.01414381f, 0.03620159f, 0.01033463f, 0.01399594f, 0.01650806f, -0.00202679f, -0.00328803f,
      -0.02329111f, -0.02204572f, -0.03433893f, 0.01819115f, 0.01876776f, 0.00352132f, -0.02261852f, -0.0021271f,
      -0.01685239f, 0.00707082f, 0.00893882f, -0.03597079f, -0.02897478f, 0.01790397f, -0.00189054f, -0.00782812f,
      0.01679492f, 0.01131928f, -0.02338723f, -0.00732064f, 0.00425937f, -0.02528145f, -0.01408323f, -0.04177788f,
      -0.00529976f, 0.01193773f, -0.00405255f, 0.02042645f, -0.03389123f, -0.02279385f, 0.01627762f, 0.02311901f,
      0.01063351f, -0.05164707f, 0.0231087f, 0.00633761f, 0.01774552f, 0.02063041f, -0.01823085f, -0.00058292f,
      0.05767727f, 0.03186193f, -0.01058707f, 0.00119207f, -0.00613069f, 0.00137152f, 0.00948873f, -0.01503408f};

  bias_data = {
      0.02f, -0.01f, -0.03f, 0.02f, 0.01f, 0.0005f, 0.01f, -0.02f,
      0.01f, 0.005f, 0.01f, -0.01f, 0.02f, 0.008f, 0.01f, -0.001f,
      -0.03f, 0.03f, 0.01f, -0.002f, 0.002f, -0.03f, 0.01f, 0.008f};

  global_weight_data = {
      0.0294202f, -0.01245436f, -0.03289853f, 0.02588101f, 0.01704302f, 0.0005645f, 0.01802721f, -0.02043311f,
      0.00065155f, 0.01990021f, -0.02261156f, -0.01354096f, 0.03066289f, 0.00368064f, 0.01333689f, -0.02616469f,
      0.03910812f, 0.00521647f, -0.01002843f, 0.00574944f, -0.00215477f, -0.03501118f, -0.00492278f, 0.02129062f,
      0.0198415f, 0.00558937f, 0.01016298f, -0.01226531f, 0.02398407f, 0.00804429f, 0.01118671f, -0.00166833f,
      -0.01615552f, -0.01540161f, 0.00378566f, 0.01431978f, -0.00993885f, 0.00361956f, 0.0116084f, -0.00427342f,
      -0.0176292f, -0.01806638f, 0.01677675f, -0.0068318f, 0.00709118f, -0.01304021f, 0.00310118f, -0.01056628f,
      -0.03345605f, 0.03466916f, 0.0154178f, -0.00299957f, 0.00297282f, -0.03894032f, 0.01632687f, 0.00817282f,
      0.02143084f, -0.01206295f, -0.0027577f, 0.0047877f, 0.00112486f, 0.01005225f, 0.03119631f, -0.01274098f,
      0.03476755f, 0.02006538f, -0.02917822f, 0.00559175f, -0.01407396f, 0.03570889f, 0.03545335f, -0.00095669f,
      -0.00425291f, 0.00163976f, -0.03921806f, 0.00625822f, -0.0246806f, 0.00815409f, 0.0092521f, 0.00433091f,
      0.00997109f, 0.00915498f, -0.02030378f, -0.00968541f, 0.00971882f, -0.00865303f, -0.05141142f, 0.02430373f,
      -0.02086174f, -0.03922447f, 0.01982334f, -0.01131777f, 0.01826571f, 0.01116992f, -0.01033904f, 0.02003466f,
      -0.01685323f, -0.00281926f, -0.01908645f, 0.00376858f, 0.03040931f, -0.01649855f, -0.00193368f, 0.02237837f,
      -0.01741545f, -0.0163137f, -0.01750915f, -0.03061339f, -0.00099388f, 0.02248359f, -0.00808812f, -0.00336211f,
      0.00031454f, 0.04020427f, -0.03191524f, -0.01187126f, 0.01208827f, -0.00885536f, -0.02643824f, -0.00441225f,
      -0.00770211f, -0.00784583f, 0.01867373f, -0.01057389f, -0.00694279f, 0.02644924f, -0.00346569f, -0.01113159f,
      -0.01246041f, -0.02784149f, -0.00033919f, 0.00444981f, -0.0196088f, 0.00031822f, -0.00622559f, -0.0112043f,
      -0.00666037f, 0.00290975f, 0.00625247f, 0.02937804f, 0.01013064f, -0.00624247f, -0.00471575f, 0.00657559f,
      -0.03100131f, 0.03663468f, -0.00701449f, -0.02267093f, 0.00842566f, -0.01073638f, -0.01930492f, 0.01421315f,
      0.03018992f, 0.00599129f, 0.00773254f, 0.0330062f, 0.00622745f, 0.00355852f, -0.02076612f, -0.02770657f,
      -0.0232522f, -0.01313687f, -0.0039819f, -0.00212091f, -0.01119681f, -0.02110178f, -0.02889993f, -0.01824146f,
      0.00702973f, -0.02526855f, 0.02776029f, 0.02142943f, -0.01821483f, 0.01552922f, 0.01018873f, -0.02794646f,
      0.0458297f, 0.01344144f, -0.00684268f, 0.00251583f, -0.00135352f, -0.01950933f, 0.00551906f, -0.00823819f,
      -0.01059994f, 0.01091084f, -0.00090001f, -0.00103661f, 0.0260281f, -0.01601363f, -0.00342217f, 0.0198724f};

  global_bias_data = {
      -0.01f, 0.01f, -0.0009f, -0.001f, 0.02f, -0.01f, -0.003f, 0.01f,
      -0.004f, 0.001f, -0.03f, 0.006f, -0.02f, 0.008f, 0.009f, 0.004f,
      -0.01f, -0.002f, -0.01f, 0.003f, 0.03f, -0.01f, -0.001f, 0.02f};
}

// Run test on tiny longformer model with batch size 1 and sequence length 4 or 8
static void RunTinyLongformerBatch1(
    std::vector<float>& mask_data,
    std::vector<int>& global_data,
    std::vector<float>& input_data,
    std::vector<float>& output_data,
    bool use_float16) {
  int batch_size = 1;
  int one_sided_attention_window_size = 2;
  int hidden_size = 8;
  int number_of_heads = 2;

  std::vector<float> weight_data;
  std::vector<float> bias_data;
  std::vector<float> global_weight_data;
  std::vector<float> global_bias_data;
  GetTinyLongformerData(weight_data, bias_data, global_weight_data, global_bias_data);

  int sequence_length = static_cast<int>(mask_data.size()) / batch_size;

  RunAttentionTest(input_data, weight_data, bias_data, mask_data, global_weight_data, global_bias_data, global_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, one_sided_attention_window_size, use_float16);
}

static void RunTinyLongformerBatch1(
    std::vector<float>& mask_data,
    std::vector<int>& global_data,
    std::vector<float>& output_data,
    bool use_float16,
    bool window_cover_whole_sequence = false) {
  // Total windows size 4 will cover the whole sequence length 4
  std::vector<float> input_data;
  if (window_cover_whole_sequence) {
    input_data = {
        -1.1354f, -0.3925f, -0.4778f, -1.2506f, 0.0692f, 1.6338f, 0.1119f, 1.4415f,
        0.5219f, 0.1777f, 0.7090f, -2.1933f, 0.5258f, -0.0639f, -0.8511f, 1.1738f,
        -1.8157f, -0.6058f, 0.4016f, -0.4158f, -0.0343f, 1.0701f, -0.2686f, 1.6685f,
        0.5219f, 0.1777f, 0.7090f, -2.1933f, 0.5258f, -0.0639f, -0.8511f, 1.1738f};
  } else {
    input_data = {
        -1.1354f, -0.3925f, -0.4778f, -1.2506f, 0.0692f, 1.6338f, 0.1119f, 1.4415f,
        0.5219f, 0.1777f, 0.7090f, -2.1933f, 0.5258f, -0.0639f, -0.8511f, 1.1738f,
        -1.8157f, -0.6058f, 0.4016f, -0.4158f, -0.0343f, 1.0701f, -0.2686f, 1.6685f,
        -1.5789f, 1.1709f, -1.4334f, 0.7868f, -0.2459f, 0.3509f, -0.1686f, 1.1181f,
        0.7612f, -2.1558f, -0.2982f, 0.8303f, 0.8898f, -0.8073f, 0.0386f, 0.7414f,
        -1.6978f, 0.7141f, 1.1334f, -1.1359f, 0.8673f, -0.8174f, 0.7631f, 0.1731f,
        -1.0536f, -0.0425f, -1.1194f, -0.6423f, 2.1825f, 0.2547f, 0.6015f, -0.1809f,
        0.5219f, 0.1777f, 0.7090f, -2.1933f, 0.5258f, -0.0639f, -0.8511f, 1.1738f};
  }
  return RunTinyLongformerBatch1(mask_data, global_data, input_data, output_data, use_float16);
}

TEST(LongformerAttentionTest, LongformerAttention_Format1_NoGlobal) {
  // last word is masked.
  std::vector<float> mask_data = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -10000.0f};

  // no global attention.
  std::vector<int> global_data = {0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<float> output_data = {
      0.04935503f, 0.09777762f, 0.080508679f, 0.043586157f, 0.021670891f, -0.060086727f, 0.011003745f, -0.043082085f,
      0.030698329f, 0.084516019f, 0.075875372f, 0.030447651f, 0.012232142f, -0.061684664f, 0.011670727f, -0.024751294f,
      0.043819577f, 0.071589649f, 0.042219087f, 0.023776785f, 0.0082963137f, -0.052200478f, 0.015257062f, -0.022881128f,
      0.042480543f, 0.075072303f, 0.024976324f, 0.02212034f, -0.0051459596f, -0.053096619f, 0.012826156f, -0.028897678f,
      0.043811984f, 0.060341693f, 0.023737304f, 0.02106005f, -0.006941377f, -0.064169027f, 0.0014586616f, -0.037011269f,
      0.035779588f, 0.05780342f, 0.0065651359f, 0.0108707f, -0.009737955f, -0.065662488f, -0.0089816339f, -0.034388386f,
      0.056027573f, 0.062186595f, -0.01182458f, 0.017458128f, -0.007595225f, -0.06538119f, -0.016503287f, -0.055929951f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  RunTinyLongformerBatch1(mask_data, global_data, output_data, false);
}

TEST(LongformerAttentionTest, LongformerAttention_Format1_GlobalStart) {
  std::vector<float> mask_data = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -10000.0f};

  // Global at the start of the sequence
  std::vector<int> global_data = {1, 1, 0, 0, 0, 0, 0, 0};

  std::vector<float> output_data = {
      -0.045565117f, 0.05013489f, -0.029825294f, 0.0034495876f, 0.054528885f, -0.011395404f, -0.01799028f, 0.007753480f,
      -0.04556668f, 0.05010877f, -0.029804626f, 0.003465367f, 0.054491505f, -0.011396205f, -0.017886724f, 0.0078192977f,
      0.043819577f, 0.071589649f, 0.042219087f, 0.023776785f, 0.0082963137f, -0.052200478f, 0.015257062f, -0.022881128f,
      0.041874513f, 0.077329829f, 0.039112404f, 0.025902566f, 0.0027558794f, -0.058892913f, 0.010357748f, -0.03357362f,
      0.041588653f, 0.074910186f, 0.03832721f, 0.024912203f, 0.0037148837f, -0.06321992f, -0.00037646585f, -0.03805575f,
      0.035857923f, 0.075681195f, 0.029317651f, 0.018762534f, 0.0036572944f, -0.064060912f, -0.007641451f, -0.03648905f,
      0.048029676f, 0.081865683f, 0.022818407f, 0.024291465f, 0.0076162969f, -0.063553929f, -0.011883155f, -0.04981004f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  RunTinyLongformerBatch1(mask_data, global_data, output_data, false);
}

TEST(LongformerAttentionTest, LongformerAttention_Format1_NoCompactMemory) {
  std::vector<float> mask_data = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -10000.0f, -10000.0f, -10000.0f};

  std::vector<int> global_data = {1, 1, 0, 0, 0, 0, 0, 0};

  std::vector<float> input_data = {
      -1.6886f, -0.0712f, 0.7017f, -1.5513f, 0.1059f, 0.8119f, 1.1139f, 0.5776f,
      -1.0874f, 0.2870f, -0.3767f, -1.8203f, 0.0116f, 1.4285f, 0.6643f, 0.8931f,
      -1.0425f, 1.0148f, -0.6387f, -1.6496f, -0.3027f, 0.3839f, 1.0230f, 1.2119f,
      0.3576f, -1.5848f, 0.1713f, -0.9284f, 1.7172f, -0.8574f, 0.8390f, 0.2855f,
      -1.7443f, 0.4251f, 1.6560f, -0.5054f, -0.8734f, 0.4769f, 0.8434f, -0.2781f,
      0.5219f, 0.1777f, 0.7090f, -2.1933f, 0.5258f, -0.0639f, -0.8511f, 1.1738f,
      0.5219f, 0.1777f, 0.7090f, -2.1933f, 0.5258f, -0.0639f, -0.8511f, 1.1738f,
      0.5219f, 0.1777f, 0.7090f, -2.1933f, 0.5258f, -0.0639f, -0.8511f, 1.1738f};

  std::vector<float> output_data = {
      -0.042999879f, 0.049746618f, -0.03921134f, 0.022377649f, 0.014597901f, -0.008852208f, -0.0051137861f, -0.0317555f,
      -0.0430619f, 0.049675107f, -0.039144009f, 0.022413466f, 0.014596507f, -0.00886563f, -0.0051382119f, -0.031742509f,
      0.0120557f, 0.092630297f, 0.054536056f, 0.030807339f, 0.021582549f, -0.10123824f, -0.013279535f, -0.079527803f,
      0.011912746f, 0.092575297f, 0.054668114f, 0.030843586f, 0.021580728f, -0.10122664f, -0.013295304f, -0.079535671f,
      0.012123509f, 0.092629217f, 0.054470878f, 0.030805945f, 0.021572994f, -0.10122325f, -0.01326612f, -0.079525866f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {onnxruntime::contrib::longformer::kUseCompactMemory, "0"},
      }};

  RunTinyLongformerBatch1(mask_data, global_data, input_data, output_data, false);
}

TEST(LongformerAttentionTest, LongformerAttention_Format1_Float16) {
  std::vector<float> mask_data = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -10000.0f};

  // Global at the start of the sequence
  std::vector<int> global_data = {1, 1, 0, 0, 0, 0, 0, 0};

  std::vector<float> output_data = {
      -0.045562744f, 0.05014038f, -0.029830933f, 0.0034484863f, 0.054504395f, -0.011383057f, -0.017990112f, 0.00774765f,
      -0.045532227f, 0.050079346f, -0.029800415f, 0.0034713745f, 0.054504395f, -0.011390686f, -0.0178833f, 0.007820129f,
      0.043823242f, 0.071594238f, 0.042205811f, 0.023788452f, 0.0083084106f, -0.052185059f, 0.015258789f, -0.022903442f,
      0.041870117f, 0.077331543f, 0.039123535f, 0.025909424f, 0.0027542114f, -0.058898926f, 0.010360718f, -0.033599854f,
      0.04156494f, 0.074890137f, 0.038330078f, 0.024917603f, 0.003715515f, -0.063232422f, -0.00037002563f, -0.03805542f,
      0.035858154f, 0.075683594f, 0.029327393f, 0.01876831f, 0.0036621094f, -0.064086914f, -0.007644653f, -0.036499023f,
      0.048034668f, 0.081848145f, 0.02279663f, 0.024291992f, 0.0076179504f, -0.063537598f, -0.011878967f, -0.049804688f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  RunTinyLongformerBatch1(mask_data, global_data, output_data, true);
}

TEST(LongformerAttentionTest, LongformerAttention_Format1_FullWindow) {
  // last word is masked.
  std::vector<float> mask_data = {0.0f, 0.0f, 0.0f, -10000.0f};

  // no global attention.
  std::vector<int> global_data = {0, 0, 0, 0};

  std::vector<float> output_data = {
      0.04935503f, 0.09777762f, 0.080508679f, 0.043586157f, 0.021670891f, -0.060086727f, 0.011003745f, -0.043082085f,
      0.049329545f, 0.097794257f, 0.08051388f, 0.043572858f, 0.021667825f, -0.060093321f, 0.011015881f, -0.043088093f,
      0.049349822f, 0.097794116f, 0.080487557f, 0.043577947f, 0.021671539f, -0.060104892f, 0.011019757f, -0.043095518f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  // One double-sided window will cover the whole sequence.
  bool window_cover_whole_sequence = true;
  RunTinyLongformerBatch1(mask_data, global_data, output_data, false, window_cover_whole_sequence);
}

}  // namespace test
}  // namespace onnxruntime
