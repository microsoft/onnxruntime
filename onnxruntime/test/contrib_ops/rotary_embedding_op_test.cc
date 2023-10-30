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

static void RunTest(
    const std::vector<float>& input_data,
    const std::vector<int64_t>& position_ids,
    const std::vector<float>& cos_cache,
    const std::vector<float>& sin_cache,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int head_size,
    int num_heads,
    int max_sequence_length,
    int64_t interleaved,
    bool use_float16,
    bool disable_cpu,
    bool disable_cuda,
    bool disable_dml) {
  //    input        : (batch_size, sequence_length, hidden_size)
  //    position ids : (1) or (batch_size, sequence_length)
  //    cos cache    : (max_sequence_length, head_size / 2)
  //    sin cache    : (max_sequence_length, head_size / 2)
  //    interleaved  : 0 = false, 1 = true

  int hidden_size = num_heads * head_size;
  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> pos_dims;
  std::vector<int64_t> cache_dims = {max_sequence_length, head_size / 2};

  assert(hidden_size != 0 && head_size != 0 && num_heads != 0 && max_sequence_length != 0);
  assert(max_sequence_length >= sequence_length);
  if (position_ids.size() == 1) {
    pos_dims = {1};
  } else {
    pos_dims = {batch_size, sequence_length};
  }

  std::string op_type = "RotaryEmbedding";
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;

  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_dml = (nullptr != DefaultDmlExecutionProvider().get()) && !disable_dml;

  if (enable_cuda && !disable_cuda) {
    execution_providers.push_back(DefaultCudaExecutionProvider());
  }
  if (enable_dml && !disable_dml) {
    execution_providers.push_back(DefaultDmlExecutionProvider());
  }
  if (!use_float16 && !disable_cpu) {
    execution_providers.push_back(DefaultCpuExecutionProvider());
  }
  if (execution_providers.size() == 0) {
    // Return early if CI pipeline does not support EP (e.g. CUDA EP for CPU CI pipeline)
    return;
  }

  OpTester test(op_type.c_str(), 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("interleaved", interleaved);

  if (!use_float16) {
    test.AddInput<float>("input", input_dims, input_data);
    test.AddInput<int64_t>("position_ids", pos_dims, position_ids);
    test.AddInput<float>("cos_cache", cache_dims, cos_cache);
    test.AddInput<float>("sin_cache", cache_dims, sin_cache);
    test.AddOutput<float>("output", input_dims, output_data);
  } else {
    test.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
    test.AddInput<int64_t>("position_ids", pos_dims, position_ids);
    test.AddInput<MLFloat16>("cos_cache", cache_dims, ToFloat16(cos_cache));
    test.AddInput<MLFloat16>("sin_cache", cache_dims, ToFloat16(sin_cache));
    test.AddOutput<MLFloat16>("output", input_dims, ToFloat16(output_data));
  }
  test.SetOutputAbsErr("output", 0.002f);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

static void RunTests(const std::vector<float>& input_data,
                     const std::vector<int64_t>& position_ids,
                     const std::vector<float>& cos_cache,
                     const std::vector<float>& sin_cache,
                     const std::vector<float>& output_data,
                     int batch_size,
                     int sequence_length,
                     int head_size = 0,
                     int num_heads = 0,
                     int max_sequence_length = 0,
                     int64_t interleaved = 0,
                     bool use_float16 = true) {
  // FP32 test for CPU
  RunTest(input_data,
          position_ids,
          cos_cache,
          sin_cache,
          output_data,
          batch_size,
          sequence_length,
          head_size,
          num_heads,
          max_sequence_length,
          interleaved,
          false, /* use_fp16 */
          false, /* disable_cpu */
          true, /* disable_cuda */
          true /* disable_dml */);

  // FP32 test for CUDA and DML
  RunTest(input_data,
          position_ids,
          cos_cache,
          sin_cache,
          output_data,
          batch_size,
          sequence_length,
          head_size,
          num_heads,
          max_sequence_length,
          interleaved,
          false, /* use_fp16 */
          false, /* disable_cpu */
          false, /* disable_cuda */
          false /* disable_dml */);

  // FP16 test for CUDA and DML
  if (use_float16) {
    RunTest(input_data,
            position_ids,
            cos_cache,
            sin_cache,
            output_data,
            batch_size,
            sequence_length,
            head_size,
            num_heads,
            max_sequence_length,
            interleaved,
            true, /* use_fp16 */
            true, /* disable_cpu */
            false, /* disable_cuda*/
            false /* disable_dml */);
  }
}

// Interleaved = true, pos ids shape = (1)
TEST(RotaryEmbeddingTest, RotaryEmbedding_Interleaved_SmallData_LlamaMSFT) {
  int batch_size = 1;
  int sequence_length = 3;
  int num_heads = 2;
  int head_size = 4;
  int max_sequence_length = 8;
  int64_t interleaved = 1;  // true

  std::vector<float> input_data = {
      -1.0408f, 0.9166f, -1.3042f, -1.1097f, -0.1320f, -0.2751f, -0.2350f, 0.0937f,
      -1.2188f, 1.1676f, -1.0574f, -0.1188f, -0.7396f, -1.2425f, -0.1752f, 0.6990f,
      -0.8110f, 0.6737f, -1.1233f, -0.0919f, -0.6861f, 0.7202f, 0.1963f, 0.6142f};

  std::vector<int64_t> position_ids = {0};

  std::vector<float> cos_cache = {
      1.0000f, 1.0000f, 0.5403f, 0.9999f, -0.4161f, 0.9998f, -0.9900f, 0.9996f,
      -0.6536f, 0.9992f, 0.2837f, 0.9988f, 0.9602f, 0.9982f, 0.7539f, 0.9976f};

  std::vector<float> sin_cache = {
      0.0000f, 0.0000f, 0.8415f, 0.0100f, 0.9093f, 0.0200f, 0.1411f, 0.0300f,
      -0.7568f, 0.0400f, -0.9589f, 0.0500f, -0.2794f, 0.0600f, 0.6570f, 0.0699f};

  std::vector<float> output_data = {
      -1.0408f, 0.9166f, -1.3042f, -1.1097f, -0.1320f, -0.2751f, -0.2350f, 0.0937f,
      -1.6411f, -0.3948f, -1.0561f, -0.1294f, 0.6460f, -1.2937f, -0.1822f, 0.6972f,
      -0.2751f, -1.0178f, -1.1212f, -0.1143f, -0.3694f, -0.9235f, 0.1840f, 0.6180f};

  RunTests(input_data,
           position_ids,
           cos_cache,
           sin_cache,
           output_data,
           batch_size,
           sequence_length,
           head_size,
           num_heads,
           max_sequence_length,
           interleaved);
}

// Interleaved = true, pos ids shape = (1)
TEST(RotaryEmbeddingTest, RotaryEmbedding_Interleaved_LargeData_LlamaMSFT) {
  int batch_size = 2;
  int sequence_length = 8;
  int num_heads = 4;
  int head_size = 6;
  int max_sequence_length = 16;
  int64_t interleaved = 1;  // true

  std::vector<float> input_data = {
      -1.0408f, 0.9166f, -1.3042f, -1.1097f, -1.2188f,
      1.1676f, -1.0190f, 0.3157f, -1.6036f, 1.8493f,
      0.0447f, 1.5853f, 0.1036f, -0.3514f, 0.2421f,
      0.6463f, 0.8730f, -0.9276f, 1.0311f, -1.9557f,
      -0.1482f, 1.7376f, 2.2039f, -0.6589f, -1.0574f,
      -0.1188f, -0.9078f, 0.3452f, -0.5713f, -0.2351f,
      -0.5912f, 1.1312f, 0.7562f, -1.2023f, -0.5833f,
      -0.4407f, 0.1766f, 1.0224f, -0.4826f, -0.5421f,
      -0.5342f, -0.6413f, 1.3314f, -0.4498f, 0.5493f,
      0.0539f, 0.2601f, 0.8570f, 1.0076f, -0.7529f,
      -0.2250f, -0.4327f, -1.5071f, -0.4586f, -1.9791f,
      0.7787f, -0.7749f, -0.1398f, 1.1414f, -0.6354f,
      0.0352f, -0.4765f, -0.0409f, 1.1993f, 0.5374f,
      -0.1930f, 2.5211f, -0.0452f, -0.3105f, -0.9407f,
      -0.0034f, 1.5199f, -0.8480f, 0.5266f, 0.0299f,
      -0.0498f, 1.0651f, 0.8860f, -1.4702f, -0.2134f,
      -0.8707f, 1.6159f, -0.2356f, 0.9444f, 0.5937f,
      0.7203f, 0.5061f, 1.5192f, -0.4897f, 0.9231f,
      0.2654f, -0.1441f, 0.5407f, -1.5476f, 0.6455f,
      -1.1382f, 0.4640f, -0.4986f, 0.1289f, 2.7631f,
      0.1405f, 1.1191f, 2.1134f, -0.9754f, 0.1757f,
      -0.1319f, -0.2735f, 0.3355f, -0.6008f, -1.1164f,
      0.2577f, -0.7226f, -0.9244f, 1.8737f, 0.6052f,
      1.1904f, 1.2195f, -0.0470f, -1.0914f, 1.0223f,
      0.3152f, 1.7528f, -0.7650f, 1.8299f, -0.2784f,
      -0.2719f, 0.1885f, 2.1432f, 0.8527f, 0.0965f,
      -0.0625f, 0.8269f, 1.0122f, -1.4482f, -0.0644f,
      0.3215f, 0.5908f, -1.4197f, 0.2113f, 0.0306f,
      0.3604f, 0.3166f, -0.8975f, -0.6393f, -1.2944f,
      -0.0243f, -0.2354f, -0.7087f, 1.1566f, 0.4296f,
      0.5599f, -0.7776f, 0.3339f, 0.1759f, 2.1108f,
      1.0702f, 0.8279f, -0.2969f, 0.7120f, -0.2068f,
      -0.1548f, 0.1553f, 0.6207f, -0.1690f, -0.5816f,
      1.2632f, 0.0695f, 1.1862f, -1.1874f, -0.7468f,
      -0.9320f, -0.8579f, -0.9647f, -0.0991f, 0.0195f,
      1.1213f, -1.4873f, -0.2043f, -1.0466f, -1.5772f,
      -0.0489f, 0.3430f, 0.1264f, 0.1519f, -1.3639f,
      -1.6593f, 1.8127f, -1.4459f, -0.2158f, -0.9792f,
      -1.4392f, 0.6508f, 0.8964f, 0.5717f, -0.2390f,
      0.6983f, -1.3416f, 0.2715f, -0.2852f, 0.6051f,
      0.2167f, -0.2181f, -1.6306f, 1.4788f, 0.2754f,
      -0.0261f, -0.4618f, -0.5646f, -1.0389f, 0.5819f,
      1.3697f, 0.0002f, 1.5333f, -1.0556f, -0.1254f,
      0.1527f, -0.5996f, -1.0962f, 1.6327f, 1.3951f,
      0.8784f, 0.3389f, 1.2907f, 0.3124f, 0.7299f,
      1.4220f, 0.3375f, 0.0438f, 1.8698f, -0.2635f,
      -2.0799f, -0.6313f, 0.4090f, -1.1458f, 0.0784f,
      -1.8848f, -1.6165f, 0.6179f, 0.9905f, -0.0729f,
      0.5054f, -0.6681f, -1.4382f, 1.7547f, -0.9605f,
      -0.4558f, -1.6105f, 0.2979f, 1.1537f, -1.5604f,
      1.2779f, -1.2514f, 0.6056f, 0.5763f, -3.3558f,
      0.2836f, 0.6909f, -0.7631f, 2.4451f, -0.3500f,
      1.3289f, -0.6494f, 0.3478f, 1.0038f, -0.2937f,
      0.9238f, -1.2185f, 0.4138f, 0.5033f, 0.9174f,
      1.8131f, 1.4436f, -0.4207f, 0.0220f, -0.6807f,
      -1.3306f, 1.5646f, 0.3338f, 0.7105f, 0.4683f,
      -0.6179f, 0.0818f, -0.0488f, -0.9810f, -1.3632f,
      0.0929f, -1.7926f, -0.2921f, -0.4792f, 0.6756f,
      -0.3413f, -0.2242f, -0.2111f, 0.6282f, 0.1667f,
      -1.4055f, 1.5895f, 1.0838f, -0.9077f, -0.8060f,
      0.7967f, -2.9351f, 2.4179f, -0.4026f, 0.6451f,
      1.6845f, -0.0901f, 0.6106f, 2.3603f, 1.3908f,
      -0.7917f, -0.6734f, -0.1213f, -1.1116f, -0.7401f,
      -0.7879f, 0.0606f, -2.3337f, -1.2603f, -1.7245f,
      -0.3533f, -0.9421f, -0.1776f, 0.3992f, -1.7142f,
      -0.5319f, -0.8848f, 0.6513f, 1.0002f, -1.4699f,
      -1.4254f, 0.7013f, 0.2414f, 0.2551f, -0.7457f,
      0.3133f, -1.0941f, -0.3682f, -0.0163f, -0.0645f,
      -0.8101f, 0.1415f, 0.0551f, 0.5873f, -0.5887f,
      -1.4733f, -0.8565f, 0.7400f, -0.5033f, 0.0553f,
      0.9265f, -0.8652f, -0.0288f, -0.2209f, 0.0610f,
      0.6776f, 0.4361f, -0.8052f, 0.3955f, 0.8988f,
      0.8238f, 0.2262f, 1.2912f, 0.6488f, 1.2114f,
      1.3569f, 0.2983f, 0.4718f, -1.1936f, 0.7928f,
      -0.8665f, 0.9468f, 1.1629f, 0.0616f, -1.3136f,
      -0.2764f, 0.0277f, -0.1126f, 0.2342f, -0.5866f,
      -1.8219f, 1.1079f, 0.5795f, -1.4249f};

  std::vector<int64_t> position_ids = {0};

  std::vector<float> cos_cache = {
      1.0000f, 1.0000f, 1.0000f, 0.5403f, 0.9989f, 1.0000f, -0.4161f, 0.9957f,
      1.0000f, -0.9900f, 0.9903f, 1.0000f, -0.6536f, 0.9828f, 1.0000f, 0.2837f,
      0.9732f, 0.9999f, 0.9602f, 0.9615f, 0.9999f, 0.7539f, 0.9477f, 0.9999f,
      -0.1455f, 0.9318f, 0.9999f, -0.9111f, 0.9140f, 0.9998f, -0.8391f, 0.8942f,
      0.9998f, 0.0044f, 0.8725f, 0.9997f, 0.8439f, 0.8488f, 0.9997f, 0.9074f,
      0.8234f, 0.9996f, 0.1367f, 0.7962f, 0.9995f, -0.7597f, 0.7673f, 0.9995f};

  std::vector<float> sin_cache = {
      0.0000f, 0.0000f, 0.0000f, 0.8415f, 0.0464f, 0.0022f, 0.9093f, 0.0927f,
      0.0043f, 0.1411f, 0.1388f, 0.0065f, -0.7568f, 0.1846f, 0.0086f, -0.9589f,
      0.2300f, 0.0108f, -0.2794f, 0.2749f, 0.0129f, 0.6570f, 0.3192f, 0.0151f,
      0.9894f, 0.3629f, 0.0172f, 0.4121f, 0.4057f, 0.0194f, -0.5440f, 0.4477f,
      0.0215f, -1.0000f, 0.4887f, 0.0237f, -0.5366f, 0.5286f, 0.0259f, 0.4202f,
      0.5675f, 0.0280f, 0.9906f, 0.6050f, 0.0302f, 0.6503f, 0.6413f, 0.0323f};

  std::vector<float> output_data = {
      -1.0408f, 0.9166f, -1.3042f, -1.1097f, -1.2188f,
      1.1676f, -1.0190f, 0.3157f, -1.6036f, 1.8493f,
      0.0447f, 1.5853f, 0.1036f, -0.3514f, 0.2421f,
      0.6463f, 0.8730f, -0.9276f, 1.0311f, -1.9557f,
      -0.1482f, 1.7376f, 2.2039f, -0.6589f, -0.4713f,
      -0.9540f, -0.9229f, 0.3027f, -0.5708f, -0.2363f,
      -1.2713f, 0.1137f, 0.8112f, -1.1659f, -0.5824f,
      -0.4419f, -0.7649f, 0.7011f, -0.4569f, -0.5639f,
      -0.5328f, -0.6424f, 1.0979f, 0.8773f, 0.5462f,
      0.0793f, 0.2582f, 0.8576f, 0.2653f, 1.2295f,
      -0.1839f, -0.4517f, -1.5052f, -0.4651f, 0.1155f,
      -2.1237f, -0.7586f, -0.2110f, 1.1441f, -0.6304f,
      0.4186f, 0.2303f, -0.1519f, 1.1903f, 0.5382f,
      -0.1906f, -1.0080f, 2.3112f, -0.2220f, -0.9655f,
      -0.0099f, 1.5198f, 0.7652f, -0.6410f, 0.0365f,
      -0.0452f, 1.0593f, 0.8929f, 1.4856f, 0.0038f,
      -1.0865f, 1.4794f, -0.2417f, 0.9428f, -0.6894f,
      -0.6293f, 0.2904f, 1.5747f, -0.4956f, 0.9199f,
      -0.2424f, 0.1801f, 0.7503f, -1.4576f, 0.6529f,
      -1.1340f, -0.6807f, -0.0252f, -0.3834f, 2.7394f,
      0.1308f, 1.1203f, -2.1196f, -0.9618f, 0.1970f,
      -0.0972f, -0.2764f, 0.3332f, -0.4522f, 1.1844f,
      0.3867f, -0.6626f, -0.9405f, 1.8656f, 0.5053f,
      -1.2361f, 1.2072f, 0.1789f, -1.1002f, 1.0129f,
      1.7702f, 0.1949f, -1.1653f, 1.6049f, -0.2755f,
      -0.2749f, 2.1087f, 0.4272f, 0.8076f, 0.2900f,
      -0.0714f, 0.8261f, -1.1016f, -1.3814f, -0.1366f,
      0.2981f, 0.6060f, -1.4132f, 0.0893f, -0.1939f,
      0.2779f, 0.3910f, -0.8906f, -0.6489f, -1.2496f,
      0.3383f, -0.0315f, -0.7461f, 1.1510f, 0.4445f,
      0.3203f, -0.9031f, 0.2727f, 0.2609f, 2.0968f,
      1.0974f, 0.7120f, -0.5164f, 0.7415f, -0.0031f,
      -0.1568f, 0.1533f, 0.5487f, -0.3357f, -0.9064f,
      1.0546f, 0.0542f, 1.1870f, -0.4045f, -1.3431f,
      -0.6094f, -1.1105f, -0.9631f, -0.1137f, -0.7219f,
      0.8582f, -1.3443f, -0.6684f, -1.0227f, -1.5929f,
      -0.2622f, 0.2264f, 0.0713f, 0.1843f, -1.3387f,
      -1.6797f, 2.3165f, 0.1009f, 0.1081f, -0.9969f,
      -1.4488f, 0.6291f, 0.8964f, 0.5717f, -0.2390f,
      0.6983f, -1.3416f, 0.2715f, -0.2852f, 0.6051f,
      0.2167f, -0.2181f, -1.6306f, 1.4788f, 0.2754f,
      -0.0261f, -0.4618f, -0.5646f, -1.0389f, 0.5819f,
      1.3697f, 0.0002f, 1.5333f, -1.0556f, -0.1254f,
      0.1527f, 0.5985f, -1.0968f, 1.5662f, 1.4693f,
      0.8776f, 0.3408f, 0.4345f, 1.2549f, 0.6631f,
      1.4543f, 0.3374f, 0.0445f, 1.2320f, 1.4311f,
      -2.0483f, -0.7272f, 0.4114f, -1.1449f, 1.6283f,
      -0.9524f, -1.6435f, 0.5422f, 0.9907f, -0.0708f,
      0.3972f, 0.7376f, -1.5947f, 1.6138f, -0.9586f,
      -0.4600f, 0.3993f, -1.5884f, 1.2934f, -1.4467f,
      1.2833f, -1.2459f, -0.7760f, 0.3108f, -3.3677f,
      -0.0287f, 0.6942f, -0.7601f, -0.6993f, 2.3690f,
      1.3834f, -0.5234f, 0.3435f, 1.0053f, 0.1604f,
      -0.9560f, -1.2641f, 0.2406f, 0.4973f, 0.9206f,
      -1.9987f, -1.1733f, -0.4197f, -0.0366f, -0.6720f,
      -1.3350f, -1.5960f, -0.1097f, 0.6386f, 0.5624f,
      -0.6184f, 0.0778f, 0.1867f, 0.9643f, -1.3629f,
      -0.0972f, -1.7907f, -0.3037f, 0.8245f, -0.0789f,
      -0.2940f, -0.2833f, -0.2165f, 0.6264f, -1.1726f,
      0.7926f, 1.3621f, 1.3586f, -0.9007f, -0.8138f,
      -2.7421f, 1.3155f, 2.4507f, 0.0507f, 0.6305f,
      1.6900f, 0.5210f, -0.3309f, 2.0630f, 1.8026f,
      -0.7859f, -0.6802f, -1.1003f, -0.1990f, -0.5391f,
      -0.9370f, 0.0857f, -2.3330f, -2.0112f, 0.7193f,
      -0.1272f, -0.9981f, -0.1818f, 0.3973f, -0.9963f,
      1.4929f, -1.0109f, 0.4304f, 1.0160f, -1.4590f,
      0.2682f, 1.5658f, 0.1762f, 0.3038f, -0.7491f,
      0.3052f, -1.1534f, -0.0478f, 0.0021f, -0.0665f,
      -0.8118f, 0.1310f, 0.2171f, 0.5485f, -0.1610f,
      -1.5784f, -0.8660f, 0.7289f, -0.4678f, 0.1937f,
      1.1287f, -0.5772f, -0.0259f, -0.2212f, 0.2479f,
      0.6336f, 0.6407f, -0.6543f, 0.3838f, 0.9039f,
      0.4724f, 0.7117f, 1.0165f, 1.0270f, 1.1908f,
      1.3750f, -0.0850f, 0.5517f, -1.3842f, 0.3703f,
      -0.8806f, 0.9336f, 0.8362f, 0.8105f, -1.1566f,
      -0.6813f, 0.0294f, -0.1122f, 0.5620f, -0.2884f,
      -2.0803f, 0.4684f, 0.6009f, -1.4160f};

  RunTests(input_data,
           position_ids,
           cos_cache,
           sin_cache,
           output_data,
           batch_size,
           sequence_length,
           head_size,
           num_heads,
           max_sequence_length,
           interleaved);
}

// Interleaved = false, pos ids shape = (1)
TEST(RotaryEmbeddingTest, RotaryEmbedding_NotInterleaved_LargeData_LlamaMSFT) {
  int batch_size = 2;
  int sequence_length = 8;
  int num_heads = 4;
  int head_size = 6;
  int max_sequence_length = 16;
  int64_t interleaved = 0;  // false

  std::vector<float> input_data = {
      -1.0408f, 0.9166f, -1.3042f, -1.1097f, -1.2188f,
      1.1676f, -1.0190f, 0.3157f, -1.6036f, 1.8493f,
      0.0447f, 1.5853f, 0.1036f, -0.3514f, 0.2421f,
      0.6463f, 0.8730f, -0.9276f, 1.0311f, -1.9557f,
      -0.1482f, 1.7376f, 2.2039f, -0.6589f, -1.0574f,
      -0.1188f, -0.9078f, 0.3452f, -0.5713f, -0.2351f,
      -0.5912f, 1.1312f, 0.7562f, -1.2023f, -0.5833f,
      -0.4407f, 0.1766f, 1.0224f, -0.4826f, -0.5421f,
      -0.5342f, -0.6413f, 1.3314f, -0.4498f, 0.5493f,
      0.0539f, 0.2601f, 0.8570f, 1.0076f, -0.7529f,
      -0.2250f, -0.4327f, -1.5071f, -0.4586f, -1.9791f,
      0.7787f, -0.7749f, -0.1398f, 1.1414f, -0.6354f,
      0.0352f, -0.4765f, -0.0409f, 1.1993f, 0.5374f,
      -0.1930f, 2.5211f, -0.0452f, -0.3105f, -0.9407f,
      -0.0034f, 1.5199f, -0.8480f, 0.5266f, 0.0299f,
      -0.0498f, 1.0651f, 0.8860f, -1.4702f, -0.2134f,
      -0.8707f, 1.6159f, -0.2356f, 0.9444f, 0.5937f,
      0.7203f, 0.5061f, 1.5192f, -0.4897f, 0.9231f,
      0.2654f, -0.1441f, 0.5407f, -1.5476f, 0.6455f,
      -1.1382f, 0.4640f, -0.4986f, 0.1289f, 2.7631f,
      0.1405f, 1.1191f, 2.1134f, -0.9754f, 0.1757f,
      -0.1319f, -0.2735f, 0.3355f, -0.6008f, -1.1164f,
      0.2577f, -0.7226f, -0.9244f, 1.8737f, 0.6052f,
      1.1904f, 1.2195f, -0.0470f, -1.0914f, 1.0223f,
      0.3152f, 1.7528f, -0.7650f, 1.8299f, -0.2784f,
      -0.2719f, 0.1885f, 2.1432f, 0.8527f, 0.0965f,
      -0.0625f, 0.8269f, 1.0122f, -1.4482f, -0.0644f,
      0.3215f, 0.5908f, -1.4197f, 0.2113f, 0.0306f,
      0.3604f, 0.3166f, -0.8975f, -0.6393f, -1.2944f,
      -0.0243f, -0.2354f, -0.7087f, 1.1566f, 0.4296f,
      0.5599f, -0.7776f, 0.3339f, 0.1759f, 2.1108f,
      1.0702f, 0.8279f, -0.2969f, 0.7120f, -0.2068f,
      -0.1548f, 0.1553f, 0.6207f, -0.1690f, -0.5816f,
      1.2632f, 0.0695f, 1.1862f, -1.1874f, -0.7468f,
      -0.9320f, -0.8579f, -0.9647f, -0.0991f, 0.0195f,
      1.1213f, -1.4873f, -0.2043f, -1.0466f, -1.5772f,
      -0.0489f, 0.3430f, 0.1264f, 0.1519f, -1.3639f,
      -1.6593f, 1.8127f, -1.4459f, -0.2158f, -0.9792f,
      -1.4392f, 0.6508f, 0.8964f, 0.5717f, -0.2390f,
      0.6983f, -1.3416f, 0.2715f, -0.2852f, 0.6051f,
      0.2167f, -0.2181f, -1.6306f, 1.4788f, 0.2754f,
      -0.0261f, -0.4618f, -0.5646f, -1.0389f, 0.5819f,
      1.3697f, 0.0002f, 1.5333f, -1.0556f, -0.1254f,
      0.1527f, -0.5996f, -1.0962f, 1.6327f, 1.3951f,
      0.8784f, 0.3389f, 1.2907f, 0.3124f, 0.7299f,
      1.4220f, 0.3375f, 0.0438f, 1.8698f, -0.2635f,
      -2.0799f, -0.6313f, 0.4090f, -1.1458f, 0.0784f,
      -1.8848f, -1.6165f, 0.6179f, 0.9905f, -0.0729f,
      0.5054f, -0.6681f, -1.4382f, 1.7547f, -0.9605f,
      -0.4558f, -1.6105f, 0.2979f, 1.1537f, -1.5604f,
      1.2779f, -1.2514f, 0.6056f, 0.5763f, -3.3558f,
      0.2836f, 0.6909f, -0.7631f, 2.4451f, -0.3500f,
      1.3289f, -0.6494f, 0.3478f, 1.0038f, -0.2937f,
      0.9238f, -1.2185f, 0.4138f, 0.5033f, 0.9174f,
      1.8131f, 1.4436f, -0.4207f, 0.0220f, -0.6807f,
      -1.3306f, 1.5646f, 0.3338f, 0.7105f, 0.4683f,
      -0.6179f, 0.0818f, -0.0488f, -0.9810f, -1.3632f,
      0.0929f, -1.7926f, -0.2921f, -0.4792f, 0.6756f,
      -0.3413f, -0.2242f, -0.2111f, 0.6282f, 0.1667f,
      -1.4055f, 1.5895f, 1.0838f, -0.9077f, -0.8060f,
      0.7967f, -2.9351f, 2.4179f, -0.4026f, 0.6451f,
      1.6845f, -0.0901f, 0.6106f, 2.3603f, 1.3908f,
      -0.7917f, -0.6734f, -0.1213f, -1.1116f, -0.7401f,
      -0.7879f, 0.0606f, -2.3337f, -1.2603f, -1.7245f,
      -0.3533f, -0.9421f, -0.1776f, 0.3992f, -1.7142f,
      -0.5319f, -0.8848f, 0.6513f, 1.0002f, -1.4699f,
      -1.4254f, 0.7013f, 0.2414f, 0.2551f, -0.7457f,
      0.3133f, -1.0941f, -0.3682f, -0.0163f, -0.0645f,
      -0.8101f, 0.1415f, 0.0551f, 0.5873f, -0.5887f,
      -1.4733f, -0.8565f, 0.7400f, -0.5033f, 0.0553f,
      0.9265f, -0.8652f, -0.0288f, -0.2209f, 0.0610f,
      0.6776f, 0.4361f, -0.8052f, 0.3955f, 0.8988f,
      0.8238f, 0.2262f, 1.2912f, 0.6488f, 1.2114f,
      1.3569f, 0.2983f, 0.4718f, -1.1936f, 0.7928f,
      -0.8665f, 0.9468f, 1.1629f, 0.0616f, -1.3136f,
      -0.2764f, 0.0277f, -0.1126f, 0.2342f, -0.5866f,
      -1.8219f, 1.1079f, 0.5795f, -1.4249f};

  std::vector<int64_t> position_ids = {0};

  std::vector<float> cos_cache = {
      1.0000f, 1.0000f, 1.0000f, 0.5403f, 0.9989f, 1.0000f, -0.4161f, 0.9957f,
      1.0000f, -0.9900f, 0.9903f, 1.0000f, -0.6536f, 0.9828f, 1.0000f, 0.2837f,
      0.9732f, 0.9999f, 0.9602f, 0.9615f, 0.9999f, 0.7539f, 0.9477f, 0.9999f,
      -0.1455f, 0.9318f, 0.9999f, -0.9111f, 0.9140f, 0.9998f, -0.8391f, 0.8942f,
      0.9998f, 0.0044f, 0.8725f, 0.9997f, 0.8439f, 0.8488f, 0.9997f, 0.9074f,
      0.8234f, 0.9996f, 0.1367f, 0.7962f, 0.9995f, -0.7597f, 0.7673f, 0.9995f};

  std::vector<float> sin_cache = {
      0.0000f, 0.0000f, 0.0000f, 0.8415f, 0.0464f, 0.0022f, 0.9093f, 0.0927f,
      0.0043f, 0.1411f, 0.1388f, 0.0065f, -0.7568f, 0.1846f, 0.0086f, -0.9589f,
      0.2300f, 0.0108f, -0.2794f, 0.2749f, 0.0129f, 0.6570f, 0.3192f, 0.0151f,
      0.9894f, 0.3629f, 0.0172f, 0.4121f, 0.4057f, 0.0194f, -0.5440f, 0.4477f,
      0.0215f, -1.0000f, 0.4887f, 0.0237f, -0.5366f, 0.5286f, 0.0259f, 0.4202f,
      0.5675f, 0.0280f, 0.9906f, 0.6050f, 0.0302f, 0.6503f, 0.6413f, 0.0323f};

  std::vector<float> output_data = {
      -1.0408f, 0.9166f, -1.3042f, -1.1097f, -1.2188f,
      1.1676f, -1.0190f, 0.3157f, -1.6036f, 1.8493f,
      0.0447f, 1.5853f, 0.1036f, -0.3514f, 0.2421f,
      0.6463f, 0.8730f, -0.9276f, 1.0311f, -1.9557f,
      -0.1482f, 1.7376f, 2.2039f, -0.6589f, -0.8618f,
      -0.0922f, -0.9073f, -0.7032f, -0.5762f, -0.2371f,
      0.6923f, 1.1571f, 0.7572f, -1.1471f, -0.5302f,
      -0.4391f, 0.5516f, 1.0461f, -0.4812f, -0.1443f,
      -0.4862f, -0.6423f, 0.6740f, -0.4614f, 0.5475f,
      1.1495f, 0.2389f, 0.8582f, -0.0259f, -0.6099f,
      -0.2230f, 1.0963f, -1.5704f, -0.4595f, 0.9507f,
      0.6696f, -0.7721f, -1.7415f, 1.2087f, -0.6387f,
      -1.1052f, -0.5243f, -0.0400f, -0.4671f, 0.4909f,
      -0.1931f, -0.1937f, -0.0447f, -0.3171f, 2.6839f,
      -0.0076f, 1.5185f, 0.8465f, 0.3737f, 0.0242f,
      -0.0703f, 1.1279f, 0.8862f, 1.2275f, -0.1786f,
      -0.8767f, -1.8072f, -0.2630f, 0.9387f, -0.8021f,
      0.7813f, 0.5001f, -1.4202f, -0.3850f, 0.9263f,
      -0.0443f, -0.2323f, 0.5480f, 1.5696f, 0.6193f,
      -1.1346f, 1.7878f, -0.5160f, 0.1192f, -2.1572f,
      0.0460f, 1.1202f, -1.4812f, -0.9082f, 0.1728f,
      -1.5132f, -0.4489f, 0.3370f, -0.1541f, -0.9266f,
      0.2416f, 0.9270f, -1.1146f, 1.8758f, -0.4312f,
      1.3714f, 1.2106f, -0.4272f, -0.8529f, 1.0328f,
      1.8441f, 1.7698f, -0.7620f, 0.2168f, 0.1322f,
      -0.2802f, 0.1460f, 2.1002f, 0.8437f, -0.1534f,
      0.4321f, 0.8360f, 0.5955f, -1.5452f, -0.0491f,
      -0.8794f, 0.2418f, -1.4203f, 0.3635f, 0.2362f,
      0.3672f, -0.1128f, -0.8664f, -0.6354f, -1.4409f,
      -0.3413f, -0.2409f, -0.3188f, 1.1054f, 0.4265f,
      0.5867f, -1.3279f, 0.3201f, 0.0125f, 1.8157f,
      1.0745f, 0.7372f, -0.2429f, 0.7100f, -0.4299f,
      -0.2304f, 0.1645f, 0.9489f, -0.1816f, -0.5968f,
      1.0394f, 0.0204f, 1.1786f, -0.3315f, -0.3997f,
      -0.9304f, -1.4268f, -1.1526f, -0.1132f, 0.1490f,
      1.3967f, -1.4634f, -0.1412f, -0.6339f, -1.5995f,
      -0.1366f, 0.7604f, 0.1514f, 0.0824f, -1.1830f,
      -1.6572f, 2.0099f, -0.9108f, -0.2256f, 0.4527f,
      -1.8254f, 0.6475f, 0.8964f, 0.5717f, -0.2390f,
      0.6983f, -1.3416f, 0.2715f, -0.2852f, 0.6051f,
      0.2167f, -0.2181f, -1.6306f, 1.4788f, 0.2754f,
      -0.0261f, -0.4618f, -0.5646f, -1.0389f, 0.5819f,
      1.3697f, 0.0002f, 1.5333f, -1.0556f, -0.1254f,
      0.1527f, -1.4979f, -1.1358f, 1.6320f, 0.2493f,
      0.8266f, 0.3424f, -0.4992f, 0.2964f, 0.7298f,
      1.8544f, 0.3516f, 0.0454f, 1.5415f, -0.2822f,
      -2.0774f, 1.2323f, 0.3963f, -1.1503f, -0.4775f,
      -1.9287f, -1.6164f, 0.3998f, 0.9020f, -0.0764f,
      -1.8059f, -0.5762f, -1.4362f, -0.2706f, -1.0183f,
      -0.4620f, 2.0891f, 0.1782f, 1.1591f, -0.8151f,
      1.3000f, -1.2464f, -0.5099f, 0.5098f, -3.3525f,
      0.4326f, 0.7414f, -0.7775f, -0.4271f, -0.3807f,
      1.3245f, 2.4936f, 0.3139f, 1.0095f, 0.2323f,
      0.8450f, -1.2244f, -0.4511f, 0.6266f, 0.9095f,
      -1.7981f, 1.5241f, -0.4121f, 0.2341f, -0.4737f,
      -1.3333f, -1.6150f, 0.4164f, 0.7100f, -0.2429f,
      -0.5656f, 0.0863f, 0.0352f, -0.7227f, -1.3613f,
      -0.0988f, -1.9114f, -0.3009f, 0.1435f, 0.7029f,
      -0.3467f, 0.5092f, -0.0828f, 0.6253f, 0.7113f,
      -1.2138f, 1.5964f, -0.8346f, -1.1515f, -0.7923f,
      -0.8254f, -3.0038f, 2.4033f, -0.3398f, 0.0922f,
      1.7053f, 1.1114f, 0.7462f, 2.3660f, -0.8409f,
      -0.6654f, -0.6530f, -0.7899f, -1.0957f, -0.7149f,
      -0.1072f, -0.1967f, -2.3416f, -1.2609f, -1.6375f,
      -0.3576f, 0.9413f, -0.5694f, 0.3954f, 0.1383f,
      -0.7477f, -0.8689f, 1.8286f, 0.8510f, -1.4793f,
      -0.1597f, 0.8541f, 0.2380f, 1.4392f, -0.5644f,
      0.3158f, -1.0686f, -0.1313f, -0.0181f, 0.2438f,
      -0.8801f, 0.1413f, -0.3587f, 0.8002f, -0.5982f,
      -1.4301f, -0.6620f, 0.7324f, -0.7250f, 0.0610f,
      0.9293f, -0.6902f, -0.0125f, -0.2089f, -0.1664f,
      0.5428f, 0.4245f, -0.7901f, 0.5665f, 0.9044f,
      0.1948f, -0.1723f, 1.2705f, 1.0303f, 1.2202f,
      1.3762f, -0.2959f, 0.7237f, -1.2077f, 0.7937f,
      -0.6705f, 0.9287f, 1.0583f, 0.0496f, -1.3118f,
      0.5556f, 0.0459f, -0.1324f, -0.5513f, -0.7409f,
      -1.8002f, 0.9892f, 0.3619f, -1.4522f};

  RunTests(input_data,
           position_ids,
           cos_cache,
           sin_cache,
           output_data,
           batch_size,
           sequence_length,
           head_size,
           num_heads,
           max_sequence_length,
           interleaved);
}

// Interleaved = false, pos ids shape = (batch_size, sequence_length)
TEST(RotaryEmbeddingTest, RotaryEmbedding_NotInterleaved_SmallData_LlamaMSFT) {
  int batch_size = 1;
  int sequence_length = 2;
  int num_heads = 3;
  int head_size = 6;
  int max_sequence_length = 4;
  int64_t interleaved = 0;  // false

  std::vector<float> input_data = {
      -1.0408f, 0.9166f, -1.3042f, -1.1097f, -1.2188f, 1.1676f, 1.0076f, -0.7529f,
      -0.2250f, -0.4327f, -1.5071f, -0.4586f, -0.8663f, -0.2656f, 0.1665f, 0.7911f,
      -0.9320f, -0.8579f, -1.0574f, -0.1188f, -0.9078f, 0.3452f, -0.5713f, -0.2351f,
      -0.8480f, 0.5266f, -1.2944f, -0.0243f, -0.2354f, -0.7087f, -0.9647f, -0.0991f,
      -0.2994f, -0.0650f, -1.5720f, -1.3211f};

  std::vector<int64_t> position_ids = {0, 1};

  std::vector<float> cos_cache = {
      1.0000f, 1.0000f, 1.0000f, 0.5403f, 0.9989f, 1.0000f, -0.4161f, 0.9957f,
      1.0000f, -0.9900f, 0.9903f, 1.0000f};

  std::vector<float> sin_cache = {
      0.0000f, 0.0000f, 0.0000f, 0.8415f, 0.0464f, 0.0022f, 0.9093f, 0.0927f, 0.0043f,
      0.1411f, 0.1388f, 0.0065f};

  std::vector<float> output_data = {
      -1.0408f, 0.9166f, -1.3042f, -1.1097f, -1.2188f, 1.1676f, 1.0076f, -0.7529f,
      -0.2250f, -0.4327f, -1.5071f, -0.4586f, -0.8663f, -0.2656f, 0.1665f, 0.7911f,
      -0.9320f, -0.8579f, -0.8618f, -0.0922f, -0.9073f, -0.7032f, -0.5762f, -0.2371f,
      -0.4377f, 0.5370f, -1.2929f, -0.7267f, -0.2107f, -0.7115f, -0.4666f, -0.0261f,
      -0.2965f, -0.8469f, -1.5749f, -1.3217f};

  RunTests(input_data,
           position_ids,
           cos_cache,
           sin_cache,
           output_data,
           batch_size,
           sequence_length,
           head_size,
           num_heads,
           max_sequence_length,
           interleaved);
}

}  // namespace test
}  // namespace onnxruntime
