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

constexpr float epsilon_ = 1e-12f;

static void RunTest(
    const std::vector<float>& input_data,
    const std::vector<int64_t>& position_ids,
    const std::vector<float>& cos_cache,
    const std::vector<float>& sin_cache,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int hidden_size,
    int head_size,
    int num_heads,
    int max_sequence_length,
    bool use_float16) {

  // When LLaMA Microsoft model:
  //    input        : (batch_size, sequence_length, hidden_size) or (batch_size, sequence_length, num_heads, head_size)
  //    position ids : (1)
  //    cos cache    : (max_sequence_length, head_size / 2)
  //    sin cache    : (max_sequence_length, head_size / 2)
  // When LLaMA Hugging Face model:
  //    input        : (batch_size, num_heads, sequence_length, head_size)
  //    position ids : (batch_size, sequence_length)
  //    cos cache    : (sequence_length, head_size)
  //    sin cache    : (sequence_length, head_size)

  std::vector<int64_t> input_dims;
  std::vector<int64_t> pos_dims;
  std::vector<int64_t> cache_dims;

  if (position_ids.size() == 1) {
    assert(hidden_size != 0 && max_sequence_length != 0 && head_size != 0);
    assert(max_sequence_length >= sequence_length);
    input_dims = {batch_size, sequence_length, hidden_size};
    pos_dims = {1};
    cache_dims = {max_sequence_length, head_size / 2};
  } else {
    assert(num_heads != 0 && head_size != 0);
    input_dims = {batch_size, num_heads, sequence_length, head_size};
    pos_dims = {batch_size, sequence_length};
    cache_dims = {sequence_length, head_size};
  }

  std::string op_type = "RotaryEmbedding";
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());

  OpTester test(op_type.c_str(), 1, onnxruntime::kMSDomain);
  if (!use_float16) {
    test.AddInput<float>("input", input_dims, input_data);
    test.AddInput<int64_t>("position_ids", pos_dims, position_ids);
    test.AddInput<float>("cos_cache", cache_dims, cos_cache);
    test.AddInput<float>("sin_cache", cache_dims, sin_cache);
    test.AddOutput<float>("output", input_dims, output_data);
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  } else {
    test.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
    test.AddInput<int64_t>("position_ids", pos_dims, position_ids);
    test.AddInput<MLFloat16>("cos_cache", cache_dims, ToFloat16(cos_cache));
    test.AddInput<MLFloat16>("sin_cache", cache_dims, ToFloat16(sin_cache));
    test.AddOutput<MLFloat16>("output", input_dims, ToFloat16(output_data));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

static void RunTests(const std::vector<float>& input_data,
                     const std::vector<int64_t>& position_ids,
                     const std::vector<float>& cos_cache,
                     const std::vector<float>& sin_cache,
                     const std::vector<float>& output_data,
                     int batch_size,
                     int sequence_length,
                     int hidden_size = 0,
                     int head_size = 0,
                     int num_heads = 0,
                     int max_sequence_length = 0,
                     bool use_float16 = true) {

    RunTest(input_data,
            position_ids,
            cos_cache,
            sin_cache,
            output_data,
            batch_size,
            sequence_length,
            hidden_size,
            head_size,
            num_heads,
            max_sequence_length,
            false /* use_fp16 */);

    if (use_float16) {
        RunTest(input_data,
                position_ids,
                cos_cache,
                sin_cache,
                output_data,
                batch_size,
                sequence_length,
                hidden_size,
                head_size,
                num_heads,
                max_sequence_length,
                true /* use_fp16 */);
    }
}

TEST(RotaryEmbeddingTest, RotaryEmbedding_Prompt_SmallData_LlamaMSFT) {
    int batch_size = 1;
    int sequence_length = 3;
    int num_heads = 2;
    int head_size = 4;
    int hidden_size = num_heads * head_size;
    int max_sequence_length = 8;

    std::vector<float> input_data = {
      -1.0408f,  0.9166f, -1.3042f, -1.1097f, -0.1320f, -0.2751f, -0.2350f,  0.0937f,
      -1.2188f,  1.1676f, -1.0574f, -0.1188f, -0.7396f, -1.2425f, -0.1752f,  0.6990f,
      -0.8110f,  0.6737f, -1.1233f, -0.0919f, -0.6861f,  0.7202f,  0.1963f,  0.6142f
    };

    std::vector<int64_t> position_ids = {0};

    // std::vector<float> cos_cache = {
    //     1.0000f,  1.0000f,  0.5403f,  0.9999f,  -0.4161f,  0.9998f
    // };

    // std::vector<float> sin_cache = {
    //     0.0000f,  0.0000f,  0.8415f,  0.0100f,  0.9093f,  0.0200f
    // };

    std::vector<float> cos_cache = {
      1.0000f,  1.0000f,  0.5403f,  0.9999f, -0.4161f,  0.9998f, -0.9900f,  0.9996f,
      -0.6536f,  0.9992f,  0.2837f,  0.9988f,  0.9602f,  0.9982f,  0.7539f,  0.9976f
    };

    std::vector<float> sin_cache = {
      0.0000f,  0.0000f,  0.8415f,  0.0100f,  0.9093f,  0.0200f,  0.1411f,  0.0300f,
      -0.7568f,  0.0400f, -0.9589f,  0.0500f, -0.2794f,  0.0600f,  0.6570f,  0.0699f
    };

    std::vector<float> output_data = {
      // -1.0000f, -1.0000f, -1.0000f, -1.0000f, -1.0000f, -1.0000f, -1.0000f, -1.0000f,
      // -1.0000f, -1.0000f, -1.0000f, -1.0000f, -1.0000f, -1.0000f, -1.0000f, -1.0000f,
      // -1.0000f, -1.0000f, -1.0000f, -1.0000f, -1.0000f, -1.0000f, -1.0000f, -1.0000f
      -1.0408f,  0.9166f, -1.3042f, -1.1097f, -0.1320f, -0.2751f, -0.2350f,  0.0937f,
      -1.6411f, -0.3948f, -1.0561f, -0.1294f,  0.6460f, -1.2937f, -0.1822f,  0.6972f,
      -0.2751f, -1.0178f, -1.1212f, -0.1143f, -0.3694f, -0.9235f,  0.1840f,  0.6180f
    };

    RunTests(input_data,
             position_ids,
             cos_cache,
             sin_cache,
             output_data,
             batch_size,
             sequence_length,
             hidden_size,
             head_size,
             num_heads,
             max_sequence_length);
}

TEST(RotaryEmbeddingTest, RotaryEmbedding_Prompt_LargeData_LlamaMSFT) {
    int batch_size = 2;
    int sequence_length = 8;
    int num_heads = 4;
    int head_size = 6;
    int hidden_size = num_heads * head_size;
    int max_sequence_length = 16;

    std::vector<float> input_data = {
            -1.0408f,      0.9166f,     -1.3042f,     -1.1097f,     -1.2188f,
             1.1676f,     -1.0190f,      0.3157f,     -1.6036f,      1.8493f,
             0.0447f,      1.5853f,      0.1036f,     -0.3514f,      0.2421f,
             0.6463f,      0.8730f,     -0.9276f,      1.0311f,     -1.9557f,
            -0.1482f,      1.7376f,      2.2039f,     -0.6589f,     -1.0574f,
            -0.1188f,     -0.9078f,      0.3452f,     -0.5713f,     -0.2351f,
            -0.5912f,      1.1312f,      0.7562f,     -1.2023f,     -0.5833f,
            -0.4407f,      0.1766f,      1.0224f,     -0.4826f,     -0.5421f,
            -0.5342f,     -0.6413f,      1.3314f,     -0.4498f,      0.5493f,
             0.0539f,      0.2601f,      0.8570f,      1.0076f,     -0.7529f,
            -0.2250f,     -0.4327f,     -1.5071f,     -0.4586f,     -1.9791f,
             0.7787f,     -0.7749f,     -0.1398f,      1.1414f,     -0.6354f,
             0.0352f,     -0.4765f,     -0.0409f,      1.1993f,      0.5374f,
            -0.1930f,      2.5211f,     -0.0452f,     -0.3105f,     -0.9407f,
            -0.0034f,      1.5199f,     -0.8480f,      0.5266f,      0.0299f,
            -0.0498f,      1.0651f,      0.8860f,     -1.4702f,     -0.2134f,
            -0.8707f,      1.6159f,     -0.2356f,      0.9444f,      0.5937f,
             0.7203f,      0.5061f,      1.5192f,     -0.4897f,      0.9231f,
             0.2654f,     -0.1441f,      0.5407f,     -1.5476f,      0.6455f,
            -1.1382f,      0.4640f,     -0.4986f,      0.1289f,      2.7631f,
             0.1405f,      1.1191f,      2.1134f,     -0.9754f,      0.1757f,
            -0.1319f,     -0.2735f,      0.3355f,     -0.6008f,     -1.1164f,
             0.2577f,     -0.7226f,     -0.9244f,      1.8737f,      0.6052f,
             1.1904f,      1.2195f,     -0.0470f,     -1.0914f,      1.0223f,
             0.3152f,      1.7528f,     -0.7650f,      1.8299f,     -0.2784f,
            -0.2719f,      0.1885f,      2.1432f,      0.8527f,      0.0965f,
            -0.0625f,      0.8269f,      1.0122f,     -1.4482f,     -0.0644f,
             0.3215f,      0.5908f,     -1.4197f,      0.2113f,      0.0306f,
             0.3604f,      0.3166f,     -0.8975f,     -0.6393f,     -1.2944f,
            -0.0243f,     -0.2354f,     -0.7087f,      1.1566f,      0.4296f,
             0.5599f,     -0.7776f,      0.3339f,      0.1759f,      2.1108f,
             1.0702f,      0.8279f,     -0.2969f,      0.7120f,     -0.2068f,
            -0.1548f,      0.1553f,      0.6207f,     -0.1690f,     -0.5816f,
             1.2632f,      0.0695f,      1.1862f,     -1.1874f,     -0.7468f,
            -0.9320f,     -0.8579f,     -0.9647f,     -0.0991f,      0.0195f,
             1.1213f,     -1.4873f,     -0.2043f,     -1.0466f,     -1.5772f,
            -0.0489f,      0.3430f,      0.1264f,      0.1519f,     -1.3639f,
            -1.6593f,      1.8127f,     -1.4459f,     -0.2158f,     -0.9792f,
            -1.4392f,      0.6508f,      0.8964f,      0.5717f,     -0.2390f,
             0.6983f,     -1.3416f,      0.2715f,     -0.2852f,      0.6051f,
             0.2167f,     -0.2181f,     -1.6306f,      1.4788f,      0.2754f,
            -0.0261f,     -0.4618f,     -0.5646f,     -1.0389f,      0.5819f,
             1.3697f,      0.0002f,      1.5333f,     -1.0556f,     -0.1254f,
             0.1527f,     -0.5996f,     -1.0962f,      1.6327f,      1.3951f,
             0.8784f,      0.3389f,      1.2907f,      0.3124f,      0.7299f,
             1.4220f,      0.3375f,      0.0438f,      1.8698f,     -0.2635f,
            -2.0799f,     -0.6313f,      0.4090f,     -1.1458f,      0.0784f,
            -1.8848f,     -1.6165f,      0.6179f,      0.9905f,     -0.0729f,
             0.5054f,     -0.6681f,     -1.4382f,      1.7547f,     -0.9605f,
            -0.4558f,     -1.6105f,      0.2979f,      1.1537f,     -1.5604f,
             1.2779f,     -1.2514f,      0.6056f,      0.5763f,     -3.3558f,
             0.2836f,      0.6909f,     -0.7631f,      2.4451f,     -0.3500f,
             1.3289f,     -0.6494f,      0.3478f,      1.0038f,     -0.2937f,
             0.9238f,     -1.2185f,      0.4138f,      0.5033f,      0.9174f,
             1.8131f,      1.4436f,     -0.4207f,      0.0220f,     -0.6807f,
            -1.3306f,      1.5646f,      0.3338f,      0.7105f,      0.4683f,
            -0.6179f,      0.0818f,     -0.0488f,     -0.9810f,     -1.3632f,
             0.0929f,     -1.7926f,     -0.2921f,     -0.4792f,      0.6756f,
            -0.3413f,     -0.2242f,     -0.2111f,      0.6282f,      0.1667f,
            -1.4055f,      1.5895f,      1.0838f,     -0.9077f,     -0.8060f,
             0.7967f,     -2.9351f,      2.4179f,     -0.4026f,      0.6451f,
             1.6845f,     -0.0901f,      0.6106f,      2.3603f,      1.3908f,
            -0.7917f,     -0.6734f,     -0.1213f,     -1.1116f,     -0.7401f,
            -0.7879f,      0.0606f,     -2.3337f,     -1.2603f,     -1.7245f,
            -0.3533f,     -0.9421f,     -0.1776f,      0.3992f,     -1.7142f,
            -0.5319f,     -0.8848f,      0.6513f,      1.0002f,     -1.4699f,
            -1.4254f,      0.7013f,      0.2414f,      0.2551f,     -0.7457f,
             0.3133f,     -1.0941f,     -0.3682f,     -0.0163f,     -0.0645f,
            -0.8101f,      0.1415f,      0.0551f,      0.5873f,     -0.5887f,
            -1.4733f,     -0.8565f,      0.7400f,     -0.5033f,      0.0553f,
             0.9265f,     -0.8652f,     -0.0288f,     -0.2209f,      0.0610f,
             0.6776f,      0.4361f,     -0.8052f,      0.3955f,      0.8988f,
             0.8238f,      0.2262f,      1.2912f,      0.6488f,      1.2114f,
             1.3569f,      0.2983f,      0.4718f,     -1.1936f,      0.7928f,
            -0.8665f,      0.9468f,      1.1629f,      0.0616f,     -1.3136f,
            -0.2764f,      0.0277f,     -0.1126f,      0.2342f,     -0.5866f,
            -1.8219f,      1.1079f,      0.5795f,     -1.4249f
    };

    std::vector<int64_t> position_ids = {0};

    std::vector<float> cos_cache = {
      1.0000f, 1.0000f, 1.0000f, 0.5403f, 0.9989f, 1.0000f, -0.4161f, 0.9957f,
      1.0000f, -0.9900f, 0.9903f, 1.0000f, -0.6536f, 0.9828f, 1.0000f, 0.2837f,
      0.9732f, 0.9999f, 0.9602f, 0.9615f, 0.9999f, 0.7539f, 0.9477f, 0.9999f,
      -0.1455f, 0.9318f, 0.9999f, -0.9111f, 0.9140f, 0.9998f, -0.8391f, 0.8942f,
      0.9998f, 0.0044f, 0.8725f, 0.9997f, 0.8439f, 0.8488f, 0.9997f, 0.9074f,
      0.8234f, 0.9996f, 0.1367f, 0.7962f, 0.9995f, -0.7597f, 0.7673f, 0.9995f
    };

    std::vector<float> sin_cache = {
      0.0000f, 0.0000f, 0.0000f, 0.8415f, 0.0464f, 0.0022f, 0.9093f, 0.0927f,
      0.0043f, 0.1411f, 0.1388f, 0.0065f, -0.7568f, 0.1846f, 0.0086f, -0.9589f,
      0.2300f, 0.0108f, -0.2794f, 0.2749f, 0.0129f, 0.6570f, 0.3192f, 0.0151f,
      0.9894f, 0.3629f, 0.0172f, 0.4121f, 0.4057f, 0.0194f, -0.5440f, 0.4477f,
      0.0215f, -1.0000f, 0.4887f, 0.0237f, -0.5366f, 0.5286f, 0.0259f, 0.4202f,
      0.5675f, 0.0280f, 0.9906f, 0.6050f, 0.0302f, 0.6503f, 0.6413f, 0.0323f
    };

    std::vector<float> output_data = {
            -1.0408f,      0.9166f,     -1.3042f,     -1.1097f,     -1.2188f,
             1.1676f,     -1.0190f,      0.3157f,     -1.6036f,      1.8493f,
             0.0447f,      1.5853f,      0.1036f,     -0.3514f,      0.2421f,
             0.6463f,      0.8730f,     -0.9276f,      1.0311f,     -1.9557f,
            -0.1482f,      1.7376f,      2.2039f,     -0.6589f,     -0.4713f,
            -0.9540f,     -0.9229f,      0.3027f,     -0.5708f,     -0.2363f,
            -1.2713f,      0.1137f,      0.8112f,     -1.1659f,     -0.5824f,
            -0.4419f,     -0.7649f,      0.7011f,     -0.4569f,     -0.5639f,
            -0.5328f,     -0.6424f,      1.0979f,      0.8773f,      0.5462f,
             0.0793f,      0.2582f,      0.8576f,      0.2653f,      1.2295f,
            -0.1839f,     -0.4517f,     -1.5052f,     -0.4651f,      0.1155f,
            -2.1237f,     -0.7586f,     -0.2110f,      1.1441f,     -0.6304f,
             0.4186f,      0.2303f,     -0.1519f,      1.1903f,      0.5382f,
            -0.1906f,     -1.0080f,      2.3112f,     -0.2220f,     -0.9655f,
            -0.0099f,      1.5198f,      0.7652f,     -0.6410f,      0.0365f,
            -0.0452f,      1.0593f,      0.8929f,      1.4856f,      0.0038f,
            -1.0865f,      1.4794f,     -0.2417f,      0.9428f,     -0.6894f,
            -0.6293f,      0.2904f,      1.5747f,     -0.4956f,      0.9199f,
            -0.2424f,      0.1801f,      0.7503f,     -1.4576f,      0.6529f,
            -1.1340f,     -0.6807f,     -0.0252f,     -0.3834f,      2.7394f,
             0.1308f,      1.1203f,     -2.1196f,     -0.9618f,      0.1970f,
            -0.0972f,     -0.2764f,      0.3332f,     -0.4522f,      1.1844f,
             0.3867f,     -0.6626f,     -0.9405f,      1.8656f,      0.5053f,
            -1.2361f,      1.2072f,      0.1789f,     -1.1002f,      1.0129f,
             1.7702f,      0.1949f,     -1.1653f,      1.6049f,     -0.2755f,
            -0.2749f,      2.1087f,      0.4272f,      0.8076f,      0.2900f,
            -0.0714f,      0.8261f,     -1.1016f,     -1.3814f,     -0.1366f,
             0.2981f,      0.6060f,     -1.4132f,      0.0893f,     -0.1939f,
             0.2779f,      0.3910f,     -0.8906f,     -0.6489f,     -1.2496f,
             0.3383f,     -0.0315f,     -0.7461f,      1.1510f,      0.4445f,
             0.3203f,     -0.9031f,      0.2727f,      0.2609f,      2.0968f,
             1.0974f,      0.7120f,     -0.5164f,      0.7415f,     -0.0031f,
            -0.1568f,      0.1533f,      0.5487f,     -0.3357f,     -0.9064f,
             1.0546f,      0.0542f,      1.1870f,     -0.4045f,     -1.3431f,
            -0.6094f,     -1.1105f,     -0.9631f,     -0.1137f,     -0.7219f,
             0.8582f,     -1.3443f,     -0.6684f,     -1.0227f,     -1.5929f,
            -0.2622f,      0.2264f,      0.0713f,      0.1843f,     -1.3387f,
            -1.6797f,      2.3165f,      0.1009f,      0.1081f,     -0.9969f,
            -1.4488f,      0.6291f,      0.8964f,      0.5717f,     -0.2390f,
             0.6983f,     -1.3416f,      0.2715f,     -0.2852f,      0.6051f,
             0.2167f,     -0.2181f,     -1.6306f,      1.4788f,      0.2754f,
            -0.0261f,     -0.4618f,     -0.5646f,     -1.0389f,      0.5819f,
             1.3697f,      0.0002f,      1.5333f,     -1.0556f,     -0.1254f,
             0.1527f,      0.5985f,     -1.0968f,      1.5662f,      1.4693f,
             0.8776f,      0.3408f,      0.4345f,      1.2549f,      0.6631f,
             1.4543f,      0.3374f,      0.0445f,      1.2320f,      1.4311f,
            -2.0483f,     -0.7272f,      0.4114f,     -1.1449f,      1.6283f,
            -0.9524f,     -1.6435f,      0.5422f,      0.9907f,     -0.0708f,
             0.3972f,      0.7376f,     -1.5947f,      1.6138f,     -0.9586f,
            -0.4600f,      0.3993f,     -1.5884f,      1.2934f,     -1.4467f,
             1.2833f,     -1.2459f,     -0.7760f,      0.3108f,     -3.3677f,
            -0.0287f,      0.6942f,     -0.7601f,     -0.6993f,      2.3690f,
             1.3834f,     -0.5234f,      0.3435f,      1.0053f,      0.1604f,
            -0.9560f,     -1.2641f,      0.2406f,      0.4973f,      0.9206f,
            -1.9987f,     -1.1733f,     -0.4197f,     -0.0366f,     -0.6720f,
            -1.3350f,     -1.5960f,     -0.1097f,      0.6386f,      0.5624f,
            -0.6184f,      0.0778f,      0.1867f,      0.9643f,     -1.3629f,
            -0.0972f,     -1.7907f,     -0.3037f,      0.8245f,     -0.0789f,
            -0.2940f,     -0.2833f,     -0.2165f,      0.6264f,     -1.1726f,
             0.7926f,      1.3621f,      1.3586f,     -0.9007f,     -0.8138f,
            -2.7421f,      1.3155f,      2.4507f,      0.0507f,      0.6305f,
             1.6900f,      0.5210f,     -0.3309f,      2.0630f,      1.8026f,
            -0.7859f,     -0.6802f,     -1.1003f,     -0.1990f,     -0.5391f,
            -0.9370f,      0.0857f,     -2.3330f,     -2.0112f,      0.7193f,
            -0.1272f,     -0.9981f,     -0.1818f,      0.3973f,     -0.9963f,
             1.4929f,     -1.0109f,      0.4304f,      1.0160f,     -1.4590f,
             0.2682f,      1.5658f,      0.1762f,      0.3038f,     -0.7491f,
             0.3052f,     -1.1534f,     -0.0478f,      0.0021f,     -0.0665f,
            -0.8118f,      0.1310f,      0.2171f,      0.5485f,     -0.1610f,
            -1.5784f,     -0.8660f,      0.7289f,     -0.4678f,      0.1937f,
             1.1287f,     -0.5772f,     -0.0259f,     -0.2212f,      0.2479f,
             0.6336f,      0.6407f,     -0.6543f,      0.3838f,      0.9039f,
             0.4724f,      0.7117f,      1.0165f,      1.0270f,      1.1908f,
             1.3750f,     -0.0850f,      0.5517f,     -1.3842f,      0.3703f,
            -0.8806f,      0.9336f,      0.8362f,      0.8105f,     -1.1566f,
            -0.6813f,      0.0294f,     -0.1122f,      0.5620f,     -0.2884f,
            -2.0803f,      0.4684f,      0.6009f,     -1.4160f
    };

    RunTests(input_data,
             position_ids,
             cos_cache,
             sin_cache,
             output_data,
             batch_size,
             sequence_length,
             hidden_size,
             head_size,
             num_heads,
             max_sequence_length);
}

// TEST(RotaryEmbeddingTest, RotaryEmbedding_PerToken_LlamaMSFT) {
//     int batch_size = 1;
//     int sequence_length = 1;
//     int num_heads = 2;
//     int head_size = 4;
//     int hidden_size = num_heads * head_size;
//     int max_sequence_length = 8;

//     std::vector<float> input_data = {

//     };

//     std::vector<int64_t> position_ids = {?}; // new_pos_id = old_pos_id + prompt_seq_len
//                                              // Using values from RotaryEmbedding_Prompt_WithPosIdsInt test: 0 + 4 = 4

//     std::vector<float> cos_cache = {

//     };

//     std::vector<float> sin_cache = {

//     };

//     std::vector<float> output_data = {

//     };

//     RunTests(input_data,
//              position_ids,
//              cos_cache,
//              sin_cache,
//              output_data,
//              epsilon_,
//              batch_size,
//              sequence_length,
//              hidden_size,
//              head_size,
//              num_heads,
//              max_sequence_length);
// }

}  // namespace test
}  // namespace onnxruntime
