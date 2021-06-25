// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm_test_vectors.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunTest(const embedlayernorm::OpData& data,
                       bool use_float16 = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;

  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = !use_float16;

  if (enable_cpu || enable_cuda) {
    // Input and output shapes
    //   Input 0 - input_ids          : (batch_size, sequence_size)
    //   Input 1 - segment_ids        : (batch_size, sequence_size)
    //   Input 2 - word_embedding     : (,hidden_size)
    //   Input 3 - position_embedding : (,hidden_size)
    //   Input 4 - segment_embedding  : (,hidden_size)
    //   Input 5 - gamma              : (hidden_size)
    //   Input 6 - beta               : (hidden_size)
    //   Input 7 - mask               : (batch_size, sequence_size)
    //   Output 0 - output            : (batch_size, sequence_size, hidden_size)
    //   Output 1 - mask_index        : (batch_size)

    std::vector<int64_t> input_ids_dims = {data.batch_size, data.sequence_size};
    std::vector<int64_t> segment_ids_dims = {data.batch_size, data.sequence_size};
    std::vector<int64_t> mask_dims = {data.batch_size, data.sequence_size};

    ASSERT_TRUE(data.word_embedding_data.size() % data.hidden_size == 0);
    std::vector<int64_t> word_embedding_dims = {
        static_cast<int64_t>(data.word_embedding_data.size() / data.hidden_size),
        data.hidden_size};

    ASSERT_TRUE(data.position_embedding_data.size() % data.hidden_size == 0);
    std::vector<int64_t> position_embedding_dims = {
        static_cast<int64_t>(data.position_embedding_data.size() / data.hidden_size),
        data.hidden_size};

    ASSERT_TRUE(data.segment_embedding_data.size() % data.hidden_size == 0);
    std::vector<int64_t> segment_embedding_dims = {
        static_cast<int64_t>(data.segment_embedding_data.size() / data.hidden_size),
        data.hidden_size};

    std::vector<int64_t> gamma_dims = {data.hidden_size};
    std::vector<int64_t> beta_dims = gamma_dims;
    std::vector<int64_t> output_dims = {data.batch_size, data.sequence_size, data.hidden_size};
    std::vector<int64_t> mask_index_dims = {data.batch_size};

    OpTester tester("EmbedLayerNormalization", 1, onnxruntime::kMSDomain);
    tester.AddInput<int32_t>("input_ids", input_ids_dims, data.input_ids_data);
    if (!data.has_segment) {
      tester.AddMissingOptionalInput<int32_t>();
    } else {
      tester.AddInput<int32_t>("segment_ids", segment_ids_dims, data.segment_ids_data);
    }
    if (use_float16) {
      tester.AddInput<MLFloat16>("word_embedding",
                                 word_embedding_dims,
                                 ToFloat16(data.word_embedding_data));
      tester.AddInput<MLFloat16>("position_embedding",
                                 position_embedding_dims,
                                 ToFloat16(data.position_embedding_data));
      if (!data.has_segment) {
        tester.AddMissingOptionalInput<MLFloat16>();
      } else {
        tester.AddInput<MLFloat16>("segment_embedding",
                                   segment_embedding_dims,
                                   ToFloat16(data.segment_embedding_data));
      }
      tester.AddInput<MLFloat16>("gamma", gamma_dims, ToFloat16(data.gamma_data));
      tester.AddInput<MLFloat16>("beta", beta_dims, ToFloat16(data.beta_data));
      tester.AddAttribute("epsilon", data.epsilon);
      if (data.has_mask) {
        tester.AddInput<int32_t>("mask", mask_dims, data.mask_data);
      }
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(data.output_data));
    } else {
      tester.AddInput<float>("word_embedding",
                             word_embedding_dims,
                             data.word_embedding_data);
      tester.AddInput<float>("position_embedding",
                             position_embedding_dims,
                             data.position_embedding_data);
      if (!data.has_segment) {
        tester.AddMissingOptionalInput<MLFloat16>();
      } else {
        tester.AddInput<float>("segment_embedding",
                               segment_embedding_dims,
                               data.segment_embedding_data);
      }
      tester.AddInput<float>("gamma", gamma_dims, data.gamma_data);
      tester.AddInput<float>("beta", beta_dims, data.beta_data);
      tester.AddAttribute("epsilon", data.epsilon);
      if (data.has_mask) {
        tester.AddInput<int32_t>("mask", mask_dims, data.mask_data);
      }
      tester.AddOutput<float>("output", output_dims, data.output_data);
    }
    tester.AddOutput<int32_t>("mask_index", mask_index_dims, data.mask_index_data);
    tester.Run();
  }
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1());
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_Float16) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1(), /*use_float16=*/true);
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch2) {
  RunTest(embedlayernorm::EmbedLayerNormBatch2());
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch2_NoMask) {
  RunTest(embedlayernorm::EmbedLayerNormBatch2(/*has_mask=*/false));
}

// BatchSize > HiddenSize to reproduce mask processing bug
TEST(EmbedLayerNormTest, EmbedLayerNormLargeBatchSmallHiddenSize) {
  RunTest(embedlayernorm::EmbedLayerNormLargeBatchSmallHiddenSize());
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch_Distill) {
  RunTest(embedlayernorm::EmbedLayerNormBatch_Distill());
}

}  // namespace test
}  // namespace onnxruntime
