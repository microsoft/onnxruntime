// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm_test_vectors.h"
#include "gtest/gtest.h"
#include "test/common/quantization_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

static void RunTest(const embedlayernorm::OpData& data,
                       float accuracy_threshold = 0.25f) {
  ASSERT_TRUE(data.word_embedding_data.size() % data.hidden_size == 0);
  ASSERT_TRUE(data.position_embedding_data.size() % data.hidden_size == 0);
  ASSERT_TRUE(data.segment_embedding_data.size() % data.hidden_size == 0);

  std::vector<int64_t> input_ids_dims = {data.batch_size, data.sequence_size};
  std::vector<int64_t> segment_ids_dims = {data.batch_size, data.sequence_size};
    std::vector<int64_t> word_embedding_dims = {
        static_cast<int64_t>(data.word_embedding_data.size() / data.hidden_size),
        data.hidden_size};
    std::vector<int64_t> position_embedding_dims = {
        static_cast<int64_t>(data.position_embedding_data.size() / data.hidden_size),
        data.hidden_size};
    std::vector<int64_t> segment_embedding_dims = {
        static_cast<int64_t>(data.segment_embedding_data.size() / data.hidden_size),
        data.hidden_size};
  std::vector<int64_t> gamma_dims = {data.hidden_size};
  std::vector<int64_t> beta_dims = {data.hidden_size};
  std::vector<int64_t> output_dims = {data.batch_size, data.sequence_size, data.hidden_size};
  std::vector<int64_t> mask_index_dims = {data.batch_size};

  float word_embedding_scale = 0.0f;
  uint8_t word_embedding_zero_point = 0;
  std::vector<uint8_t> word_embedding_data_quant =
      QuantizeLinear<uint8_t, /*symmetric=*/false>(
          data.word_embedding_data, word_embedding_scale, word_embedding_zero_point);

  float position_embedding_scale = 0.0f;
  uint8_t position_embedding_zero_point = 0;
  std::vector<uint8_t> position_embedding_data_quant =
      QuantizeLinear<uint8_t, /*symmetric=*/false>(
          data.position_embedding_data, position_embedding_scale, position_embedding_zero_point);

  float segment_embedding_scale = 0.0f;
  uint8_t segment_embedding_zero_point = 0;
  std::vector<uint8_t> segment_embedding_data_quant;
  if (data.has_segment) {
    segment_embedding_data_quant =
        QuantizeLinear<uint8_t, /*symmetric=*/false>(
            data.segment_embedding_data, segment_embedding_scale, segment_embedding_zero_point);
  }

  float gamma_scale = 0.0f;
  uint8_t gamma_zero_point = 0;
  std::vector<uint8_t> gamma_data_quant =
      QuantizeLinear<uint8_t, /*symmetric=*/false>(
          data.gamma_data, gamma_scale, gamma_zero_point);

  float beta_scale = 0.0f;
  uint8_t beta_zero_point = 0;
  std::vector<uint8_t> beta_data_quant =
      QuantizeLinear<uint8_t, /*symmetric=*/false>(
          data.beta_data, beta_scale, beta_zero_point);

  OpTester tester("QEmbedLayerNormalization", 1, onnxruntime::kMSDomain);

  // Operator inputs passed in at int32_t:
  tester.AddInput<int32_t>("input_ids", input_ids_dims, data.input_ids_data);
  if (data.has_segment) {
    tester.AddInput<int32_t>("segment_ids", segment_ids_dims, data.segment_ids_data);
  } else {
    tester.AddMissingOptionalInput<int32_t>();
  }

  // Quantized initializer inputs:
  tester.AddInput<uint8_t>("word_embedding_data",
                           word_embedding_dims,
                           word_embedding_data_quant);
  tester.AddInput<uint8_t>("position_embedding_data",
                           position_embedding_dims,
                           position_embedding_data_quant);
  if (data.has_segment) {
    tester.AddInput<uint8_t>("segment_embedding_data",
                             segment_embedding_dims,
                             segment_embedding_data_quant);
  } else {
    tester.AddMissingOptionalInput<uint8_t>();
  }
  tester.AddInput<uint8_t>("gamma",
                           gamma_dims,
                           gamma_data_quant);
  tester.AddInput<uint8_t>("beta",
                           beta_dims,
                           beta_data_quant);
  if (data.has_mask) {
    std::vector<int64_t> mask_dims = {data.batch_size, data.sequence_size};
    tester.AddInput<int32_t>("mask", mask_dims, data.mask_data);
  } else {
    tester.AddMissingOptionalInput<int32_t>();
  }

  // Quantized scales:
  tester.AddInput<float>("word_embedding_scale",
                         /*dims=*/{},
                         {word_embedding_scale});
  tester.AddInput<float>("position_embedding_scale",
                         /*dims=*/{},
                         {position_embedding_scale});
  if (data.has_segment) {
    tester.AddInput<float>("segment_embedding_scale",
                           /*dims=*/{},
                           {segment_embedding_scale});
  } else {
    tester.AddMissingOptionalInput<float>();
  }
  tester.AddInput<float>("gamma_scale",
                         /*dims=*/{},
                         {gamma_scale});
  tester.AddInput<float>("beta_scale",
                         /*dims=*/{},
                         {beta_scale});

  // Quantized zero points:
  tester.AddInput<uint8_t>("word_embedding_zero_point",
                           /*dims=*/{},
                           {word_embedding_zero_point});
  tester.AddInput<uint8_t>("position_embedding_zero_point",
                           /*dims=*/{},
                           {position_embedding_zero_point});
  if (data.has_segment) {
    tester.AddInput<uint8_t>("segment_embedding_zero_point",
                             /*dims=*/{},
                             {segment_embedding_zero_point});
  } else {
    tester.AddMissingOptionalInput<uint8_t>();
  }
  tester.AddInput<uint8_t>("gamma_zero_point",
                           /*dims=*/{},
                           {gamma_zero_point});
  tester.AddInput<uint8_t>("beta_zero_point",
                           /*dims=*/{},
                           {beta_zero_point});
  // Outputs:
  tester.AddOutput<float>("output", output_dims, data.output_data);
  tester.AddOutput<int32_t>("mask_index", mask_index_dims, data.mask_index_data);

  // Floating point test vectors are quantized, passed through the operator,
  // and dequantized. This dance will result in some loss in precision, ensure
  // the test framework accounts for this loss:
  tester.SetOutputAbsErr("output", accuracy_threshold);

  // Attributes:
  tester.AddAttribute("epsilon", embedlayernorm::kEpsilon);

  tester.Run();
}

}  // namespace

TEST(QEmbedLayerNormTest, EmbedLayerNormBatch1) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1());
}

TEST(QEmbedLayerNormTest, EmbedLayerNormBatch1_Float16) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1(), /*use_float16=*/true);
}

TEST(QEmbedLayerNormTest, EmbedLayerNormBatch2) {
  RunTest(embedlayernorm::EmbedLayerNormBatch2());
}

TEST(QEmbedLayerNormTest, EmbedLayerNormBatch2_NoMask) {
  RunTest(embedlayernorm::EmbedLayerNormBatch2(/*has_mask=*/false));
}

// BatchSize > HiddenSize to reproduce mask processing bug
TEST(QEmbedLayerNormTest, EmbedLayerNormLargeBatchSmallHiddenSize) {
  RunTest(embedlayernorm::EmbedLayerNormLargeBatchSmallHiddenSize());
}

TEST(QEmbedLayerNormTest, EmbedLayerNormBatch_Distill) {
  RunTest(embedlayernorm::EmbedLayerNormBatch_Distill());
}

}  // namespace test
}  // namespace onnxruntime
