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

template <typename WeightT>
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

  quantization::Params<WeightT> word_embedding_params;
  std::vector<WeightT> word_embedding_data_quant =
      QuantizeLinearTestVector<WeightT>(data.word_embedding_data,
                                        word_embedding_params);

  quantization::Params<WeightT> position_embedding_params;
  std::vector<WeightT> position_embedding_data_quant =
      QuantizeLinearTestVector<WeightT>(data.position_embedding_data,
                                        position_embedding_params);

  quantization::Params<WeightT> segment_embedding_params = {};
  std::vector<WeightT> segment_embedding_data_quant;
  if (data.has_segment) {
    segment_embedding_data_quant = QuantizeLinearTestVector<WeightT>(
        data.segment_embedding_data, segment_embedding_params);
  }

  quantization::Params<WeightT> gamma_params;
  std::vector<WeightT> gamma_data_quant =
      QuantizeLinearTestVector<WeightT>(data.gamma_data, gamma_params);

  quantization::Params<WeightT> beta_params;
  std::vector<WeightT> beta_data_quant =
      QuantizeLinearTestVector<WeightT>(data.beta_data, beta_params);

  OpTester tester("QEmbedLayerNormalization", 1, onnxruntime::kMSDomain);

  // Operator inputs passed in at int32_t:
  tester.AddInput<int32_t>("input_ids", input_ids_dims, data.input_ids_data);
  if (data.has_segment) {
    tester.AddInput<int32_t>("segment_ids", segment_ids_dims, data.segment_ids_data);
  } else {
    tester.AddOptionalInputEdge<int32_t>();
  }

  // Quantized initializer inputs:
  tester.AddInput<WeightT>("word_embedding_data",
                           word_embedding_dims,
                           word_embedding_data_quant,
                           /*is_initializer=*/true);
  tester.AddInput<WeightT>("position_embedding_data",
                           position_embedding_dims,
                           position_embedding_data_quant,
                           /*is_initializer=*/true);
  if (data.has_segment) {
    tester.AddInput<WeightT>("segment_embedding_data",
                             segment_embedding_dims,
                             segment_embedding_data_quant,
                             /*is_initializer=*/true);
  } else {
    tester.AddOptionalInputEdge<WeightT>();
  }
  tester.AddInput<WeightT>("gamma",
                           gamma_dims,
                           gamma_data_quant,
                           /*is_initializer=*/true);
  tester.AddInput<WeightT>("beta",
                           beta_dims,
                           beta_data_quant,
                           /*is_initializer=*/true);
  if (data.has_mask) {
    std::vector<int64_t> mask_dims = {data.batch_size, data.sequence_size};
    tester.AddInput<int32_t>("mask", mask_dims, data.mask_data);
  } else {
    tester.AddOptionalInputEdge<int32_t>();
  }

  // Quantized scales:
  tester.AddInput<float>("word_embedding_scale",
                         /*dims=*/{},
                         {word_embedding_params.scale},
                         /*is_initializer=*/true);
  tester.AddInput<float>("position_embedding_scale",
                         /*dims=*/{},
                         {position_embedding_params.scale},
                         /*is_initializer=*/true);
  if (data.has_segment) {
    tester.AddInput<float>("segment_embedding_scale",
                           /*dims=*/{},
                           {segment_embedding_params.scale},
                           /*is_initializer=*/true);
  } else {
    tester.AddOptionalInputEdge<float>();
  }
  tester.AddInput<float>("gamma_scale",
                         /*dims=*/{},
                         {gamma_params.scale},
                         /*is_initializer=*/true);
  tester.AddInput<float>("beta_scale",
                         /*dims=*/{},
                         {beta_params.scale},
                         /*is_initializer=*/true);

  // Quantized zero points:
  tester.AddInput<WeightT>("word_embedding_zero_point",
                           /*dims=*/{},
                           {word_embedding_params.zero_point},
                           /*is_initializer=*/true);
  tester.AddInput<WeightT>("position_embedding_zero_point",
                           /*dims=*/{},
                           {position_embedding_params.zero_point},
                           /*is_initializer=*/true);
  if (data.has_segment) {
    tester.AddInput<WeightT>("segment_embedding_zero_point",
                             /*dims=*/{},
                             {segment_embedding_params.zero_point},
                             /*is_initializer=*/true);
  } else {
    tester.AddOptionalInputEdge<WeightT>();
  }
  tester.AddInput<WeightT>("gamma_zero_point",
                           /*dims=*/{},
                           {gamma_params.zero_point},
                           /*is_initializer=*/true);
  tester.AddInput<WeightT>("beta_zero_point",
                           /*dims=*/{},
                           {beta_params.zero_point},
                           /*is_initializer=*/true);
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
  RunTest<int8_t>(embedlayernorm::EmbedLayerNormBatch1());
  RunTest<uint8_t>(embedlayernorm::EmbedLayerNormBatch1());
}

TEST(QEmbedLayerNormTest, EmbedLayerNormBatch1_Float16) {
  RunTest<int8_t>(embedlayernorm::EmbedLayerNormBatch1(), /*use_float16=*/true);
  RunTest<uint8_t>(embedlayernorm::EmbedLayerNormBatch1(), /*use_float16=*/true);
}

TEST(QEmbedLayerNormTest, EmbedLayerNormBatch2) {
  RunTest<int8_t>(embedlayernorm::EmbedLayerNormBatch2());
  RunTest<uint8_t>(embedlayernorm::EmbedLayerNormBatch2());
}

TEST(QEmbedLayerNormTest, EmbedLayerNormBatch2_NoMask) {
  RunTest<int8_t>(embedlayernorm::EmbedLayerNormBatch2(/*has_mask=*/false));
  RunTest<uint8_t>(embedlayernorm::EmbedLayerNormBatch2(/*has_mask=*/false));
}

// BatchSize > HiddenSize to reproduce mask processing bug
TEST(QEmbedLayerNormTest, EmbedLayerNormLargeBatchSmallHiddenSize) {
  RunTest<int8_t>(embedlayernorm::EmbedLayerNormLargeBatchSmallHiddenSize());
  RunTest<uint8_t>(embedlayernorm::EmbedLayerNormLargeBatchSmallHiddenSize());
}

TEST(QEmbedLayerNormTest, EmbedLayerNormBatch_Distill) {
  RunTest<int8_t>(embedlayernorm::EmbedLayerNormBatch_Distill());
  RunTest<uint8_t>(embedlayernorm::EmbedLayerNormBatch_Distill());
}

}  // namespace test
}  // namespace onnxruntime
