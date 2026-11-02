// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm_test_vectors.h"
#include "gtest/gtest.h"
#include <cmath>
#include <type_traits>
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(USE_CUDA)
template <typename T>
static void CheckCudaHighOffsetVariance() {
  if (!HasCudaEnvironment(0)) GTEST_SKIP() << "CUDA EP unavailable";

  constexpr float epsilon = 1e-5f;
  const auto native_value = [](float value) {
    if constexpr (std::is_same_v<T, float>) {
      return value;
    } else {
      return T(value).ToFloat();
    }
  };
  for (int hidden_size : {1, 3, 4, 5, 128, 129, 320, 768, 769, 8192, 8193}) {
    SCOPED_TRACE(hidden_size);
    for (bool has_segment : {false, true}) {
      SCOPED_TRACE(has_segment);
      const std::vector<int64_t> shape{1, 2, hidden_size};
      const std::vector<int64_t> table_shape{2, hidden_size};
      const std::vector<int64_t> parameter_shape{hidden_size};
      std::vector<float> pattern{1000.01f, 1000.02f, 999.99f, 1000.0f};
      float position_value = 128.0f;
      if constexpr (std::is_same_v<T, MLFloat16>) {
        pattern = {60000.0f, 60032.0f, 59968.0f, 60000.0f};
        position_value = -59872.0f;
      }
      std::vector<float> word(2 * hidden_size);
      std::vector<float> position(2 * hidden_size, has_segment ? position_value : 0.0f);
      std::vector<float> segment(hidden_size, 512.0f);
      std::vector<float> gamma(hidden_size);
      std::vector<float> beta(hidden_size);
      std::vector<float> sum(2 * hidden_size);
      std::vector<float> expected(2 * hidden_size);
      for (int i = 0; i < hidden_size; ++i) {
        word[i] = pattern[i % 4];
        word[hidden_size + i] = std::is_same_v<T, float> ? -1000.0f : pattern[0];
        if constexpr (std::is_same_v<T, MLFloat16>) {
          segment[i] = i % 2 == 0 ? 0.25f : 0.5f;
        }
        gamma[i] = i % 2 == 0 ? 0.5f : -1.25f;
        beta[i] = i % 2 == 0 ? 0.25f : -0.75f;
      }
      for (int row = 0; row < 2; ++row) {
        double mean = 0.0;
        for (int i = 0; i < hidden_size; ++i) {
          const int index = row * hidden_size + i;
          const float value = native_value(native_value(word[index] + (has_segment ? segment[i] : 0.0f)) +
                                           position[index]);
          sum[index] = value;
          mean += static_cast<double>(value);
        }
        mean /= hidden_size;
        double variance = 0.0;
        for (int i = 0; i < hidden_size; ++i) {
          const double delta = static_cast<double>(sum[row * hidden_size + i]) - mean;
          variance += delta * delta;
        }
        variance /= hidden_size;
        const double inv_std = 1.0 / std::sqrt(variance + static_cast<double>(epsilon));
        for (int i = 0; i < hidden_size; ++i) {
          const int index = row * hidden_size + i;
          expected[index] = static_cast<float>((static_cast<double>(sum[index]) - mean) *
                                                   inv_std * gamma[i] +
                                               beta[i]);
        }
      }

      OpTester tester("EmbedLayerNormalization", 1, kMSDomain);
      tester.AddAttribute("epsilon", epsilon);
      tester.AddInput<int32_t>("input_ids", {1, 2}, {0, 1});
      if (has_segment) {
        tester.AddInput<int32_t>("segment_ids", {1, 2}, {0, 0});
      } else {
        tester.AddOptionalInputEdge<int32_t>();
      }
      tester.AddInput<T>("word_embedding", table_shape, std::vector<T>(word.begin(), word.end()));
      tester.AddInput<T>("position_embedding", table_shape, std::vector<T>(position.begin(), position.end()));
      if (has_segment) {
        tester.AddInput<T>("segment_embedding", {1, hidden_size}, std::vector<T>(segment.begin(), segment.end()));
      } else {
        tester.AddOptionalInputEdge<T>();
      }
      tester.AddInput<T>("gamma", parameter_shape, std::vector<T>(gamma.begin(), gamma.end()));
      tester.AddInput<T>("beta", parameter_shape, std::vector<T>(beta.begin(), beta.end()));
      tester.AddInput<int32_t>("mask", {1, 2}, {1, 0});
      tester.AddOutput<T>("output", shape, std::vector<T>(expected.begin(), expected.end()));
      tester.AddOutput<int32_t>("mask_index", {1}, {1});
      tester.AddOutput<T>("embedding_sum", shape, std::vector<T>(sum.begin(), sum.end()));
      if constexpr (std::is_same_v<T, float>) {
        tester.SetOutputAbsErr("output", 1e-4f);
        tester.SetOutputRelErr("output", 1e-4f);
      }
      tester.SetOutputAbsErr("embedding_sum", 0.0f);
      tester.SetOutputRelErr("embedding_sum", 0.0f);
      SessionOptions options;
      ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1"));
      std::vector<std::unique_ptr<IExecutionProvider>> providers;
      providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
    }
  }
}

TEST(EmbedLayerNormTest, CudaHighOffsetVariance) {
  CheckCudaHighOffsetVariance<float>();
}

TEST(EmbedLayerNormTest, CudaFloat16HighOffsetVariance) {
  CheckCudaHighOffsetVariance<MLFloat16>();
}
#endif

static void RunTest(const embedlayernorm::OpData& data,
                    bool use_float16 = false,
                    bool sum_output = false,
                    bool broadcast_position_ids = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;

  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_dml = DefaultDmlExecutionProvider().get() != nullptr;
  bool enable_cpu = !use_float16;

  if (enable_cpu || enable_cuda || enable_dml) {
    // Input and output shapes
    //   Input 0 - input_ids          : (batch_size, sequence_size)
    //   Input 1 - segment_ids        : (batch_size, sequence_size)
    //   Input 2 - word_embedding     : (,hidden_size)
    //   Input 3 - position_embedding : (,hidden_size)
    //   Input 4 - segment_embedding  : (,hidden_size)
    //   Input 5 - gamma              : (hidden_size)
    //   Input 6 - beta               : (hidden_size)
    //   Input 7 - mask               : (batch_size, sequence_size)
    //   Input 8 - position ids       : (batch_size, sequence_size) or (1, sequence_size)
    //   Output 0 - output            : (batch_size, sequence_size, hidden_size)
    //   Output 1 - mask_index        : (batch_size)
    //   Output 2 - embedding_sum     : (batch_size, sequence_size, hidden_size)

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
      tester.AddOptionalInputEdge<int32_t>();
    } else {
      tester.AddInput<int32_t>("segment_ids", segment_ids_dims, data.segment_ids_data);
    }
    if (use_float16) {
      tester.AddInput<MLFloat16>("word_embedding",
                                 word_embedding_dims,
                                 ToFloat16(data.word_embedding_data),
                                 /*is_initializer=*/true);
      tester.AddInput<MLFloat16>("position_embedding",
                                 position_embedding_dims,
                                 ToFloat16(data.position_embedding_data),
                                 /*is_initializer=*/true);
      if (!data.has_segment) {
        tester.AddOptionalInputEdge<MLFloat16>();
      } else {
        tester.AddInput<MLFloat16>("segment_embedding",
                                   segment_embedding_dims,
                                   ToFloat16(data.segment_embedding_data),
                                   /*is_initializer=*/true);
      }
      tester.AddInput<MLFloat16>("gamma",
                                 gamma_dims,
                                 ToFloat16(data.gamma_data),
                                 /*is_initializer=*/true);
      tester.AddInput<MLFloat16>("beta",
                                 beta_dims,
                                 ToFloat16(data.beta_data),
                                 /*is_initializer=*/true);
      tester.AddAttribute("epsilon", data.epsilon);
      if (data.has_mask && data.mask_data.size()) {
        tester.AddInput<int32_t>("mask", mask_dims, data.mask_data);
      }
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(data.output_data));
    } else {
      tester.AddInput<float>("word_embedding",
                             word_embedding_dims,
                             data.word_embedding_data,
                             /*is_initializer=*/true);
      tester.AddInput<float>("position_embedding",
                             position_embedding_dims,
                             data.position_embedding_data,
                             /*is_initializer=*/true);
      if (!data.has_segment) {
        tester.AddOptionalInputEdge<MLFloat16>();
      } else {
        tester.AddInput<float>("segment_embedding",
                               segment_embedding_dims,
                               data.segment_embedding_data,
                               /*is_initializer=*/true);
      }
      tester.AddInput<float>("gamma", gamma_dims, data.gamma_data, /*is_initializer=*/true);
      tester.AddInput<float>("beta", beta_dims, data.beta_data, /*is_initializer=*/true);
      tester.AddAttribute("epsilon", data.epsilon);
      if (data.has_mask && data.mask_data.size()) {
        tester.AddInput<int32_t>("mask", mask_dims, data.mask_data);
      }
      tester.AddOutput<float>("output", output_dims, data.output_data);
    }
    tester.AddAttribute("mask_index_type", static_cast<int64_t>(data.mask_index_type));
    if (data.mask_index_data.size()) {
      tester.AddOutput<int32_t>("mask_index", mask_index_dims, data.mask_index_data);
    } else {
      tester.AddOptionalOutputEdge<int32_t>();
    }
    if (sum_output) {
      std::vector<int64_t> embedding_sum_output_dims = output_dims;
      if (use_float16) {
        tester.AddOutput<MLFloat16>("embedding_sum", embedding_sum_output_dims, ToFloat16(data.embedding_sum_data));
      } else {
        tester.AddOutput<float>("embedding_sum", embedding_sum_output_dims, data.embedding_sum_data);
      }
    }
    if (data.position_ids_data.size() != 0) {
      if (broadcast_position_ids) {
        std::vector<int64_t> position_ids_dims = {1, data.sequence_size};
        tester.AddInput<int32_t>("position_ids", position_ids_dims, data.position_ids_data);
      } else {
        tester.AddInput<int32_t>("position_ids", input_ids_dims, data.position_ids_data);
      }
    }

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    } else if (enable_dml) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultDmlExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    } else {
      if (sum_output) {
        // OpenVINO EP only supports 2 outputs for EmbedLayerNormalization, skip when 3rd output (embedding_sum) is requested
        tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
      } else {
        tester.Run();
      }
    }
  }
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1());
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_Float16) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1(), /*use_float16=*/true);
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_PositionIds) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1_PositionIds());
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_PositionIdsDiffOrder) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1_PositionIds(true));
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch3_PositionIds_BroadCast) {
  RunTest(embedlayernorm::EmbedLayerNormBatch3_PositionIds_BroadCast(),
          /*use_float16=*/false,
          /*sum_output=*/false,
          /*broadcast_position_ids=*/true);
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_EmbeddingSum) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1_EmbeddingSum(), false, true);
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_EmbeddingSum_Float16) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1_EmbeddingSum(), true, true);
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_EmbeddingSum_NoMaskIndex) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1_EmbeddingSum_NoMaskIndex(),
          /* use_float16 = */ false,
          /* sum_output = */ true);
}

// Regression test: shape inference with mask_index_type=0 and 2 outputs (no embedding_sum)
// must not write to output index 2, which would be out-of-bounds.
// Before the fix in EmbedLayerNormalizationShapeInference (shape_inference_functions.cc),
// this case triggered a heap OOB write when getNumOutputs()==2 and mask_index_type==0.
TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_NoMaskIndex_NoSumOutput) {
  RunTest(embedlayernorm::EmbedLayerNormBatch1_EmbeddingSum_NoMaskIndex(),
          /* use_float16 = */ false,
          /* sum_output = */ false);
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

// Regression test: negative position_ids must be rejected to not cause OOB read.
TEST(EmbedLayerNormTest, EmbedLayerNormNegativePositionIds) {
  int batch_size = 1;
  int sequence_size = 2;
  int hidden_size = 4;

  std::vector<int64_t> input_ids_dims = {batch_size, sequence_size};
  std::vector<int64_t> position_ids_dims = {batch_size, sequence_size};

  // 6 words (rows) x hidden_size=4
  std::vector<int64_t> word_embedding_dims = {6, hidden_size};
  // 3 positions x hidden_size=4
  std::vector<int64_t> position_embedding_dims = {3, hidden_size};
  // 2 segments x hidden_size=4
  std::vector<int64_t> segment_embedding_dims = {2, hidden_size};

  std::vector<int64_t> gamma_dims = {hidden_size};
  std::vector<int64_t> beta_dims = {hidden_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_size, hidden_size};
  std::vector<int64_t> mask_index_dims = {batch_size};

  OpTester tester("EmbedLayerNormalization", 1, onnxruntime::kMSDomain);
  tester.AddInput<int32_t>("input_ids", input_ids_dims, {1, 3});
  tester.AddInput<int32_t>("segment_ids", input_ids_dims, {0, 1});
  tester.AddInput<float>("word_embedding", word_embedding_dims,
                         {0.2f, 0.1f, 0.4f, -0.6f,
                          0.3f, 0.2f, 0.5f, 0.6f,
                          0.6f, 0.7f, 0.0f, -0.1f,
                          0.8f, 0.6f, 0.9f, 1.2f,
                          0.1f, 0.3f, 0.5f, 0.9f,
                          1.0f, -2.0f, 1.1f, 0.8f},
                         /*is_initializer=*/true);
  tester.AddInput<float>("position_embedding", position_embedding_dims,
                         {0.1f, 0.1f, 0.4f, 0.6f,
                          0.6f, 0.0f, 0.8f, 0.6f,
                          0.3f, 0.9f, -2.0f, 0.8f},
                         /*is_initializer=*/true);
  tester.AddInput<float>("segment_embedding", segment_embedding_dims,
                         {0.3f, 0.4f, 0.9f, 0.1f,
                          0.7f, 0.3f, 0.5f, 0.2f},
                         /*is_initializer=*/true);
  tester.AddInput<float>("gamma", gamma_dims, {0.25f, 0.15f, 0.45f, -0.66f},
                         /*is_initializer=*/true);
  tester.AddInput<float>("beta", beta_dims, {0.6f, 0.2f, 0.5f, -0.6f},
                         /*is_initializer=*/true);
  tester.AddAttribute("epsilon", embedlayernorm::kEpsilon);
  // Skip mask (input 7) - add optional edge
  tester.AddOptionalInputEdge<int32_t>();
  // Negative position_ids - this is the malicious input
  tester.AddInput<int32_t>("position_ids", position_ids_dims, {-5, 1});

  // Dummy outputs (won't be checked since we expect failure)
  tester.AddOutput<float>("output", output_dims, std::vector<float>(batch_size * sequence_size * hidden_size, 0.0f));
  tester.AddOutput<int32_t>("mask_index", mask_index_dims, {0});

  // Run CPU only - expect failure due to negative position_ids
  tester.Run(OpTester::ExpectResult::kExpectFailure, "", {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kDmlExecutionProvider, kOpenVINOExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
