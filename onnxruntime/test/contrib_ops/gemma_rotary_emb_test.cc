// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include "gtest/gtest.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include <cstdlib>  // For rand() and srand()

namespace onnxruntime {
namespace test {

constexpr auto k_random_data_min = -1.0f;
constexpr auto k_random_data_max = 1.0f;

namespace {
enum class TensorType {
  kFloat,
  kFloat16,
  kBFloat16
};
}  // anonymous namespace

static void calculateExpectedOutput(const std::vector<float>& emb_data,
                                    const std::vector<MLFloat16>& q_data,
                                    const std::vector<MLFloat16>& q_rot_data,
                                    const std::vector<MLFloat16>& k_data,
                                    const std::vector<MLFloat16>& k_rot_data,
                                    const std::vector<int64_t>& mul_dim,
                                    std::vector<MLFloat16>& output1,
                                    std::vector<MLFloat16>& output2) {
  for (int64_t i = 0; i < mul_dim[0]; ++i) {
    for (int64_t j = 0; j < mul_dim[1]; ++j) {
      for (int64_t k = 0; k < mul_dim[2]; ++k) {
        for (int64_t l = 0; l < mul_dim[3]; ++l) {
          int64_t embIdx = i * mul_dim[1] * mul_dim[3] + k * mul_dim[3] + l;
          int64_t mulIdx = i * mul_dim[1] * mul_dim[2] * mul_dim[3] + j * mul_dim[2] * mul_dim[3] + k * mul_dim[3] + l;

          MLFloat16 sin_val = static_cast<MLFloat16>(sin(emb_data[embIdx]));
          MLFloat16 cos_val = static_cast<MLFloat16>(cos(emb_data[embIdx]));
          MLFloat16 q_val = static_cast<MLFloat16>(q_data[mulIdx]);
          MLFloat16 q_rot_val = static_cast<MLFloat16>(q_rot_data[mulIdx]);
          MLFloat16 k_val = static_cast<MLFloat16>(k_data[mulIdx]);
          MLFloat16 k_rot_val = static_cast<MLFloat16>(k_rot_data[mulIdx]);
          output1.push_back(static_cast<MLFloat16>(q_val * cos_val + q_rot_val * sin_val));
          output2.push_back(static_cast<MLFloat16>(k_val * cos_val + k_rot_val * sin_val));
        }
      }
    }
  }
}

static void RunTest() {
  std::string op_type = "GemmaRotaryEmbedding";
  std::vector<int64_t> emb_dim = {1, 2, 2};
  std::vector<int64_t> mul_dim = {1, 3, 2, 2};
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;

  int min_cuda_architecture = 530;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);

  if (enable_cuda) {
    execution_providers.push_back(DefaultCudaExecutionProvider());
  }

  if (execution_providers.size() == 0) {
    // Return early if CI pipeline does not support EP (e.g. CUDA EP for CPU CI pipeline)
    return;
  }

  OpTester test(op_type.c_str(), 1, onnxruntime::kMSDomain);

  // create rand inputs
  RandomValueGenerator random{};
  const std::vector<float> emb_data = random.Uniform<float>(emb_dim, k_random_data_min, k_random_data_max);
  const std::vector<MLFloat16> q = random.Uniform<MLFloat16>(mul_dim, k_random_data_min, k_random_data_max);
  const std::vector<MLFloat16> q_rot = random.Uniform<MLFloat16>(mul_dim, k_random_data_min, k_random_data_max);
  const std::vector<MLFloat16> k = random.Uniform<MLFloat16>(mul_dim, k_random_data_min, k_random_data_max);
  const std::vector<MLFloat16> k_rot = random.Uniform<MLFloat16>(mul_dim, k_random_data_min, k_random_data_max);

  std::vector<MLFloat16> output1;
  std::vector<MLFloat16> output2;

  calculateExpectedOutput(emb_data, q, q_rot, k, k_rot, mul_dim, output1, output2);

  test.AddInput<float>("emb", emb_dim, emb_data);
  test.AddInput<MLFloat16>("q_data", mul_dim, q);
  test.AddInput<MLFloat16>("q_rot_data", mul_dim, q_rot);
  test.AddInput<MLFloat16>("k_data", mul_dim, k);
  test.AddInput<MLFloat16>("k_rot_data", mul_dim, k_rot);
  test.AddOutput<MLFloat16>("output1", mul_dim, output1);
  test.AddOutput<MLFloat16>("output2", mul_dim, output2);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(GemmaRotaryEmbeddingTest, GemmaRotaryEmbedding_Small) {
  RunTest();
}

}  // namespace test
}  // namespace onnxruntime