// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#ifdef USE_CUDA
namespace {

int Parity(size_t value) {
  int parity = 0;
  while (value != 0) {
    parity ^= 1;
    value &= value - 1;
  }
  return parity;
}

std::vector<MLFloat16> DenseReference(
    const std::vector<MLFloat16>& input,
    const std::vector<MLFloat16>& sign,
    int64_t last_dimension,
    int64_t block_size) {
  std::vector<MLFloat16> output(input.size());
  const float normalization = 1.0f / std::sqrt(static_cast<float>(block_size));
  const int64_t row_count = static_cast<int64_t>(input.size()) / last_dimension;

  for (int64_t row = 0; row < row_count; ++row) {
    const int64_t row_offset = row * last_dimension;
    for (int64_t block_offset = 0; block_offset < last_dimension; block_offset += block_size) {
      for (int64_t column = 0; column < block_size; ++column) {
        float sum = 0.0f;
        for (int64_t index = 0; index < block_size; ++index) {
          const float hadamard = Parity(static_cast<size_t>(index & column)) == 0 ? 1.0f : -1.0f;
          sum += input[static_cast<size_t>(row_offset + block_offset + index)].ToFloat() *
                 sign[static_cast<size_t>(block_offset + index)].ToFloat() * hadamard;
        }
        output[static_cast<size_t>(row_offset + block_offset + column)] = MLFloat16(sum * normalization);
      }
    }
  }
  return output;
}

void RunTest(const std::vector<int64_t>& shape, int64_t block_size, bool use_default_attribute) {
  if (!HasCudaEnvironment(900)) {
    GTEST_SKIP() << "FusedHadamardTransform test requires CUDA SM90 or newer.";
  }

  int64_t element_count = 1;
  for (int64_t dimension : shape) {
    element_count *= dimension;
  }
  const int64_t last_dimension = shape.back();

  std::vector<MLFloat16> input(static_cast<size_t>(element_count));
  for (int64_t index = 0; index < element_count; ++index) {
    const float value = static_cast<float>((index * 37 + 11) % 257 - 128) / 128.0f;
    input[static_cast<size_t>(index)] = MLFloat16(value);
  }

  std::vector<MLFloat16> sign(static_cast<size_t>(last_dimension));
  for (int64_t index = 0; index < last_dimension; ++index) {
    sign[static_cast<size_t>(index)] = MLFloat16(((index * 13 + 5) % 7) < 3 ? -1.0f : 1.0f);
  }

  OpTester tester("FusedHadamardTransform", 1, kMSDomain);
  if (!use_default_attribute) {
    tester.AddAttribute<int64_t>("block_size", block_size);
  }
  tester.AddInput<MLFloat16>("X", shape, input);
  tester.AddInput<MLFloat16>("sign", {last_dimension}, sign);
  tester.AddOutput<MLFloat16>("Y", shape, DenseReference(input, sign, last_dimension, block_size));
  tester.SetOutputAbsErr("Y", 0.003f);
  tester.SetOutputRelErr("Y", 0.003f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace

TEST(FusedHadamardTransformTest, SmallDenseReference) {
  RunTest({2, 16}, 8, false);
}

TEST(FusedHadamardTransformTest, Sm90Float16Width5120MultipleRows) {
  RunTest({3, 5120}, 1024, true);
}
#endif

}  // namespace test
}  // namespace onnxruntime